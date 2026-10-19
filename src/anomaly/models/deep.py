"""Phase 4: sequence models on 20-bar windows (LSTM autoencoder and Transformer forecaster).

Both share one training loop:
  - inputs are scaled with train-split statistics only
  - the last ``holdout_frac`` of the train dates is held out for early stopping;
    the val split is never used for training or stopping, because it has injected anomalies
  - the weights from the best holdout epoch are kept
"""

from __future__ import annotations

import copy
import time
from abc import abstractmethod

import numpy as np
import pandas as pd
import torch
from torch import nn

from anomaly.models.base import AnomalyModel
from anomaly.models.preprocess import FeatureScaler
from anomaly.models.windows import make_windows


class WindowModel(AnomalyModel):
    """Shared fitting and scoring for models that read a window of bars and score its last bar."""

    def __init__(
        self,
        features: list[str],
        window: int = 20,
        scaling: str = "per_ticker",
        min_train_rows_per_ticker: int = 250,
        batch_size: int = 512,
        max_epochs: int = 30,
        patience: int = 4,
        learning_rate: float = 1e-3,
        holdout_frac: float = 0.1,
        seed: int = 42,
        torch_threads: int = 4,
    ):
        self.features = list(features)
        self.window = window
        self.scaler = FeatureScaler(scaling, min_train_rows_per_ticker)  # type: ignore[arg-type]
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.holdout_frac = holdout_frac
        self.seed = seed
        self.torch_threads = torch_threads
        self.net: nn.Module | None = None
        self.history: list[dict[str, float]] = []
        self.best_epoch = 0
        self.train_windows = 0
        self.holdout_windows = 0

    # ----- what each model defines -----

    @abstractmethod
    def build_network(self, n_features: int) -> nn.Module:
        """The PyTorch module for this model."""

    @abstractmethod
    def errors(self, net: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(training loss for the batch, anomaly score for each window's last bar)."""

    # ----- shared machinery -----

    def _setup_torch(self) -> None:
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.torch_threads)
        torch.use_deterministic_algorithms(True, warn_only=True)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters()) if self.net is not None else 0

    def _batch_errors(self, x: np.ndarray, train: bool = False) -> tuple[float, np.ndarray]:
        """Mean loss and per-window scores over ``x``, in batches, without updating weights."""
        assert self.net is not None
        self.net.eval()
        losses, scores = [], []
        with torch.no_grad():
            for i in range(0, len(x), self.batch_size * 4):
                xb = torch.from_numpy(x[i : i + self.batch_size * 4])
                loss, score = self.errors(self.net, xb)
                losses.append(float(loss) * len(xb))
                scores.append(score.numpy())
        return sum(losses) / max(len(x), 1), np.concatenate(scores) if scores else np.empty(0)

    def fit(self, train: pd.DataFrame) -> "WindowModel":
        self._setup_torch()
        self.scaler.fit(train.dropna(subset=self.features), self.features)
        x, rows = make_windows(train, self.features, self.scaler, self.window)
        last_day = train["timestamp"].to_numpy()[rows]
        days = np.sort(np.unique(last_day))
        cutoff = days[int(len(days) * (1 - self.holdout_frac)) - 1]
        fit_x, hold_x = x[last_day <= cutoff], x[last_day > cutoff]
        self.train_windows, self.holdout_windows = len(fit_x), len(hold_x)

        self.net = self.build_network(len(self.features))
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        gen = torch.Generator().manual_seed(self.seed)
        fit_t = torch.from_numpy(fit_x)
        best_loss, best_state, stale = float("inf"), None, 0
        self.history = []
        for epoch in range(1, self.max_epochs + 1):
            start = time.perf_counter()
            self.net.train()
            perm = torch.randperm(len(fit_t), generator=gen)
            total = 0.0
            for i in range(0, len(perm), self.batch_size):
                xb = fit_t[perm[i : i + self.batch_size]]
                loss, _ = self.errors(self.net, xb)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                total += loss.item() * len(xb)
            hold_loss, _ = self._batch_errors(hold_x)
            self.history.append(
                {
                    "epoch": epoch,
                    "train_loss": total / len(fit_t),
                    "holdout_loss": hold_loss,
                    "seconds": time.perf_counter() - start,
                }
            )
            if hold_loss < best_loss - 1e-6:
                best_loss, best_state, stale, self.best_epoch = hold_loss, copy.deepcopy(self.net.state_dict()), 0, epoch
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Score each bar that has a full window of history in ``df``.

        Pass the whole history (not just the val or test rows) so that the first
        bars of a split still get their earlier context.
        """
        if self.net is None:
            raise RuntimeError("score called before fit")
        torch.set_num_threads(self.torch_threads)
        out = np.full(len(df), np.nan)
        x, rows = make_windows(df, self.features, self.scaler, self.window)
        if len(x):
            _, scores = self._batch_errors(x)
            out[rows] = scores
        return out


# ---------- LSTM autoencoder ----------


class _LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)
        summary = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)  # the whole window squeezed into one vector
        decoded, _ = self.decoder(summary)
        return self.output(decoded)


class LSTMAutoencoderModel(WindowModel):
    """Compress the window into a small vector, rebuild it, and score how badly the latest bar was rebuilt.

    Trained on (mostly) normal history, it learns to rebuild normal patterns.
    An unusual latest bar can't pass through the bottleneck, so its
    reconstruction error is large. Unlike Isolation Forest, the error keeps
    growing with the size of the anomaly instead of saturating.
    """

    name = "lstm_autoencoder"

    def __init__(self, features: list[str], hidden_size: int = 32, num_layers: int = 1, **kwargs):
        super().__init__(features, **kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def build_network(self, n_features: int) -> nn.Module:
        return _LSTMAutoencoder(n_features, self.hidden_size, self.num_layers)

    def errors(self, net: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sq = (net(x) - x) ** 2
        return sq.mean(), sq[:, -1, :].mean(dim=1)  # train on the whole window, score the last bar


# ---------- Transformer forecaster ----------


class _TransformerForecaster(nn.Module):
    def __init__(self, n_features: int, context: int, d_model: int, n_heads: int, n_layers: int, ff_dim: int, dropout: float):
        super().__init__()
        self.input = nn.Linear(n_features, d_model)
        self.position = nn.Parameter(torch.randn(1, context, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ff_dim, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, n_features)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.input(context) + self.position)
        return self.head(h[:, -1])


class TransformerForecastModel(WindowModel):
    """Predict the latest bar's features from the bars before it, and score the prediction error.

    It's a forecaster rather than an autoencoder on purpose: a Transformer
    asked to rebuild its own input can learn to copy it through attention,
    which would make every bar look normal. A forecaster can't see the bar it
    scores, so a surprise shows up as error.
    """

    name = "transformer"

    def __init__(
        self,
        features: list[str],
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 64,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(features, **kwargs)
        self.arch = {"d_model": d_model, "n_heads": n_heads, "n_layers": n_layers, "ff_dim": ff_dim, "dropout": dropout}

    def build_network(self, n_features: int) -> nn.Module:
        return _TransformerForecaster(n_features, self.window - 1, **self.arch)

    def errors(self, net: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sq = (net(x[:, :-1]) - x[:, -1]) ** 2
        return sq.mean(), sq.mean(dim=1)
