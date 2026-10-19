"""Sliding windows and the two sequence models, on tiny synthetic data so the tests stay fast."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from anomaly.models.base import AnomalyModel
from anomaly.models.deep import LSTMAutoencoderModel, TransformerForecastModel
from anomaly.models.preprocess import FeatureScaler
from anomaly.models.windows import make_windows

FEATURES = ["a", "b"]


def _panel(n: int = 300, tickers=("AAA", "BBB"), seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2020-01-01", periods=n)
    frames = [
        pd.DataFrame(
            {
                "ticker": t,
                "timestamp": days,
                "a": np.sin(np.arange(n) / 5) + rng.normal(0, 0.1, n),  # smooth, so it's learnable
                "b": rng.normal(0, 1, n),
            }
        )
        for t in tickers
    ]
    return pd.concat(frames, ignore_index=True)


def _fitted_scaler(df: pd.DataFrame) -> FeatureScaler:
    return FeatureScaler("global").fit(df, FEATURES)


# ---------- windows ----------


def test_windows_end_at_the_scored_bar_and_never_cross_tickers():
    df = _panel(n=10).sample(frac=1, random_state=0)  # shuffled input order
    scaler = _fitted_scaler(df)
    x, rows = make_windows(df, FEATURES, scaler, length=4)
    assert x.shape == (2 * (10 - 3), 4, 2)
    flat = df.reset_index(drop=True)
    z = scaler.transform(flat)
    for window, row in zip(x, rows):
        ticker, ts = flat.loc[row, ["ticker", "timestamp"]]
        history = flat[(flat["ticker"] == ticker) & (flat["timestamp"] <= ts)].sort_values("timestamp").tail(4)
        np.testing.assert_allclose(window, z[history.index.to_numpy()])  # same ticker, past bars only, ending at row


def test_windows_with_missing_values_are_skipped():
    df = _panel(n=10, tickers=("AAA",))
    df.loc[5, "a"] = np.nan
    x, rows = make_windows(df, FEATURES, _fitted_scaler(df.dropna()), length=3)
    assert not np.isnan(x).any()
    assert sorted(rows.tolist()) == [2, 3, 4, 8, 9]  # windows ending at 5, 6, 7 contain the gap


def test_short_tickers_get_no_windows():
    df = _panel(n=3, tickers=("AAA",))
    x, rows = make_windows(df, FEATURES, _fitted_scaler(df), length=5)
    assert x.shape == (0, 5, 2) and len(rows) == 0


# ---------- models ----------

SMALL = dict(window=8, batch_size=64, max_epochs=3, patience=2, learning_rate=5e-3, holdout_frac=0.2, torch_threads=1)


@pytest.mark.parametrize(
    "cls, arch",
    [(LSTMAutoencoderModel, {"hidden_size": 8}), (TransformerForecastModel, {"d_model": 8, "n_heads": 2, "n_layers": 1, "ff_dim": 16})],
)
def test_model_trains_scores_and_reproduces(cls, arch, tmp_path):
    df = _panel()
    train = df[df["timestamp"] < df["timestamp"].quantile(0.7)]
    model = cls(FEATURES, **arch, seed=0, **SMALL).fit(train)

    assert 1 <= len(model.history) <= SMALL["max_epochs"] and 1 <= model.best_epoch <= len(model.history)
    assert model.history[-1]["train_loss"] < model.history[0]["train_loss"] * 1.5
    assert model.train_windows > 0 and model.holdout_windows > 0

    scores = model.score(df)
    first_bars = df.groupby("ticker").head(SMALL["window"] - 1).index
    assert np.isnan(scores[first_bars]).all()  # not enough history yet
    assert np.isfinite(np.delete(scores, first_bars)).all()

    again = cls(FEATURES, **arch, seed=0, **SMALL).fit(train).score(df)
    np.testing.assert_allclose(again, scores, rtol=1e-5)  # same seed, same scores

    model.save(tmp_path / "m.joblib")
    np.testing.assert_allclose(AnomalyModel.load(tmp_path / "m.joblib").score(df), scores, rtol=1e-6)


@pytest.mark.parametrize(
    "cls, arch",
    [(LSTMAutoencoderModel, {"hidden_size": 8}), (TransformerForecastModel, {"d_model": 8, "n_heads": 2, "n_layers": 1, "ff_dim": 16})],
)
def test_error_keeps_growing_with_the_size_of_the_anomaly(cls, arch):
    """Unlike Isolation Forest (see test_models.py), a bigger anomaly always scores higher."""
    df = _panel()
    model = cls(FEATURES, **arch, seed=0, **SMALL).fit(df)
    last = df.index[df["ticker"] == "AAA"][-1]
    scores = []
    for size in (0.0, 4.0, 12.0, 100.0):
        shocked = df.copy()
        shocked.loc[last, "b"] = size
        scores.append(model.score(shocked)[last])
    assert scores[0] < scores[1] < scores[2] < scores[3]


def test_copy_slope_measures_how_much_of_a_big_input_is_rebuilt():
    df = _panel(n=400)
    df["split"] = np.where(df["timestamp"] < df["timestamp"].quantile(0.5), "train", "val")
    rng = np.random.default_rng(1)
    spikes = rng.choice(df.index, 40, replace=False)
    df.loc[spikes, "b"] = rng.choice([-6.0, 6.0], 40)  # big values so the fit has enough points
    model = LSTMAutoencoderModel(FEATURES, hidden_size=8, seed=0, **SMALL).fit(df[df["split"] == "train"])
    slope = model.copy_slope(df, feature="b", min_abs=3.0)
    assert np.isfinite(slope) and -1.5 < slope < 1.5
    assert np.isfinite(model.copy_slope(df, feature="b", min_abs=3.0, split="val"))
    assert np.isnan(model.copy_slope(df, feature="b", min_abs=1e9))  # nothing that large: no slope


def test_scoring_before_fit_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        LSTMAutoencoderModel(FEATURES, **SMALL).score(_panel())
