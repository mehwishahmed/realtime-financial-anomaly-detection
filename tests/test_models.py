"""Model inputs (derived features, train-only scaling) and the Isolation Forest wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from anomaly.config import ConfigError, load_config
from anomaly.models.base import AnomalyModel
from anomaly.models.isolation_forest import IsolationForestModel
from anomaly.models.preprocess import FeatureScaler, add_derived_features

from conftest import REPO_ROOT

FEATURES = ["a", "b"]


def _panel(n: int = 400, tickers=("AAA", "BBB"), seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2020-01-01", periods=n)
    frames = []
    for i, t in enumerate(tickers):
        frames.append(
            pd.DataFrame(
                {
                    "ticker": t,
                    "timestamp": days,
                    "a": rng.normal(10 * i, 1 + i, n),  # each ticker has its own level and spread
                    "b": rng.normal(0, 1, n),
                    "split": ["train"] * (n // 2) + ["test"] * (n - n // 2),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# ---------- derived features ----------


def test_derived_features_use_previous_volatility():
    df = pd.DataFrame(
        {
            "ticker": ["A"] * 4,
            "timestamp": pd.bdate_range("2024-01-01", periods=4),
            "log_ret": [0.01, 0.02, 0.10, 0.0],
            "gap": [0.0, 0.01, 0.05, 0.0],
            "vol_20": [0.01, 0.01, 0.05, 0.05],  # the spike day's own vol is inflated
        }
    )
    out = add_derived_features(df.iloc[::-1])  # row order in, row order out
    assert out.index.tolist() == [3, 2, 1, 0]
    out = out.sort_index()
    assert np.isnan(out.loc[0, "ret_z"])
    assert out.loc[2, "ret_z"] == pytest.approx(10.0)  # 0.10 / yesterday's 0.01, not today's 0.05
    assert out.loc[2, "gap_z"] == pytest.approx(5.0)


# ---------- scaler ----------


def test_per_ticker_scaling_uses_train_statistics_only():
    df = _panel()
    train = df[df["split"] == "train"]
    scaler = FeatureScaler("per_ticker", min_rows=50).fit(train, FEATURES)
    z = scaler.transform(train)
    for t in ("AAA", "BBB"):
        rows = (train["ticker"] == t).to_numpy()
        np.testing.assert_allclose(z[rows].mean(axis=0), 0, atol=1e-5)
        np.testing.assert_allclose(z[rows].std(axis=0, ddof=1), 1, atol=1e-4)

    shocked = df.copy()
    shocked.loc[shocked["split"] == "test", "a"] += 1000  # wild test data
    refit = FeatureScaler("per_ticker", min_rows=50).fit(shocked[shocked["split"] == "train"], FEATURES)
    np.testing.assert_allclose(refit.transform(train), z)  # test rows can't influence the statistics


def test_short_history_and_unseen_tickers_fall_back_to_global():
    df = _panel(tickers=("AAA", "BBB", "NEW"))
    train = df[(df["split"] == "train") & ~((df["ticker"] == "NEW") & (df.index % 400 > 20))]
    scaler = FeatureScaler("per_ticker", min_rows=50).fit(train, FEATURES)
    unseen = df[df["ticker"] == "AAA"].head(3).assign(ticker="ZZZ")
    assert scaler.fallback_tickers(pd.concat([df, unseen])) == ["NEW", "ZZZ"]
    expected = (unseen[FEATURES] - scaler.global_mean) / scaler.global_std
    np.testing.assert_allclose(scaler.transform(unseen), expected.to_numpy(), rtol=1e-5)


def test_scaler_handles_nan_and_constant_columns():
    df = _panel()
    df["b"] = 5.0
    scaler = FeatureScaler("global").fit(df, FEATURES)
    row = df.head(1).copy()
    row.loc[row.index[0], "a"] = np.nan
    z = scaler.transform(row)
    assert np.isnan(z[0, 0]) and z[0, 1] == 0.0  # constant column: no divide-by-zero


# ---------- Isolation Forest ----------


def _model(seed: int = 0) -> IsolationForestModel:
    return IsolationForestModel(FEATURES, n_estimators=50, max_samples=128, min_train_rows_per_ticker=50, seed=seed, n_jobs=1)


def test_outliers_score_higher_and_runs_are_reproducible():
    df = _panel()
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"].copy()
    test.iloc[5, [test.columns.get_loc("a"), test.columns.get_loc("b")]] = [8.0, 8.0]  # unusual on both features
    m1 = _model().fit(train)
    s1 = m1.score(test)
    assert s1[5] == s1.max()
    np.testing.assert_array_equal(s1, _model().fit(train).score(test))  # same seed, same scores
    assert not np.array_equal(s1, _model(seed=1).fit(train).score(test))


def test_scores_saturate_beyond_the_training_range():
    """Documents a known Isolation Forest limitation (seen in Phase 3 as weak volume-burst recall).

    Trees can only split inside the range of values seen in training, so any
    value past that range ends up in the same leaf as the most extreme
    training point: a 100-sigma move scores the same as a 4-sigma one.
    """
    df = _panel()
    train = df[df["split"] == "train"]
    model = _model().fit(train)
    row = df[df["split"] == "test"].head(1).copy()
    scores = {b: model.score(row.assign(b=b))[0] for b in (0.0, 4.0, 12.0, 100.0)}
    assert scores[4.0] > scores[0.0]
    assert scores[4.0] == scores[12.0] == scores[100.0]


def test_missing_inputs_get_no_score_and_save_load_round_trips(tmp_path):
    df = _panel()
    model = _model().fit(df[df["split"] == "train"])
    test = df[df["split"] == "test"].copy()
    test.iloc[0, test.columns.get_loc("a")] = np.nan
    frame = model.score_frame(test)
    assert list(frame.columns) == ["ticker", "timestamp", "score"] and len(frame) == len(test)
    assert np.isnan(frame.loc[0, "score"]) and frame["score"].iloc[1:].notna().all()

    model.save(tmp_path / "m.joblib")
    loaded = AnomalyModel.load(tmp_path / "m.joblib")
    np.testing.assert_array_equal(loaded.score(test), model.score(test))


def test_model_config_validation(tmp_path):
    raw = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text())
    raw["models"]["isolation_forest"]["grid"]["feature_set"] = ["base", "made_up"]
    path = REPO_ROOT / "configs" / f"_test_{tmp_path.name}.yaml"
    path.write_text(yaml.safe_dump(raw))
    try:
        with pytest.raises(ConfigError, match="made_up"):
            load_config(path)
    finally:
        path.unlink()
