"""Score normalization, weight search, the ensemble model, the rule wrapper, and the README table generator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from anomaly.evaluation.baselines import zscore_rule_scores
from anomaly.models.base import AnomalyModel
from anomaly.models.ensemble import EnsembleModel, ScoreNormalizer, combine, weight_grid
from anomaly.models.rules import ZScoreRuleModel
from anomaly.reporting import END, START, replace_block, results_markdown


# ---------- normalization ----------


def test_robust_z_uses_median_and_mad_and_keeps_extremes():
    train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan])
    nz = ScoreNormalizer("robust_z").fit(train)
    assert nz.center == 3.0 and nz.scale == pytest.approx(1.4826)
    out = nz.transform(np.array([3.0, 1000.0, np.nan]))
    assert out[0] == 0.0 and out[1] > 600 and np.isnan(out[2])  # no saturation


def test_rank_is_bounded_and_saturates():
    nz = ScoreNormalizer("rank").fit(np.arange(100.0))
    out = nz.transform(np.array([-5.0, 49.0, 99.0, 1e9, np.nan]))
    assert out[0] == 0.0 and out[1] == pytest.approx(0.5) and out[2] == 1.0 and out[3] == 1.0
    assert np.isnan(out[4])


def test_normalizer_only_sees_what_it_was_fit_on():
    nz = ScoreNormalizer("robust_z").fit(np.array([0.0, 1.0, 2.0]))
    before = nz.transform(np.array([5.0]))
    nz.transform(np.array([1e6, -1e6]))  # transforming other data doesn't change the fit
    assert nz.transform(np.array([5.0])) == before


def test_constant_training_scores_do_not_divide_by_zero():
    nz = ScoreNormalizer("robust_z").fit(np.ones(10))
    assert np.isfinite(nz.transform(np.array([2.0]))).all()


# ---------- weights and combining ----------


def test_weight_grid_covers_the_simplex():
    grid = weight_grid(4, 0.1)
    assert len(grid) == 286  # (10 + 3 choose 3)
    assert all(abs(sum(w) - 1) < 1e-9 for w in grid)
    assert (1.0, 0.0, 0.0, 0.0) in grid and (0.0, 0.0, 0.0, 1.0) in grid
    assert len(weight_grid(2, 0.25)) == 5


def test_combine_rescales_weights_over_members_that_scored():
    norm = np.array([[1.0, 3.0], [2.0, np.nan], [np.nan, np.nan]])
    out = combine(norm, np.array([0.5, 0.5]))
    assert out[0] == 2.0
    assert out[1] == 2.0  # only the first member scored, so it gets all the weight
    assert np.isnan(out[2])
    assert combine(np.array([[np.nan, 4.0]]), np.array([0.0, 1.0]))[0] == 4.0  # a zero-weight member's NaN is ignored
    assert np.isnan(combine(np.array([[4.0, np.nan]]), np.array([0.0, 1.0]))[0])  # only a zero-weight member scored


# ---------- ensemble model ----------


class _Const(AnomalyModel):
    """A stand-in member whose score is a column of the input."""

    def __init__(self, col: str):
        self.col = col
        self.name = col

    def fit(self, train):
        return self

    def score(self, df):
        return df[self.col].to_numpy(dtype=float)


def test_ensemble_model_matches_manual_combination():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"ticker": "A", "timestamp": pd.bdate_range("2024-01-01", periods=200), "a": rng.normal(size=200), "b": rng.exponential(size=200) * 10})
    train, test = df.iloc[:150], df.iloc[150:]
    model = EnsembleModel({"a": _Const("a"), "b": _Const("b")}, {"a": 0.7, "b": 0.3}, "robust_z").fit(train)
    na = ScoreNormalizer("robust_z").fit(train["a"].to_numpy())
    nb = ScoreNormalizer("robust_z").fit(train["b"].to_numpy())
    expected = 0.7 * na.transform(test["a"].to_numpy()) + 0.3 * nb.transform(test["b"].to_numpy())
    np.testing.assert_allclose(model.score(test), expected)


def test_ensemble_rejects_mismatched_weights_and_unfitted_use():
    with pytest.raises(ValueError, match="weights"):
        EnsembleModel({"a": _Const("a")}, {"b": 1.0})
    with pytest.raises(RuntimeError, match="before fit"):
        EnsembleModel({"a": _Const("a")}, {"a": 1.0}).combine_raw(np.ones((2, 1)))


# ---------- rule wrapper ----------


def test_rule_model_matches_the_baseline_in_any_row_order():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2024-01-01", periods=30)
    df = pd.concat(
        [
            pd.DataFrame({"ticker": t, "timestamp": days, "log_ret": rng.normal(0, 0.01, 30), "gap": rng.normal(0, 0.005, 30),
                          "volume_z": rng.normal(size=30), "vol_20": 0.01})
            for t in ("A", "B")
        ],
        ignore_index=True,
    )
    expected = zscore_rule_scores(df)["score"].to_numpy()
    shuffled = df.sample(frac=1, random_state=1)
    got = ZScoreRuleModel().score(shuffled)
    np.testing.assert_allclose(got, expected[shuffled.index.to_numpy()], equal_nan=True)


# ---------- README table ----------


def test_results_table_is_generated_from_the_summary():
    rows = []
    for model, pr in (("baseline_zscore_rule", 0.3), ("ensemble", 0.32), ("not_listed", 0.9)):
        for track in ("synthetic", "real"):
            rows.append({"model": model, "split": "test", "track": track, "pr_auc": pr, "event_recall": 0.5,
                         "false_alarms_per_day_all_tickers": 0.5, "false_alarms_per_ticker_month": 0.1})
    rows.append({**rows[0], "split": "val", "pr_auc": 0.99})
    table = results_markdown(pd.DataFrame(rows))
    assert "| Z-score rule | 0.300 |" in table and "0.320" in table
    assert "0.990" not in table and "not_listed" not in table  # val rows and unknown models are left out
    text = f"intro\n{START}\nold table\n{END}\noutro"
    assert replace_block(text, "NEW") == f"intro\n{START}\nNEW\n{END}\noutro"
    with pytest.raises(ValueError, match="markers"):
        replace_block("no markers here", "NEW")
