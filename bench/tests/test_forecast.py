"""The forecasting benchmark: that it runs the page's own script, and that its rule decides what it says it does."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dcn import forecast, forecast_learn
from dcn.forecast_learn import choose

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "bench" / "results" / "forecast.csv"


def per_series(kind_gain: dict[str, float], units: int = 11, series: int = 20, seed: int = 0) -> pd.DataFrame:
    """Regret of both orders on every series, the kind order better by `kind_gain` per score."""
    rng = np.random.default_rng(seed)
    rows = []
    for metric, gain in kind_gain.items():
        for u in range(units):
            for s in range(series):
                fixed = abs(rng.normal(0.05, 0.02))
                for strategy, regret in (("fixed", fixed), ("kind", max(fixed - gain + rng.normal(0, 0.002), 0))):
                    rows.append({"metric": metric, "strategy": strategy, "dataset": f"c{u}", "unit": f"c{u}",
                                 "series": f"s{s}", "cell": "level", "pick": "TSM2", "regret": regret})
    return pd.DataFrame(rows)


def test_a_kind_order_better_on_one_score_and_level_on_the_other_replaces_the_fixed_one():
    decision = choose(per_series({"r2": 0.02, "mae": 0.0}))
    assert decision["replaced"] and decision["better_on"] == ["r2"]


def test_a_kind_order_worse_on_either_score_does_not():
    assert not choose(per_series({"r2": 0.02, "mae": -0.02}))["replaced"]


def test_level_everywhere_keeps_the_fixed_order():
    assert not choose(per_series({"r2": 0.0, "mae": 0.0}))["replaced"]


def test_the_benchmark_runs_the_script_the_page_writes():
    forecast._init()
    ns = forecast._NS
    assert ns["FORECAST"] is True
    assert list(ns["SHORTLIST"]) == forecast.CODES
    rng = np.random.default_rng(1)
    t = np.arange(60)
    values = (20 + 5 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.5, 60)).tolist()
    forecast.BY_NAME["test"] = forecast.Collection("test", "test", "monthly", "test", "a test series")
    try:
        rows = forecast.run_series(("test", "s1", values, False, set()))
    finally:
        del forecast.BY_NAME["test"]
    methods = [r["method"] for r in rows]
    assert methods == ["NAIVE-LAST", "NAIVE-AVERAGE", "NAIVE-SEASON", *forecast.CODES]
    assert rows[0]["period"] == 12 and rows[0]["season"] == 12, "monthly dates: a year of 12 rows"
    assert rows[0]["cell"] == "seasonal"
    assert all(r["status"] == "ok" for r in rows), rows
    # The same numbers as the script's own evaluate(), which the page runs.
    ns["SEASON"] = 12
    folds = ns["forecast_folds"](60)
    for row in rows:
        method = next(m for code, name, needs, m in ns["forecast_runs"](np.asarray(values)) if code == row["method"])
        scores = [ns["r2_score"](np.asarray(values)[test], method(np.asarray(values), train, test)) for train, test in folds]
        assert row["r2"] == pytest.approx(np.mean(scores)), row["method"]


@pytest.mark.skipif(not RESULTS.exists(), reason="the forecasting run has not been committed")
def test_the_published_evidence_is_the_committed_run(tmp_path):
    out = tmp_path / "forecast.json"
    forecast_learn.main(["--results", str(RESULTS), "--out", str(out), "--report", str(tmp_path / "lodo.csv")])
    assert json.loads(out.read_text()) == json.loads((ROOT / "site" / "taxonomy" / "forecast.json").read_text())
