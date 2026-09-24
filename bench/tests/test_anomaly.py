"""The anomaly benchmark: its units, its detectors, and that the published evidence is the committed run."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dcn import anomaly, anomaly_learn

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "bench" / "results" / "anomaly.csv"


def test_tables_cut_from_one_source_are_one_unit():
    units = {anomaly.unit_of(n) for n in anomaly.NAMES}
    assert len(anomaly.NAMES) == 47 and len(units) == 39
    assert anomaly.unit_of("42_WBC") == anomaly.unit_of("43_WDBC") == "wisconsin_breast"
    assert anomaly.unit_of("16_http") == anomaly.unit_of("34_smtp") == "kddcup99"


def test_the_width_cut_offs_match_the_page():
    assert [anomaly.kind_of(c) for c in (3, 10, 11, 50, 51, 500)] == ["low", "low", "mid", "mid", "high", "high"]
    page = (ROOT / "site" / "js" / "anomalies.js").read_text()
    assert f"LOW_COLUMNS = {anomaly.LOW_COLUMNS};" in page and f"HIGH_COLUMNS = {anomaly.HIGH_COLUMNS};" in page


@pytest.mark.parametrize("code", list(anomaly.DETECTORS))
def test_every_detector_scores_an_obvious_outlier_highest(code):
    rng = np.random.default_rng(0)
    X = np.r_[rng.normal(0, 1, (200, 4)), [[9.0, 9.0, 9.0, 9.0]]]
    score = anomaly.DETECTORS[code][1](X)
    assert len(score) == 201
    if code == "CL2":   # DBSCAN only says noise or not
        assert score[-1] == 1.0
    else:
        assert np.argmax(score) == 200, code


def test_robust_covariance_refuses_too_few_rows():
    with pytest.raises(ValueError, match="twice as many rows"):
        anomaly.DETECTORS["PR6"][1](np.random.default_rng(0).normal(size=(30, 20)))


@pytest.mark.skipif(not RESULTS.exists(), reason="the anomaly run has not been committed")
def test_the_published_evidence_is_the_committed_run(tmp_path):
    out = tmp_path / "anomaly.json"
    anomaly_learn.main(["--results", str(RESULTS), "--out", str(out), "--report", str(tmp_path / "lodo.csv")])
    assert json.loads(out.read_text()) == json.loads((ROOT / "site" / "taxonomy" / "anomaly.json").read_text())
