"""The drift benchmark: scenarios that do what they say, checkers that behave, a fair summary."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from dcn import drift as D

ROOT = Path(__file__).resolve().parents[2]


def test_every_checker_in_the_taxonomy_is_measured_or_says_why_not():
    codes = [c["c"] for c in json.loads((ROOT / "site" / "taxonomy" / "drift.json").read_text())["checkers"]]
    measured = {c.code for c in D.checkers()}
    assert not measured & set(D.NOT_MEASURED)
    assert measured | set(D.NOT_MEASURED) == set(codes)


def test_the_committed_run_has_one_dataset_per_family_and_every_case():
    from dcn.significance import family_of
    results = pd.read_csv(ROOT / "bench" / "results" / "drift.csv")
    names = results.dataset.unique()
    assert len({family_of(n) for n in names}) == len(names)
    per_case = results.groupby(["dataset", "rep", "scenario"]).checker.nunique()
    assert (per_case == len(D.checkers())).all(), "every checker ran on every case"
    assert set(results.scenario) == set(D.SCENARIOS)


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(0)
    n = 4000
    a = rng.normal(size=n)
    X = np.column_stack([a, 0.9 * a + 0.3 * rng.normal(size=n), rng.normal(size=n), rng.integers(0, 3, n)])
    y = (X[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(int)
    y[rng.random(n) < 0.3] = 2           # a third class, so the most common class is clear
    split = D.split_rows(n, rng)
    split["feature"] = 0
    split["correlated"] = D.most_correlated(X[split["ref"]], [0, 1, 2])
    return X, y, split


def windows(world, scenario, seed=1):
    X, y, split = world
    return D.make_windows(X, y, scenario, np.random.default_rng(seed), split)


def test_no_drift_is_just_fresh_rows(world):
    X, y, split = world
    cur, labels, rows, feature = windows(world, "none")
    assert len(cur) == D.CUR_ROWS and feature is None
    assert set(rows) <= set(split["pool"])
    assert (cur == X[rows]).all() and (labels == y[rows]).all()


def test_shift_and_scale_touch_one_feature_by_the_stated_amount(world):
    X, y, split = world
    sd = X[split["ref"], 0].std()
    cur, _, rows, feature = windows(world, "shift")
    assert feature == 0
    assert np.allclose(cur[:, 0] - X[rows, 0], 0.3 * sd)
    assert (cur[:, 1:] == X[rows, 1:]).all()

    cur, _, rows, _ = windows(world, "scale")
    centre = np.median(X[split["ref"], 0])
    assert np.allclose(cur[:, 0] - centre, 1.5 * (X[rows, 0] - centre))
    assert (cur[:, 1:] == X[rows, 1:]).all()


def test_breaking_a_correlation_leaves_every_feature_the_same_on_its_own(world):
    X, y, split = world
    assert split["correlated"] in (0, 1), "the feature with a partner is chosen"
    cur, _, rows, feature = windows(world, "correlation")
    f = split["correlated"]
    assert feature == f
    assert (np.sort(cur[:, f]) == np.sort(X[rows, f])).all()
    before = abs(np.corrcoef(X[rows, 0], X[rows, 1])[0, 1])
    after = abs(np.corrcoef(cur[:, 0], cur[:, 1])[0, 1])
    assert before > 0.9 and after < 0.2


def test_selection_and_label_shift_change_who_is_drawn(world):
    X, y, split = world
    pool = split["pool"]
    cur, _, _, _ = windows(world, "selection")
    assert cur[:, 0].mean() > X[pool, 0].mean() + 0.1
    _, labels, _, _ = windows(world, "label_shift")
    common = pd.Series(y[split["ref"]]).mode().iloc[0]
    assert (labels == common).mean() < (y[pool] == common).mean() - 0.05


def test_concept_drift_changes_labels_only_above_the_median(world):
    X, y, split = world
    cur, labels, rows, _ = windows(world, "concept")
    assert (cur == X[rows]).all(), "the features do not move"
    changed = labels != y[rows]
    above = cur[:, 0] > np.median(X[split["ref"], 0])
    assert not changed[~above].any()
    assert 0.2 < changed[above].mean() < 0.4


def gaussian_case(shift=0.0, seed=3, d=5):
    rng = np.random.default_rng(seed)
    ref = rng.normal(size=(D.REF_ROWS, d))
    cur = rng.normal(size=(D.CUR_ROWS, d)) + shift
    errors = np.zeros(D.REF_ROWS + D.CUR_ROWS, dtype=int)
    return D.Case("g", 0, "x", ref, cur, np.zeros(d, dtype=bool),
                  errors[:D.REF_ROWS], errors[D.REF_ROWS:])


def test_feature_checkers_are_quiet_without_drift_and_fire_on_a_large_one():
    quiet, loud = gaussian_case(0.0), gaussian_case(1.0)
    for checker in D.checkers():
        if checker.sees == "errors":
            continue
        assert not checker.run(quiet)[0], f"{checker.code} fired on two samples of one Gaussian"
        assert checker.run(loud)[0], f"{checker.code} missed a shift of one standard deviation"


def test_error_checkers_fire_when_the_model_breaks():
    rng = np.random.default_rng(5)
    case = gaussian_case()
    case.ref_errors = (rng.random(D.REF_ROWS) < 0.05).astype(int)
    case.cur_errors = (rng.random(D.CUR_ROWS) < 0.6).astype(int)
    for checker in D.checkers():
        if checker.sees == "errors":
            assert checker.run(case)[0], f"{checker.code} missed the error rate going from 5% to 60%"

    case.cur_errors = (rng.random(D.CUR_ROWS) < 0.05).astype(int)
    steady = {c.code: c.run(case)[0] for c in D.checkers() if c.code in ("DR-C1", "DR-C4", "DR-C6")}
    assert not any(steady.values()), steady


def test_fhddm_is_given_correct_predictions_because_river_fires_on_a_falling_mean():
    from river.drift.binary import FHDDM
    rng = np.random.default_rng(0)
    errors = np.r_[rng.random(500) < 0.05, rng.random(250) < 0.5].astype(int)
    raw = FHDDM()
    fired = False
    for x in errors:
        raw.update(bool(x))
        fired |= raw.drift_detected
    assert not fired, "given errors, river's FHDDM does not notice the model getting worse"

    case = gaussian_case()
    case.ref_errors, case.cur_errors = errors[:500], errors[500:]
    fhddm = next(c for c in D.checkers() if c.code == "DR-C9")
    assert fhddm.run(case)[0]


def test_cramer_von_mises_matches_scipy_and_survives_ties():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=300), rng.normal(0.2, size=200)
    assert D.cvm_pvalue(a, b) == pytest.approx(stats.cramervonmises_2samp(a, b).pvalue, rel=1e-9)
    # Mostly zeros, two samples of the same thing. scipy 1.17 puts p below
    # 1e-9 here; the version here must not.
    rng = np.random.default_rng(2)
    a = np.where(rng.random(500) < 0.6, 0, rng.exponential(size=500))
    b = np.where(rng.random(250) < 0.6, 0, rng.exponential(size=250))
    assert D.cvm_pvalue(a, b) > 0.05
    assert stats.ks_2samp(a, b).pvalue > 0.05, "KS, which handles ties, agrees nothing changed"


def test_the_summary_counts_each_dataset_once():
    rows = []
    # Dataset "a" was cut four times and fired every time on shift; "b" once and never did.
    for rep in range(4):
        for scenario in D.SCENARIOS:
            rows.append({"dataset": "a", "rep": rep, "scenario": scenario, "checker": "X",
                         "fired": int(scenario == "shift" or (scenario == "none" and rep == 0)), "seconds": 0.1})
    for scenario in D.SCENARIOS:
        rows.append({"dataset": "b", "rep": 0, "scenario": scenario, "checker": "X", "fired": 0, "seconds": 0.1})
    out = D.summarise(pd.DataFrame(rows), n_boot=200)["X"]
    assert out["caught"]["shift"] == 0.5                      # (1 + 0) / 2, not 4 / 5
    assert out["false_alarm"] == pytest.approx(0.125)         # (1/4 + 0) / 2
    assert out["net"] == pytest.approx(((1 / 6 - 0.25) + 0) / 2, abs=1e-3)


def test_a_case_is_the_same_every_time():
    rng_frame = np.random.default_rng(2)
    frame = pd.DataFrame(rng_frame.normal(size=(2500, 3)), columns=["a", "b", "c"])
    frame["target"] = (frame.a > 0).map({True: "yes", False: "no"})
    first = [(c.scenario, c.cur.sum(), c.cur_errors.sum()) for c in D.cases_for("t", 0, frame)]
    again = [(c.scenario, c.cur.sum(), c.cur_errors.sum()) for c in D.cases_for("t", 0, frame)]
    assert first == again
    assert [s for s, _, _ in first] == list(D.SCENARIOS)
