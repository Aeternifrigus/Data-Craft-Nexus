"""The statistics that decide whether a difference is published as real."""
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from dcn.significance import (average_ranks, bootstrap_ci, cliques, collapse_seeds, critical_difference,
                              by_family, family_of, friedman, holm, model_ranking, paired, seed_stability)


@pytest.mark.parametrize("k,q", [(2, 1.960), (3, 2.343), (4, 2.569), (5, 2.728), (10, 3.164)])
def test_critical_difference_uses_demsars_table(k, q):
    # Demšar (2006), table 5(a): q_0.05 for the Nemenyi test.
    n = 30
    assert critical_difference(k, n) == pytest.approx(q * math.sqrt(k * (k + 1) / (6 * n)), abs=2e-3)


def test_critical_difference_shrinks_with_more_datasets():
    assert critical_difference(20, 100) < critical_difference(20, 20)
    assert critical_difference(20, 80) == pytest.approx(critical_difference(20, 20) / 2)


def test_average_ranks_put_the_best_first_and_share_ties():
    matrix = pd.DataFrame({"a": [0.9, 0.8], "b": [0.7, 0.8], "c": [0.5, 0.1]}, index=["d1", "d2"])
    ranks = average_ranks(matrix)
    assert list(ranks.index) == ["a", "b", "c"]
    assert ranks["a"] == pytest.approx((1 + 1.5) / 2)
    assert ranks["b"] == pytest.approx((2 + 1.5) / 2)
    assert ranks["c"] == pytest.approx(3.0)


def test_a_model_that_did_not_finish_ranks_last():
    matrix = pd.DataFrame({"a": [0.9], "b": [np.nan], "c": [np.nan], "d": [0.1]}, index=["d1"])
    ranks = average_ranks(matrix)
    assert ranks["d"] == 2
    assert ranks["b"] == ranks["c"] == 3.5   # tied for last


def test_friedman_matches_scipy():
    rng = np.random.default_rng(1)
    matrix = pd.DataFrame(rng.normal(size=(12, 4)), columns=list("abcd"))
    expected = stats.friedmanchisquare(*[matrix[c] for c in matrix.columns])
    got = friedman(matrix)
    # scipy ranks ascending, we rank descending; the statistic is symmetric.
    assert got["chi2"] == pytest.approx(expected.statistic)
    assert got["p"] == pytest.approx(expected.pvalue)


def test_holm_on_a_worked_example():
    adjusted = holm({"x": 0.01, "y": 0.04, "z": 0.03})
    assert adjusted["x"] == pytest.approx(0.03)
    assert adjusted["z"] == pytest.approx(0.06)
    assert adjusted["y"] == pytest.approx(0.06)   # never below a smaller p's adjustment


def test_paired_counts_and_direction():
    a = [0.0, 0.0, 0.1, 0.02, 0.0]
    b = [0.1, 0.2, 0.0, 0.02, 0.3]
    out = paired(a, b)
    assert (out["wins"], out["losses"], out["ties"]) == (3, 1, 1)
    assert out["median_difference"] > 0      # a had less regret
    assert 0 <= out["p"] <= 1


def test_paired_ignores_missing_datasets():
    out = paired([0.0, np.nan, 0.1], [0.1, 0.2, np.nan])
    assert out["datasets"] == 1


def test_bootstrap_interval_contains_the_median_and_is_repeatable():
    values = np.random.default_rng(3).exponential(size=40)
    low, high = bootstrap_ci(values)
    assert low <= np.median(values) <= high
    assert (low, high) == bootstrap_ci(values)


def test_cliques_are_maximal_groups_within_the_difference():
    ranks = pd.Series({"a": 1.0, "b": 1.5, "c": 2.4, "d": 4.0})
    assert cliques(ranks, 1.0) == [["a", "b"], ["b", "c"]]
    assert cliques(ranks, 3.0) == [["a", "b", "c", "d"]]


def test_seeds_are_averaged_into_one_row_per_model():
    results = pd.DataFrame([
        {"dataset": "d", "task": "classification", "model": "m", "seed": 0, "score": 0.8, "seconds": 1.0,
         "status": "ok", "score_std": 0.0},
        {"dataset": "d", "task": "classification", "model": "m", "seed": 1, "score": 0.6, "seconds": 3.0,
         "status": "ok", "score_std": 0.0},
        {"dataset": "d", "task": "classification", "model": "n", "seed": 0, "score": None, "seconds": None,
         "status": "timeout", "score_std": None},
        {"dataset": "d", "task": "classification", "model": "n", "seed": 1, "score": 0.5, "seconds": 2.0,
         "status": "ok", "score_std": 0.0},
    ])
    out = collapse_seeds(results).set_index("model")
    assert len(out) == 2
    assert out.loc["m", "score"] == pytest.approx(0.7)
    assert out.loc["m", "seeds"] == 2
    assert out.loc["n", "score"] == pytest.approx(0.5)   # the seed that finished
    assert out.loc["n", "status"] == "ok"


def test_a_single_seed_run_passes_through_unchanged():
    results = pd.DataFrame([{"dataset": "d", "task": "regression", "model": "m", "seed": 0, "score": 0.5,
                             "status": "ok"}])
    out = collapse_seeds(results)
    assert out.score.tolist() == [0.5]


def test_model_ranking_reports_who_cannot_be_separated_from_the_best():
    rows = []
    rng = np.random.default_rng(0)
    for i in range(30):
        for model, level in [("strong", 0.9), ("close", 0.89), ("weak", 0.5)]:
            rows.append({"dataset": f"d{i}", "task": "classification", "model": model,
                         "score": level + rng.normal(scale=0.02), "status": "ok"})
    out = model_ranking(pd.DataFrame(rows), "classification")
    assert out["datasets"] == 30
    assert out["friedman"]["p"] < 1e-6
    assert list(out["ranks"])[-1] == "weak"
    assert "weak" not in out["tied_with_best"]
    assert "close" in out["tied_with_best"]


def test_seed_stability_counts_flips_and_changing_winners():
    rows = []
    # d1: the same best model under both seeds, and the pick beats the reference both times.
    # d2: the winner changes with the seed, and so does the pick against the reference.
    for seed, scores in [(0, {"a": 0.9, "b": 0.8, "BASE-HGB": 0.7}), (1, {"a": 0.91, "b": 0.8, "BASE-HGB": 0.72})]:
        rows += [{"dataset": "d1", "task": "classification", "model": m, "seed": seed, "score": s, "status": "ok"}
                 for m, s in scores.items()]
    for seed, scores in [(0, {"a": 0.9, "b": 0.8, "BASE-HGB": 0.85}), (1, {"a": 0.8, "b": 0.9, "BASE-HGB": 0.85})]:
        rows += [{"dataset": "d2", "task": "classification", "model": m, "seed": seed, "score": s, "status": "ok"}
                 for m, s in scores.items()]
    # d3 ran once, so it says nothing about seeds.
    rows += [{"dataset": "d3", "task": "classification", "model": "a", "seed": 0, "score": 0.5, "status": "ok"}]
    picks = pd.DataFrame([{"dataset": "d1", "task": "classification", "prior_model": "a"},
                          {"dataset": "d2", "task": "classification", "prior_model": "a"}])
    out = seed_stability(pd.DataFrame(rows), picks, "prior")["classification"]
    assert out["datasets"] == 2 and out["seeds"] == 2
    assert out["same_best"] == 0.5
    assert (out["compared"], out["flips"]) == (2, 1)
    assert out["score_spread"] > 0


def test_datasets_from_one_generator_are_one_family():
    assert family_of("581_fri_c3_500_25") == family_of("607_fri_c4_1000_50") == "friedman"
    assert family_of("strogatz_glider2") == "strogatz"
    assert family_of("1199_BNG_echoMonths") == "bng"
    # Collected datasets stand alone, even when their names share a prefix.
    assert family_of("analcatdata_apnea2") == "analcatdata_apnea2"
    assert family_of("nikuradse_1") != family_of("nikuradse_2")


def test_a_family_counts_once():
    frame = pd.DataFrame({"dataset": ["581_fri_c3_500_25", "607_fri_c4_1000_50", "yeast"],
                          "regret": [0.1, 0.3, 0.5]})
    units = by_family(frame, ["regret"])
    assert len(units) == 2
    assert units.loc["friedman", "regret"] == pytest.approx(0.2)


def test_model_ranking_ranks_families_not_datasets():
    rows = []
    # Ten sister datasets where "a" wins, and three collected ones where "b" does.
    for i in range(10):
        rows += [{"dataset": f"{600 + i}_fri_c0_500_5", "task": "regression", "model": m, "score": s, "status": "ok"}
                 for m, s in [("a", 0.9), ("b", 0.8), ("c", 0.1)]]
    for name in ["x", "y", "z"]:
        rows += [{"dataset": name, "task": "regression", "model": m, "score": s, "status": "ok"}
                 for m, s in [("a", 0.8), ("b", 0.9), ("c", 0.1)]]
    out = model_ranking(pd.DataFrame(rows), "regression")
    assert (out["datasets"], out["units"]) == (13, 4)
    assert list(out["ranks"])[0] == "b", "ten sisters must not outvote three independent datasets"
