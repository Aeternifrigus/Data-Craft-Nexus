"""Tuned boosting goes before the order only by the rule that decides which order ships."""
import numpy as np
import pandas as pd

from dcn.learn import choose_lead


def table(tuned_gain: dict[str, float], n: int = 30, seed: int = 0) -> pd.DataFrame:
    """Scores per dataset: `best`, the order's pick, and tuned boosting's, with a chosen edge per task."""
    rng = np.random.default_rng(seed)
    rows = []
    for task, gain in tuned_gain.items():
        for i in range(n):
            prior = 0.8 + rng.normal(0, 0.01)
            tuned = prior + gain + rng.normal(0, 0.005)
            rows.append({"dataset": f"{task}_{i}", "task": task, "best": max(prior, tuned, 0.85),
                         "prior": prior, "tuned": tuned})
    return pd.DataFrame(rows)


def test_better_on_one_task_and_no_worse_on_the_other_goes_first():
    lead = choose_lead(table({"classification": 0.02, "regression": 0.0}), "prior")
    assert lead["led"] and lead["better_on"] == ["classification"]
    assert lead["code"] == "BASE-HGB-TUNED"


def test_worse_anywhere_does_not():
    lead = choose_lead(table({"classification": 0.02, "regression": -0.02}), "prior")
    assert not lead["led"] and not lead["no_worse"]


def test_level_everywhere_does_not():
    lead = choose_lead(table({"classification": 0.0, "regression": 0.0}), "prior")
    assert not lead["led"]


def test_nothing_to_decide_without_the_reference():
    assert choose_lead(table({"classification": 0.0}).drop(columns="tuned"), "prior") is None


# TabPFN against tuned boosting, by the rule written down before TabPFN ran.
from dcn.learn import choose_small_lead  # noqa: E402


def small_table(tabpfn_gain: dict[str, float], n: int = 30, skipped: int = 10, seed: int = 1) -> pd.DataFrame:
    """Tuned boosting's and TabPFN's scores, with TabPFN missing on the `skipped` largest datasets per task.

    A task with no edge is an exact tie on every dataset, so "no worse" does not hang on the noise.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for task, gain in tabpfn_gain.items():
        for i in range(n + skipped):
            tuned = 0.8 + rng.normal(0, 0.01)
            tabpfn = tuned + gain + (rng.normal(0, 0.005) if gain else 0.0) if i < n else np.nan
            rows.append({"dataset": f"{task}_{i}", "task": task, "best": np.nanmax([tuned, tabpfn, 0.85]),
                         "prior": tuned - 0.01, "tuned": tuned, "tabpfn": tabpfn})
    return pd.DataFrame(rows)


def test_tabpfn_better_on_one_task_and_no_worse_on_the_other_goes_first_on_small_tables():
    lead = choose_small_lead(small_table({"classification": 0.02, "regression": 0.0}), max_rows=1000)
    assert lead["led"] and lead["better_on"] == ["classification"]
    assert lead["code"] == "BASE-TABPFN" and lead["against"] == "tuned" and lead["max_rows"] == 1000


def test_tabpfn_is_judged_only_where_it_ran():
    lead = choose_small_lead(small_table({"classification": 0.02, "regression": 0.0}, n=30, skipped=25))
    assert all(t["units"] == 30 for t in lead["tasks"].values())


def test_tabpfn_worse_anywhere_does_not_go_first():
    lead = choose_small_lead(small_table({"classification": 0.02, "regression": -0.02}))
    assert not lead["led"] and not lead["no_worse"]


def test_tabpfn_level_everywhere_does_not_go_first():
    assert not choose_small_lead(small_table({"classification": 0.0, "regression": 0.0}))["led"]


def test_nothing_to_decide_before_tabpfn_has_run():
    table = small_table({"classification": 0.0})
    assert choose_small_lead(table.drop(columns="tabpfn")) is None
    assert choose_small_lead(table.assign(tabpfn=np.nan)) is None
