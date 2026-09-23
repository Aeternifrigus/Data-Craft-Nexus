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
