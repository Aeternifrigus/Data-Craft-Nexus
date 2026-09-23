"""What messy data did: losses, detection and regret computed as described, and the published block reproducible."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dcn import messy as M

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "bench" / "results"


def runs(condition, scores):
    """Result rows: {(dataset, task): {model: score or None for a failure}}."""
    rows = []
    for (dataset, task), models in scores.items():
        for model, score in models.items():
            rows.append({"dataset": dataset, "task": task, "model": model, "seed": 0, "condition": condition,
                         "score": score, "status": "ok" if score is not None else "timeout"})
    return pd.DataFrame(rows)


def test_a_first_pick_that_failed_counts_as_the_worst_model():
    table = pd.DataFrame({"A": [0.9, 0.8], "B": [0.7, np.nan], "BASE-HGB": [0.85, 0.6]},
                         index=pd.MultiIndex.from_tuples([("d1", "t"), ("d2", "t")], names=["dataset", "task"]))
    first = pd.Series({("d1", "t"): "B", ("d2", "t"): "B"})
    out = M.regrets(table, first)
    assert out.regret_first.tolist() == pytest.approx([0.2, 0.2])      # d2: B failed, scored as the worst (0.6)
    assert out.regret_boosting.tolist() == pytest.approx([0.05, 0.2])


def test_loss_detection_and_regret_are_measured_as_described():
    clean = runs("clean", {("581_fri_c3_500_25", "regression"): {"A": 0.9, "BASE-HGB": 0.8},
                           ("607_fri_c4_1000_50", "regression"): {"A": 0.7, "BASE-HGB": 0.8},
                           ("yeast", "regression"): {"A": 0.5, "BASE-HGB": 0.6}})
    damaged = runs("dirty_5", {("581_fri_c3_500_25", "regression"): {"A": 0.8, "BASE-HGB": 0.8},
                               ("607_fri_c4_1000_50", "regression"): {"A": 0.5, "BASE-HGB": 0.7},
                               ("yeast", "regression"): {"A": 0.5, "BASE-HGB": None}})
    picks = pd.DataFrame([
        {"dataset": d, "task": "regression", "condition": c, "first": "A",
         "signature": "A11 A21 A31 A41 A51 " + ("A64" if c == "dirty_5" and d != "yeast" else "A61"),
         "flags": "", "numeric": 0 if d == "yeast" else 3}
        for d in ["581_fri_c3_500_25", "607_fri_c4_1000_50", "yeast"] for c in ["clean", "dirty_5"]])
    out = M.analyse(damaged, clean, picks)
    entry = out["conditions"]["dirty_5"]
    # yeast has no numeric column, so no junk went in: detection is judged on the other two.
    assert entry["damaged"] == 2 and entry["flagged"] == 1.0 and entry["flagged_clean"] == 0.0
    task = entry["tasks"]["regression"]
    assert task["datasets"] == 3 and task["units"] == 2, "the two Friedman datasets are one unit"
    # Model A lost 0.1 and 0.2 on the Friedman sisters (one unit, 0.15) and nothing on yeast.
    assert task["models"]["A"]["median"] == pytest.approx(0.075)
    assert task["first_changed"] == 0.0


def test_the_published_block_is_the_committed_run():
    evidence = json.loads((ROOT / "site" / "taxonomy" / "evidence.json").read_text())
    if not evidence.get("messy") or not (RESULTS / "messy.csv").exists():
        pytest.skip("no messy run committed")
    again = M.evidence(RESULTS / "messy.csv", RESULTS / "full.csv", picks_cache=RESULTS / "messy-picks.csv")
    assert json.loads(json.dumps(again)) == evidence["messy"]


def test_an_order_learned_from_damaged_runs_picks_what_survives_the_damage():
    """Clean, model A is always best; damaged, B is. The quality order learns that from the other datasets."""
    names = [f"d{i}" for i in range(12)]
    def frame(condition, a, b):
        rows = []
        for i, name in enumerate(names):
            for model, score in (("LM2", a + 0.001 * i), ("TR1", b + 0.001 * i), ("PR1", 0.5)):
                rows.append({"dataset": name, "task": "classification", "model": model, "seed": 0,
                             "condition": condition, "score": score, "status": "ok", "eligible": True,
                             "signature": "A11 A21 A31 A41 A51 A61", "rows": 500, "features": 5})
        return pd.DataFrame(rows)
    clean, damaged = frame("clean", 0.9, 0.8), frame("missing_30", 0.6, 0.7)
    out = M.quality_order(damaged, clean)["missing_30"]
    task = out["tasks"]["classification"]
    assert task["fixed"] == pytest.approx(0.1) and task["quality"] == pytest.approx(0.0)
    assert task["wins"] == 12 and out["replaces"]
    # Label noise cannot be seen in a file, so no order could act on it.
    assert not M.quality_order(frame("labels_10", 0.6, 0.7), clean)["labels_10"]["replaces"]
