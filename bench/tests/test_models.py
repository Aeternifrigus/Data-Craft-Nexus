"""The registry has to cover the taxonomy, and every pipeline has to fit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dcn.models import BY_CODE, NOT_RUNNABLE, SPECS, build_pipeline, runnable_codes

ROOT = Path(__file__).resolve().parents[2]
TAXONOMY = json.loads((ROOT / "site" / "taxonomy" / "models.json").read_text())["models"]
TABULAR_TASKS = {"number", "category"}


def sample_frame(task: str, n: int = 120) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "num1": rng.normal(size=n),
        "num2": rng.integers(0, 50, size=n).astype(float),
        "cat1": rng.choice(["a", "b", "c"], size=n),
        "cat2": rng.choice(["x", "y"], size=n),
    })
    signal = frame.num1 + (frame.cat1 == "a") * 2
    y = (signal > signal.median()).astype(int).to_numpy() if task == "classification" else signal.to_numpy()
    return frame, y, ["num1", "num2"], ["cat1", "cat2"]


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: f"{s.code}-{s.name}")
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_every_pipeline_fits_and_predicts(spec, task):
    if task not in spec.tasks:
        pytest.skip(f"{spec.code} is not for {task}")
    frame, y, numeric, categorical = sample_frame(task)
    pipe = build_pipeline(spec, task, numeric, categorical)
    pipe.fit(frame, y)
    pred = pipe.predict(frame)
    assert len(pred) == len(y)
    assert np.isfinite(np.asarray(pred, dtype=float)).all()


def test_every_tabular_model_is_either_runnable_or_explained():
    undecided = []
    for model in TAXONOMY:
        if model["p"] not in ("SL",) or not TABULAR_TASKS.intersection(model["task"]):
            continue
        if model["c"] in BY_CODE or model["c"] in NOT_RUNNABLE:
            continue
        undecided.append(f'{model["c"]} {model["n"]}')
    assert undecided == [], "new tabular models need an estimator or a reason in NOT_RUNNABLE"


def test_registry_codes_exist_in_the_taxonomy():
    known = {m["c"] for m in TAXONOMY}
    assert [c for c in BY_CODE if c not in known] == []
    assert [c for c in NOT_RUNNABLE if c not in known] == []


def test_the_taxonomy_offers_every_model_for_the_tasks_it_can_actually_do():
    """A model with a working regressor must be offered for "a number".

    The benchmark found AdaBoost and both SVMs tagged classification-only,
    so the site never offered them for a numeric target. AdaBoost went on to
    beat every recommended model on nine regression datasets.
    """
    task_code = {"classification": "category", "regression": "number"}
    by_code = {m["c"]: m for m in TAXONOMY}
    missing = []
    for code, spec in BY_CODE.items():
        tagged = by_code[code]["task"]
        for task in spec.tasks:
            if task_code[task] not in tagged:
                missing.append(f'{code} {spec.name}: runs {task} but is not tagged "{task_code[task]}"')
    assert missing == []


def test_both_tasks_have_a_reasonable_field_of_candidates():
    assert len(runnable_codes("classification")) >= 10
    assert len(runnable_codes("regression")) >= 10
