"""The script the site hands out must be the benchmark, not a lookalike.

site/js/export.js writes a Python script that runs the shortlist on a user's
file. These tests generate it with the site's own JavaScript and hold it to
the benchmark: every estimator, the tuning candidates, the preprocessing and
the cross-validation, down to the score.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dcn.datasets import Dataset
from dcn.models import BASELINE, BY_CODE, hgb_candidates
from dcn.run import evaluate

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
TASKS = {"category": "classification", "number": "regression"}


def generate(csv: Path, target: str, task: str, codes: list[str], order: str = "A21") -> str:
    proc = subprocess.run(["node", str(ROOT / "bench" / "tools" / "js_script.mjs"), str(csv), target, task, order,
                           *codes], capture_output=True, text=True, check=True, cwd=ROOT)
    return proc.stdout


def load(script: str) -> dict:
    namespace = {"__name__": "dcn_shortlist"}
    exec(compile(script, "dcn_shortlist.py", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("task,csv,target", [("category", "balanced.csv", "label"), ("number", "numeric.csv", "y")])
def test_every_estimator_in_the_script_is_the_benchmarks(task, csv, target):
    bench_task = TASKS[task]
    codes = [c for c, spec in BY_CODE.items() if bench_task in spec.tasks]
    script = load(generate(FIXTURES / csv, target, task, codes))
    assert list(script["SHORTLIST"]) == codes
    for code, (name, scale, needs, build) in script["SHORTLIST"].items():
        spec = BY_CODE[code]
        assert name == spec.name, code
        assert scale == spec.scale, f"{code}: scaled in one and not the other"
        ours, theirs = build(), spec.build(bench_task)
        assert type(ours) is type(theirs), code
        assert repr(ours.get_params()) == repr(theirs.get_params()), code


def test_the_tuning_candidates_are_the_benchmarks():
    script = load(generate(FIXTURES / "balanced.csv", "label", "category", ["TR2"]))
    assert script["hgb_candidates"]() == hgb_candidates()


def test_models_the_script_cannot_run_are_named_not_dropped():
    text = generate(FIXTURES / "balanced.csv", "label", "category", ["TR2", "NN9"])
    assert "Recommended but not runnable on a table here: NN9." in text
    assert list(load(text)["SHORTLIST"]) == ["TR2"]


def test_order_that_matters_is_split_by_time():
    script = load(generate(FIXTURES / "numeric.csv", "y", "number", ["LM1"], order="A22"))
    assert script["ORDERED"] is True
    assert type(script["splits"](np.zeros(50))).__name__ == "TimeSeriesSplit"


@pytest.mark.parametrize("task,csv,target,codes", [
    ("category", "balanced.csv", "label", ["TR1", "LM2"]),
    ("number", "numeric.csv", "y", ["LM3", "TR1"]),
])
def test_the_script_scores_exactly_what_the_benchmark_scores(task, csv, target, codes):
    """Same preprocessing, same folds, same estimators: the same numbers."""
    bench_task = TASKS[task]
    script = load(generate(FIXTURES / csv, target, task, codes))
    frame = pd.read_csv(FIXTURES / csv, dtype=str, keep_default_na=False)
    X, y = script["prepare"](frame)
    cv = script["splits"](y)

    typed = pd.read_csv(FIXTURES / csv).rename(columns={target: "target"})
    dataset = Dataset(name=csv, task=bench_task, frame=typed, categorical=[], types_known=True)
    for code in codes:
        name, scale, needs, build = script["SHORTLIST"][code]
        from sklearn.model_selection import cross_val_score
        ours = float(np.mean(cross_val_score(script["pipeline"](build(), scale), X, y, cv=cv,
                                             scoring=script["SCORING"])))
        theirs = evaluate(dataset, BY_CODE[code], bench_task, script["NUMERIC"], script["CATEGORICAL"],
                          folds=5, budget=300)
        assert theirs["status"] == "ok", theirs
        assert ours == pytest.approx(theirs["score"], abs=1e-9), code


def test_the_script_runs_from_the_command_line(tmp_path):
    path = tmp_path / "dcn_shortlist.py"
    path.write_text(generate(FIXTURES / "balanced.csv", "label", "category", ["TR1", "LM2", "EN4"]))
    proc = subprocess.run([sys.executable, str(path), str(FIXTURES / "balanced.csv")], capture_output=True,
                          text=True, timeout=600, env={"OMP_NUM_THREADS": "1", "PATH": "/usr/bin:/bin"})
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "balanced accuracy" in out
    for code in ["TR1", "LM2", "EN4", "BASE-HGB-TUNED"]:
        assert code in out, code
    assert BASELINE.code + "-TUNED" in out
