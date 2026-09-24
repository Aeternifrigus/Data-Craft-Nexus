"""The script the site hands out must be the benchmark, not a lookalike.

site/js/export.js writes a Python script that runs the shortlist on a user's
file. These tests generate it with the site's own JavaScript and hold it to
the benchmark: every estimator, the tuning candidates, the preprocessing and
the cross-validation, down to the score. The page can also run the script in
the browser (site/js/verify.js); the Python it runs there is run here too.
"""
from __future__ import annotations

import io
import json
import os
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


def generate(csv: Path, target: str, task: str, codes: list[str], order: str = "A21", cost: str | None = None) -> str:
    """The script the page writes. `cost` answers "What does a wrong answer cost?" (site/js/costs.js)."""
    env = {**os.environ, "DCN_COST": cost} if cost else None
    proc = subprocess.run(["node", str(ROOT / "bench" / "tools" / "js_script.mjs"), str(csv), target, task, order,
                           *codes], capture_output=True, text=True, check=True, cwd=ROOT, env=env)
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


def run_in_page(script: str, csv_text: str) -> tuple[list[dict], list[dict]]:
    """What the "Run it here" worker does, minus Pyodide: the page's runner, verbatim."""
    runner = subprocess.run(["node", str(ROOT / "bench" / "tools" / "js_runner.mjs")], capture_output=True,
                            text=True, check=True, cwd=ROOT).stdout
    streamed = []
    scope = {"SCRIPT": script, "CSV": csv_text, "report_row": lambda text: streamed.append(json.loads(text))}
    body, last = runner.rstrip().rsplit("\n", 1)
    exec(compile(body, "runner.py", "exec"), scope)
    return streamed, json.loads(eval(last, scope))   # the last line is the value Pyodide hands back


def test_run_it_here_gives_what_the_downloaded_script_gives():
    csv = FIXTURES / "balanced.csv"
    text = generate(csv, "label", "category", ["TR1", "LM2"])
    streamed, final = run_in_page(text, csv.read_text())
    assert streamed == final, "every row the page shows as it arrives is in the final result"
    assert [r["code"] for r in final] == ["TR1", "LM2", "BASE-HGB-TUNED"]

    script = load(text)
    direct = script["evaluate"](script["load"](str(csv)), progress=lambda row: None)
    for mine, theirs in zip(final, direct):
        assert mine["code"] == theirs["code"] and mine["status"] == theirs["status"] == "ok"
        assert mine["score"] == pytest.approx(theirs["score"], abs=1e-12)


def test_the_page_copy_needs_no_encoding():
    """The worker hands over text the page already decoded, whatever the file's encoding was."""
    text = generate(FIXTURES / "balanced.csv", "label", "category", ["TR1"]).replace(
        '"encoding": "utf-8"', '"encoding": "cp1250"')
    script = load(text)
    from_path = script["load"](str(FIXTURES / "balanced.csv"))
    from_text = script["load"](io.StringIO((FIXTURES / "balanced.csv").read_text()))
    assert from_text.equals(from_path)


# "What does a wrong answer cost?" -----------------------------------------------------------------------------------

def fold_scores(script, code, metric, proba=False):
    """The shortlist model's score in each of the script's folds, computed by hand with the named metric."""
    from sklearn.base import clone
    frame = script["load"](str(script["_csv"]))
    X, y = script["prepare"](frame)
    name, scale, needs, build = script["SHORTLIST"][code]
    model = script["pipeline"](build(), scale)
    out = []
    for train, test in script["splits"](y).split(X, y):
        fitted = clone(model).fit(X.iloc[train], y[train])
        guess = fitted.predict_proba(X.iloc[test]) if proba else fitted.predict(X.iloc[test])
        out.append(metric(y[test], guess))
    return out


@pytest.mark.parametrize("task,csv,target,code,cost,scoring,metric,proba,sign", [
    ("category", "imbalanced.csv", "churn", "LM2", "rows", "accuracy", "accuracy_score", False, 1),
    ("category", "imbalanced.csv", "churn", "LM2", "rank", "roc_auc", "roc_auc_score", "positive", 1),
    ("category", "balanced.csv", "label", "LM2", "rank", "roc_auc_ovr", "roc_auc_score", "ovr", 1),
    ("category", "balanced.csv", "label", "LM2", "chances", "neg_log_loss", "log_loss", True, -1),
    ("number", "numeric.csv", "y", "LM3", "absolute", "neg_mean_absolute_error", "mean_absolute_error", False, -1),
    ("number", "numeric.csv", "y", "LM3", "percent", "neg_mean_absolute_percentage_error",
     "mean_absolute_percentage_error", False, -1),
])
def test_each_answer_is_scored_by_the_score_the_page_names(task, csv, target, code, cost, scoring, metric, proba, sign):
    """The script's score for each answer is the metric the page describes, with the sign it says."""
    from sklearn import metrics
    from sklearn.model_selection import cross_val_score
    script = load(generate(FIXTURES / csv, target, task, [code], cost=cost))
    script["_csv"] = FIXTURES / csv
    assert script["SCORING"] == scoring
    assert ("shown negative" in script["METRIC"]) == (sign < 0)
    fn = getattr(metrics, metric)
    by_hand = {
        False: lambda: fold_scores(script, code, fn),
        True: lambda: fold_scores(script, code, fn, proba=True),
        "positive": lambda: fold_scores(script, code, lambda t, p: fn(t, p[:, 1]), proba=True),
        "ovr": lambda: fold_scores(script, code, lambda t, p: fn(t, p, multi_class="ovr"), proba=True),
    }[proba]()
    frame = script["load"](str(FIXTURES / csv))
    X, y = script["prepare"](frame)
    name, scale, needs, build = script["SHORTLIST"][code]
    ours = cross_val_score(script["pipeline"](build(), scale), X, y, cv=script["splits"](y), scoring=script["SCORING"])
    assert list(ours) == pytest.approx([sign * v for v in by_hand], abs=1e-12)


def test_tuned_boosting_gives_chances_when_a_score_needs_them():
    """Log loss and ROC AUC ask a classifier for its chances; the script's tuned boosting has to count as one."""
    from sklearn.base import is_classifier
    from sklearn.model_selection import cross_val_score
    script = load(generate(FIXTURES / "imbalanced.csv", "churn", "category", ["TR1"], cost="chances"))
    X, y = script["prepare"](script["load"](str(FIXTURES / "imbalanced.csv")))
    model = script["pipeline"](script["TunedHGB"]("classification"), False)
    assert is_classifier(model) and not is_classifier(script["TunedHGB"]("regression"))
    scores = cross_val_score(model, X, y, cv=script["splits"](y), scoring=script["SCORING"], error_score="raise")
    assert np.all(np.isfinite(scores)) and np.all(scores < 0)


def fraud(tmp_path, n=600, seed=0):
    """A yes or no target with signal in it: fraud is rare, and likelier as x1 grows."""
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    chance = 1 / (1 + np.exp(-(-3 + 2 * x1)))
    frame = pd.DataFrame({"x1": x1.round(4), "x2": x2.round(4),
                          "outcome": np.where(rng.random(n) < chance, "fraud", "ok")})
    path = tmp_path / "fraud.csv"
    frame.to_csv(path, index=False)
    return path


def test_a_priced_miss_is_scored_by_what_the_mistakes_cost(tmp_path):
    from sklearn.base import clone
    csv = fraud(tmp_path)
    script = load(generate(csv, "outcome", "category", ["LM2", "SV1"], cost="miss:10"))
    assert script["POSITIVE"] == "fraud" and script["MISS_COST"] == 10
    assert script["THRESHOLD"] == pytest.approx(1 / 11)
    frame = script["load"](str(csv))
    X, y = script["prepare"](frame)
    assert np.array_equal(y, (frame["outcome"] == "fraud").astype(int).to_numpy()), "1 is the case a miss is about"

    train, test = next(script["splits"](y).split(X, y))
    for code in ["LM2", "SV1"]:
        name, scale, needs, build = script["SHORTLIST"][code]
        fitted = clone(script["pipeline"](build(), scale)).fit(X.iloc[train], y[train])
        if hasattr(fitted, "predict_proba"):
            flagged = fitted.predict_proba(X.iloc[test])[:, 1] > 1 / 11
        else:   # a model that gives no chances is scored on its plain answer
            flagged = fitted.predict(X.iloc[test]) == 1
        truth = y[test]
        by_hand = (10 * np.sum((truth == 1) & ~flagged) + np.sum((truth == 0) & flagged)) / len(truth)
        assert script["cost_score"](fitted, X.iloc[test], truth) == pytest.approx(-by_hand), code


def test_the_threshold_reported_is_the_cheapest_on_the_file(tmp_path, capsys):
    from sklearn.base import clone
    csv = fraud(tmp_path)
    script = load(generate(csv, "outcome", "category", ["SV1", "LM2"], cost="miss:10"))
    frame = script["load"](str(csv))
    # SV1 scored best but gives no chances, so the threshold is LM2's.
    results = [{"code": "SV1", "name": "SVM (linear)", "status": "ok", "score": -0.1},
               {"code": "LM2", "name": "Logistic Regression", "status": "ok", "score": -0.2}]
    found = script["threshold_report"](results, frame)
    out = capsys.readouterr().out
    assert found["code"] == "LM2" and "Threshold, for Logistic Regression" in out

    X, y = script["prepare"](frame)
    name, scale, needs, build = script["SHORTLIST"]["LM2"]
    p = np.zeros(len(y))
    for train, test in script["splits"](y).split(X, y):
        p[test] = clone(script["pipeline"](build(), scale)).fit(X.iloc[train], y[train]).predict_proba(X.iloc[test])[:, 1]
    grid = np.linspace(0, 1, 1001)
    costs = np.array([script["cost_per_row"](y, p > t) for t in grid])
    assert found["cost"] == pytest.approx(costs.min(), abs=1e-12)
    cheapest = grid[np.isclose(costs, costs.min(), rtol=0, atol=1e-12)]
    assert found["best"] == pytest.approx(cheapest[np.argmin(np.abs(cheapest - 1 / 11))])
    assert costs.min() <= script["cost_per_row"](y, p > 0.5), "no dearer than the default cut"


def test_a_priced_miss_runs_from_the_command_line(tmp_path):
    csv = fraud(tmp_path)
    path = tmp_path / "dcn_shortlist.py"
    path.write_text(generate(csv, "outcome", "category", ["LM2"], cost="miss:10"))
    proc = subprocess.run([sys.executable, str(path), str(csv)], capture_output=True, text=True, timeout=600,
                          env={"OMP_NUM_THREADS": "1", "PATH": "/usr/bin:/bin"})
    assert proc.returncode == 0, proc.stderr
    assert "cost per row (shown negative: closer to zero is better)" in proc.stdout
    assert "flag a row as 'fraud' when its chance is above" in proc.stdout


def test_an_answer_the_page_does_not_offer_is_refused():
    with pytest.raises(subprocess.CalledProcessError):
        generate(FIXTURES / "balanced.csv", "label", "category", ["TR1"], cost="miss:5")   # three classes
