"""Fit every runnable model on every dataset and record what it scored.

Every model is run, not only the ones the taxonomy recommends. That is the
point: without the models it ruled out, there is no way to ask whether it
threw away a winner.

Results are appended row by row to a CSV, so a run can be stopped and resumed
and nothing is lost. Each fit has a time budget; what runs out is recorded as
a timeout rather than dropped, because "too slow to be worth recommending" is
itself a finding.

  python -m dcn.run --task classification --limit 20 --out results/pilot.csv
"""
from __future__ import annotations

import argparse
import csv
import signal
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

from . import datasets as ds
from .csvread import parse_csv
from .models import BASELINE, BY_CODE, build_pipeline, runnable_codes
from .profile import profile_data, signature
from .recommend import conflict, load_taxonomy, rank_models

FIELDS = [
    "dataset", "task", "rows", "features", "subsampled", "signature", "flags",
    "model", "model_name", "eligible", "ruled_out_why", "recommended_rank",
    "score", "score_std", "seconds", "status", "detail",
]
TASK_CODE = {"classification": "category", "regression": "number"}
SCORING = {"classification": "balanced_accuracy", "regression": "r2"}


class Timeout(Exception):
    pass


@contextmanager
def time_budget(seconds: int):
    def handle(signum, frame):
        raise Timeout()
    old = signal.signal(signal.SIGALRM, handle)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def profile_dataset(dataset: ds.Dataset) -> dict:
    """Measure the dataset exactly as the site would, from its CSV form."""
    parsed = parse_csv(dataset.to_csv_text())
    profile = profile_data(parsed.head, parsed.body)
    sig = signature(profile, "target", "A21")
    numeric = [c.name for c in profile.columns if c.numeric and c.name != "target"]
    categorical = [c.name for c in profile.columns if not c.numeric and c.name != "target"]
    return {"signature": sig, "numeric": numeric, "categorical": categorical, "profile": profile}


def evaluate(dataset: ds.Dataset, spec, task: str, numeric, categorical, folds: int, budget: int) -> dict:
    X = dataset.frame.drop(columns=["target"])
    y = dataset.frame["target"].to_numpy()
    if task == "classification":
        counts = pd.Series(y).value_counts()
        if counts.min() < folds:  # too few examples of a class to split
            return {"status": "skipped", "detail": f"a class has only {counts.min()} rows"}
        # Some libraries (XGBoost) insist on integer class labels.
        y = LabelEncoder().fit_transform(y)
        cv = StratifiedKFold(folds, shuffle=True, random_state=0)
    else:
        y = y.astype(float)
        cv = KFold(folds, shuffle=True, random_state=0)

    pipe = build_pipeline(spec, task, numeric, categorical)
    started = time.time()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            with time_budget(budget):
                scores = cross_val_score(pipe, X, y, cv=cv, scoring=SCORING[task], error_score="raise")
        return {"status": "ok", "score": float(np.mean(scores)), "score_std": float(np.std(scores)),
                "seconds": round(time.time() - started, 2)}
    except Timeout:
        return {"status": "timeout", "seconds": budget, "detail": f"over {budget}s"}
    except Exception as exc:  # a model that cannot handle this data is a result too
        return {"status": "error", "seconds": round(time.time() - started, 2),
                "detail": f"{type(exc).__name__}: {exc}"[:300]}


def done_pairs(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    with path.open() as fh:
        return {(row["dataset"], row["model"]) for row in csv.DictReader(fh)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=["classification", "regression", "both"], default="both")
    ap.add_argument("--limit", type=int, default=0, help="datasets per task (0 = all)")
    ap.add_argument("--out", default="results/raw.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--budget", type=int, default=120, help="seconds per model per dataset")
    ap.add_argument("--max-rows", type=int, default=5000, help="subsample larger datasets")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    taxonomy = load_taxonomy()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    already = done_pairs(out)
    new_file = not out.exists()

    tasks = ["classification", "regression"] if args.task == "both" else [args.task]
    with out.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()

        for task in tasks:
            candidates = ds.candidates(task=task)
            if args.limit:
                # Spread across the size range rather than taking the smallest.
                idx = np.linspace(0, len(candidates) - 1, args.limit).astype(int)
                candidates = candidates.iloc[idx]

            for _, row in candidates.iterrows():
                name = row.dataset
                codes = [c for c in runnable_codes(task)]
                if all((name, c) in already for c in codes + [BASELINE.code]):
                    continue
                try:
                    dataset = ds.load(name)
                except Exception as exc:
                    print(f"{name}: could not load ({exc})", file=sys.stderr, flush=True)
                    continue

                subsampled = False
                if len(dataset.frame) > args.max_rows:
                    dataset.frame = dataset.frame.sample(args.max_rows, random_state=args.seed).reset_index(drop=True)
                    subsampled = True

                measured = profile_dataset(dataset)
                sig = measured["signature"]
                ranking = rank_models(taxonomy, sig, TASK_CODE[task], limit=99)
                rank_of = {m["c"]: i + 1 for i, m in enumerate(ranking.items)}
                ruled = {m["c"]: m["why"] for m in ranking.ruled_out}

                print(f"{name} [{task}] {len(dataset.frame)}x{dataset.n_features} "
                      f"{' '.join(sig['codes'])} -> {len(ranking.items)} eligible, {len(ruled)} ruled out",
                      flush=True)

                for code in codes + [BASELINE.code]:
                    if (name, code) in already:
                        continue
                    spec = BASELINE if code == BASELINE.code else BY_CODE[code]
                    result = evaluate(dataset, spec, task, measured["numeric"], measured["categorical"],
                                      args.folds, args.budget)
                    writer.writerow({
                        "dataset": name, "task": task, "rows": len(dataset.frame),
                        "features": dataset.n_features, "subsampled": subsampled,
                        "signature": " ".join(sig["codes"]), "flags": " ".join(sig["flags"]),
                        "model": code, "model_name": spec.name,
                        "eligible": code in rank_of, "ruled_out_why": ruled.get(code, ""),
                        "recommended_rank": rank_of.get(code, ""),
                        "score": result.get("score", ""), "score_std": result.get("score_std", ""),
                        "seconds": result.get("seconds", ""), "status": result["status"],
                        "detail": result.get("detail", ""),
                    })
                    fh.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
