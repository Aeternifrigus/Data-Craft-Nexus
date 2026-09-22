"""Turn a benchmark run into what the site shows.

Writes site/taxonomy/evidence.json: the headline numbers, a line per model,
and the per-dataset table. The page reads it like any other taxonomy file, so
a recommendation can be shown next to what it was worth on real data.

Everything here comes from results/, nothing is typed by hand.

  python -m dcn.evidence --results results/after-fixes.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .analyze import BASELINE, METRIC, per_dataset
from .models import BY_CODE, NOT_RUNNABLE
from .recommend import load_taxonomy

ROOT = Path(__file__).resolve().parents[2]


def model_lines(results: pd.DataFrame, table: pd.DataFrame) -> dict:
    """Per model: how often it was the best choice, and how it did when recommended."""
    ok = results[results.status == "ok"]
    best_by_dataset = ok.loc[ok.groupby(["dataset", "task"]).score.idxmax()]
    wins = best_by_dataset.groupby(["model", "task"]).size()
    ranked_first = table.groupby(["first_model", "task"]).size()

    out: dict[str, dict] = {}
    for (model, task), group in ok.groupby(["model", "task"]):
        if model == BASELINE:
            continue
        datasets = int(group.dataset.nunique())
        # Regret: how far below the best available model this one landed.
        merged = group.merge(table[["dataset", "task", "best"]], on=["dataset", "task"], how="left")
        regret = (merged.best - merged.score).dropna()
        entry = out.setdefault(model, {"name": group.model_name.iloc[0], "tasks": {}})
        entry["tasks"][task] = {
            "datasets": datasets,
            "was_best": int(wins.get((model, task), 0)),
            "recommended_first": int(ranked_first.get((model, task), 0)),
            "median_regret": round(float(regret.median()), 4),
            "median_seconds": round(float(group.seconds.median()), 2),
            "failed": int((results[(results.model == model) & (results.task == task)].status != "ok").sum()),
        }
    return out


def build(results_path: Path, run_label: str) -> dict:
    results = pd.read_csv(results_path)
    table = per_dataset(results)
    taxonomy = load_taxonomy()
    names = {m["c"]: m["n"] for m in taxonomy["MODELS"]}

    headline = {}
    for task, group in table.groupby("task"):
        headline[task] = {
            "datasets": int(len(group)),
            "metric": METRIC[task],
            "first": round(float(group.regret_first.median()), 4),
            "top4": round(float(group.regret_top4.median()), 4),
            "boosting": round(float(group.regret_boosting.median()), 4),
            "random_eligible": round(float(group.regret_eligible_mean.median()), 4),
            "first_was_best": round(float((group.regret_first <= 1e-9).mean()), 3),
            "top4_was_best": round(float((group.regret_top4 <= 1e-9).mean()), 3),
            "boosting_was_best": round(float((group.regret_boosting <= 1e-9).mean()), 3),
            "top4_beats_boosting": round(float((group.regret_top4 < group.regret_boosting - 1e-9).mean()), 3),
        }

    datasets = [{
        "dataset": row.dataset,
        "task": row.task,
        "rows": int(row.rows),
        "features": int(row.features),
        "signature": row.signature,
        "best": {"model": row.best_model, "name": names.get(row.best_model, row.best_model),
                 "score": round(float(row.best), 4)},
        "first": {"model": row.first_model, "name": names.get(row.first_model, row.first_model),
                  "score": round(float(row.first), 4)},
        "top4": round(float(row.top4), 4),
        "boosting": round(float(row.boosting), 4) if pd.notna(row.boosting) else None,
        "eligible": int(row.n_eligible),
    } for row in table.itertuples()]

    return {
        "run": run_label,
        "source": "PMLB (Penn Machine Learning Benchmarks)",
        "source_url": "https://github.com/EpistasisLab/pmlb",
        "how": ("Every runnable model was fitted on every dataset, five-fold cross-validated, "
                "including the models the instrument rules out. Regret is how far a choice "
                "landed below the best model that ran."),
        "model_runs": int(len(results)),
        "not_runnable": {code: why for code, why in NOT_RUNNABLE.items()},
        "runnable": sorted(BY_CODE),
        "headline": headline,
        "models": model_lines(results, table),
        "datasets": datasets,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results/after-fixes.csv")
    ap.add_argument("--label", default="40 PMLB datasets, 760 model runs")
    ap.add_argument("--out", default=str(ROOT / "site" / "taxonomy" / "evidence.json"))
    args = ap.parse_args(argv)

    evidence = build(Path(args.results), args.label)
    Path(args.out).write_text(json.dumps(evidence, indent=1) + "\n")
    print(f"wrote {args.out}: {len(evidence['datasets'])} datasets, {len(evidence['models'])} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
