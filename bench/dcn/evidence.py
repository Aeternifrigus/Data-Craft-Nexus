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

from .analyze import BASELINE as BASELINE_CODE, METRIC, per_dataset
from .models import BASELINE
from .meta import FEATURES
from .models import BY_CODE, NOT_RUNNABLE  # noqa: F401
from .ranking import load_ranking
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
        if model == BASELINE_CODE:
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


def ranking_evidence(lodo_path: Path) -> dict | None:
    """How the learned order did against the alternatives, leave-one-dataset-out."""
    if not lodo_path.exists():
        return None
    table = pd.read_csv(lodo_path)
    ranking = load_ranking() or {}
    out = {"chosen": ranking.get("chosen"), "trained_on": ranking.get("trained_on"), "tasks": {}}
    for task, group in table.groupby("task"):
        out["tasks"][task] = {"datasets": int(len(group)), "metric": METRIC[task], "strategies": {}}
        for key, label in [("current", "counting matched coordinates"), ("prior", "learned from the benchmark"),
                           ("prior_fit", "learned, with dataset interactions"), ("boosting", "always use boosting")]:
            if key not in group:
                continue
            regret = (group.best - group[key]).dropna()
            out["tasks"][task]["strategies"][key] = {
                "label": label,
                "median_regret": round(float(regret.median()), 4),
                "was_best": round(float((regret <= 1e-9).mean()), 3),
                "within_one_point": round(float((regret <= 0.01).mean()), 3),
            }
    return out


def load_meta(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def build(results_path: Path, run_label: str) -> dict:
    results = pd.read_csv(results_path)
    meta = load_meta(results_path.parent / "meta.csv")
    table = per_dataset(results)
    taxonomy = load_taxonomy()
    names = {m["c"]: m["n"] for m in taxonomy["MODELS"]}
    names[BASELINE_CODE] = BASELINE.name   # the baseline is not one of the taxonomy models

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

    # Every model's score on every dataset, so the page can answer questions
    # the summary did not anticipate: what won on datasets like yours, how a
    # model did on the ones closest to what you uploaded.
    scores = {}
    for (dataset, task), group in results[results.status == "ok"].groupby(["dataset", "task"]):
        scores[f"{dataset}|{task}"] = {row.model: round(float(row.score), 4) for row in group.itertuples()}

    meta_by_name = {}
    if meta is not None:
        for row in meta.itertuples():
            meta_by_name[f"{row.dataset}|{row.task}"] = {f: float(getattr(row, f)) for f in FEATURES}

    datasets = [{
        "dataset": row.dataset,
        "task": row.task,
        "rows": int(row.rows),
        "features": int(row.features),
        "signature": row.signature,
        "meta": meta_by_name.get(f"{row.dataset}|{row.task}"),
        "scores": scores.get(f"{row.dataset}|{row.task}", {}),
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
        "meta_features": FEATURES,
        # Mean and spread of each feature across the benchmark, so a distance
        # between datasets can be computed in the page without shipping pandas.
        "meta_scale": ({f: {"mean": round(float(meta[f].mean()), 4),
                            "std": round(float(meta[f].std() or 1.0), 4)} for f in FEATURES}
                       if meta is not None else None),
        "headline": headline,
        "ranking": ranking_evidence(results_path.parent / "ranking-lodo.csv"),
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
