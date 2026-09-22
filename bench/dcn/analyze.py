"""Turn a benchmark run into the numbers that judge the recommendations.

The question is not "is the recommended model good" but "how much worse than
the best available model is it". That gap, the regret, is what a user pays for
following the advice.

Four things are compared on every dataset:
  first       the model the site puts first
  top four    the best of the four the site shows
  boosting    histogram gradient boosting, the "just use boosting" reflex
  eligible    the average of every model the site considers eligible, which is
              what picking at random from its list would give you

Writes results/summary.json (committed, small) and prints a report.

  python -m dcn.analyze --results results/pilot.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASELINE = "BASE-HGB"
METRIC = {"classification": "balanced accuracy", "regression": "R²"}


def _why(ruled_out: pd.DataFrame) -> str:
    """Why the best non-recommended model was not offered.

    Two different failures hide here: a rule ruled the model out, or the
    taxonomy never lists it for this task at all (AdaBoost and the SVMs are
    tagged classification-only, though both have regressors).
    """
    if not len(ruled_out):
        return ""
    why = ruled_out.loc[ruled_out.score.idxmax()].ruled_out_why
    return why if isinstance(why, str) and why.strip() else "is not tagged for this task in the taxonomy"


def per_dataset(results: pd.DataFrame) -> pd.DataFrame:
    ok = results[results.status == "ok"]
    rows = []
    for (dataset, task), group in ok.groupby(["dataset", "task"]):
        models = group[group.model != BASELINE]
        eligible = models[models.eligible]
        ruled_out = models[~models.eligible]
        if eligible.empty:
            continue
        ranked = eligible.sort_values("recommended_rank")
        shown = ranked[ranked.recommended_rank <= 4]
        baseline = group[group.model == BASELINE]
        best_row = group.loc[group.score.idxmax()]

        rows.append({
            "dataset": dataset,
            "task": task,
            "rows": int(group.rows.iloc[0]),
            "features": int(group.features.iloc[0]),
            "signature": group.signature.iloc[0],
            "best_model": best_row.model,
            "best": best_row.score,
            "first_model": ranked.iloc[0].model,
            "first": ranked.iloc[0].score,
            "top4": shown.score.max() if len(shown) else np.nan,
            "boosting": baseline.score.iloc[0] if len(baseline) else np.nan,
            "eligible_mean": eligible.score.mean(),
            "eligible_worst": eligible.score.min(),
            "best_ruled_out": ruled_out.score.max() if len(ruled_out) else np.nan,
            "best_ruled_out_model": ruled_out.loc[ruled_out.score.idxmax()].model if len(ruled_out) else "",
            "ruled_out_why": _why(ruled_out),
            "n_eligible": len(eligible),
            "n_ruled_out": len(ruled_out),
        })
    table = pd.DataFrame(rows)
    for column in ["first", "top4", "boosting", "eligible_mean"]:
        table[f"regret_{column}"] = table.best - table[column]
    return table


def summarise(table: pd.DataFrame, results: pd.DataFrame) -> dict:
    def stats(task_table: pd.DataFrame) -> dict:
        out = {"datasets": int(len(task_table))}
        for column in ["first", "top4", "boosting", "eligible_mean"]:
            regret = task_table[f"regret_{column}"].dropna()
            out[column] = {
                "median_regret": round(float(regret.median()), 4),
                "mean_regret": round(float(regret.mean()), 4),
                "was_best": round(float((regret <= 1e-9).mean()), 3),
                "within_one_point": round(float((regret <= 0.01).mean()), 3),
            }
        beat = (task_table.regret_first < task_table.regret_boosting - 1e-9).mean()
        out["first_beats_boosting"] = round(float(beat), 3)
        out["top4_beats_boosting"] = round(float((task_table.regret_top4 < task_table.regret_boosting - 1e-9).mean()), 3)
        return out

    # Where the rules threw away the winner.
    thrown = table[table.best_ruled_out > table.top4 + 1e-9]
    losses = [{
        "dataset": row.dataset, "model": row.best_ruled_out_model, "why": row.ruled_out_why,
        "lost": round(float(row.best_ruled_out - row.top4), 4),
    } for row in thrown.sort_values("best_ruled_out", ascending=False).itertuples()]
    by_reason: dict[str, dict] = {}
    for loss in losses:
        entry = by_reason.setdefault(loss["why"], {"datasets": 0, "worst_loss": 0.0, "models": set()})
        entry["datasets"] += 1
        entry["worst_loss"] = max(entry["worst_loss"], loss["lost"])
        entry["models"].add(loss["model"])
    for entry in by_reason.values():
        entry["models"] = sorted(entry["models"])

    first_picks = (table.groupby(["task", "first_model"]).size().reset_index(name="times")
                   .sort_values("times", ascending=False))

    return {
        "datasets": int(len(table)),
        "model_runs": int(len(results)),
        "failed_runs": int((results.status != "ok").sum()),
        "metric": METRIC,
        "by_task": {task: stats(group) for task, group in table.groupby("task")},
        "ruled_out_winners": losses,
        "ruled_out_by_reason": by_reason,
        "most_recommended_first": [
            {"task": r.task, "model": r.first_model, "times": int(r.times)} for r in first_picks.itertuples()
        ],
    }


def report(table: pd.DataFrame, summary: dict) -> str:
    lines = [f"{summary['datasets']} datasets, {summary['model_runs']} model runs, "
             f"{summary['failed_runs']} did not finish", ""]
    for task, stats in summary["by_task"].items():
        lines.append(f"{task} ({METRIC[task]}, {stats['datasets']} datasets)")
        lines.append(f"{'':22}{'median regret':>14}{'was best':>10}{'within 1pt':>12}")
        for key, label in [("first", "site's first pick"), ("top4", "best of the four shown"),
                           ("boosting", "always boosting"), ("eligible_mean", "random eligible model")]:
            s = stats[key]
            lines.append(f"  {label:20}{s['median_regret']:>14.3f}{s['was_best']:>10.0%}{s['within_one_point']:>12.0%}")
        lines.append(f"  the first pick beats boosting on {stats['first_beats_boosting']:.0%} of datasets, "
                     f"the best of four on {stats['top4_beats_boosting']:.0%}")
        lines.append("")
    if summary["ruled_out_winners"]:
        lines.append("models the rules ruled out that would have won:")
        for loss in summary["ruled_out_winners"]:
            lines.append(f"  {loss['dataset']}: {loss['model']} by {loss['lost']:.3f}, ruled out because it {loss['why']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results/pilot.csv")
    ap.add_argument("--out", default="results/summary.json")
    ap.add_argument("--table", default="results/per_dataset.csv")
    args = ap.parse_args(argv)

    results = pd.read_csv(args.results)
    table = per_dataset(results)
    summary = summarise(table, results)

    Path(args.out).write_text(json.dumps(summary, indent=1) + "\n")
    table.round(4).to_csv(args.table, index=False)
    print(report(table, summary))
    print(f"\nwrote {args.out} and {args.table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
