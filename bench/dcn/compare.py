"""Compare two benchmark runs, dataset by dataset.

A run on its own says how good the advice is. Two runs say whether a change
to the taxonomy helped, and where it helped or hurt, which is the only way to
tell a fix from a rearrangement.

  python -m dcn.compare --before results/pilot.csv --after results/after-fixes.csv

Writes results/delta.csv (one row per dataset) and prints the summary.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .analyze import per_dataset

COLUMNS = ["first", "top4", "boosting"]


def compare(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    a = per_dataset(before).set_index(["dataset", "task"])
    b = per_dataset(after).set_index(["dataset", "task"])
    shared = a.index.intersection(b.index)
    rows = []
    for key in shared:
        old, new = a.loc[key], b.loc[key]
        row = {"dataset": key[0], "task": key[1]}
        for column in COLUMNS:
            row[f"{column}_before"] = old[column]
            row[f"{column}_after"] = new[column]
            # Positive means the change made the advice better on this dataset.
            row[f"{column}_delta"] = new[column] - old[column]
        row["first_model_before"] = old.first_model
        row["first_model_after"] = new.first_model
        row["eligible_before"] = old.n_eligible
        row["eligible_after"] = new.n_eligible
        row["ruled_out_winner_before"] = old.best_ruled_out > old.top4 + 1e-9
        row["ruled_out_winner_after"] = new.best_ruled_out > new.top4 + 1e-9
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["task", "dataset"])


def report(delta: pd.DataFrame) -> str:
    lines = []
    for task, group in delta.groupby("task"):
        lines.append(f"{task} ({len(group)} datasets)")
        for column in COLUMNS:
            d = group[f"{column}_delta"]
            better, worse = int((d > 1e-9).sum()), int((d < -1e-9).sum())
            lines.append(f"  {column:9} median change {d.median():+.3f}   better on {better}, worse on {worse}, "
                         f"unchanged on {len(d) - better - worse}")
        changed = group[group.first_model_before != group.first_model_after]
        lines.append(f"  the first recommendation changed on {len(changed)} of {len(group)} datasets")
        lines.append(f"  a ruled-out model beat the advice on {int(group.ruled_out_winner_before.sum())} before, "
                     f"{int(group.ruled_out_winner_after.sum())} after")
        lines.append("")
    biggest = delta.reindex(delta.first_delta.abs().sort_values(ascending=False).index).head(8)
    lines.append("largest changes to the first recommendation:")
    for row in biggest.itertuples():
        lines.append(f"  {row.dataset:34} {row.first_model_before} -> {row.first_model_after}"
                     f"   {row.first_before:.3f} -> {row.first_after:.3f}  ({row.first_delta:+.3f})")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before", default="results/pilot.csv")
    ap.add_argument("--after", default="results/after-fixes.csv")
    ap.add_argument("--out", default="results/delta.csv")
    args = ap.parse_args(argv)

    delta = compare(pd.read_csv(args.before), pd.read_csv(args.after))
    delta.round(4).to_csv(args.out, index=False)
    print(report(delta))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
