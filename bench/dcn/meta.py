"""Meta-features: the handful of numbers that say what a dataset is like.

Used for two things the site needs and could not do before:

  the map    the plot used only axes 1 to 3, so every labelled table with
             independent rows landed on the same point. These coordinates
             actually separate datasets.
  neighbours "on datasets like yours" needs a distance between datasets, and
             a distance needs numbers rather than codes.

Everything here is computed from the same CSV reading and profiling the site
uses, so a dataset the user uploads is measured exactly like the benchmark
datasets it gets compared against.

  python -m dcn.meta --out results/meta.csv
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from . import datasets as ds
from .csvread import parse_csv
from .profile import measure_drift, profile_data

# The order matters: site/js/nearest.js reads the same list.
FEATURES = ["log_rows", "log_features", "numeric_share", "categorical_share",
            "missing_share", "noise_share", "majority_share", "drift_psi"]


def meta_features(head: list[str], body: list[list[str]], target: str = "target") -> dict:
    """What a dataset looks like, in numbers, from its parsed CSV."""
    profile = profile_data(head, body)
    features = [c for c in profile.columns if c.name != target]
    n = max(len(features), 1)

    numeric = sum(1 for c in features if c.numeric and not c.date_like)
    text_or_date = sum(1 for c in features if c.text_like or c.date_like)
    categorical = n - numeric - text_or_date

    target_col = next((c for c in profile.columns if c.name == target), None)
    majority = 0.0
    if target_col is not None and not target_col.numeric:
        counts: dict[str, int] = {}
        for value in target_col.values:
            if value != "":
                counts[value] = counts.get(value, 0) + 1
        if counts:
            majority = max(counts.values()) / sum(counts.values())

    drift = measure_drift(profile, target)

    return {
        "rows": profile.n,
        "features": n,
        "log_rows": round(math.log10(max(profile.n, 1)), 4),
        "log_features": round(math.log10(n), 4),
        "numeric_share": round(numeric / n, 4),
        "categorical_share": round(categorical / n, 4),
        "missing_share": round(sum(c.missing for c in features) / n, 4),
        "noise_share": round(sum(c.dirty_rate for c in features) / n, 4),
        "majority_share": round(majority, 4),
        "drift_psi": round(min(drift["psi"], 5.0), 4) if drift else 0.0,
    }


def for_dataset(name: str) -> dict:
    dataset = ds.load(name)
    parsed = parse_csv(dataset.to_csv_text())
    return {"dataset": name, "task": dataset.task, **meta_features(parsed.head, parsed.body)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results/after-fixes.csv",
                    help="which run's datasets to measure")
    ap.add_argument("--out", default="results/meta.csv")
    args = ap.parse_args(argv)

    names = sorted(pd.read_csv(args.results).dataset.unique())
    rows = []
    for name in names:
        try:
            rows.append(for_dataset(name))
            print(f"{name}: {rows[-1]['rows']}x{rows[-1]['features']}", flush=True)
        except Exception as exc:
            print(f"{name}: skipped ({exc})", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}: {len(frame)} datasets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
