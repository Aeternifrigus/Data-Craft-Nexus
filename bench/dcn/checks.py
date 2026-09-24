"""What the page's "Before you trust a score" checks catch, and how often they
fire when nothing was planted.

Both measurements run the page's own JavaScript (tools/js_checks.mjs) on the
CSV a user would upload, each benchmark dataset loaded and subsampled the way
the runner does it:

  clean     every benchmark dataset as it is. A flag here is either a false
            alarm or a real property of the dataset, so the datasets are
            listed by name for a reader to judge.
  planted   the 40 datasets the messy-data benchmark damaged (20 per task),
            each with one problem planted in a column called "extra":
              leak       the target under another name: labels renamed, or
                         the number rescaled
              leak_1     the same, with 1% of its rows changed
              leak_5     the same, with 5% of its rows changed
              id_run     the row numbers 1..n, shuffled
              id_code    a short random code per row
              copies_2   2% of the rows copied and appended

The date check is a rule, not an estimate (a date column and rows declared
independent), so there is nothing to measure. The thresholds in
site/js/checks.js were set before this ran and are not tuned to it.

  python -m dcn.checks --out results/checks.csv
"""
from __future__ import annotations

import argparse
import json
import string
import subprocess
import tempfile
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import datasets as ds

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parents[1] / "results"
MAX_ROWS = 5000     # the runner's default
SEED = 0
TASK_CODE = {"classification": "category", "regression": "number"}
PLANTED = ["leak", "leak_1", "leak_5", "id_run", "id_code", "copies_2"]
# What a planted problem has to be flagged as to count as caught.
EXPECTED = {"leak": "leak", "leak_1": "leak", "leak_5": "leak", "id_run": "id", "id_code": "id",
            "copies_2": "duplicates"}
COLUMN = "extra"
LEAK_THRESHOLDS = [0.95, 0.97, 0.99]


def load(name: str) -> tuple[pd.DataFrame, str]:
    """A benchmark dataset as the runner sees it: subsampled to 5,000 rows with seed 0."""
    dataset = ds.load(name)
    frame = dataset.frame
    if len(frame) > MAX_ROWS:
        frame = frame.sample(MAX_ROWS, random_state=SEED).reset_index(drop=True)
    return frame, dataset.task


def plant(frame: pd.DataFrame, condition: str, task: str, name: str) -> pd.DataFrame:
    """One problem, planted in a copy of the dataset. Deterministic per dataset."""
    rng = np.random.default_rng(zlib.crc32(f"{name}:{condition}".encode()))
    out = frame.copy()
    n = len(out)
    y = out["target"]
    if condition.startswith("leak"):
        if task == "classification":
            labels = sorted(y.astype(str).unique())
            renamed = dict(zip(labels, [f"t{k}" for k in rng.permutation(len(labels))]))
            extra = y.astype(str).map(renamed).to_numpy(dtype=object)
        else:
            extra = (y.astype(float) * 1.7 + 3).round(6).to_numpy(dtype=object)
        share = {"leak": 0.0, "leak_1": 0.01, "leak_5": 0.05}[condition]
        changed = rng.choice(n, size=int(round(share * n)), replace=False)
        for i in changed:
            # Another row's value: what a proxy recorded with some mistakes looks like.
            j = int(rng.integers(n))
            while extra[j] == extra[i] and len(set(extra)) > 1:
                j = int(rng.integers(n))
            extra[i] = extra[j]
        out.insert(out.columns.get_loc("target"), COLUMN, extra)
    elif condition == "id_run":
        out.insert(0, COLUMN, rng.permutation(n) + 1)
    elif condition == "id_code":
        alphabet = np.array(list(string.ascii_uppercase + string.digits))
        codes = set()
        while len(codes) < n:
            codes.add("K" + "".join(rng.choice(alphabet, 7)))
        out.insert(0, COLUMN, rng.permutation(sorted(codes)))
    elif condition == "copies_2":
        # Copies scattered through the file, and the file kept within the
        # 5,000 rows the page reads: rows dropped to make room for them.
        k = max(2, int(round(0.02 * n)))
        keep = out.iloc[np.sort(rng.choice(n, size=min(n, MAX_ROWS - k), replace=False))] if n + k > MAX_ROWS else out
        copies = keep.iloc[rng.choice(len(keep), size=k, replace=False)]
        out = pd.concat([keep, copies], ignore_index=True).iloc[rng.permutation(len(keep) + k)].reset_index(drop=True)
    else:
        raise ValueError(condition)
    return out


def run_js(files: list[dict]) -> list[dict]:
    """The page's checks on each file, through node."""
    proc = subprocess.run(["node", str(ROOT / "bench" / "tools" / "js_checks.mjs")], input=json.dumps(files),
                          capture_output=True, text=True, check=True)
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


def measure(names: list[tuple[str, str]], planted_on: list[str]) -> pd.DataFrame:
    rows, files = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for name, task in names:
            frame, loaded_task = load(name)
            assert loaded_task == task, (name, loaded_task, task)
            conditions = ["clean"] + (PLANTED if name in planted_on else [])
            for condition in conditions:
                data = frame if condition == "clean" else plant(frame, condition, task, name)
                path = Path(tmp) / f"{name}__{condition}.csv"
                data.to_csv(path, index=False)
                files.append({"path": str(path), "target": "target", "task": TASK_CODE[task]})
                rows.append({"dataset": name, "task": task, "condition": condition})
        results = run_js(files)
    for row, result in zip(rows, results):
        flags = result["flags"]
        best = result.get("best") or {}
        row["best_column"] = best.get("column") or ""
        row["best_score"] = round(best["score"], 4) if best.get("score") is not None else None
        row["flags"] = ";".join(f"{f['kind']}:{f['column'] or ''}" for f in flags)
        if row["condition"] == "clean":
            row["caught"] = ""
        else:
            want = EXPECTED[row["condition"]]
            row["caught"] = int(any(f["kind"] == want and (want == "duplicates" or f["column"] == COLUMN)
                                    for f in flags))
    return pd.DataFrame(rows)


def evidence(table: pd.DataFrame) -> dict:
    """What the page says under each flag, and the evidence tab's table."""
    table = table.assign(flags=table["flags"].fillna("").astype(str))
    clean = table[table.condition == "clean"]
    fired = {kind: sorted(clean[clean["flags"].str.contains(f"(?:^|;){kind}:", regex=True)].dataset)
             for kind in ["leak", "id", "duplicates"]}
    planted = table[table.condition != "clean"]
    return {
        "how": ("The page's own checks, run on every benchmark dataset as a user would upload it, and on the 40 "
                "datasets the messy-data benchmark damaged with one problem planted in each."),
        "datasets": int(clean.dataset.nunique()),
        "fired": fired,
        "planted_on": int(planted.dataset.nunique()),
        "caught": {c: int(planted[planted.condition == c].caught.astype(int).sum()) for c in PLANTED},
        # What another leak threshold would have done, from the best single
        # column on each file. Reported, not used to pick the threshold.
        "leak_thresholds": {f"{thr:.2f}": {
            "clean": int((clean.best_score >= thr).sum()),
            **{c: int(((planted.condition == c) & (planted.best_column == COLUMN) & (planted.best_score >= thr)).sum())
               for c in ["leak", "leak_1", "leak_5"]},
        } for thr in LEAK_THRESHOLDS},
        "planted": {
            "leak": "the target under another name",
            "leak_1": "the target under another name, 1% of rows changed",
            "leak_5": "the target under another name, 5% of rows changed",
            "id_run": "the row numbers, shuffled",
            "id_code": "a short random code per row",
            "copies_2": "2% of the rows copied",
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", default=str(RESULTS / "per_dataset.csv"), help="the benchmark's datasets")
    ap.add_argument("--messy", default=str(RESULTS / "messy.csv"), help="the 40 datasets problems are planted in")
    ap.add_argument("--out", default=str(RESULTS / "checks.csv"))
    args = ap.parse_args(argv)

    per = pd.read_csv(args.full)
    names = list(per[["dataset", "task"]].drop_duplicates().itertuples(index=False, name=None))
    planted_on = sorted(pd.read_csv(args.messy).dataset.unique())
    table = measure(names, planted_on)
    table.to_csv(args.out, index=False)
    summary = evidence(table)
    print(f"{summary['datasets']} clean datasets:")
    for kind, which in summary["fired"].items():
        print(f"  {kind:10} fired on {len(which):3}  {', '.join(which[:8])}{' ...' if len(which) > 8 else ''}")
    print(f"planted in {summary['planted_on']} datasets, caught:")
    for c in PLANTED:
        print(f"  {c:10} {summary['caught'][c]:3} of {summary['planted_on']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
