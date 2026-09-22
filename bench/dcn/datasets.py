"""Datasets for the benchmark, from PMLB.

PMLB (Penn Machine Learning Benchmarks) collects real datasets from UCI,
OpenML and elsewhere in one shape: a TSV with a `target` column. It is used
here because every file is a genuine dataset someone actually collected, not
something generated to make the taxonomy look good.

PMLB encodes categorical columns as integers, which would make every dataset
look like a table of numbers. Each dataset's metadata.yaml records the real
type of each column, so categorical columns are restored to labels before the
profiler sees them. Datasets whose metadata has not been filled in are used
with the types marked unknown, and the runner records which is which.

Everything is cached under bench/cache/ (git-ignored).
"""
from __future__ import annotations

import gzip
import io
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RAW = "https://raw.githubusercontent.com/EpistasisLab/pmlb/master"
MEDIA = "https://media.githubusercontent.com/media/EpistasisLab/pmlb/master"
CACHE = Path(__file__).resolve().parents[1] / "cache"
TIMEOUT = 60


def _fetch(url: str, cache_name: str, binary: bool = False) -> bytes:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / cache_name
    if path.exists():
        return path.read_bytes()
    with urllib.request.urlopen(url, timeout=TIMEOUT) as res:
        data = res.read()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def summary() -> pd.DataFrame:
    """One row per dataset: size, task, number of classes, imbalance."""
    data = _fetch(f"{RAW}/pmlb/all_summary_stats.tsv", "all_summary_stats.tsv")
    return pd.read_csv(io.BytesIO(data), sep="\t")


@dataclass
class Dataset:
    name: str
    task: str                    # "classification" or "regression"
    frame: pd.DataFrame          # features plus a "target" column
    categorical: list[str]       # columns restored to labels
    types_known: bool            # whether metadata listed the column types

    @property
    def n_rows(self) -> int:
        return len(self.frame)

    @property
    def n_features(self) -> int:
        return self.frame.shape[1] - 1

    def to_csv_text(self) -> str:
        """The dataset as a CSV, so the profiler reads exactly what a user would upload."""
        return self.frame.to_csv(index=False)


def _metadata_types(name: str) -> dict[str, str]:
    """Column name to type from metadata.yaml, without a YAML dependency."""
    try:
        text = _fetch(f"{RAW}/datasets/{name}/metadata.yaml", f"meta/{name}.yaml").decode("utf-8", "replace")
    except Exception:
        return {}
    types: dict[str, str] = {}
    block = text.split("features:", 1)
    if len(block) < 2:
        return {}
    current = None
    for line in block[1].splitlines():
        m = re.match(r"\s*-\s*name:\s*(.+?)\s*$", line)
        if m:
            current = m.group(1).strip().strip("'\"")
            continue
        m = re.match(r"\s*type:\s*(\w+)", line)
        if m and current:
            types[current] = m.group(1).lower()
            current = None
    return types


def load(name: str) -> Dataset:
    raw = _fetch(f"{MEDIA}/datasets/{name}/{name}.tsv.gz", f"data/{name}.tsv.gz", binary=True)
    frame = pd.read_csv(io.BytesIO(gzip.decompress(raw)), sep="\t")
    if "target" not in frame.columns:
        raise ValueError(f"{name}: no target column")

    types = _metadata_types(name)
    categorical = []
    for col in frame.columns:
        if col == "target":
            continue
        kind = types.get(col)
        if kind in ("categorical", "binary", "nominal", "ordinal"):
            categorical.append(col)
            # Integer codes are labels, not quantities: make them look like labels
            # so the profiler sees the dataset the way its collector meant it.
            frame[col] = frame[col].map(lambda v: f"{col}_{v}" if pd.notna(v) else "")

    task = "classification" if _is_classification(name, frame) else "regression"
    if task == "classification":
        frame["target"] = frame["target"].map(lambda v: f"class_{v}" if pd.notna(v) else "")

    return Dataset(name=name, task=task, frame=frame, categorical=categorical, types_known=bool(types))


def _is_classification(name: str, frame: pd.DataFrame) -> bool:
    row = summary().set_index("dataset").get("task")
    try:
        return str(row.loc[name]) == "classification"
    except Exception:
        target = frame["target"]
        return target.nunique() <= 20 and pd.api.types.is_integer_dtype(target)


def candidates(task: str | None = None, min_rows: int = 200, max_rows: int = 20000,
               max_features: int = 100) -> pd.DataFrame:
    """The datasets worth running: real sizes, but small enough to finish."""
    df = summary()
    df = df[(df.n_instances >= min_rows) & (df.n_instances <= max_rows) & (df.n_features <= max_features)]
    if task:
        df = df[df.task == task]
    # GAMETES datasets are simulated genetics, not collected data, and the
    # deprecated entries duplicate datasets already in the list.
    df = df[~df.dataset.str.startswith("GAMETES")]
    df = df[~df.dataset.str.startswith("_deprecated")]
    return df.sort_values("n_instances").reset_index(drop=True)


def cache_manifest() -> dict:
    """What has been downloaded, so a run can be reproduced from the cache alone."""
    files = sorted(p.relative_to(CACHE).as_posix() for p in CACHE.rglob("*") if p.is_file())
    return {"cache_dir": str(CACHE), "files": files}


if __name__ == "__main__":  # a quick look at what is available
    df = candidates()
    print(df.groupby("task").size().to_dict())
    print(json.dumps(df.head(5)[["dataset", "n_instances", "n_features", "task"]].to_dict("records"), indent=1))
