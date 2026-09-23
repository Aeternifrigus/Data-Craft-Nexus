"""The learned order, ported from site/js/ranking.js.

The weights come from learn.py and live in site/taxonomy/ranking.json. The
benchmark has to rank models exactly as the page does, or it would be scoring
advice nobody is given, so this mirrors the JavaScript and the agreement test
compares the two.

Three kinds of order can be in use ("chosen" in ranking.json):

  prior       one number per model: its average percentile rank
  prior_fit   the prior plus fitted interactions with the dataset's features
  prior_knn   the prior blended with what the model was worth on the
              benchmark datasets nearest to this one

The blend for prior_knn is (S * prior + sum w * rank) / (S + sum w) over the k
nearest benchmark datasets, where rank is the model's percentile rank on that
dataset and w = exp(-(distance / h)^2). Close neighbours pull the score toward
what happened on them; far ones barely move it off the prior. S, k and h are
fixed in learn.py before anything is evaluated, not tuned on the results.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRIOR = 0.5
FLAGS = ["A31", "A32", "A38", "A41", "A42", "A43", "A51", "A52", "A53", "A54", "A61", "A62", "A64"]


def load_ranking(root: Path = ROOT) -> dict | None:
    path = root / "site" / "taxonomy" / "ranking.json"
    return json.loads(path.read_text()) if path.exists() else None


def ranking_features(sig: dict) -> dict[str, float]:
    codes = set(sig["codes"])
    rows = max(sig.get("rows") or 0, 1)
    features = max(sig.get("features") or 0, 1)
    out = {
        "log_rows": math.log10(rows),
        "log_features": math.log10(features),
        "features_per_row": math.log10(features / rows),
    }
    for code in FLAGS:
        out[f"is_{code}"] = 1.0 if code in codes else 0.0
    return out


def has_learned_ranking(ranking: dict | None, task: str) -> bool:
    return bool(ranking and ranking.get("tasks", {}).get(task))


def meta_vector(meta: dict, names: list[str]) -> list[float]:
    """A dataset's meta-features in a fixed order, at the four decimals both sides store."""
    return [round(float(meta.get(name) or 0.0), 4) for name in names]


def meta_distance(x: list[float], y: list[float], names: list[str], scale: dict) -> float:
    """Mirrors metaDistance() in site/js/ranking.js and distance() in site/js/nearest.js."""
    total = 0.0
    for i, name in enumerate(names):
        spread = scale.get(name) or 1.0
        diff = (x[i] - y[i]) / spread
        total += diff * diff
    return math.sqrt(total / len(names))


def neighbours_of(block: dict, meta: dict) -> list[tuple[dict, float]]:
    """The k nearest benchmark datasets and their kernel weights, nearest first."""
    names, scale = block["features"], block["scale"]
    x = meta_vector(meta, names)
    scored = [(meta_distance(x, d["meta"], names, scale), d["dataset"], d) for d in block["datasets"]]
    scored.sort(key=lambda item: (item[0], item[1]))
    h = block["bandwidth"]
    return [(d, math.exp(-((dist / h) ** 2))) for dist, _, d in scored[:block["k"]]]


def ranking_neighbours(ranking: dict | None, task: str, meta: dict | None):
    """Neighbours for the order in use, or None when it does not use them."""
    task_weights = (ranking or {}).get("tasks", {}).get(task)
    block = (task_weights or {}).get("neighbours")
    if not block or meta is None or ranking.get("chosen") != "prior_knn":
        return None
    return neighbours_of(block, meta)


def blend(prior: float, code: str, neighbours: list[tuple[dict, float]], strength: float) -> float:
    total = weight = 0.0
    for d, w in neighbours:
        rank = d["ranks"].get(code)
        if rank is None:
            continue
        total += w * rank
        weight += w
    return (strength * prior + total) / (strength + weight)


def learned_score(ranking: dict | None, model: dict, task: str, features: dict,
                  neighbours: list | None = None) -> float | None:
    task_weights = (ranking or {}).get("tasks", {}).get(task)
    if not task_weights:
        return None
    prior = task_weights["prior"].get(model["c"])
    if prior is None:
        return None
    if neighbours is not None:
        return blend(prior, model["c"], neighbours, task_weights["neighbours"]["strength"])
    score = prior
    family = (ranking.get("families") or {}).get(model["c"])
    for name, value in features.items():
        score += task_weights.get("weights", {}).get(f"{family}|{name}", 0.0) * value
    return score
