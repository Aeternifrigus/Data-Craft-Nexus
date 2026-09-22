"""The learned order, ported from site/js/ranking.js.

The weights come from learn.py and live in site/taxonomy/ranking.json. The
benchmark has to rank models exactly as the page does, or it would be scoring
advice nobody is given, so this mirrors the JavaScript and the agreement test
compares the two.
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


def learned_score(ranking: dict | None, model: dict, task: str, features: dict) -> float | None:
    task_weights = (ranking or {}).get("tasks", {}).get(task)
    if not task_weights:
        return None
    prior = task_weights["prior"].get(model["c"])
    if prior is None:
        return None
    score = prior
    family = (ranking.get("families") or {}).get(model["c"])
    for name, value in features.items():
        score += task_weights.get("weights", {}).get(f"{family}|{name}", 0.0) * value
    return score
