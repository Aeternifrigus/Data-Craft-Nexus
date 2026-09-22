"""The ranking, ported from site/js/recommend.js.

The benchmark has to score the recommendations the site actually makes, so
this mirrors it exactly: the same conflicts rule models out, the same
coordinate counting ranks what is left, and ties are reported rather than
broken. tests/test_agreement.py checks both sides against each other.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

STRUCTURE = ["A21", "A22", "A23", "A24", "A25", "A26"]
MODALITY = ["A31", "A32", "A33", "A34", "A35", "A36", "A37", "A38"]
MIXED_OK = ["A31", "A32", "A33", "A38"]
PARADIGMS = {"A11": ["SL"], "A12": ["USL", "SSL"]}

ROOT = Path(__file__).resolve().parents[2]


def load_taxonomy(root: Path = ROOT) -> dict:
    read = lambda name: json.loads((root / "site" / "taxonomy" / f"{name}.json").read_text())  # noqa: E731
    axes, math, models, drift, pipelines, instrument = (read(n) for n in
                                                        ("axes", "math", "models", "drift", "pipelines", "instrument"))
    return {
        "AXES": axes["axes"], "CODES": axes["codes"],
        "MATH_DOMAINS": math["domains"], "MATH": math["formulas"],
        "MODEL_DOMAINS": models["domains"], "MODELS": models["models"],
        "DRIFT_DOMAINS": drift["domains"], "DRIFTS": drift["checkers"],
        "PIPELINE_DOMAINS": pipelines["domains"], "PIPELINES": pipelines["pipelines"],
        "STAGES": pipelines["stages"], "TASKS": instrument["tasks"], "REFERENCE": instrument["reference"],
    }


def paradigm_of(sig: dict) -> str:
    return "SL" if sig["codes"][0] == "A11" else "USL"


def paradigm_label(p: str) -> str:
    return {"SL": "supervised", "USL": "unsupervised"}.get(p, "self/semi-supervised")


def _code_of(model: dict, family: list[str]):
    return next((d for d in model["data"] if d in family), None)


def conflict(model: dict, sig: dict):
    """Why a model cannot be used here, or None."""
    supervision, structure, modality = sig["codes"][0], sig["codes"][1], sig["codes"][2]

    if model["p"] not in PARADIGMS.get(supervision, []):
        if model["p"] == "RL":
            return "needs an environment to interact with, not a table"
        return f"needs {paradigm_label(model['p'])} data"

    wants = _code_of(model, STRUCTURE)
    if wants and wants != "A21" and wants != structure:
        return f"needs {wants} data (yours is {structure})"

    wants_modality = _code_of(model, MODALITY)
    if wants_modality and wants_modality != modality and wants_modality != "A38":
        if not (modality == "A38" and wants_modality in MIXED_OK):
            return f"needs {wants_modality} data (yours is {modality})"
    return None


def caution(model: dict, sig: dict):
    structure = sig["codes"][1]
    if structure == "A22" and _code_of(model, STRUCTURE) == "A21":
        return "assumes rows are independent: split by time, not at random, and build lag features yourself"
    if "A54" in sig["flags"] or sig["codes"][4] == "A54":
        return "your data shifts across the file, so hold out the most recent rows and watch for drift after deployment"
    return None


def match_codes(sig: dict) -> list[str]:
    return [*sig["codes"], *sig["flags"]]


@dataclass
class Ranking:
    items: list[dict]
    candidates: int
    tied: int
    top_score: int
    ruled_out: list[dict]


def rank_models(taxonomy: dict, sig: dict, task: str, limit: int = 4) -> Ranking:
    codes = match_codes(sig)
    usable, ruled_out = [], []
    for model in taxonomy["MODELS"]:
        if task not in model["task"]:
            continue
        why = conflict(model, sig)
        if why:
            ruled_out.append({"c": model["c"], "n": model["n"], "why": why})
            continue
        hits = [d for d in model["data"] if d in codes]
        usable.append({**model, "hits": hits, "score": len(hits), "of": len(model["data"]),
                       "caution": caution(model, sig)})

    usable.sort(key=lambda m: -m["score"])
    top = usable[0]["score"] if usable else 0
    return Ranking(
        items=usable[:limit],
        candidates=len(usable),
        tied=sum(1 for m in usable if m["score"] == top),
        top_score=top,
        ruled_out=ruled_out,
    )


def rank_drifts(taxonomy: dict, sig: dict, limit: int = 4) -> Ranking:
    codes = match_codes(sig)
    usable = []
    for checker in taxonomy["DRIFTS"]:
        hits = [f for f in checker["fits"] if f in codes]
        if not hits:
            continue
        usable.append({**checker, "hits": hits, "score": len(hits), "of": len(checker["fits"])})
    usable.sort(key=lambda d: -d["score"])
    top = usable[0]["score"] if usable else 0
    return Ranking(usable[:limit], len(usable), sum(1 for d in usable if d["score"] == top), top, [])


def rank_pipelines(taxonomy: dict, sig: dict, task: str, limit: int = 3) -> Ranking:
    codes = match_codes(sig)
    ranked = []
    for pipeline in taxonomy["PIPELINES"]:
        specific = [d for d in pipeline["data"] if "x" not in d]
        hits = [d for d in specific if d in codes]
        task_hit = "any" in pipeline["task"] or task in pipeline["task"]
        ranked.append({**pipeline, "hits": hits, "of": len(specific), "score": len(hits) + (1 if task_hit else 0)})
    ranked.sort(key=lambda p: -p["score"])
    top = ranked[0]["score"] if ranked else 0
    return Ranking(ranked[:limit], len(ranked), sum(1 for p in ranked if p["score"] == top), top, [])
