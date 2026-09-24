"""The ranking, ported from site/js/recommend.js.

The benchmark has to score the recommendations the site actually makes, so
this mirrors it exactly: the same conflicts rule models out, the same learned
order (or, for tasks the benchmark did not cover, the same coordinate
counting) ranks what is left, and ties are reported rather than broken.
tests/test_agreement.py checks both sides against each other.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .ranking import has_learned_ranking, learned_score, load_ranking, ranking_features, ranking_neighbours

STRUCTURE = ["A21", "A22", "A23", "A24", "A25", "A26"]
MODALITY = ["A31", "A32", "A33", "A34", "A35", "A36", "A37", "A38"]
MIXED_OK = ["A31", "A32", "A33", "A38"]
# A table of categories feeds a numeric model for independent rows once they
# are one-hot encoded. Mirrors encodes() in site/js/recommend.js.
ENCODED_OK = {"A32": ["A31"]}


def _encodes(model: dict, modality: str) -> bool:
    return (_code_of(model, MODALITY) in ENCODED_OK.get(modality, [])
            and _code_of(model, STRUCTURE) in (None, "A21"))
PARADIGMS = {"A11": ["SL"], "A12": ["USL", "SSL"]}

ROOT = Path(__file__).resolve().parents[2]


def load_taxonomy(root: Path = ROOT) -> dict:
    read = lambda name: json.loads((root / "site" / "taxonomy" / f"{name}.json").read_text())  # noqa: E731
    axes, math, models, drift, pipelines, instrument = (read(n) for n in
                                                        ("axes", "math", "models", "drift", "pipelines", "instrument"))
    evidence_path = root / "site" / "taxonomy" / "evidence.json"
    forecast_path = root / "site" / "taxonomy" / "forecast.json"
    return {
        # What the benchmarks measured, as the site reads it (taxonomy.js).
        "EVIDENCE": json.loads(evidence_path.read_text()) if evidence_path.exists() else None,
        "FORECAST": json.loads(forecast_path.read_text()) if forecast_path.exists() else None,
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

    # Croston's method and TSB forecast demand, which is never negative.
    series = sig.get("series")
    if model["c"] in ("TSM4", "TSM5") and series and not series["features"]["nonNegative"]:
        return "forecasts demand, which is never negative, and yours has negative values"
    # A forecaster reads only the target's own past, so the other columns cannot rule it out.
    if series and model.get("dom") == "Time series":
        return None

    wants_modality = _code_of(model, MODALITY)
    if wants_modality and wants_modality != modality and wants_modality != "A38":
        mixed = modality == "A38" and wants_modality in MIXED_OK
        encoded = _encodes(model, modality)
        if not mixed and not encoded:
            return f"needs {wants_modality} data (yours is {modality})"
    return None


def caution(model: dict, sig: dict):
    structure, modality = sig["codes"][1], sig["codes"][2]
    if structure == "A22" and _code_of(model, STRUCTURE) == "A21":
        return "assumes rows are independent: split by time, not at random, and build lag features yourself"
    if _encodes(model, modality):
        return "built for numbers: one-hot encode your categories first, as the take-home script does"
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


_RANKING_CACHE: dict = {}


def _ranking() -> dict | None:
    if "value" not in _RANKING_CACHE:
        _RANKING_CACHE["value"] = load_ranking()
    return _RANKING_CACHE["value"]


def rank_models(taxonomy: dict, sig: dict, task: str, limit: int = 4, ranking: dict | None = None,
                meta: dict | None = None) -> Ranking:
    """`meta` is the dataset's meta-features (meta.py), which the neighbour order needs."""
    codes = match_codes(sig)
    ranking = ranking if ranking is not None else _ranking()
    forecast = forecast_order(taxonomy, sig.get("series")) if task == "forecast" else None
    learned = forecast is None and has_learned_ranking(ranking, task)
    features = ranking_features(sig) if learned else {}
    neighbours = ranking_neighbours(ranking, task, meta) if learned else None

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
                       "caution": caution(model, sig),
                       "evidence_score": forecast["prior"].get(model["c"]) if forecast
                       else learned_score(ranking, model, task, features, neighbours) if learned else None})
    learned = learned or forecast is not None

    def sort_key(m):
        # Models the benchmark ran come first, ordered by what they were worth;
        # everything else falls back to the coordinate count.
        if learned:
            return (0 if m["evidence_score"] is not None else 1,
                    -(m["evidence_score"] if m["evidence_score"] is not None else 0),
                    -m["score"])
        return (0, 0, -m["score"])

    usable.sort(key=sort_key)
    top = usable[0] if usable else None
    if top is None:
        tied = 0
    elif learned and top["evidence_score"] is not None:
        tied = sum(1 for m in usable
                   if m["evidence_score"] is not None and abs(m["evidence_score"] - top["evidence_score"]) < 1e-9)
    else:
        tied = sum(1 for m in usable if m["evidence_score"] is None and m["score"] == top["score"])

    return Ranking(
        items=usable[:limit],
        candidates=len(usable),
        tied=tied,
        top_score=top["score"] if top else 0,
        ruled_out=ruled_out,
    )


def forecast_order(taxonomy: dict, series: dict | None) -> dict | None:
    """The prior forecasters are ordered by: per kind when that passed its rule. Mirrors forecastOrder()."""
    forecast = taxonomy.get("FORECAST")
    if not forecast:
        return None
    metric = series["metric"] if series and series.get("metric") in forecast["prior"] else "r2"
    kind = series.get("kind") if series else None
    by_kind = forecast["chosen"] == "kind" and kind and forecast["kind_prior"].get(metric, {}).get(kind)
    return {"prior": by_kind or forecast["prior"][metric], "by": "kind" if by_kind else "fixed", "kind": kind,
            "metric": metric}


# How the thing will run decides which drift checkers and pipelines can be
# used at all: mirrors operatingConflict() in site/js/recommend.js.
PIPELINE_STAGES = {
    "data": {"label": "getting data in", "domains": ["ETL", "Feature", "Labeling"]},
    "train": {"label": "training and evaluating", "domains": ["Training", "Evaluation"]},
    "ship": {"label": "shipping and watching",
             "domains": ["Deployment", "Inference", "Monitoring", "Retraining", "ABTesting"]},
    "llm": {"label": "work with language models", "domains": ["RAG", "FineTuning"]},
}


def operating_conflict(entry: dict, ops: dict | None) -> str | None:
    needs = entry.get("needs")
    if not needs or not ops:
        return None
    mode, labels = ops.get("mode"), ops.get("labels")
    if needs.get("modes") and mode and mode not in needs["modes"]:
        shape = "a stream" if mode == "streaming" else "batch scoring"
        return f"{needs.get('note') or 'is built for the other shape'}, so it does not fit {shape}"
    if needs.get("labels") and labels and labels not in needs["labels"]:
        never = "never arrive" if labels == "none" else "arrive later"
        return needs.get("note") or f"needs labels, and yours {never}"
    return None


def pipeline_conflict(pipeline: dict, ops: dict | None) -> str | None:
    operating = operating_conflict(pipeline, ops)
    if operating:
        return operating
    stage = PIPELINE_STAGES.get((ops or {}).get("stage"))
    if not stage:
        return None
    if pipeline["p"] in stage["domains"]:
        return None
    return f"belongs to a different part of the work than {stage['label']}"


def drift_codes(codes: list[str]) -> list[str]:
    """A mixed table (A38) has numeric and categorical columns: checkers for either apply."""
    return list(dict.fromkeys([*codes, "A31", "A32"])) if "A38" in codes else codes


def drift_measure(taxonomy: dict, code: str) -> dict | None:
    """What the drift benchmark measured for a checker, or None: driftMeasure() in recommend.js."""
    return (((taxonomy.get("EVIDENCE") or {}).get("drift") or {}).get("checkers") or {}).get(code)


def rank_drifts(taxonomy: dict, sig: dict, limit: int = 4) -> Ranking:
    """Usable checkers, measured ones first by net score, the rest by coordinates matched."""
    codes = drift_codes(match_codes(sig))
    ops = sig.get("ops")
    measured = bool(((taxonomy.get("EVIDENCE") or {}).get("drift") or {}).get("checkers"))
    usable, ruled_out = [], []
    for checker in taxonomy["DRIFTS"]:
        hits = [f for f in checker["fits"] if f in codes]
        if not hits:
            continue
        why = operating_conflict(checker, ops)
        if why:
            ruled_out.append({"c": checker["c"], "n": checker["n"], "why": why})
            continue
        measure = drift_measure(taxonomy, checker["c"]) if measured else None
        usable.append({**checker, "hits": hits, "score": len(hits), "of": len(checker["fits"]),
                       "measure": measure, "evidence_score": measure["net"] if measure else None})

    def sort_key(d):
        if measured:
            known = d["evidence_score"] is not None
            return (0 if known else 1, -(d["evidence_score"] if known else 0), -d["score"])
        return (0, 0, -d["score"])

    usable.sort(key=sort_key)
    top = usable[0] if usable else None
    if top is None:
        tied = 0
    elif measured and top["evidence_score"] is not None:
        tied = sum(1 for d in usable
                   if d["evidence_score"] is not None and abs(d["evidence_score"] - top["evidence_score"]) < 1e-9)
    else:
        tied = sum(1 for d in usable if d["evidence_score"] is None and d["score"] == top["score"])
    return Ranking(usable[:limit], len(usable), tied, top["score"] if top else 0, ruled_out)


def rank_pipelines(taxonomy: dict, sig: dict, task: str, limit: int = 3) -> Ranking:
    codes = match_codes(sig)
    ops = sig.get("ops")
    ranked, ruled_out = [], []
    for pipeline in taxonomy["PIPELINES"]:
        why = pipeline_conflict(pipeline, ops)
        if why:
            ruled_out.append({"c": pipeline["c"], "n": pipeline["n"], "why": why})
            continue
        specific = [d for d in pipeline["data"] if "x" not in d]
        hits = [d for d in specific if d in codes]
        task_hit = "any" in pipeline["task"] or task in pipeline["task"]
        ranked.append({**pipeline, "hits": hits, "of": len(specific), "score": len(hits) + (1 if task_hit else 0)})
    ranked.sort(key=lambda p: -p["score"])
    top = ranked[0]["score"] if ranked else 0
    return Ranking(ranked[:limit], len(ranked), sum(1 for p in ranked if p["score"] == top), top, ruled_out)
