"""Turn a benchmark run into what the site shows.

Writes site/taxonomy/evidence.json: the headline numbers, a line per model,
the per-dataset table, what the drift benchmark measured, and what messy data
did. The page reads it like any other taxonomy file, so a recommendation can
be shown next to what it was worth on real data.

Everything here comes from results/, nothing is typed by hand.

  python -m dcn.evidence --results results/after-fixes.csv
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze import BASELINE as BASELINE_CODE, METRIC, per_dataset
from .models import BASELINE, REFERENCE_NAMES, TUNED, is_reference
from .meta import FEATURES
from .models import BY_CODE, NOT_RUNNABLE  # noqa: F401
from .learn import comparisons, regret_units
from .ranking import load_ranking
from .recommend import load_taxonomy
from .significance import bootstrap_ci, by_family, collapse_seeds, family_of, model_ranking, seed_stability

ROOT = Path(__file__).resolve().parents[2]


def model_lines(results: pd.DataFrame, table: pd.DataFrame) -> dict:
    """Per model: how often it was the best choice, and how it did when recommended."""
    ok = results[results.status == "ok"]
    best_by_dataset = ok.loc[ok.groupby(["dataset", "task"]).score.idxmax()]
    wins = best_by_dataset.groupby(["model", "task"]).size()
    ranked_first = table.groupby(["first_model", "task"]).size()

    out: dict[str, dict] = {}
    for (model, task), group in ok.groupby(["model", "task"]):
        if is_reference(model):   # the baseline and the references are never recommended
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


STRATEGY_LABELS = [("current", "counting matched coordinates"), ("prior", "learned from the benchmark"),
                   ("prior_fit", "learned, with dataset interactions"),
                   ("prior_knn", "learned, weighted toward datasets like yours"), ("boosting", "always use boosting"),
                   ("tuned", "always use boosting, tuned"), ("tabpfn", "always use TabPFN")]


def _r(x, digits=4):
    return None if x is None or pd.isna(x) else round(float(x), digits)


def _p(x):
    """A p-value to four significant figures, so 4e-17 does not round to zero."""
    return None if x is None or pd.isna(x) else float(f"{float(x):.4g}")


def ranking_evidence(lodo_path: Path) -> dict | None:
    """How the learned order did against the alternatives, leave-one-dataset-out.

    Each median comes with a 95% interval from resampling the datasets, and
    the order in use is tested against every alternative, paired over the
    same datasets, so the page can say which differences are more than luck.
    """
    if not lodo_path.exists():
        return None
    table = pd.read_csv(lodo_path)
    ranking = load_ranking() or {}
    chosen = ranking.get("chosen")
    tests = comparisons(table, chosen) if chosen in table else {}
    out = {"chosen": chosen, "trained_on": ranking.get("trained_on"), "level": 0.95,
           "choice": ranking.get("choice", []), "lead": ranking.get("lead"),
           "small_lead": ranking.get("small_lead"), "tasks": {}}
    for task, group in table.groupby("task"):
        # Every number is over independent units: a synthetic family's
        # datasets are averaged into one, as the tests in learn.py count them.
        units = regret_units(group, [key for key, _ in STRATEGY_LABELS])
        out["tasks"][task] = {"datasets": int(len(group)), "units": int(len(units)), "metric": METRIC[task],
                              "strategies": {}, "against_chosen": {}}
        for key, label in STRATEGY_LABELS:
            if key not in units:
                continue
            regret = units[key].dropna()
            if regret.empty:
                continue
            low, high = bootstrap_ci(regret.to_numpy())
            out["tasks"][task]["strategies"][key] = {
                "label": label,
                "units": int(len(regret)),    # fewer than the task's when a reference skipped datasets
                "median_regret": _r(regret.median()),
                "ci": [_r(low), _r(high)],
                "was_best": round(float((regret <= 1e-9).mean()), 3),
                "within_one_point": round(float((regret <= 0.01).mean()), 3),
            }
        # How far each strategy lands from the best of everything that ran,
        # references included: the question "how much is left on the table".
        if "ceiling" in group:
            out["tasks"][task]["ceiling"] = {}
            keys = [k for k, _ in STRATEGY_LABELS if k in group and group[k].notna().any()]
            gaps = by_family(group.assign(**{k: group.ceiling - group[k] for k in keys}), keys)
            for key in keys:
                gap = gaps[key].dropna()
                low, high = bootstrap_ci(gap.to_numpy())
                out["tasks"][task]["ceiling"][key] = {"units": int(len(gap)), "median_gap": _r(gap.median()),
                                                      "ci": [_r(low), _r(high)],
                                                      "at_ceiling": round(float((gap <= 1e-9).mean()), 3)}
        for other, test in tests.get(task, {}).items():
            out["tasks"][task]["against_chosen"][other] = {
                "units": test.get("units"),
                "wins": test["wins"], "ties": test["ties"], "losses": test["losses"],
                "median_difference": _r(test["median_difference"]),
                "ci": [_r(test["ci"][0]), _r(test["ci"][1])],
                "p": _p(test["p"]), "p_holm": _p(test["p_holm"]),
            }
    return out


def model_significance(results: pd.DataFrame, table: pd.DataFrame, names: dict) -> dict:
    """Which models the benchmark can actually tell apart, per task."""
    out = {}
    for task, group in table.groupby("task"):
        on_page = results[(results.task == task) & results.dataset.isin(set(group.dataset))]
        out[task] = model_ranking(on_page, task)
        out[task]["names"] = {code: names.get(code, code) for code in out[task]["ranks"]}
        out[task]["critical_difference"] = _r(out[task]["critical_difference"])
        friedman = out[task]["friedman"]
        friedman["chi2"], friedman["p"] = _r(friedman["chi2"], 3), _p(friedman["p"])
    return out


def seeds_evidence(raw: pd.DataFrame, lodo_path: Path) -> dict:
    """How much the split alone moves things, on the datasets run under several seeds."""
    picks = pd.read_csv(lodo_path) if lodo_path.exists() else None
    chosen = (load_ranking() or {}).get("chosen")
    if picks is not None and f"{chosen}_model" not in picks:
        picks = None
    return seed_stability(raw, picks, chosen)


def meta_distance(a: dict, b: dict, names: list[str], scale: dict) -> float:
    """Distance between two datasets' meta-features. Mirrors distance() in site/js/nearest.js."""
    total = 0.0
    for name in names:
        spread = (scale.get(name) or {}).get("std") or 1.0
        diff = ((a.get(name) or 0.0) - (b.get(name) or 0.0)) / spread
        total += diff * diff
    return math.sqrt(total / len(names))


def coverage(datasets: list[dict], names: list[str], scale: dict | None) -> dict:
    """What the benchmark covered, so the page can say when an upload is outside it.

    Two things per task: the smallest dataset tested, and how far a benchmark
    dataset usually is from its nearest other benchmark dataset. An upload
    further from every benchmark dataset than 95% of benchmark datasets are
    from each other is not like anything that was tested. A dataset's own
    synthetic family does not count as its neighbour. Computed from the
    rounded values the page carries, so the page can recompute it.
    """
    out = {}
    if not scale:
        return out
    for task in sorted({d["task"] for d in datasets}):
        rows = [d for d in datasets if d["task"] == task and d.get("meta")]
        # Nearest *other kind* of dataset: a synthetic dataset's sisters are
        # always close, and would make the threshold meaninglessly tight.
        nearest = [min((meta_distance(d["meta"], o["meta"], names, scale) for o in rows
                        if o["family"] != d["family"]), default=None) for d in rows] if len(rows) > 1 else []
        nearest = [x for x in nearest if x is not None]
        out[task] = {
            "datasets": len(rows),
            "min_rows": int(min(d["rows"] for d in rows)) if rows else None,
            "max_rows": int(max(d["rows"] for d in rows)) if rows else None,
            "nearest_median": round(float(np.quantile(nearest, 0.5)), 4) if nearest else None,
            "nearest_p95": round(float(np.quantile(nearest, 0.95)), 4) if nearest else None,
        }
    return out


def drift_evidence(path: Path) -> dict | None:
    """What the drift benchmark (drift.py) measured, when it has been run."""
    if not path.exists():
        return None
    from .drift import evidence as drift_block
    return drift_block(pd.read_csv(path))


def messy_evidence(results_path: Path) -> dict | None:
    """What damaging the data did (messy.py), when the damaged runs are there."""
    folder = results_path.parent
    if not (folder / "messy.csv").exists():
        return None
    from .messy import evidence as messy_block
    return messy_block(folder / "messy.csv", results_path, picks_cache=folder / "messy-picks.csv")


def load_meta(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def build(results_path: Path, run_label: str, ranked_by: str = "counting matched coordinates") -> dict:
    raw = pd.read_csv(results_path)
    results = collapse_seeds(raw)
    meta = load_meta(results_path.parent / "meta.csv")
    table = per_dataset(results)
    taxonomy = load_taxonomy()
    names = {m["c"]: m["n"] for m in taxonomy["MODELS"]}
    names.update(REFERENCE_NAMES)   # the baseline and the references are not taxonomy models

    headline = {}
    for task, group in table.groupby("task"):
        # Over independent units, like every other number on the page: a
        # synthetic family's datasets averaged into one.
        units = by_family(group, ["regret_first", "regret_top4", "regret_boosting", "regret_eligible_mean"])
        headline[task] = {
            "datasets": int(len(group)),
            "units": int(len(units)),
            "metric": METRIC[task],
            "first": round(float(units.regret_first.median()), 4),
            "top4": round(float(units.regret_top4.median()), 4),
            "boosting": round(float(units.regret_boosting.median()), 4),
            "random_eligible": round(float(units.regret_eligible_mean.median()), 4),
            "first_was_best": round(float((units.regret_first <= 1e-9).mean()), 3),
            "top4_was_best": round(float((units.regret_top4 <= 1e-9).mean()), 3),
            "boosting_was_best": round(float((units.regret_boosting <= 1e-9).mean()), 3),
            "top4_beats_boosting": round(float((units.regret_top4 < units.regret_boosting - 1e-9).mean()), 3),
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
        "family": family_of(row.dataset),
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

    # Mean and spread of each feature across the benchmark, so a distance
    # between datasets can be computed in the page without shipping pandas.
    meta_scale = ({f: {"mean": round(float(meta[f].mean()), 4),
                       "std": round(float(meta[f].std() or 1.0), 4)} for f in FEATURES}
                  if meta is not None else None)

    return {
        "run": run_label,
        "source": "PMLB (Penn Machine Learning Benchmarks)",
        "source_url": "https://github.com/EpistasisLab/pmlb",
        "how": ("Every runnable model was fitted on every dataset, five-fold cross-validated, "
                "including the models the instrument rules out. Regret is how far a choice "
                "landed below the best model that ran."),
        "model_runs": int(len(raw)),
        # Which order the site used while the run was recorded, so the page can
        # say what "shown first" meant at the time.
        "ranked_by": ranked_by,
        "not_runnable": {code: why for code, why in NOT_RUNNABLE.items()},
        "runnable": sorted(BY_CODE),
        "meta_features": FEATURES,
        "meta_scale": meta_scale,
        "coverage": coverage(datasets, FEATURES, meta_scale),
        "headline": headline,
        "ranking": ranking_evidence(results_path.parent / "ranking-lodo.csv"),
        # How the tuned reference was tuned, in words, for the page to state.
        "tuning": TUNED.notes,
        "significance": model_significance(results, table, names),
        "seeds": seeds_evidence(raw, results_path.parent / "ranking-lodo.csv"),
        "models": model_lines(results, table),
        "datasets": datasets,
        # The drift checkers, from their own benchmark (drift.py).
        "drift": drift_evidence(results_path.parent / "drift.csv"),
        # The same datasets damaged on purpose, and the ones that came with gaps (messy.py).
        "messy": messy_evidence(results_path),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results/after-fixes.csv")
    ap.add_argument("--label", default="40 PMLB datasets, 760 model runs")
    ap.add_argument("--ranked-by", default="counting matched coordinates",
                    help="which order the site used while this run was recorded")
    ap.add_argument("--out", default=str(ROOT / "site" / "taxonomy" / "evidence.json"))
    args = ap.parse_args(argv)

    evidence = build(Path(args.results), args.label, args.ranked_by)
    Path(args.out).write_text(json.dumps(evidence, indent=1) + "\n")
    print(f"wrote {args.out}: {len(evidence['datasets'])} datasets, {len(evidence['models'])} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
