"""Learn the order to show models in, instead of counting matched coordinates.

The benchmark showed that counting how many of a model's three codes appear in
the signature ranks models by how *describable* they are, not by how well they
do. This learns the order from the run instead.

Target: a model's percentile rank among the models that ran on the same
dataset. Rank rather than score, because R² and balanced accuracy are not
comparable and a single catastrophic R² would dominate a regression.

Three candidates, simplest first:

  prior         what a model is worth on average, shrunk toward the middle
                so a model seen on few datasets is not trusted too far
  prior+fit     the prior plus ridge-fitted interactions between the dataset's
                measured features and the model's family
  prior+knn     the prior blended with what each model was worth on the
                benchmark datasets nearest to this one (see ranking.py)

All are judged leave-one-dataset-out, with synthetic families held out whole
(family_of in significance.py): the held-out dataset, and any dataset
generated from the same function, never contributes to the weights, the
neighbours or the scale that rank it. The tests then count each family once. They are compared
against counting coordinates and against always reaching for boosting.

The rule for which one ships was fixed before the full run (choose()): the
plain prior, unless a richer order is no worse on either task and better by
more than luck (paired Wilcoxon, Holm-corrected over the two tasks) on at
least one. The neighbour order's settings are fixed below for the same
reason: tuning them on the results and then reporting the results would
flatter it.

  python -m dcn.learn --results results/after-fixes.csv
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .meta import FEATURES as META_FEATURES
from .models import BASELINE, TABPFN_CODE, TUNED, is_reference
from .ranking import blend, meta_distance, meta_vector, neighbours_of
from .recommend import conflict, load_taxonomy, rank_models
from .significance import by_family, collapse_seeds, family_of, holm, paired

# The neighbour order, fixed in advance rather than tuned on the results.
KNN_K = 10              # neighbours consulted
KNN_STRENGTH = 4.0      # how many datasets' worth of trust the prior keeps
# Bandwidth: the median distance from a benchmark dataset to its nearest
# other one, so "close" means as close as benchmark datasets usually are.
ALPHA = 0.05

ROOT = Path(__file__).resolve().parents[2]
TASK_CODE = {"classification": "category", "regression": "number"}

# Everything the page can compute for itself, from the signature and the shape.
FLAGS = ["A31", "A32", "A38", "A41", "A42", "A43", "A51", "A52", "A53", "A54", "A61", "A62", "A64"]


def dataset_features(signature: str, rows: int, features: int) -> dict[str, float]:
    codes = set(signature.split())
    out = {f"is_{code}": float(code in codes) for code in FLAGS}
    out["log_rows"] = math.log10(max(rows, 1))
    out["log_features"] = math.log10(max(features, 1))
    out["features_per_row"] = math.log10(max(features, 1) / max(rows, 1))
    return out


FEATURE_NAMES = sorted(dataset_features("A31", 100, 10))


def build_frame(results: pd.DataFrame, taxonomy: dict) -> pd.DataFrame:
    """One row per dataset and model, with the target and the features."""
    results = collapse_seeds(results)
    # The learned order is fitted on the taxonomy's models only: the baseline
    # and the references are what it is judged against, not what it chooses from.
    ok = results[(results.status == "ok") & ~results.model.map(is_reference)].copy()
    family = {m["c"]: m["dom"] for m in taxonomy["MODELS"]}
    rows = []
    for (dataset, task), group in ok.groupby(["dataset", "task"]):
        # Percentile rank inside the dataset: 1.0 is the best model that ran.
        ranks = group.score.rank(pct=True)
        features = dataset_features(group.signature.iloc[0], int(group.rows.iloc[0]), int(group.features.iloc[0]))
        for row, target in zip(group.itertuples(), ranks):
            rows.append({
                "dataset": dataset, "task": task, "model": row.model, "family": family.get(row.model, "?"),
                "target": float(target), "score": row.score, **features,
            })
    return pd.DataFrame(rows)


def fit_prior(train: pd.DataFrame, strength: float = 4.0) -> dict[str, float]:
    """Average percentile rank per model, shrunk toward 0.5 by how little we saw."""
    prior = {}
    for model, group in train.groupby("model"):
        n = len(group)
        prior[model] = float((group.target.sum() + strength * 0.5) / (n + strength))
    return prior


def fit_interactions(train: pd.DataFrame, prior: dict[str, float], alpha: float = 10.0):
    """Ridge on (family × dataset feature), fitted to what the prior leaves over."""
    families = sorted(train.family.unique())
    columns = [(f, name) for f in families for name in FEATURE_NAMES]
    index = {c: i for i, c in enumerate(columns)}

    X = np.zeros((len(train), len(columns)))
    residual = train.target.to_numpy() - train.model.map(lambda m: prior.get(m, 0.5)).to_numpy()
    for i, row in enumerate(train.itertuples()):
        for name in FEATURE_NAMES:
            X[i, index[(row.family, name)]] = getattr(row, name)

    model = Ridge(alpha=alpha, fit_intercept=False).fit(X, residual)
    weights = {f"{f}|{name}": float(w) for (f, name), w in zip(columns, model.coef_) if abs(w) > 1e-6}
    return weights


def predict(model_code: str, family: str, features: dict, prior: dict, weights: dict) -> float:
    value = prior.get(model_code, 0.5)
    for name, x in features.items():
        value += weights.get(f"{family}|{name}", 0.0) * x
    return value


def load_meta(path: Path) -> pd.DataFrame | None:
    """Meta-features per dataset (meta.py), which the neighbour order needs."""
    return pd.read_csv(path) if path.exists() else None


def meta_scale(meta: pd.DataFrame) -> dict[str, float]:
    """Spread of each meta-feature, as evidence.py publishes it for the page."""
    return {f: round(float(meta[f].std() or 1.0), 4) for f in META_FEATURES}


def neighbour_block(train: pd.DataFrame, meta: pd.DataFrame, task: str, scale: dict) -> dict:
    """The benchmark datasets a neighbour order can draw on, with their ranks."""
    by_name = {(r.dataset, r.task): r for r in meta.itertuples()}
    datasets = []
    for (dataset, _), group in train[train.task == task].groupby(["dataset", "task"]):
        row = by_name.get((dataset, task))
        if row is None:
            continue
        datasets.append({
            "dataset": dataset,
            "meta": meta_vector({f: getattr(row, f) for f in META_FEATURES}, META_FEATURES),
            "ranks": {m: round(float(r), 4) for m, r in zip(group.model, group.target)},
        })
    # "Close" is measured against other kinds of data: a Friedman dataset's
    # sister is always near it, and would make every bandwidth tiny.
    nearest = [min((meta_distance(d["meta"], o["meta"], META_FEATURES, scale) for o in datasets
                    if family_of(o["dataset"]) != family_of(d["dataset"])), default=1.0)
               for d in datasets] if len(datasets) > 1 else [1.0]
    bandwidth = round(float(np.median(nearest)), 4) or 1.0
    return {"k": KNN_K, "strength": KNN_STRENGTH, "bandwidth": bandwidth,
            "features": META_FEATURES, "scale": scale, "datasets": datasets}


def evaluate(frame: pd.DataFrame, taxonomy: dict, results: pd.DataFrame, meta: pd.DataFrame | None = None) -> dict:
    """Leave one dataset out, rank its models with weights fitted on the rest."""
    present = set(results.model)
    reference_keys = ["boosting"] + [key for key, code in REFERENCE_KEYS.items() if code in present]
    keys = ["current", "prior", "prior_fit"] + (["prior_knn"] if meta is not None else []) + reference_keys
    scores = {k: [] for k in keys + ["oracle"]}
    per_dataset = []
    family = {m["c"]: m["dom"] for m in taxonomy["MODELS"]}

    results = collapse_seeds(results)
    families = frame.dataset.map(family_of)
    for (dataset, task), group in frame.groupby(["dataset", "task"]):
        # Leave the dataset's whole family out, not just the dataset.
        train = frame[(families != family_of(dataset)) & (frame.task == task)]
        prior = fit_prior(train)
        weights = fit_interactions(train, prior)
        features = {name: group.iloc[0][name] for name in FEATURE_NAMES}

        neighbours = None
        if meta is not None:
            own = meta[(meta.dataset == dataset) & (meta.task == task)]
            others = meta[~((meta.dataset.map(family_of) == family_of(dataset)) & (meta.task == task))]
            if len(own):
                block = neighbour_block(train, others, task, meta_scale(others))
                neighbours = (block, neighbours_of(block, {f: own.iloc[0][f] for f in META_FEATURES}))

        best = group.score.max()
        run = results[(results.dataset == dataset) & (results.task == task)]
        def reference(code):
            row = run[(run.model == code) & (run.status == "ok")]
            return float(row.score.iloc[0]) if len(row) else np.nan

        boosting = reference(BASELINE.code)

        # The order the learned one replaced: the first model by coordinates
        # matched. The empty ranking matters. Without it rank_models reads the
        # committed ranking.json, which was fitted on every dataset including
        # this one, and "current" would quietly become an in-sample prior.
        signature = {"codes": run.signature.iloc[0].split(), "flags": []}
        ranking = rank_models(taxonomy, signature, TASK_CODE[task], limit=99, ranking={"tasks": {}})
        eligible = [m["c"] for m in ranking.items if m["c"] in set(group.model)]
        by_model = dict(zip(group.model, group.score))

        def pick(scorer):
            ordered = sorted(eligible, key=scorer, reverse=True)
            return (by_model[ordered[0]], ordered[0]) if ordered else (np.nan, "")

        chosen_models = {
            "current": (by_model[eligible[0]], eligible[0]) if eligible else (np.nan, ""),
            "prior": pick(lambda m: prior.get(m, 0.5)),
            "prior_fit": pick(lambda m: predict(m, family.get(m, "?"), features, prior, weights)),
            "boosting": (boosting, BASELINE.code),
        }
        if meta is not None:
            chosen_models["prior_knn"] = (
                pick(lambda m: blend(prior.get(m, 0.5), m, neighbours[1], KNN_STRENGTH))
                if neighbours else chosen_models["prior"])
        for key, code in REFERENCE_KEYS.items():
            if key in keys:
                chosen_models[key] = (reference(code), code)
        picks = {k: v[0] for k, v in chosen_models.items()}
        # The ceiling: the best of everything that ran here, references included.
        ceiling = float(np.nanmax([best] + [picks[k] for k in reference_keys]))

        for key, value in list(picks.items()) + [("oracle", best)]:
            scores[key].append(best - value if value is not None else np.nan)
        # The score each strategy's first pick got, and which model that was,
        # so a later analysis can follow the same pick across seeds.
        per_dataset.append({"dataset": dataset, "task": task, "family": family_of(dataset), "best": best,
                            "ceiling": ceiling,
                            **{k: picks[k] for k in keys},
                            **{f"{k}_model": chosen_models[k][1] for k in keys}})

    table = pd.DataFrame(per_dataset)
    summary = {}
    for task, group in table.groupby("task"):
        summary[task] = {}
        units = by_family(group.assign(**{k: group.best - group[k] for k in keys}), keys)
        for key in keys:
            regret = units[key].dropna()
            summary[task][key] = {
                "median_regret": round(float(regret.median()), 4),
                "mean_regret": round(float(regret.mean()), 4),
                "was_best": round(float((regret <= 1e-9).mean()), 3),
                "within_one_point": round(float((regret <= 0.01).mean()), 3),
            }
    return {"summary": summary, "table": table}


# Strategies that are a reference model's score rather than a pick from the
# taxonomy, and the model each one is.
REFERENCE_KEYS = {"tuned": TUNED.code, "tabpfn": TABPFN_CODE}
STRATEGIES = ["current", "prior", "prior_fit", "prior_knn", "boosting", *REFERENCE_KEYS]
CANDIDATES = ["prior", "prior_fit", "prior_knn"]   # simplest first


def choose(table: pd.DataFrame) -> tuple[str, list[dict]]:
    """Which learned order ships, by the rule fixed before the full run.

    Start from the plain prior. A richer candidate replaces the current choice
    only if its median regret is no worse on either task and it is better by
    more than luck on at least one: paired Wilcoxon, Holm-corrected over the
    tasks, p below ALPHA, with more wins than losses.
    """
    decisions = []
    chosen = "prior"
    for candidate in CANDIDATES[1:]:
        if candidate not in table:
            continue
        no_worse, tests = True, {}
        for task, group in table.groupby("task"):
            units = regret_units(group, [candidate, chosen])
            regret_new, regret_old = units[candidate].to_numpy(), units[chosen].to_numpy()
            if np.nanmedian(regret_new) > np.nanmedian(regret_old) + 1e-9:
                no_worse = False
            tests[task] = paired(regret_new, regret_old)
        adjusted = holm({task: test["p"] for task, test in tests.items()})
        better = [task for task, test in tests.items()
                  if adjusted[task] < ALPHA and test["wins"] > test["losses"]]
        decisions.append({
            "candidate": candidate, "against": chosen, "replaced": bool(no_worse and better),
            "no_worse": no_worse,
            "tasks": {task: {"wins": test["wins"], "losses": test["losses"], "ties": test["ties"],
                             "p_holm": float(f"{adjusted[task]:.4g}")} for task, test in tests.items()},
        })
        if no_worse and better:
            chosen = candidate
    return chosen, decisions


def choose_lead(table: pd.DataFrame, chosen: str) -> dict | None:
    """Should a reference be shown before the order's first pick?

    The same rule as choose(), applied to tuned boosting against the order in
    use: no worse median regret on either task, and better by more than luck on
    at least one. choose() was written for the learned orders before the full
    run; this applies it to the reference afterwards, when tuned boosting's
    result was already known, and the page says so.
    """
    if "tuned" not in table or chosen not in table:
        return None
    no_worse, tests = True, {}
    for task, group in table.groupby("task"):
        units = regret_units(group, ["tuned", chosen])
        if np.nanmedian(units["tuned"]) > np.nanmedian(units[chosen]) + 1e-9:
            no_worse = False
        tests[task] = paired(units["tuned"].to_numpy(), units[chosen].to_numpy())
    adjusted = holm({task: test["p"] for task, test in tests.items()})
    better = [task for task, test in tests.items() if adjusted[task] < ALPHA and test["wins"] > test["losses"]]
    return {
        "candidate": "tuned", "code": REFERENCE_KEYS["tuned"], "against": chosen,
        "led": bool(no_worse and better), "no_worse": no_worse, "better_on": better,
        "tasks": {task: {"wins": test["wins"], "losses": test["losses"], "ties": test["ties"],
                         "p_holm": float(f"{adjusted[task]:.4g}")} for task, test in tests.items()},
    }


def describe(decision: dict) -> str:
    verdict = "replaces" if decision["replaced"] else "does not replace"
    detail = ", ".join(f"{task}: better on {d['wins']}, worse on {d['losses']}, p = {d['p_holm']:.3g}"
                       for task, d in decision["tasks"].items())
    worse = "" if decision["no_worse"] else "; worse median regret on a task"
    return f"{decision['candidate']} {verdict} {decision['against']} ({detail}{worse})"


# What the order in use is compared with: the two it claims to beat (counting
# coordinates, default boosting) and the two references it makes no claim to
# beat but a reader will ask about (tuned boosting, TabPFN).
REFERENCES = ["current", "boosting", "tuned", "tabpfn"]


def regret_units(group: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Regret per independent unit: a synthetic family's datasets averaged into one."""
    keys = [k for k in dict.fromkeys(keys) if k in group]
    return by_family(group.assign(**{k: group.best - group[k] for k in keys}), keys)


def comparisons(table: pd.DataFrame, chosen: str) -> dict:
    """The order in use against the two references, per task, Holm-corrected over the two.

    The learned variants are not in this family: whether one of them replaces
    the prior is choose()'s question, tested there.
    """
    out = {}
    for task, group in table.groupby("task"):
        tests = {}
        for other in REFERENCES:
            if other not in group or other == chosen or group[other].isna().all():
                continue
            # Only where both ran: TabPFN skips large datasets, and a family's
            # unit is then averaged over the members it ran on, for both sides.
            both = group[group[chosen].notna() & group[other].notna()]
            units = regret_units(both, [chosen, other])
            tests[other] = paired(units[chosen].to_numpy(), units[other].to_numpy())
            tests[other]["units"] = int(len(units))
        adjusted = holm({other: t["p"] for other, t in tests.items()})
        for other, t in tests.items():
            t["p_holm"] = adjusted[other]
        out[task] = tests
    return out


def report_comparisons(table: pd.DataFrame, chosen: str) -> str:
    lines = [f"{chosen} against the references, paired over independent units (Wilcoxon, Holm-adjusted):"]
    for task, tests in comparisons(table, chosen).items():
        lines.append(f"  {task}")
        for other, t in tests.items():
            lines.append(f"    vs {other:10} better on {t['wins']:>3}, worse on {t['losses']:>3}, tied {t['ties']:>3}"
                         f"   p = {t['p_holm']:.3g}")
    return "\n".join(lines)


def export(frame: pd.DataFrame, taxonomy: dict, out: Path, chosen: str, meta: pd.DataFrame | None = None,
           decisions: list[dict] | None = None, lead: dict | None = None) -> dict:
    """Fit on everything and write the weights the site will use."""
    family = {m["c"]: m["dom"] for m in taxonomy["MODELS"]}
    payload = {
        "version": 1,
        "chosen": chosen,
        # A reference shown before the order's first pick, when it passed the
        # same rule against the order (choose_lead()).
        "lead": lead,
        "trained_on": {"datasets": int(frame.dataset.nunique()), "rows": int(len(frame))},
        "target": "percentile rank of a model among those that ran on the same dataset",
        "features": FEATURE_NAMES,
        "families": {},
        "tasks": {},
        # How the order in use was chosen, by the rule in choose().
        "choice": decisions or [],
    }
    for task, group in frame.groupby("task"):
        prior = fit_prior(group)
        weights = fit_interactions(group, prior) if chosen == "prior_fit" else {}
        payload["tasks"][TASK_CODE[task]] = {
            "prior": {k: round(v, 4) for k, v in sorted(prior.items())},
            "weights": {k: round(v, 5) for k, v in sorted(weights.items())},
            "default_prior": 0.5,
            "datasets": int(group.dataset.nunique()),
        }
        if chosen == "prior_knn" and meta is not None:
            payload["tasks"][TASK_CODE[task]]["neighbours"] = neighbour_block(group, meta, task, meta_scale(meta))
    payload["families"] = {m["c"]: family[m["c"]] for m in taxonomy["MODELS"]}
    out.write_text(json.dumps(payload, indent=1) + "\n")
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results/after-fixes.csv")
    ap.add_argument("--out", default=str(ROOT / "site" / "taxonomy" / "ranking.json"))
    ap.add_argument("--report", default="results/ranking-lodo.csv")
    ap.add_argument("--meta", default=None, help="meta-features per dataset (default: meta.csv next to the results)")
    args = ap.parse_args(argv)

    results = pd.read_csv(args.results)
    meta = load_meta(Path(args.meta) if args.meta else Path(args.results).parent / "meta.csv")
    taxonomy = load_taxonomy()
    frame = build_frame(results, taxonomy)
    if meta is not None:
        meta = meta.merge(frame[["dataset", "task"]].drop_duplicates(), on=["dataset", "task"])
    evaluation = evaluate(frame, taxonomy, results, meta)

    units = frame.dataset.map(family_of).nunique()
    print(f"{frame.dataset.nunique()} datasets ({units} independent), {len(frame)} model results, "
          f"leave-one-dataset-out with families held out whole\n")
    for task, stats in evaluation["summary"].items():
        print(f"{task}")
        print(f"{'':26}{'median regret':>14}{'was best':>10}{'within 1pt':>12}")
        for key, label in [("current", "counting coordinates"), ("prior", "learned prior"),
                           ("prior_fit", "prior + interactions"), ("prior_knn", "prior + neighbours"),
                           ("boosting", "always boosting"), ("tuned", "boosting, tuned"), ("tabpfn", "TabPFN")]:
            if key not in stats:
                continue
            s = stats[key]
            print(f"  {label:24}{s['median_regret']:>14.3f}{s['was_best']:>10.0%}{s['within_one_point']:>12.0%}")
        print()

    chosen, decisions = choose(evaluation["table"])
    for decision in decisions:
        print(describe(decision))
    print(f"chosen: {chosen}")
    lead = choose_lead(evaluation["table"], chosen)
    if lead:
        print(describe({**lead, "replaced": lead["led"]}).replace("replaces", "goes before")
              .replace("does not replace", "does not go before"))
    print()
    print(report_comparisons(evaluation["table"], chosen))

    evaluation["table"].round(4).to_csv(args.report, index=False)
    payload = export(frame, taxonomy, Path(args.out), chosen, meta, decisions, lead)
    print(f"wrote {args.out} ({len(payload['tasks'])} tasks) and {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
