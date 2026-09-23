"""Learn the order to show models in, instead of counting matched coordinates.

The benchmark showed that counting how many of a model's three codes appear in
the signature ranks models by how *describable* they are, not by how well they
do. This learns the order from the run instead.

Target: a model's percentile rank among the models that ran on the same
dataset. Rank rather than score, because R² and balanced accuracy are not
comparable and a single catastrophic R² would dominate a regression.

Two candidates, because with 20 datasets per task the simpler one may well win:

  prior         what a model is worth on average, shrunk toward the middle
                so a model seen on few datasets is not trusted too far
  prior+fit     the prior plus ridge-fitted interactions between the dataset's
                measured features and the model's family

Both are judged leave-one-dataset-out: the held-out dataset never contributes
to the weights that rank it. They are compared against what the site does now
and against always reaching for boosting.

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

from .models import BASELINE
from .recommend import conflict, load_taxonomy, rank_models
from .significance import collapse_seeds, holm, paired

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
    ok = results[(results.status == "ok") & (results.model != BASELINE.code)].copy()
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


def evaluate(frame: pd.DataFrame, taxonomy: dict, results: pd.DataFrame) -> dict:
    """Leave one dataset out, rank its models with weights fitted on the rest."""
    scores = {k: [] for k in ["current", "prior", "prior_fit", "boosting", "oracle"]}
    per_dataset = []
    family = {m["c"]: m["dom"] for m in taxonomy["MODELS"]}

    results = collapse_seeds(results)
    for (dataset, task), group in frame.groupby(["dataset", "task"]):
        train = frame[(frame.dataset != dataset) & (frame.task == task)]
        prior = fit_prior(train)
        weights = fit_interactions(train, prior)
        features = {name: group.iloc[0][name] for name in FEATURE_NAMES}

        best = group.score.max()
        run = results[(results.dataset == dataset) & (results.task == task)]
        boosting_row = run[(run.model == BASELINE.code) & (run.status == "ok")]
        boosting = float(boosting_row.score.iloc[0]) if len(boosting_row) else np.nan

        # The order the learned one replaced: the first model by coordinates
        # matched. The empty ranking matters. Without it rank_models reads the
        # committed ranking.json, which was fitted on every dataset including
        # this one, and "current" would quietly become an in-sample prior.
        signature = {"codes": run.signature.iloc[0].split(), "flags": []}
        ranking = rank_models(taxonomy, signature, TASK_CODE[task], limit=99, ranking={"tasks": {}})
        eligible = [m["c"] for m in ranking.items if m["c"] in set(group.model)]
        by_model = dict(zip(group.model, group.score))
        current = by_model.get(eligible[0]) if eligible else np.nan

        def pick(scorer):
            ordered = sorted(eligible, key=scorer, reverse=True)
            return by_model[ordered[0]] if ordered else np.nan

        prior_pick = pick(lambda m: prior.get(m, 0.5))
        fit_pick = pick(lambda m: predict(m, family.get(m, "?"), features, prior, weights))

        for key, value in [("current", current), ("prior", prior_pick), ("prior_fit", fit_pick),
                           ("boosting", boosting), ("oracle", best)]:
            scores[key].append(best - value if value is not None else np.nan)
        per_dataset.append({"dataset": dataset, "task": task, "best": best, "current": current,
                            "prior": prior_pick, "prior_fit": fit_pick, "boosting": boosting})

    table = pd.DataFrame(per_dataset)
    summary = {}
    for task, group in table.groupby("task"):
        summary[task] = {}
        for key in ["current", "prior", "prior_fit", "boosting"]:
            regret = (group.best - group[key]).dropna()
            summary[task][key] = {
                "median_regret": round(float(regret.median()), 4),
                "mean_regret": round(float(regret.mean()), 4),
                "was_best": round(float((regret <= 1e-9).mean()), 3),
                "within_one_point": round(float((regret <= 0.01).mean()), 3),
            }
    return {"summary": summary, "table": table}


STRATEGIES = ["current", "prior", "prior_fit", "boosting"]


def comparisons(table: pd.DataFrame, chosen: str) -> dict:
    """The order in use against every alternative, per task, with Holm's correction."""
    out = {}
    for task, group in table.groupby("task"):
        regret = {key: (group.best - group[key]).to_numpy() for key in STRATEGIES if key in group}
        tests = {other: paired(regret[chosen], regret[other]) for other in regret if other != chosen}
        adjusted = holm({other: t["p"] for other, t in tests.items()})
        for other, t in tests.items():
            t["p_holm"] = adjusted[other]
        out[task] = tests
    return out


def report_comparisons(table: pd.DataFrame, chosen: str) -> str:
    lines = [f"{chosen} against each alternative, paired over datasets (Wilcoxon, Holm-adjusted):"]
    for task, tests in comparisons(table, chosen).items():
        lines.append(f"  {task}")
        for other, t in tests.items():
            lines.append(f"    vs {other:10} better on {t['wins']:>3}, worse on {t['losses']:>3}, tied {t['ties']:>3}"
                         f"   p = {t['p_holm']:.3g}")
    return "\n".join(lines)


def export(frame: pd.DataFrame, taxonomy: dict, out: Path, chosen: str) -> dict:
    """Fit on everything and write the weights the site will use."""
    family = {m["c"]: m["dom"] for m in taxonomy["MODELS"]}
    payload = {
        "version": 1,
        "chosen": chosen,
        "trained_on": {"datasets": int(frame.dataset.nunique()), "rows": int(len(frame))},
        "target": "percentile rank of a model among those that ran on the same dataset",
        "features": FEATURE_NAMES,
        "families": {},
        "tasks": {},
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
    payload["families"] = {m["c"]: family[m["c"]] for m in taxonomy["MODELS"]}
    out.write_text(json.dumps(payload, indent=1) + "\n")
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results/after-fixes.csv")
    ap.add_argument("--out", default=str(ROOT / "site" / "taxonomy" / "ranking.json"))
    ap.add_argument("--report", default="results/ranking-lodo.csv")
    args = ap.parse_args(argv)

    results = pd.read_csv(args.results)
    taxonomy = load_taxonomy()
    frame = build_frame(results, taxonomy)
    evaluation = evaluate(frame, taxonomy, results)

    print(f"{frame.dataset.nunique()} datasets, {len(frame)} model results, leave-one-dataset-out\n")
    for task, stats in evaluation["summary"].items():
        print(f"{task}")
        print(f"{'':26}{'median regret':>14}{'was best':>10}{'within 1pt':>12}")
        for key, label in [("current", "counting coordinates"), ("prior", "learned prior"),
                           ("prior_fit", "prior + interactions"), ("boosting", "always boosting")]:
            s = stats[key]
            print(f"  {label:24}{s['median_regret']:>14.3f}{s['was_best']:>10.0%}{s['within_one_point']:>12.0%}")
        print()

    # Pick by median regret across both tasks, ties going to the simpler model.
    def total(key):
        return sum(stats[key]["median_regret"] for stats in evaluation["summary"].values())
    chosen = "prior" if total("prior") <= total("prior_fit") else "prior_fit"
    print(f"chosen: {chosen} (prior {total('prior'):.3f} vs prior+interactions {total('prior_fit'):.3f}, "
          f"counting coordinates {total('current'):.3f}, boosting {total('boosting'):.3f})\n")
    print(report_comparisons(evaluation["table"], chosen))

    evaluation["table"].round(4).to_csv(args.report, index=False)
    payload = export(frame, taxonomy, Path(args.out), chosen)
    print(f"wrote {args.out} ({len(payload['tasks'])} tasks) and {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
