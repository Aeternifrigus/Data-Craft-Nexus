"""Should the width of a table decide which anomaly detector comes first?

anomaly.py fitted every detector without labels on ADBench's tables and scored
each against the labels. This asks the page's question: does reading the table
(how many columns it has, the only thing about it the page can measure that the
mathematics says should matter) pick a better first detector than always
showing the same one?

Two orders, fitted on the other units and judged on the one held out:

  fixed   one order for every table: each detector's average percentile rank
          among the detectors that ran, over all training tables
  kind    the same, kept separately for narrow (at most 10 columns), middling
          (11 to 50) and wide tables (more than 50), and shrunk toward the fixed
          order by KIND_STRENGTH tables' worth

Regret is how far a pick landed below the best detector on that table, in ROC
AUC and in average precision. A unit's regret is its tables' mean; tables cut
from one source (the four Wisconsin breast-cancer tables, say) are one unit.

The rule, fixed before the run, is choose() from forecast_learn.py: the kind
order replaces the fixed one only if its median regret over units is no worse
under either score and better by more than luck under at least one (paired
Wilcoxon, Holm over the two scores, more wins than losses).

  python -m dcn.anomaly_learn --results results/anomaly.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .anomaly import DETECTORS, HIGH_COLUMNS, LOW_COLUMNS
from .forecast_learn import choose as choose_rule
from .significance import holm, paired  # noqa: F401  (the rule's tests, through choose_rule)

ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = list(DETECTORS)
METRICS = ["auc", "ap"]
KIND_STRENGTH = 5.0     # fixed before the run
KINDS = {
    "low": f"narrow (at most {LOW_COLUMNS} columns)",
    "mid": f"middling ({LOW_COLUMNS + 1} to {HIGH_COLUMNS} columns)",
    "high": f"wide (more than {HIGH_COLUMNS} columns)",
}


def load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame[frame.status == "ok"].copy()


def wide(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    table = frame.pivot_table(index=["dataset", "unit", "kind"], columns="detector", values=metric, aggfunc="first")
    table = table[[c for c in CANDIDATES if c in table]]
    return table[table.notna().sum(axis=1) >= 2]


def fit_fixed(ranks: pd.DataFrame, strength: float = 4.0) -> dict[str, float]:
    return {c: float((ranks[c].sum() + strength * 0.5) / (ranks[c].notna().sum() + strength)) for c in ranks.columns}


def fit_kind(ranks: pd.DataFrame, fixed: dict[str, float]) -> dict[str, dict[str, float]]:
    return {kind: {c: float((g[c].sum() + KIND_STRENGTH * fixed[c]) / (g[c].notna().sum() + KIND_STRENGTH))
                   for c in ranks.columns}
            for kind, g in ranks.groupby(level="kind")}


def pick(table: pd.DataFrame, order) -> pd.Series:
    out = {}
    for index, row in table.iterrows():
        prior = order(index[2])
        ran = [c for c in CANDIDATES if c in row and pd.notna(row[c])]
        out[index] = max(ran, key=lambda c: prior.get(c, 0.5)) if ran else None
    return pd.Series(out)


def evaluate(frame: pd.DataFrame) -> pd.DataFrame:
    """Leave one unit out: both orders fitted on the other units, judged on the held-out one."""
    rows = []
    for metric in METRICS:
        table = wide(frame, metric)
        ranks = table.rank(axis=1, pct=True)
        units = table.index.get_level_values("unit")
        for unit in sorted(set(units)):
            train, test = ranks[units != unit], table[units == unit]
            fixed = fit_fixed(train)
            kind = fit_kind(train, fixed)
            for strategy, order in (("fixed", lambda k: fixed), ("kind", lambda k: kind.get(k, fixed))):
                chosen = pick(test, order)
                best = test.max(axis=1)
                for index, code in chosen.items():
                    rows.append({"metric": metric, "strategy": strategy, "dataset": index[0], "unit": index[1],
                                 "kind": index[2], "pick": code, "regret": float(best[index] - test.loc[index, code])})
    return pd.DataFrame(rows)


def per_kind(frame: pd.DataFrame, per_dataset: pd.DataFrame) -> dict:
    out = {}
    table = wide(frame, "auc")
    for kind in KINDS:
        rows = table[table.index.get_level_values("kind") == kind]
        if not len(rows):
            continue
        winners = rows.idxmax(axis=1).value_counts(normalize=True)
        entry = {
            "datasets": int(len(rows)), "units": int(rows.index.get_level_values("unit").nunique()),
            "best_share": {c: round(float(winners.get(c, 0.0)), 3) for c in CANDIDATES if c in rows},
            "median_auc": {c: round(float(rows[c].median()), 3) for c in CANDIDATES if c in rows},
            "worse_than_chance": {c: round(float((rows[c] < 0.5).mean()), 3) for c in CANDIDATES if c in rows},
        }
        for metric in METRICS:
            here = per_dataset[(per_dataset.metric == metric) & (per_dataset.kind == kind)]
            entry[f"median_regret_{metric}"] = {s: round(float(g.regret.median()), 4) for s, g in here.groupby("strategy")}
        out[kind] = entry
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results/anomaly.csv")
    ap.add_argument("--out", default=str(ROOT / "site" / "taxonomy" / "anomaly.json"))
    ap.add_argument("--report", default="results/anomaly-lodo.csv")
    args = ap.parse_args(argv)

    frame = load(Path(args.results))
    raw = pd.read_csv(args.results)
    per_dataset = evaluate(frame)
    # choose() reads units from a "unit" column and scores from "metric"; the same rule as forecasting.
    decision = choose_rule(per_dataset.assign(series=per_dataset.dataset), METRICS)
    kinds = per_kind(frame, per_dataset)

    priors, kind_priors = {}, {}
    for metric in METRICS:
        table = wide(frame, metric)
        ranks = table.rank(axis=1, pct=True)
        fixed = fit_fixed(ranks)
        priors[metric] = {c: round(v, 4) for c, v in sorted(fixed.items())}
        kind_priors[metric] = {k: {c: round(v, 4) for c, v in sorted(p.items())} for k, p in fit_kind(ranks, fixed).items()}
    units = {m: per_dataset[per_dataset.metric == m].pivot_table(index="unit", columns="strategy", values="regret",
                                                                aggfunc="mean") for m in METRICS}
    datasets = frame.drop_duplicates("dataset")
    payload = {
        "version": 1,
        "chosen": "kind" if decision["replaced"] else "fixed",
        "decision": decision,
        "fixed_before": "the kinds, their cut-offs, KIND_STRENGTH and the rule were committed before the run",
        "kinds": KINDS,
        "thresholds": {"low_columns": LOW_COLUMNS, "high_columns": HIGH_COLUMNS, "kind_strength": KIND_STRENGTH},
        "prior": priors,
        "kind_prior": kind_priors,
        "per_kind": kinds,
        "median_unit_regret": {m: {s: round(float(u[s].median()), 4) for s in u.columns} for m, u in units.items()},
        "datasets": int(len(datasets)),
        "units": int(datasets.unit.nunique()),
        "failures": {c: int(((raw.detector == c) & (raw.status != "ok")).sum()) for c in CANDIDATES},
        "runs": int(len(raw)),
    }
    Path(args.out).write_text(json.dumps(payload, indent=1) + "\n")
    per_dataset.to_csv(args.report, index=False)

    print(f"{len(datasets)} tables ({datasets.unit.nunique()} independent units), leave one unit out\n")
    for metric, d in decision["metrics"].items():
        print(f"{metric:4} kind vs fixed: better on {d['wins']}, worse on {d['losses']}, tied {d['ties']} units, "
              f"p = {d['p_holm']:.3g} (Holm); median unit regret {d['median_regret']}")
    print(f"kind {'replaces' if decision['replaced'] else 'does not replace'} fixed\n")
    for kind, e in kinds.items():
        print(f"{kind:5} {e['datasets']:3} tables, {e['units']:2} units, best {e['best_share']}, median AUC {e['median_auc']}")
    print(f"\nwrote {args.out} and {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
