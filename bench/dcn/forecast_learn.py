"""Should the kind of series decide which forecaster comes first?

forecast.py ran the page's forecasting methods on real series. This asks the
question the page exists for: does measuring a series (intermittent, seasonal,
trending: dcn/series.py) pick a better first forecaster than always showing
the same one?

Two orders, both fitted on other collections and judged on the one held out:

  fixed   one order for every series: each forecaster's average percentile
          rank among the forecasters that ran, over all training series
  kind    the same, kept separately for each kind of series and shrunk toward
          the fixed order by KIND_STRENGTH series' worth, so a kind seen
          rarely is not trusted far

Regret is how far a pick landed below the best forecaster on that series: in
R² for the page's default score, and in mean absolute error divided by doing
nothing's (carrying the last value forward) on the same rows, for "every unit
of error costs the same". A collection's regret is its series' mean, and the
M5 collections, cut from one table, count as one unit.

The rule, fixed before the run (choose()): the kind order replaces the fixed
order only if its median regret over units is no worse under either score and
it is better by more than luck under at least one: paired Wilcoxon over units,
Holm-corrected over the two scores, more wins than losses. It is the rule every
other change of order on the page has had to pass.

The first run could not decide: the orders disagree only on intermittent
demand, which came from four collections, and four cannot give Wilcoxon a p
below 0.125. The confirmation run (confirm(), bench/README.md "Forecasting,
second run") freezes both orders as the first run fitted them and judges them
on intermittent series from nine collections the first run never saw, by the
same rule. Its verdict is the one the page follows.

  python -m dcn.forecast_learn --results results/forecast.csv --confirm results/forecast-2.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .series import CELLS, INTERMITTENT_ADI, LUMPY_CV2, STRENGTH
from .significance import holm, paired

ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = ["TSM1", "TSM2", "TSM4", "TSM5"]     # what the page orders: the forecasting models the script runs
NOTHING = ["NAIVE-LAST", "NAIVE-SEASON", "NAIVE-AVERAGE"]
REFERENCE = "BASE-HGB-TUNED"
METRICS = ["r2", "mae"]
KIND_STRENGTH = 20.0    # fixed before the run
ALPHA = 0.05


def load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame[frame.status == "ok"].copy()


def wide(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One row per series, one column per method, higher is better."""
    table = frame.pivot_table(index=["dataset", "unit", "series", "cell"], columns="method", values=metric,
                              aggfunc="first")
    return table if metric == "r2" else -table


def targets(table: pd.DataFrame) -> pd.DataFrame:
    """Each candidate's percentile rank among the candidates that ran on a series: 1.0 is the best."""
    ran = table[[c for c in CANDIDATES if c in table]]
    return ran.rank(axis=1, pct=True)


def regret(table: pd.DataFrame, metric: str, pick: pd.Series, frame: pd.DataFrame) -> pd.Series:
    """How far each series' pick landed below its best candidate."""
    ran = table[[c for c in CANDIDATES if c in table]]
    best = ran.max(axis=1)
    got = pd.Series([ran.loc[i, p] if isinstance(p, str) else np.nan for i, p in pick.items()], index=pick.index)
    gap = best - got
    if metric == "mae":   # in units of doing nothing's error on the same rows
        scale = -table["NAIVE-LAST"] if "NAIVE-LAST" in table else np.nan
        gap = gap / scale.where(scale > 0)
    return gap


def fit_fixed(ranks: pd.DataFrame, strength: float = 4.0) -> dict[str, float]:
    return {c: float((ranks[c].sum() + strength * 0.5) / (ranks[c].notna().sum() + strength))
            for c in ranks.columns}


def fit_kind(ranks: pd.DataFrame, fixed: dict[str, float]) -> dict[str, dict[str, float]]:
    out = {}
    for cell, group in ranks.groupby(level="cell"):
        out[cell] = {c: float((group[c].sum() + KIND_STRENGTH * fixed[c]) / (group[c].notna().sum() + KIND_STRENGTH))
                     for c in ranks.columns}
    return out


def pick(table: pd.DataFrame, order) -> pd.Series:
    """The first candidate by `order` among those that ran on each series."""
    out = {}
    for index, row in table.iterrows():
        prior = order(index[3])
        ran = [c for c in CANDIDATES if c in row and pd.notna(row[c])]
        out[index] = max(ran, key=lambda c: prior.get(c, 0.5)) if ran else None
    return pd.Series(out)


def evaluate(frame: pd.DataFrame) -> dict:
    """Leave one unit out: both orders fitted on the other units, judged on the held-out one."""
    per_series = []
    for metric in METRICS:
        table = wide(frame, metric)
        table = table[table[[c for c in CANDIDATES if c in table]].notna().sum(axis=1) >= 2]
        ranks = targets(table)
        units = table.index.get_level_values("unit")
        for unit in sorted(set(units)):
            train, test = ranks[units != unit], table[units == unit]
            fixed = fit_fixed(train)
            kind = fit_kind(train, fixed)
            picks = {"fixed": pick(test, lambda cell: fixed),
                     "kind": pick(test, lambda cell: kind.get(cell, fixed))}
            for strategy, chosen in picks.items():
                gaps = regret(test, metric, chosen, frame)
                for (dataset, u, series, cell), gap in gaps.items():
                    per_series.append({"metric": metric, "strategy": strategy, "dataset": dataset, "unit": u,
                                       "series": series, "cell": cell, "pick": chosen[(dataset, u, series, cell)],
                                       "regret": gap})
    return {"per_series": pd.DataFrame(per_series)}


def unit_regret(per_series: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = per_series[per_series.metric == metric]
    return rows.pivot_table(index="unit", columns="strategy", values="regret", aggfunc="mean")


def choose(per_series: pd.DataFrame, metrics: list[str] | None = None) -> dict:
    """The rule fixed before the run: kind replaces fixed only if no worse on both scores and better on one."""
    tests, no_worse = {}, True
    for metric in metrics or METRICS:
        units = unit_regret(per_series, metric).dropna()
        if units["kind"].median() > units["fixed"].median() + 1e-9:
            no_worse = False
        tests[metric] = paired(units["kind"].to_numpy(), units["fixed"].to_numpy())
        tests[metric]["median"] = {"kind": float(units["kind"].median()), "fixed": float(units["fixed"].median())}
    adjusted = holm({m: t["p"] for m, t in tests.items()})
    better = [m for m, t in tests.items() if adjusted[m] < ALPHA and t["wins"] > t["losses"]]
    return {
        "candidate": "kind", "against": "fixed", "replaced": bool(no_worse and better), "no_worse": no_worse,
        "better_on": better,
        "metrics": {m: {"units": t["datasets"], "wins": t["wins"], "losses": t["losses"], "ties": t["ties"],
                        "p_holm": float(f"{adjusted[m]:.4g}"),
                        "median_regret": {k: round(v, 4) for k, v in t["median"].items()}}
                    for m, t in tests.items()},
    }


def frozen_orders(frame: pd.DataFrame) -> dict:
    """Both orders as the first run fits them on all of its series, per score."""
    out = {}
    for metric in METRICS:
        table = wide(frame, metric)
        table = table[table[[c for c in CANDIDATES if c in table]].notna().sum(axis=1) >= 2]
        ranks = targets(table)
        fixed = fit_fixed(ranks)
        out[metric] = {"fixed": fixed, "kind": fit_kind(ranks, fixed)}
    return out


def confirm(first: pd.DataFrame, second: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """The confirmation run: both orders frozen from the first run, judged on the second run's collections."""
    orders = frozen_orders(first)
    rows = []
    for metric in METRICS:
        table = wide(second, metric)
        table = table[table[[c for c in CANDIDATES if c in table]].notna().sum(axis=1) >= 2]
        fixed, kind = orders[metric]["fixed"], orders[metric]["kind"]
        for strategy, order in (("fixed", lambda cell: fixed), ("kind", lambda cell: kind.get(cell, fixed))):
            chosen = pick(table, order)
            for (dataset, unit, series, cell), gap in regret(table, metric, chosen, second).items():
                rows.append({"metric": metric, "strategy": strategy, "dataset": dataset, "unit": unit,
                             "series": series, "cell": cell, "pick": chosen[(dataset, unit, series, cell)],
                             "regret": gap})
    per_series = pd.DataFrame(rows)
    return choose(per_series), per_series


def per_kind(frame: pd.DataFrame, per_series: pd.DataFrame) -> dict:
    """What happened on each kind of series: who won, how often nothing beat every model, what each order cost."""
    out = {}
    table = wide(frame, "r2")
    for cell in CELLS:
        rows = table[table.index.get_level_values("cell") == cell]
        ran = rows[[c for c in CANDIDATES if c in rows]]
        ran = ran[ran.notna().sum(axis=1) >= 2]
        if not len(ran):
            continue
        winners = ran.idxmax(axis=1).value_counts(normalize=True)
        nothing = rows.loc[ran.index, [c for c in NOTHING if c in rows]].max(axis=1)
        entry = {
            "series": int(len(ran)),
            "units": int(ran.index.get_level_values("unit").nunique()),
            "best_share": {c: round(float(winners.get(c, 0.0)), 3) for c in CANDIDATES if c in ran},
            "nothing_beat_every_model": round(float((nothing > ran.max(axis=1) + 1e-12).mean()), 3),
        }
        if REFERENCE in rows:
            both = rows.loc[ran.index, REFERENCE].notna()
            if both.any():
                entry["reference_beat_every_model"] = round(float(
                    (rows.loc[ran.index][both][REFERENCE] > ran[both].max(axis=1) + 1e-12).mean()), 3)
                entry["reference_series"] = int(both.sum())
        for metric in METRICS:
            here = per_series[(per_series.metric == metric) & (per_series.cell == cell)]
            entry[f"median_regret_{metric}"] = {s: round(float(g.regret.median()), 4)
                                                for s, g in here.groupby("strategy")}
        out[cell] = entry
    return out


def export(frame: pd.DataFrame, decision: dict, kinds: dict, per_series: pd.DataFrame, out: Path,
           confirmation: dict | None = None) -> dict:
    """Fit both orders on every series and write what the page reads."""
    priors, kind_priors = {}, {}
    for metric in METRICS:
        table = wide(frame, metric)
        table = table[table[[c for c in CANDIDATES if c in table]].notna().sum(axis=1) >= 2]
        ranks = targets(table)
        fixed = fit_fixed(ranks)
        priors[metric] = {c: round(v, 4) for c, v in sorted(fixed.items())}
        kind_priors[metric] = {cell: {c: round(v, 4) for c, v in sorted(p.items())}
                               for cell, p in fit_kind(ranks, fixed).items()}
    summary = {}
    for metric in METRICS:
        units = unit_regret(per_series, metric)
        summary[metric] = {s: round(float(units[s].median()), 4) for s in units.columns}
    series = frame.drop_duplicates(["dataset", "series"])
    # The confirmation run's verdict, when there is one, is what the page follows.
    verdict = confirmation["decision"]["replaced"] if confirmation else decision["replaced"]
    payload = {
        "version": 1,
        "chosen": "kind" if verdict else "fixed",
        "confirmation": confirmation,
        "decision": decision,
        "fixed_before": "the kinds, their thresholds, KIND_STRENGTH and the rule were committed before the run",
        "thresholds": {"intermittent_adi": INTERMITTENT_ADI, "lumpy_cv2": LUMPY_CV2, "strength": STRENGTH,
                       "kind_strength": KIND_STRENGTH},
        "kinds": CELLS,
        "prior": priors,
        "kind_prior": kind_priors,
        "per_kind": kinds,
        "median_unit_regret": summary,
        "series": int(len(series)),
        "units": int(series.unit.nunique()),
        "collections": sorted(series.dataset.unique().tolist()),
        "runs": int(len(frame)),
    }
    out.write_text(json.dumps(payload, indent=1) + "\n")
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results/forecast.csv")
    ap.add_argument("--out", default=str(ROOT / "site" / "taxonomy" / "forecast.json"))
    ap.add_argument("--report", default="results/forecast-lodo.csv")
    ap.add_argument("--confirm", default=None, help="the confirmation run's results (results/forecast-2.csv)")
    ap.add_argument("--confirm-report", default="results/forecast-2-picks.csv")
    args = ap.parse_args(argv)

    frame = load(Path(args.results))
    evaluation = evaluate(frame)
    per_series = evaluation["per_series"]
    decision = choose(per_series)
    kinds = per_kind(frame, per_series)

    series = frame.drop_duplicates(["dataset", "series"])
    print(f"{len(series)} series from {series.dataset.nunique()} collections "
          f"({series.unit.nunique()} independent units), leave one unit out\n")
    for metric, d in decision["metrics"].items():
        print(f"{metric:4} kind vs fixed: better on {d['wins']}, worse on {d['losses']}, tied {d['ties']} units, "
              f"p = {d['p_holm']:.3g} (Holm); median unit regret {d['median_regret']}")
    print(f"kind {'replaces' if decision['replaced'] else 'does not replace'} fixed\n")
    for cell, entry in kinds.items():
        print(f"{cell:15} {entry['series']:4} series, {entry['units']:2} units, best {entry['best_share']}, "
              f"nothing beat every model on {entry['nothing_beat_every_model']:.0%}")

    confirmation = None
    if args.confirm and Path(args.confirm).exists():
        second = load(Path(args.confirm))
        verdict, picks = confirm(frame, second)
        seen = second.drop_duplicates(["dataset", "series"])
        print(f"\nconfirmation: {len(seen)} series from {seen.dataset.nunique()} new collections "
              f"({seen.unit.nunique()} units), both orders frozen from the first run")
        for metric, d in verdict["metrics"].items():
            print(f"{metric:4} kind vs fixed: better on {d['wins']}, worse on {d['losses']}, tied {d['ties']} units, "
                  f"p = {d['p_holm']:.3g} (Holm); median unit regret {d['median_regret']}")
        print(f"kind {'replaces' if verdict['replaced'] else 'does not replace'} fixed")
        confirmation = {
            "decision": verdict,
            "series": int(len(seen)), "units": int(seen.unit.nunique()),
            "collections": sorted(seen.dataset.unique().tolist()),
            "per_kind": per_kind(second, picks),
            "per_unit": {m: {u: {s: round(float(v), 4) for s, v in row.items()}
                             for u, row in unit_regret(picks, m).iterrows()} for m in METRICS},
            "runs": int(len(second)),
        }
        picks.to_csv(args.confirm_report, index=False)

    per_series.to_csv(args.report, index=False)
    export(frame, decision, kinds, per_series, Path(args.out), confirmation)
    print(f"\nwrote {args.out} and {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
