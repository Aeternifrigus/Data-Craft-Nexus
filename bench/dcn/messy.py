"""What messy data does to the models, to the profiler, and to the advice.

corrupt.py damages clean datasets in known ways; run.py --corrupt runs every
model on the damaged copies. This compares those runs with the clean run of
the same datasets (the first seed of results/full.csv) and answers:

  damage      how much each model's score falls, per kind of damage: the
              median over independent units of clean minus damaged
  detected    whether the profiler notices: the share of damaged datasets
              whose signature carries A62 (missing values) or A64 (dirty
              values). Label noise cannot be seen in a file, and is reported
              as what it is
  order       whether the order the site uses still holds: the regret of its
              first pick, and of default boosting, on the clean datasets and
              on the damaged ones, and how often the first pick changes

The first pick is replayed here, clean and damaged, with the order that
ships now, exactly as the runner makes it: load, subsample, measure, damage,
measure again, rank.

It also looks at the datasets in the full run that arrived with missing
values of their own (natural): whether the order's leave-one-dataset-out
regret there differs from the rest.

  python -m dcn.messy --messy results/messy.csv --clean results/full.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import datasets as ds
from .corrupt import CONDITIONS, damage
from .models import BASELINE
from .recommend import load_taxonomy, rank_models
from .run import TASK_CODE, profile_dataset
from .significance import bootstrap_ci, by_family, family_of

MAX_ROWS = 5000     # the runner's default
FLAG = {"missing_10": "A62", "missing_30": "A62", "dirty_5": "A64", "labels_10": None}


def load(messy_path: Path, clean_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    messy = pd.read_csv(messy_path)
    clean = pd.read_csv(clean_path)
    if "condition" in clean:
        clean = clean[clean.condition.fillna("clean").isin(["clean", ""])]
    clean = clean[(clean.seed.astype(int) == 0) & clean.dataset.isin(messy.dataset.unique())
                  & clean.model.isin(messy.model.unique())]
    return messy, clean


def picks(names: list[tuple[str, str]], conditions: list[str], taxonomy: dict | None = None) -> pd.DataFrame:
    """The order's first pick for each dataset, clean and under each condition, as the runner makes it."""
    taxonomy = taxonomy or load_taxonomy()
    rows = []
    for name, task in names:
        dataset = ds.load(name)
        if len(dataset.frame) > MAX_ROWS:
            dataset.frame = dataset.frame.sample(MAX_ROWS, random_state=0).reset_index(drop=True)
        clean_frame = dataset.frame
        measured = profile_dataset(dataset)
        for condition in ["clean", *conditions]:
            if condition != "clean":
                dataset.frame = damage(clean_frame, condition, measured["numeric"])
                now = profile_dataset(dataset)
                dataset.frame = clean_frame
            else:
                now = measured
            ranking = rank_models(taxonomy, now["signature"], TASK_CODE[task], limit=1, meta=now["meta"])
            sig = now["signature"]
            rows.append({"dataset": name, "task": task, "condition": condition,
                         "first": ranking.items[0]["c"] if ranking.items else None,
                         "signature": " ".join(sig["codes"]), "flags": " ".join(sig["flags"]),
                         "numeric": len(measured["numeric"])})
    return pd.DataFrame(rows)


def scores(frame: pd.DataFrame) -> pd.DataFrame:
    """dataset x model scores for the runs that finished."""
    ok = frame[frame.status == "ok"]
    return ok.pivot_table(index=["dataset", "task"], columns="model", values="score", aggfunc="first")


def regrets(table: pd.DataFrame, first: pd.Series) -> pd.DataFrame:
    """Per dataset: best score, and how far the first pick and default boosting landed below it.

    A first pick that did not finish on a dataset is scored as the worst model
    that did, which is what following the advice would have got.
    """
    best = table.max(axis=1)
    worst = table.min(axis=1)
    chosen = [table.at[idx, code] if code in table.columns and pd.notna(table.at[idx, code]) else worst[idx]
              for idx, code in zip(table.index, first.reindex(table.index))]
    return pd.DataFrame({"regret_first": best - np.array(chosen, dtype=float),
                         "regret_boosting": best - table.get(BASELINE.code, pd.Series(np.nan, index=table.index))},
                        index=table.index)


def median_units(values: pd.DataFrame, column: str) -> float:
    frame = values.reset_index()
    return round(float(by_family(frame, [column])[column].median()), 4)


def analyse(messy: pd.DataFrame, clean: pd.DataFrame, chosen: pd.DataFrame) -> dict:
    clean_scores = scores(clean)
    out = {"conditions": {}, "datasets": {}, "units": {}}
    for task, group in clean.groupby("task"):
        out["datasets"][task] = int(group.dataset.nunique())
        out["units"][task] = int(group.dataset.map(family_of).nunique())

    first_of = {cond: g.set_index(["dataset", "task"])["first"] for cond, g in chosen.groupby("condition")}
    signature_of = {cond: g.set_index(["dataset", "task"]) for cond, g in chosen.groupby("condition")}
    clean_regret = regrets(clean_scores, first_of["clean"])

    for condition, runs in messy.groupby("condition"):
        damaged = scores(runs)
        common = damaged.index.intersection(clean_scores.index)
        before, after = clean_scores.loc[common], damaged.loc[common]
        loss = (before - after)                     # positive: the damage cost this much
        entry = {"description": CONDITIONS[condition], "tasks": {}}

        # Did the profiler notice?
        flag = FLAG[condition]
        sigs = signature_of[condition].loc[common]
        clean_sigs = signature_of["clean"].loc[common]
        if flag:
            # Junk only goes into numeric columns: a table without one was not damaged.
            damaged_any = sigs.numeric > 0 if condition == "dirty_5" else pd.Series(True, index=sigs.index)
            entry["flag"] = flag
            entry["damaged"] = int(damaged_any.sum())
            entry["flagged"] = round(float(sigs[damaged_any].signature.str.contains(flag).mean()), 3)
            entry["flagged_clean"] = round(float(clean_sigs[damaged_any].signature.str.contains(flag).mean()), 3)
        else:
            entry["flag"] = None

        after_regret = regrets(after, first_of[condition].reindex(common))
        changed = (first_of[condition].reindex(common) != first_of["clean"].reindex(common))
        for task in sorted({t for _, t in common}):
            rows = [idx for idx in common if idx[1] == task]
            per_model = {}
            for model in loss.columns:
                values = loss.loc[rows, model].dropna()
                if len(values):
                    frame = values.rename("loss").reset_index()
                    units = by_family(frame, ["loss"])["loss"]
                    per_model[model] = {"median": round(float(units.median()), 4), "units": int(len(units))}
            all_loss = loss.loc[rows].stack().rename("loss").reset_index()
            all_units = by_family(all_loss.groupby("dataset", as_index=False).loss.median(), ["loss"])["loss"]
            reg_clean, reg_after = clean_regret.loc[rows], after_regret.loc[rows]
            diff = (reg_after.regret_first - reg_clean.regret_first).rename("diff").reset_index()
            diff_units = by_family(diff, ["diff"])["diff"]
            detection = {}
            if flag:
                hit = damaged_any.loc[rows]
                detection = {"damaged": int(hit.sum()),
                             "flagged": round(float(sigs.loc[rows][hit].signature.str.contains(flag).mean()), 3)}
            entry["tasks"][task] = {
                **detection,
                "datasets": len(rows),
                "units": int(len(all_units)),
                "median_loss": round(float(all_units.median()), 4),
                "models": dict(sorted(per_model.items(), key=lambda kv: kv[1]["median"])),
                "first_regret_clean": median_units(reg_clean, "regret_first"),
                "first_regret": median_units(reg_after, "regret_first"),
                "boosting_regret_clean": median_units(reg_clean, "regret_boosting"),
                "boosting_regret": median_units(reg_after, "regret_boosting"),
                "first_regret_change_ci": [round(x, 4) for x in bootstrap_ci(diff_units.to_numpy())],
                "first_changed": round(float(changed.loc[rows].mean()), 3),
            }
        out["conditions"][condition] = entry
    return out


def natural(lodo_path: Path, full: pd.DataFrame, chosen_key: str) -> dict | None:
    """Datasets that came with missing values: the order's held-out regret there and elsewhere."""
    if not lodo_path.exists():
        return None
    lodo = pd.read_csv(lodo_path)
    sig = full.drop_duplicates(["dataset", "task"]).set_index(["dataset", "task"]).signature
    lodo["missing"] = [("A62" in sig.get((d, t), "")) or ("A64" in sig.get((d, t), ""))
                       for d, t in zip(lodo.dataset, lodo.task)]
    out = {"strategy": chosen_key, "tasks": {}}
    for task, group in lodo.groupby("task"):
        keys = [k for k in [chosen_key, "boosting", "tuned"] if k in group]
        units = by_family(group.assign(**{k: group.best - group[k] for k in keys}).assign(
            missing=group.missing.astype(float)), keys + ["missing"])
        with_missing = units[units.missing > 0]
        complete = units[units.missing == 0]
        # Is the order's regret on these units different from the rest? Two
        # independent groups of units, so Mann-Whitney, two-sided.
        p = (float(stats.mannwhitneyu(with_missing[chosen_key], complete[chosen_key]).pvalue)
             if len(with_missing) >= 2 and len(complete) >= 2 else None)
        out["tasks"][task] = {
            "datasets": int(group.missing.sum()),
            "names": sorted(group[group.missing].dataset),
            "units": int(len(with_missing)),
            "complete_units": int(len(complete)),
            **{f"{k}_regret": round(float(with_missing[k].median()), 4) if len(with_missing) else None for k in keys},
            **{f"{k}_regret_complete": round(float(complete[k].median()), 4) for k in keys},
            "p": round(p, 4) if p is not None else None,
        }
    return out


def evidence(messy_path: Path, clean_path: Path, taxonomy: dict | None = None,
             picks_cache: Path | None = None) -> dict:
    messy, clean = load(messy_path, clean_path)
    names = sorted(set(zip(clean.dataset, clean.task)) & set(zip(messy.dataset, messy.task)))
    conditions = sorted(messy.condition.unique())
    if picks_cache and picks_cache.exists():
        chosen = pd.read_csv(picks_cache)
    else:
        chosen = picks(names, conditions, taxonomy)
        if picks_cache:
            chosen.to_csv(picks_cache, index=False)
    ranking = json.loads((Path(__file__).resolve().parents[2] / "site" / "taxonomy" / "ranking.json").read_text())
    block = analyse(messy, clean, chosen)
    full = pd.read_csv(clean_path)
    block["natural"] = natural(clean_path.parent / "ranking-lodo.csv", full, ranking.get("chosen", "prior"))
    counts = " and ".join(f"{n} {task}" for task, n in block["datasets"].items())
    block["how"] = (f"{counts} benchmark datasets, damaged on purpose and run again with every model: cells "
                    "blanked at random, junk text in numeric columns, and training labels swapped for other rows' "
                    "labels. Every model is scored on the true labels, and compared with its own score on the "
                    "same dataset clean.")
    return block


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--messy", default="results/messy.csv")
    ap.add_argument("--clean", default="results/full.csv")
    ap.add_argument("--picks", default="results/messy-picks.csv", help="the replayed first picks (cached)")
    args = ap.parse_args(argv)
    block = evidence(Path(args.messy), Path(args.clean), picks_cache=Path(args.picks))
    for condition, entry in block["conditions"].items():
        print(f"\n{condition}: {entry['description']}")
        if entry["flag"]:
            print(f"  profiler flagged {entry['flag']} on {entry['flagged']:.0%} of {entry['damaged']} damaged datasets"
                  f" (clean: {entry['flagged_clean']:.0%})")
        for task, t in entry["tasks"].items():
            print(f"  {task} ({t['datasets']} datasets, {t['units']} units): median loss {t['median_loss']:.4f}")
            print(f"    first pick regret {t['first_regret_clean']:.4f} -> {t['first_regret']:.4f}"
                  f" (change CI {t['first_regret_change_ci']}), boosting {t['boosting_regret_clean']:.4f}"
                  f" -> {t['boosting_regret']:.4f}, first pick changed on {t['first_changed']:.0%}")
            models = list(t["models"].items())
            print("    least hurt: " + ", ".join(f"{m} {v['median']:.4f}" for m, v in models[:4]))
            print("    most hurt:  " + ", ".join(f"{m} {v['median']:.4f}" for m, v in models[-4:]))
    print(json.dumps(block["natural"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
