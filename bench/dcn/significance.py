"""Is a difference more than luck?

A median over 20 datasets can move a long way when a few datasets change, so
every comparison the site publishes gets three things here:

  an interval   a bootstrap over datasets: resample which datasets were in
                the benchmark and see how far the median moves
  a paired test the Wilcoxon signed-rank test on the per-dataset differences,
                since both strategies were judged on the same datasets; with
                Holm's correction when several are compared against one
  a ranking     the Friedman test over every model at once, and the Nemenyi
                critical difference: two models whose average ranks differ by
                less than it cannot be told apart on this benchmark

The last two follow Demšar, "Statistical Comparisons of Classifiers over
Multiple Data Sets", JMLR 7 (2006), the usual reference for comparing
learners across datasets.

Repeated cross-validation seeds are averaged into one score per dataset and
model first (collapse_seeds). The seeds measure how much a single split can
move a score; they are not extra datasets, and counting them as such would
make every difference look more certain than it is.

The same goes for datasets generated from one function. PMLB has 54
regression datasets from Friedman's benchmark functions and 14 from
Strogatz's equations: sisters that share a winner. Each such family counts
once (family_of, by_family), so 101 regression datasets are 35 independent
units, and a test that counted all 101 would claim far more certainty than
the data holds.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
from scipy import stats

LEVEL = 0.95

SYNTHETIC_FAMILIES = [(re.compile(r"^\d+_fri_c\d"), "friedman"), (re.compile(r"^strogatz_"), "strogatz"),
                      (re.compile(r"^feynman_"), "feynman"), (re.compile(r"BNG"), "bng")]


def family_of(dataset: str) -> str:
    """The family a dataset was generated in, or the dataset itself when it stands alone."""
    for pattern, family in SYNTHETIC_FAMILIES:
        if pattern.search(dataset):
            return family
    return dataset


def by_family(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """One row per independent unit: the mean of each column over a family's datasets."""
    return frame.assign(family=frame.dataset.map(family_of)).groupby("family")[columns].mean()


def collapse_seeds(results: pd.DataFrame) -> pd.DataFrame:
    """One row per dataset, task and model: the mean score over the seeds that finished.

    A model that finished on no seed keeps the status of its first attempt, so
    a timeout is still reported as a timeout.
    """
    if "seed" not in results or results.groupby(["dataset", "task", "model"]).size().max() == 1:
        out = results.copy()
        out["seeds"] = (out.status == "ok").astype(int)
        return out

    rows = []
    for _, group in results.groupby(["dataset", "task", "model"], sort=False):
        ok = group[group.status == "ok"]
        first = (ok if len(ok) else group).iloc[0].copy()
        if len(ok):
            first["score"] = float(ok.score.astype(float).mean())
            first["score_std"] = float(ok.score.astype(float).std(ddof=0)) if len(ok) > 1 else first.get("score_std")
            first["seconds"] = float(ok.seconds.astype(float).mean())
        first["seeds"] = len(ok)
        rows.append(first)
    return pd.DataFrame(rows).reset_index(drop=True)


def bootstrap_ci(values, statistic=np.median, n: int = 10_000, level: float = LEVEL,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile interval for a statistic of per-dataset values, resampling datasets."""
    values = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if len(values) == 0:
        return (math.nan, math.nan)
    if len(values) == 1:
        return (float(values[0]), float(values[0]))
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n, len(values)), replace=True)
    estimates = statistic(draws, axis=1)
    tail = (1 - level) / 2
    return (float(np.quantile(estimates, tail)), float(np.quantile(estimates, 1 - tail)))


def paired(regret_a, regret_b, tolerance: float = 1e-9) -> dict:
    """Compare two strategies judged on the same datasets. Lower regret wins.

    Differences within `tolerance` count as ties and are left out of the
    Wilcoxon test, which is what its default zero handling does too.
    """
    a = np.asarray(regret_a, dtype=float)
    b = np.asarray(regret_b, dtype=float)
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    diff = b - a                     # positive: a had less regret than b
    wins = int((diff > tolerance).sum())
    losses = int((diff < -tolerance).sum())
    ties = int(len(diff) - wins - losses)
    nonzero = diff[np.abs(diff) > tolerance]
    p = float(stats.wilcoxon(nonzero).pvalue) if len(nonzero) >= 1 else 1.0
    low, high = bootstrap_ci(diff) if len(diff) else (math.nan, math.nan)
    return {
        "datasets": int(len(diff)),
        "wins": wins, "ties": ties, "losses": losses,
        "median_difference": float(np.median(diff)) if len(diff) else math.nan,
        "ci": [low, high],
        "p": p,
    }


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values: several comparisons against one control."""
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted, running = {}, 0.0
    for i, (key, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[key] = running
    return adjusted


def average_ranks(matrix: pd.DataFrame) -> pd.Series:
    """Average rank per column over the rows; rank 1 is the best (highest) score.

    A missing score is a model that did not finish in its time budget. It is
    ranked last on that dataset, tied with any other model that did not
    finish, because not finishing is what a user following the advice gets.
    """
    filled = matrix.astype(float).fillna(-np.inf)
    ranks = filled.rank(axis=1, ascending=False, method="average")
    return ranks.mean(axis=0).sort_values()


def friedman(matrix: pd.DataFrame) -> dict:
    """Do the models differ at all? The Friedman test on the rank matrix."""
    ranks = matrix.astype(float).fillna(-np.inf).rank(axis=1, ascending=False, method="average")
    n, k = ranks.shape
    if n < 2 or k < 3:
        return {"datasets": int(n), "models": int(k), "chi2": math.nan, "p": math.nan}
    chi2, p = stats.friedmanchisquare(*[ranks[c].to_numpy() for c in ranks.columns])
    return {"datasets": int(n), "models": int(k), "chi2": float(chi2), "p": float(p)}


def critical_difference(k: int, n: int, alpha: float = 0.05) -> float:
    """Nemenyi critical difference for k models over n datasets.

    q is the studentized range quantile over sqrt(2), the table in Demšar
    (2006): 2.569 for four models, 3.164 for ten.
    """
    if k < 2 or n < 1:
        return math.nan
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / math.sqrt(2)
    return float(q * math.sqrt(k * (k + 1) / (6 * n)))


def cliques(ranks: pd.Series, cd: float) -> list[list[str]]:
    """Groups of models whose average ranks all lie within the critical difference.

    Each group is maximal: the bars a critical-difference diagram draws.
    """
    names = list(ranks.index)
    values = ranks.to_numpy()
    groups = []
    for i in range(len(names)):
        j = i
        while j + 1 < len(names) and values[j + 1] - values[i] <= cd + 1e-12:
            j += 1
        if j > i:
            group = names[i:j + 1]
            if not any(set(group) <= set(g) for g in groups):
                groups.append(group)
    return groups


def score_matrix(results: pd.DataFrame, task: str, decimals: int = 4) -> pd.DataFrame:
    """Independent units by models: each model's score, a family's averaged.

    Scores are compared at the four decimals the site publishes, so a
    difference too small to show counts as a tie, and a family's mean is
    rounded to six, so the ranks can be recomputed from the page exactly.
    A model that finished on none of a unit's datasets has no score there.
    """
    run = results[results.task == task]
    scores = run.assign(score=np.where(run.status == "ok", run.score.astype(float).round(decimals), np.nan))
    matrix = scores.pivot_table(index="dataset", columns="model", values="score", aggfunc="first", dropna=False)
    return matrix.groupby(matrix.index.map(family_of)).mean().round(6)


def model_ranking(results: pd.DataFrame, task: str, alpha: float = 0.05) -> dict:
    """Friedman, the average rank of every model, and which ones cannot be separated."""
    matrix = score_matrix(results, task)
    # A model has to have been attempted on every dataset of the task to be
    # ranked against the others; one that was only run on some cannot be.
    matrix = matrix.loc[:, matrix.notna().any(axis=0)]
    run = results[results.task == task]
    attempted = run.groupby("model").dataset.nunique()
    matrix = matrix[[c for c in matrix.columns if attempted.get(c, 0) == run.dataset.nunique()]]
    ranks = average_ranks(matrix)
    n, k = matrix.shape
    cd = critical_difference(k, n, alpha)
    best = float(ranks.iloc[0])
    return {
        "datasets": int(results[results.task == task].dataset.nunique()),
        "units": int(n),
        "alpha": alpha,
        "friedman": friedman(matrix),
        "critical_difference": cd,
        "ranks": {model: round(float(r), 4) for model, r in ranks.items()},
        "tied_with_best": [m for m, r in ranks.items() if r - best <= cd + 1e-12],
        "cliques": cliques(ranks, cd),
    }


def seed_stability(raw: pd.DataFrame, picks: pd.DataFrame | None = None, chosen: str | None = None,
                   reference: str = "BASE-HGB") -> dict:
    """How much the luck of one cross-validation split can move the results.

    Uses the datasets that were run under two or more seeds, per task:

      score_spread   the median, over datasets and models, of how far a
                     model's score moves between seeds (standard deviation)
      same_best      the share of those datasets where the best model is the
                     same under every seed
      flips          datasets where the order in use's first pick and
                     `reference` swap places between seeds: the comparison
                     the headline rests on, decided by the split
    """
    ok = raw[raw.status == "ok"].copy()
    if "seed" not in ok:
        return {}
    ok["score"] = ok.score.astype(float)
    counts = ok.groupby(["dataset", "task"]).seed.nunique()
    repeated = counts[counts >= 2]
    out = {}
    for task in sorted({t for _, t in repeated.index}):
        names = [d for d, t in repeated.index if t == task]
        runs = ok[(ok.task == task) & ok.dataset.isin(names)]
        spread = runs.groupby(["dataset", "model"]).score.std(ddof=0)
        same_best, flips, compared = 0, 0, 0
        for dataset, group in runs.groupby("dataset"):
            by_seed = group.pivot_table(index="seed", columns="model", values="score", aggfunc="first")
            winners = {tuple(sorted(by_seed.columns[by_seed.loc[s] == by_seed.loc[s].max()])) for s in by_seed.index}
            same_best += len(winners) == 1
            if picks is not None and chosen:
                row = picks[(picks.dataset == dataset) & (picks.task == task)]
                model = row[f"{chosen}_model"].iloc[0] if len(row) else None
                if model and model in by_seed and reference in by_seed and model != reference:
                    diff = (by_seed[model] - by_seed[reference]).dropna()
                    signs = {np.sign(round(float(x), 4)) for x in diff if round(float(x), 4) != 0}
                    compared += 1
                    flips += len(signs) > 1
        out[task] = {
            "datasets": len(names),
            "seeds": int(runs.seed.nunique()),
            "score_spread": round(float(spread.median()), 4) if len(spread) else None,
            "score_spread_p90": round(float(spread.quantile(0.9)), 4) if len(spread) else None,
            "same_best": round(same_best / len(names), 3) if names else None,
            "compared": compared,
            "flips": flips,
        }
    return out
