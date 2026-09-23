"""How often each drift checker catches drift, and how often it cries wolf.

The site recommends drift checkers from the taxonomy, and until now it could
only say which ones match the data's coordinates. This puts them through the
same kind of test the models get: drift of a known kind is injected into real
data, and every checker is asked whether it sees it.

A case is one dataset from PMLB, shuffled with a seed and cut in three:

  train       up to 2000 rows a model is fitted on, for the checkers that
              watch its errors
  reference   500 rows the model never saw: what "normal" looks like
  current     250 rows drawn from the rest, changed by the scenario

Scenarios, each applied to one feature chosen at random among the continuous
ones (20 or more distinct values), unless said otherwise:

  none         nothing changes. Any alarm is a false alarm
  shift        the feature moves by 0.3 of its standard deviation
  scale        its spread widens by half, around the same centre
  correlation  its values are shuffled between rows: every feature looks
               exactly as before on its own, but its relationship with the
               others is gone. The feature with the strongest correlation to
               another is chosen, so there is a relationship to lose
  selection    rows are drawn more often the higher the feature is, the way
               a change in who shows up changes the data
  label_shift  the most common class is drawn half as often
  concept      the features stay exactly as they were, but where the feature
               is above its median, 30% of the labels change: the thing
               being predicted means something else now

Every checker is run as its card in site/taxonomy/drift.json says, at the
threshold the card states. Where a card leaves the threshold to the user, the
choice made here is written next to the checker (Checker.note), and a
statistic with no natural scale is calibrated on the reference window itself.
Tests on single features are run on every feature and corrected for how many
there are (Bonferroni), which is what DR-M9's card says any multi-feature use
needs. The checkers that watch a model's errors are given the error stream of
scikit-learn's gradient boosting at its defaults: the reference rows, then the
current rows, one at a time, and count as having fired only if they fire on
the current rows. river's detectors run at their defaults.

The measures:

  false alarm   the share of `none` cases in which the checker fired
  caught        per scenario, the share of cases in which it fired
  net           mean caught over the six drift scenarios, minus the false
                alarm rate. Declared before the run, and what the site
                orders drift checkers by

Rates are computed per dataset (over its repetitions) and then averaged over
datasets, so a dataset counts once however many times it was cut. Datasets
cut from one table (family_of in significance.py: the mfeat, waveform and
Garvan thyroid families, among others) are represented by one member each,
and datasets with missing values are left out.

Run:  python -m dcn.drift --reps 5 --out results/drift.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import time
import warnings
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist, jensenshannon

from .significance import family_of

REF_ROWS = 500
CUR_ROWS = 250
TRAIN_ROWS = 2000
MIN_ROWS = 2000
CONTINUOUS = 20          # distinct values for a feature to count as continuous
ALPHA = 0.05
PERMUTATIONS = 200

SCENARIOS = {
    "none": "nothing changes",
    "shift": "one feature moves by 0.3 standard deviations",
    "scale": "one feature's spread widens by half",
    "correlation": "one feature's link to the others is broken, each feature unchanged on its own",
    "selection": "rows are drawn more often the higher one feature is",
    "label_shift": "the most common class is drawn half as often",
    "concept": "features unchanged, but 30% of labels change in half the space",
}
DRIFT_SCENARIOS = [s for s in SCENARIOS if s != "none"]

# --------------------------------------------------------------------------
# Cases


@dataclass
class Case:
    """One comparison: a reference window, a current window, and a model's view of both."""
    dataset: str
    rep: int
    scenario: str
    ref: np.ndarray                     # rows by features, numeric
    cur: np.ndarray
    discrete: np.ndarray                # per feature: few enough values to count as categories
    ref_errors: np.ndarray | None = None   # 1 where the model was wrong
    cur_errors: np.ndarray | None = None
    feature: int | None = None          # the feature the scenario changed


def numeric_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Features as numbers: categorical columns by their codes, in order of first appearance."""
    cols = [c for c in frame.columns if c != "target"]
    out = np.empty((len(frame), len(cols)))
    for j, col in enumerate(cols):
        values = frame[col]
        if pd.api.types.is_numeric_dtype(values):
            out[:, j] = values.to_numpy(dtype=float)
        else:
            out[:, j] = pd.factorize(values)[0]
    return out, cols


def continuous_features(X: np.ndarray) -> list[int]:
    return [j for j in range(X.shape[1]) if len(np.unique(X[:, j])) >= CONTINUOUS]


def most_correlated(X: np.ndarray, candidates: list[int]) -> int:
    """The candidate feature with the strongest absolute correlation to any other feature."""
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(np.abs(corr))
    np.fill_diagonal(corr, 0)
    return max(candidates, key=lambda j: (corr[j].max(), -j))


def seed_for(dataset: str, rep: int) -> np.random.Generator:
    return np.random.default_rng([rep, zlib.crc32(dataset.encode())])


def weighted_rows(rng, pool: np.ndarray, weights: np.ndarray, size: int) -> np.ndarray:
    p = weights / weights.sum()
    return rng.choice(pool, size=size, replace=False, p=p)


def make_windows(X: np.ndarray, y: np.ndarray, scenario: str, rng: np.random.Generator,
                 split: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, int | None]:
    """The current window and its labels under a scenario, and the feature it changed."""
    ref, pool = split["ref"], split["pool"]
    feature = split["feature"]
    sd = X[ref, feature].std() or 1.0

    if scenario in ("selection", "label_shift"):
        if scenario == "selection":
            z = (X[pool, feature] - X[ref, feature].mean()) / sd
            weights = np.exp(0.5 * np.clip(z, -6, 6))
        else:
            common = pd.Series(y[ref]).mode().iloc[0]
            weights = np.where(y[pool] == common, 0.5, 1.0)
        rows = weighted_rows(rng, pool, weights, CUR_ROWS)
        return X[rows].copy(), y[rows].copy(), rows, (feature if scenario == "selection" else None)

    rows = rng.choice(pool, size=CUR_ROWS, replace=False)
    cur, labels = X[rows].copy(), y[rows].copy()
    if scenario == "none":
        return cur, labels, rows, None
    if scenario == "shift":
        cur[:, feature] += 0.3 * sd
        return cur, labels, rows, feature
    if scenario == "scale":
        centre = np.median(X[ref, feature])
        cur[:, feature] = centre + 1.5 * (cur[:, feature] - centre)
        return cur, labels, rows, feature
    if scenario == "correlation":
        f = split["correlated"]
        cur[:, f] = rng.permutation(cur[:, f])
        return cur, labels, rows, f
    if scenario == "concept":
        classes = np.unique(y)
        above = cur[:, feature] > np.median(X[ref, feature])
        change = above & (rng.random(len(cur)) < 0.3)
        for i in np.flatnonzero(change):
            others = classes[classes != labels[i]]
            labels[i] = rng.choice(others)
        return cur, labels, rows, feature
    raise ValueError(f"unknown scenario {scenario}")


# --------------------------------------------------------------------------
# The checkers. Each takes a Case and returns (fired, statistic).


def _standardise(case: Case) -> tuple[np.ndarray, np.ndarray]:
    mean = case.ref.mean(axis=0)
    sd = case.ref.std(axis=0)
    keep = sd > 0
    return (case.ref[:, keep] - mean[keep]) / sd[keep], (case.cur[:, keep] - mean[keep]) / sd[keep]


def _varying(case: Case) -> list[int]:
    """Features that are not constant across both windows: a test on a constant says nothing."""
    both = np.vstack([case.ref, case.cur])
    return [j for j in range(both.shape[1]) if np.ptp(both[:, j]) > 0]


def _bonferroni(pvalues: list[float]) -> tuple[bool, float]:
    if not pvalues:
        return False, 1.0
    smallest = float(min(pvalues))
    return smallest < ALPHA / len(pvalues), smallest


def _bins(reference: np.ndarray, discrete: bool) -> np.ndarray:
    """Bin edges from the reference: its own values when there are few, deciles otherwise."""
    if discrete:
        values = np.unique(reference)
        return np.concatenate([[-np.inf], (values[:-1] + values[1:]) / 2, [np.inf]])
    inner = np.unique(np.quantile(reference, np.linspace(0.1, 0.9, 9)))
    return np.concatenate([[-np.inf], inner, [np.inf]])


def _proportions(ref: np.ndarray, cur: np.ndarray, discrete: bool, floor: float = 1e-4):
    edges = _bins(ref, discrete)
    p = np.histogram(ref, edges)[0].astype(float)
    q = np.histogram(cur, edges)[0].astype(float)
    p, q = p / p.sum(), q / q.sum()
    return np.maximum(p, floor), np.maximum(q, floor), edges


def per_feature(case: Case, statistic) -> list[float]:
    return [statistic(case.ref[:, j], case.cur[:, j], bool(case.discrete[j])) for j in _varying(case)]


def ks(case):
    return _bonferroni(per_feature(case, lambda a, b, d: stats.ks_2samp(a, b).pvalue))


def psi_value(a, b, d):
    p, q, _ = _proportions(a, b, d)
    return float(np.sum((q - p) * np.log(q / p)))


def psi(case):
    worst = max(per_feature(case, psi_value), default=0.0)
    return worst > 0.25, worst


def chi_square(case):
    def p(a, b, d):
        edges = _bins(a, d)
        table = np.array([np.histogram(a, edges)[0], np.histogram(b, edges)[0]])
        table = table[:, table.sum(axis=0) > 0]
        if table.shape[1] < 2:
            return 1.0
        return float(stats.chi2_contingency(table)[1])
    return _bonferroni(per_feature(case, p))


def kl_value(a, b, d):
    p, q, _ = _proportions(a, b, d)
    return float(np.sum(p * np.log(p / q)))


def calibrated(case: Case, value, splits: int = 20) -> tuple[bool, float]:
    """For a statistic with no natural scale: the threshold comes from the reference itself.

    The reference is split at random into a pretend current window and the
    rest, 20 times; the threshold is the 95th percentile of the largest
    value over features seen between those parts, where nothing changed.
    """
    features = _varying(case)
    worst = max((value(case.ref[:, j], case.cur[:, j], bool(case.discrete[j])) for j in features), default=0.0)
    rng = np.random.default_rng(len(case.ref))
    calm = []
    for _ in range(splits):
        idx = rng.permutation(len(case.ref))
        a, b = case.ref[idx[len(case.cur):]], case.ref[idx[:len(case.cur)]]
        calm.append(max((value(a[:, j], b[:, j], bool(case.discrete[j])) for j in features), default=0.0))
    return worst > float(np.quantile(calm, 0.95)), worst


def kl(case):
    return calibrated(case, kl_value)


def js_value(a, b, d):
    p, q, _ = _proportions(a, b, d, floor=0.0)
    return float(jensenshannon(p, q, base=2) ** 2)


def js(case):
    worst = max(per_feature(case, js_value), default=0.0)
    return worst > 0.1, worst


def wasserstein_value(a, b, d):
    return float(stats.wasserstein_distance(a, b) / (a.std() or 1.0))


def wasserstein(case):
    return calibrated(case, wasserstein_value)


def anova(case):
    return _bonferroni(per_feature(case, lambda a, b, d: float(stats.f_oneway(a, b).pvalue)))


def cvm_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Cramér-von Mises T from the two empirical CDFs at every pooled value (Anderson, 1962)."""
    nx, ny = len(a), len(b)
    z = np.concatenate([a, b])
    gap = (np.searchsorted(np.sort(a), z, side="right") / nx
           - np.searchsorted(np.sort(b), z, side="right") / ny)
    return nx * ny / (nx + ny) ** 2 * float(np.sum(gap ** 2))


def cvm_pvalue(a: np.ndarray, b: np.ndarray, draws: int = 2000) -> float:
    """Two-sample Cramér-von Mises, with tied values handled.

    scipy.stats.cramervonmises_2samp (1.17) gives tied values their average
    rank and then treats the ranks as distinct. On a column that is mostly
    zeros it reports p below 1e-9 for two samples of the same data. The
    statistic here is computed from the empirical CDFs instead, which is the
    same number when there are no ties.

    The p-value is scipy's own limiting distribution for a continuous
    feature. For one with few distinct values that distribution is wrong in
    the direction that matters: a two-valued column's statistic has a heavier
    tail than it allows, so at the small levels a correction for many
    features asks for, it fires too often. There the p-value comes from
    permuting the rows instead, which is exact whatever the ties.
    """
    from scipy.stats._hypotests import _cdf_cvm_inf
    nx, ny = len(a), len(b)
    n, k = nx + ny, nx * ny
    z = np.concatenate([a, b])
    values, inverse = np.unique(z, return_inverse=True)
    t = cvm_statistic(a, b)
    if len(values) >= CONTINUOUS:
        et = (1 + 1 / n) / 6
        vt = (n + 1) * (4 * k * n - 3 * (nx ** 2 + ny ** 2) - 2 * k) / (45 * n ** 2 * 4 * k)
        tn = 1 / 6 + (t - et) / np.sqrt(45 * vt)
        # Below zero the limiting CDF is zero (scipy returns NaN there, which max() would turn into p = 0).
        return 1.0 if tn <= 0 else float(min(1.0, max(0.0, 1.0 - _cdf_cvm_inf(tn))))

    # Permutations, all at once: which rows land in the first sample, counted per distinct value.
    rng = np.random.default_rng(n)
    onehot = np.zeros((n, len(values)))
    onehot[np.arange(n), inverse] = 1.0
    total = onehot.sum(axis=0)
    first = np.argsort(rng.random((draws, n)), axis=1)[:, :nx]
    in_first = np.zeros((draws, n))
    np.put_along_axis(in_first, first, 1.0, axis=1)
    counts = in_first @ onehot
    gap = np.cumsum(counts, axis=1) / nx - np.cumsum(total - counts, axis=1) / ny
    null = k / n ** 2 * (gap ** 2 @ total)
    return float((1 + np.sum(null >= t - 1e-12)) / (1 + draws))


def cramer_von_mises(case):
    return _bonferroni(per_feature(case, lambda a, b, d: cvm_pvalue(a, b)))


def anderson_darling(case):
    # scipy reports p-values between 0.001 and 0.25 and clips outside them.
    # A clipped 0.001 means "0.001 or smaller", which is below every
    # Bonferroni threshold used here except on the widest datasets, where it
    # is counted as significant: the error that makes is at most 0.001.
    def p(a, b, d):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = stats.anderson_ksamp([a, b])
        return 0.0 if res.pvalue <= 0.001 else float(res.pvalue)
    return _bonferroni(per_feature(case, p))


def _permutation_test(K: np.ndarray, n_ref: int, statistic, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(K)
    labels = np.zeros(n, dtype=bool)
    labels[:n_ref] = True
    observed = statistic(K, labels)
    count = 0
    for _ in range(PERMUTATIONS):
        count += statistic(K, rng.permutation(labels)) >= observed - 1e-12
    return observed, (count + 1) / (PERMUTATIONS + 1)


def _block_means(M: np.ndarray, a: np.ndarray):
    fa = a.astype(float)
    fb = 1.0 - fa
    na, nb = fa.sum(), fb.sum()
    Ma, Mb = M @ fa, M @ fb
    return fa @ Ma / na ** 2, fb @ Mb / nb ** 2, fa @ Mb / (na * nb)


def mmd(case):
    ref, cur = _standardise(case)
    Z = np.vstack([ref, cur])
    D2 = cdist(Z, Z, "sqeuclidean")
    bandwidth = np.median(D2[np.triu_indices(len(Z), 1)]) or 1.0
    K = np.exp(-D2 / bandwidth)

    def stat(M, a):
        xx, yy, xy = _block_means(M, a)
        return xx + yy - 2 * xy
    value, p = _permutation_test(K, len(ref), stat, seed=len(Z))
    return p < ALPHA, float(value)


def energy(case):
    ref, cur = _standardise(case)
    Z = np.vstack([ref, cur])
    D = cdist(Z, Z)

    def stat(M, a):
        xx, yy, xy = _block_means(M, a)
        return 2 * xy - xx - yy
    value, p = _permutation_test(D, len(ref), stat, seed=len(Z))
    return p < ALPHA, float(value)


def mahalanobis(case):
    """The distance between the window means, scaled by the reference covariance, as a chi-square."""
    ref, cur = _standardise(case)
    d = ref.shape[1]
    cov = np.cov(ref, rowvar=False) + 1e-3 * np.eye(d)
    delta = cur.mean(axis=0) - ref.mean(axis=0)
    n1, n2 = len(ref), len(cur)
    t2 = float(n1 * n2 / (n1 + n2) * delta @ np.linalg.solve(cov, delta))
    return stats.chi2.sf(t2, d) < ALPHA, t2


def pca_reconstruction(case):
    """PCA fitted on half the reference; alarm when too many current rows reconstruct worse than
    the other half's 95th percentile (a one-sided binomial test at 0.05)."""
    from sklearn.decomposition import PCA
    ref, cur = _standardise(case)
    rng = np.random.default_rng(len(ref))
    idx = rng.permutation(len(ref))
    fit, calib = ref[idx[: len(ref) // 2]], ref[idx[len(ref) // 2:]]
    pca = PCA(n_components=0.9, svd_solver="full").fit(fit)

    def error(A):
        return ((A - pca.inverse_transform(pca.transform(A))) ** 2).sum(axis=1)
    threshold = np.quantile(error(calib), 0.95)
    above = int((error(cur) > threshold).sum())
    p = stats.binomtest(above, len(cur), 0.05, alternative="greater").pvalue
    return p < ALPHA, above / len(cur)


def adversarial(case):
    """A classifier told to separate reference rows from current ones; AUC above 0.6 is drift."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    Z = np.vstack([case.ref, case.cur])
    side = np.r_[np.zeros(len(case.ref)), np.ones(len(case.cur))]
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    auc = float(np.mean(cross_val_score(HistGradientBoostingClassifier(random_state=0), Z, side,
                                        cv=cv, scoring="roc_auc")))
    return auc > 0.6, auc


def error_drift(case):
    """The error rate on the current window, 20% or more above the reference window's."""
    base, now = float(case.ref_errors.mean()), float(case.cur_errors.mean())
    return now > 1.2 * base if base > 0 else now > 0, now - base


def streaming(make, feed: str = "errors"):
    """Run a river detector over the error stream: reference rows, then current rows.

    `feed="correct"` gives it 1 for a correct prediction instead. FHDDM needs
    that: river's documentation says 1 means an error, but its code fires
    when the stream's mean falls, which is the paper's reading (Pesaranghader
    and Viktor, 2016) with 1 meaning a correct prediction. Given errors, it
    fires when the model gets better.
    """
    binary = make.__module__.startswith("river.drift.binary")

    def run(case):
        detector = make()
        stream = np.r_[case.ref_errors, case.cur_errors].astype(int)
        if feed == "correct":
            stream = 1 - stream
        start = len(case.ref_errors)
        fired_at = None
        for i, x in enumerate(stream):
            detector.update(bool(x) if binary else float(x))
            if detector.drift_detected and i >= start:
                fired_at = i - start
                break
        return fired_at is not None, (np.nan if fired_at is None else float(fired_at))
    return run


def _river(name):
    from river import drift
    from river.drift import binary
    return getattr(binary, name, None) or getattr(drift, name)


@dataclass
class Checker:
    code: str
    sees: str        # "features" (one at a time), "joint" (all together) or "errors" (needs labels)
    run: object
    note: str = ""


def checkers() -> list[Checker]:
    return [
        Checker("DR-M1", "features", ks, "scipy.stats.ks_2samp on each feature, Bonferroni at 0.05"),
        Checker("DR-M2", "features", psi, "deciles of the reference (or its values, when it has few), "
                                         "alarm above 0.25 on any feature"),
        Checker("DR-M3", "features", chi_square, "counts in the PSI bins, scipy.stats.chi2_contingency, "
                                                 "Bonferroni at 0.05"),
        Checker("DR-M4", "features", kl, "PSI bins, KL(reference || current), largest over features; the "
                                         "card says to calibrate the threshold on a stable period, so it is the "
                                         "95th percentile of the same number between random parts of the reference"),
        Checker("DR-M5", "features", js, "PSI bins, the divergence in base 2 (scipy returns its square "
                                         "root), alarm above 0.1 on any feature"),
        Checker("DR-M6", "features", wasserstein, "in reference standard deviations, largest over features; the "
                                                  "card sets thresholds in each feature's own units, which cannot be "
                                                  "done for many datasets at once, so it is calibrated like DR-M4"),
        Checker("DR-M7", "features", anova, "scipy.stats.f_oneway on each feature, Bonferroni at 0.05"),
        Checker("DR-M8", "errors", error_drift, "gradient boosting's error rate on the current window, "
                                                "alarm at 20% above the reference window's"),
        Checker("DR-M10", "joint", mmd, "RBF kernel on standardised features, median-distance bandwidth, "
                                        f"{PERMUTATIONS} permutations, 0.05"),
        Checker("DR-M12", "features", cramer_von_mises, "on each feature, Bonferroni at 0.05; ties handled, and "
                                                        "p by permutation on features with few values (cvm_pvalue())"),
        Checker("DR-M13", "features", anderson_darling, "scipy.stats.anderson_ksamp on each feature, "
                                                        "Bonferroni at 0.05"),
        Checker("DR-M14", "joint", energy, f"energy distance on standardised features, {PERMUTATIONS} "
                                           "permutations, 0.05"),
        Checker("DR-C1", "errors", streaming(_river("DDM")), "river DDM, defaults, on the error stream"),
        Checker("DR-C2", "errors", streaming(_river("EDDM")), "river EDDM, defaults, on the error stream"),
        Checker("DR-C3", "errors", streaming(_river("PageHinkley")), "river PageHinkley, defaults, on the error stream"),
        Checker("DR-C4", "errors", streaming(_river("ADWIN")), "river ADWIN, defaults, on the error stream"),
        Checker("DR-C5", "errors", streaming(_kswin_seeded), "river KSWIN, defaults (seed 0), on the error stream"),
        Checker("DR-C6", "errors", streaming(_river("HDDMA")), "river HDDM_A, defaults, on the error stream"),
        Checker("DR-C7", "errors", streaming(_river("HDDMW")), "river HDDM_W, defaults, on the error stream"),
        Checker("DR-C9", "errors", streaming(_river("FHDDM"), feed="correct"),
                "river FHDDM, defaults, on the stream of correct predictions (see streaming())"),
        Checker("DR-MV1", "joint", mahalanobis, "the distance between window means under the reference "
                                                "covariance, as a chi-square with one degree per feature, 0.05"),
        Checker("DR-MV2", "joint", pca_reconstruction, "PCA keeping 90% of the variance, fitted on half the "
                                                       "reference and calibrated on the other half's 95th "
                                                       "percentile; one-sided binomial test at 0.05"),
        Checker("DR-CV1", "joint", adversarial, "gradient boosting, five-fold AUC, alarm above 0.6"),
    ]


# KSWIN draws its reference sample at random; fix it so a run repeats.
def _kswin_seeded():
    from river.drift import KSWIN
    return KSWIN(seed=0)


NOT_MEASURED = {
    "DR-M9": "not a checker on its own: every test above turns its statistic into a p-value, "
             "corrected for the number of features as this card says it must be",
    "DR-M11": "no implementation among the benchmark's dependencies (Alibi Detect has one)",
    "DR-C8": "river, which supplies the other streaming detectors, does not ship it",
    "DR-DL1": "made for images and audio (A35, A36), and needs a network trained per window; "
              "the benchmark is tables",
    "DR-DL2": "made for images and audio (A35, A36), and needs a network trained per window; "
              "the benchmark is tables",
}


# --------------------------------------------------------------------------
# Running


FIELDS = ["dataset", "rep", "scenario", "checker", "sees", "fired", "statistic", "seconds"]


def choose_datasets(limit: int | None = None) -> list[str]:
    """Classification datasets of 2000 rows or more with a continuous feature, one per family.

    Datasets with missing values are left out: every checker would need its
    own way of handling them, and missing data has its own benchmark (corrupt.py).
    """
    from .datasets import candidates, load
    names, seen = [], set()
    for name in sorted(candidates("classification", min_rows=MIN_ROWS).dataset):
        family = family_of(name)
        if family in seen:
            continue
        X, _ = numeric_matrix(load(name).frame)
        if np.isnan(X).any() or not continuous_features(X):
            continue
        seen.add(family)
        names.append(name)
        if limit and len(names) >= limit:
            break
    return names


def split_rows(n: int, rng: np.random.Generator) -> dict:
    idx = rng.permutation(n)
    ref = idx[:REF_ROWS]
    rest = idx[REF_ROWS:]
    half = len(rest) // 2
    return {"ref": ref, "pool": rest[:half], "train": rest[half:][:TRAIN_ROWS]}


def cases_for(dataset: str, rep: int, frame: pd.DataFrame, scenarios=None):
    """Every scenario's case for one cut of a dataset, sharing the cut and the model."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    X, _ = numeric_matrix(frame)
    y = pd.factorize(frame["target"])[0]
    rng = seed_for(dataset, rep)
    split = split_rows(len(X), rng)
    ref = split["ref"]
    continuous = [j for j in continuous_features(X[ref]) if X[ref, j].std() > 0]
    split["feature"] = int(rng.choice(continuous))
    split["correlated"] = most_correlated(X[ref], continuous)
    discrete = np.array([len(np.unique(X[ref, j])) < CONTINUOUS for j in range(X.shape[1])])

    model = HistGradientBoostingClassifier(random_state=0).fit(X[split["train"]], y[split["train"]])
    ref_errors = (model.predict(X[ref]) != y[ref]).astype(int)

    for scenario in scenarios or SCENARIOS:
        case_rng = np.random.default_rng([rep, zlib.crc32(dataset.encode()), zlib.crc32(scenario.encode())])
        cur, labels, rows, feature = make_windows(X, y, scenario, case_rng, split)
        cur_errors = (model.predict(cur) != labels).astype(int)
        yield Case(dataset, rep, scenario, X[ref], cur, discrete, ref_errors, cur_errors, feature)


def run_case(case: Case, only: list[str] | None = None) -> list[dict]:
    rows = []
    for checker in checkers():
        if only and checker.code not in only:
            continue
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fired, value = checker.run(case)
        rows.append({
            "dataset": case.dataset, "rep": case.rep,
            "scenario": case.scenario, "checker": checker.code, "sees": checker.sees,
            "fired": int(bool(fired)), "statistic": None if value is None or
            (isinstance(value, float) and math.isnan(value)) else round(float(value), 6),
            "seconds": round(time.perf_counter() - start, 4),
        })
    return rows


def done(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path, usecols=["dataset", "rep"])
    return set(zip(frame.dataset, frame.rep.astype(int)))


def rerun(path: Path, codes: list[str]) -> None:
    """Recompute some checkers on every case already in a results file, replacing their rows.

    Cases are seeded, so a checker changed after a run sees exactly the
    windows the others saw.
    """
    from .datasets import load
    results = pd.read_csv(path)
    keep = results[~results.checker.isin(codes)]
    fresh = []
    for name, group in results.groupby("dataset", sort=False):
        frame = load(name).frame
        for rep in sorted(group.rep.unique()):
            for case in cases_for(name, int(rep), frame):
                fresh.extend(run_case(case, only=codes))
    merged = pd.concat([keep, pd.DataFrame(fresh, columns=FIELDS)])
    order = {code: i for i, code in enumerate(c.code for c in checkers())}
    merged = merged.assign(_scenario=merged.scenario.map(list(SCENARIOS).index), _checker=merged.checker.map(order))
    merged = merged.sort_values(["dataset", "rep", "_scenario", "_checker"], kind="stable")
    merged.drop(columns=["_scenario", "_checker"]).to_csv(path, index=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/drift.csv")
    ap.add_argument("--rerun", default=None, help="comma-separated checker codes to recompute in --out")
    args = ap.parse_args(argv)

    from .datasets import load
    out = Path(args.out)
    if args.rerun:
        rerun(out, args.rerun.split(","))
        return 0
    finished = done(out)
    new = not out.exists()
    names = choose_datasets(args.limit)
    print(f"{len(names)} datasets: {', '.join(names)}", flush=True)
    with out.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        for name in names:
            frame = load(name).frame
            for rep in range(args.reps):
                if (name, rep) in finished:
                    continue
                start = time.perf_counter()
                rows = [row for case in cases_for(name, rep, frame) for row in run_case(case)]
                writer.writerows(rows)
                fh.flush()
                print(f"{name} rep {rep}: {len(rows)} checks in {time.perf_counter() - start:.0f}s", flush=True)
    return 0


# --------------------------------------------------------------------------
# Summarising


def summarise(results: pd.DataFrame, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Per checker: false alarms, what it caught in each scenario, and the net score.

    Every rate is a mean over datasets of the dataset's own rate over its
    repetitions, so each dataset counts once. The interval on the net score
    resamples datasets.
    """
    per_dataset = (results.groupby(["checker", "dataset", "scenario"]).fired.mean()
                   .unstack("scenario").reindex(columns=list(SCENARIOS)))
    out = {}
    rng = np.random.default_rng(seed)
    for code, table in per_dataset.groupby(level="checker"):
        table = table.droplevel("checker")
        caught = {s: round(float(table[s].mean()), 3) for s in DRIFT_SCENARIOS}
        false_alarm = float(table["none"].mean())
        net_per_dataset = table[DRIFT_SCENARIOS].mean(axis=1) - table["none"]
        draws = rng.choice(net_per_dataset.to_numpy(), size=(n_boot, len(net_per_dataset)), replace=True).mean(axis=1)
        out[code] = {
            "false_alarm": round(false_alarm, 3),
            "caught": caught,
            "caught_mean": round(float(table[DRIFT_SCENARIOS].mean(axis=1).mean()), 3),
            "net": round(float(net_per_dataset.mean()), 3),
            "net_ci": [round(float(np.quantile(draws, 0.025)), 3), round(float(np.quantile(draws, 0.975)), 3)],
            "seconds": round(float(results[results.checker == code].seconds.median()), 4),
        }
    return out


def evidence(results: pd.DataFrame) -> dict:
    """The block evidence.json carries for the drift checkers."""
    notes = {c.code: c.note for c in checkers()}
    sees = {c.code: c.sees for c in checkers()}
    measured = summarise(results)
    for code, entry in measured.items():
        entry["how"] = notes.get(code, "")
        entry["sees"] = sees.get(code, "")
    return {
        "how": (f"Drift of a known kind was injected into real datasets from PMLB and every checker was asked "
                f"whether it saw it: a reference window of {REF_ROWS} rows against a current window of "
                f"{CUR_ROWS}, each dataset cut {int(results.rep.nunique())} ways at random."),
        "datasets": sorted(results.dataset.unique()),
        "cases": int(results.groupby(["dataset", "rep", "scenario"]).ngroups),
        "scenarios": SCENARIOS,
        "score": "net: the share of the six kinds of drift caught, on average, minus the false alarm rate",
        "checkers": dict(sorted(measured.items(), key=lambda kv: -kv[1]["net"])),
        "not_measured": NOT_MEASURED,
    }


if __name__ == "__main__":
    raise SystemExit(main())
