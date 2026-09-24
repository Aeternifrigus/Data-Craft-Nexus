"""Run the anomaly detectors the page can recommend on real tables with known anomalies.

"Unusual records" used to put its cards in coordinate order. This measures
them: every detector fitted without labels on each of ADBench's 47 classical
tables (Han et al., NeurIPS 2022), and scored against the labels afterwards,
by ROC AUC and average precision, as ADBench does.

What was fixed before any result existed is in bench/README.md, "Anomalies:
what the data has to show".

  python -m dcn.anomaly --out results/anomaly.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "bench" / "cache" / "anomaly"
URL = "https://raw.githubusercontent.com/Minqi824/ADBench/main/adbench/datasets/Classical/{name}.npz"
SEED = 0
MAX_ROWS = 5000      # larger tables are subsampled, stratified, keeping at least MIN_ANOMALIES anomalies
MIN_ANOMALIES = 20

# ADBench's 47 classical tables, and the source each was cut from. Tables cut
# from one source share a unit and are counted once, as the benchmark's
# dataset families are.
FAMILIES = {
    "annthyroid": "thyroid", "thyroid": "thyroid",                     # UCI thyroid disease
    "cardio": "cardiotocography", "Cardiotocography": "cardiotocography",
    "breastw": "wisconsin_breast", "WBC": "wisconsin_breast", "WDBC": "wisconsin_breast",
    "WPBC": "wisconsin_breast",                                        # the Wisconsin breast cancer tables
    "landsat": "statlog_landsat", "satellite": "statlog_landsat", "satimage-2": "statlog_landsat",
    "http": "kddcup99", "smtp": "kddcup99",                            # KDD Cup 1999 intrusions
}
NAMES = ["1_ALOI", "2_annthyroid", "3_backdoor", "4_breastw", "5_campaign", "6_cardio", "7_Cardiotocography",
         "8_celeba", "9_census", "10_cover", "11_donors", "12_fault", "13_fraud", "14_glass", "15_Hepatitis",
         "16_http", "17_InternetAds", "18_Ionosphere", "19_landsat", "20_letter", "21_Lymphography",
         "22_magic.gamma", "23_mammography", "24_mnist", "25_musk", "26_optdigits", "27_PageBlocks",
         "28_pendigits", "29_Pima", "30_satellite", "31_satimage-2", "32_shuttle", "33_skin", "34_smtp",
         "35_SpamBase", "36_speech", "37_Stamps", "38_thyroid", "39_vertebral", "40_vowels", "41_Waveform",
         "42_WBC", "43_WDBC", "44_Wilt", "45_wine", "46_WPBC", "47_yeast"]

# The kinds of table the evidence is kept for, by how many columns it has:
# fixed before the run (site/js/anomalies.js holds the same cut-offs).
LOW_COLUMNS, HIGH_COLUMNS = 10, 50


def kind_of(columns: int) -> str:
    return "low" if columns <= LOW_COLUMNS else "high" if columns > HIGH_COLUMNS else "mid"


def short(name: str) -> str:
    return name.split("_", 1)[1]


def unit_of(name: str) -> str:
    return FAMILIES.get(short(name), short(name))


def load(name: str, seed: int = SEED) -> tuple[np.ndarray, np.ndarray, int]:
    """The table's features and labels (1 = anomaly), subsampled when large. Returns (X, y, rows before)."""
    import urllib.request
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.npz"
    if not path.exists():
        path.write_bytes(urllib.request.urlopen(URL.format(name=name), timeout=600).read())
    data = np.load(path, allow_pickle=True)
    X, y = np.asarray(data["X"], dtype=float), np.asarray(data["y"]).astype(int)
    rows = len(y)
    if rows > MAX_ROWS:
        rng = np.random.default_rng(seed)
        bad, good = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        keep_bad = min(len(bad), max(MIN_ANOMALIES, round(MAX_ROWS * len(bad) / rows)))
        idx = np.r_[rng.choice(bad, keep_bad, replace=False), rng.choice(good, MAX_ROWS - keep_bad, replace=False)]
        idx.sort()
        X, y = X[idx], y[idx]
    return X, y, rows


def _scaled(X):
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit_transform(X)


def _iforest(X):
    from sklearn.ensemble import IsolationForest
    return -IsolationForest(random_state=SEED).fit(X).score_samples(X)


def _ocsvm(X):
    from sklearn.svm import OneClassSVM
    Z = _scaled(X)
    return -OneClassSVM().fit(Z).decision_function(Z)


def _lof(X):
    from sklearn.neighbors import LocalOutlierFactor
    lof = LocalOutlierFactor()
    lof.fit(_scaled(X))
    return -lof.negative_outlier_factor_


def _knn(X, k: int = 5):
    from sklearn.neighbors import NearestNeighbors
    Z = _scaled(X)
    distances, _ = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)
    return distances[:, k]            # the first column is the row itself


def _mcd(X):
    from sklearn.covariance import MinCovDet
    Z = _scaled(X)
    if Z.shape[0] <= 2 * Z.shape[1]:
        raise ValueError(f"needs more than twice as many rows as columns, has {Z.shape[0]} for {Z.shape[1]}")
    return MinCovDet(random_state=SEED).fit(Z).mahalanobis(Z)


def _dbscan(X):
    from sklearn.cluster import DBSCAN
    return (DBSCAN().fit(_scaled(X)).labels_ == -1).astype(float)


# Taxonomy code -> (name, detector). Every one is fitted without labels, at
# scikit-learn's defaults, on standardised columns except the forest, which
# does not care about scale. Higher scores mean more unusual.
DETECTORS = {
    "TR4": ("Isolation Forest", _iforest),
    "SV3": ("One-Class SVM", _ocsvm),
    "IB3": ("Local Outlier Factor", _lof),
    "IB4": ("k-NN distance", _knn),
    "PR6": ("Robust covariance (Minimum Covariance Determinant)", _mcd),
    "CL2": ("DBSCAN", _dbscan),
}

FIELDS = ["dataset", "unit", "rows", "sampled", "columns", "kind", "anomaly_share", "detector", "status",
          "auc", "ap", "seconds", "detail"]


def run_one(name: str) -> list[dict]:
    from sklearn.metrics import average_precision_score, roc_auc_score
    X, y, rows = load(name)
    base = {"dataset": short(name), "unit": unit_of(name), "rows": rows, "sampled": len(y), "columns": X.shape[1],
            "kind": kind_of(X.shape[1]), "anomaly_share": round(float(y.mean()), 4)}
    out = []
    for code, (label, detect) in DETECTORS.items():
        row = {**base, "detector": code}
        started = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score = np.asarray(detect(X), dtype=float)
            if not np.isfinite(score).all():
                raise ValueError("gave scores that are not finite")
            row.update(status="ok", auc=float(roc_auc_score(y, score)), ap=float(average_precision_score(y, score)))
        except Exception as exc:  # a detector that cannot handle a table is a result too
            row.update(status="error", detail=f"{type(exc).__name__}: {exc}"[:200])
        row["seconds"] = round(time.time() - started, 2)
        out.append(row)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/anomaly.csv")
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args(argv)
    out = Path(args.out)
    done = set()
    if out.exists():
        with out.open() as fh:
            done = {row["dataset"] for row in csv.DictReader(fh)}
    todo = [n for n in NAMES if short(n) not in done]
    new_file = not out.exists()
    from multiprocessing import Pool
    with out.open("a", newline="") as fh, Pool(args.jobs) as pool:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for rows in pool.imap_unordered(run_one, todo):
            writer.writerows(rows)
            fh.flush()
            best = max((r for r in rows if r["status"] == "ok"), key=lambda r: r["auc"], default=None)
            print(f"{rows[0]['dataset']} ({rows[0]['kind']}, {rows[0]['columns']} columns): "
                  f"best {best['detector'] if best else '-'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
