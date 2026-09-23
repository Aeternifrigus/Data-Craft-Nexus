"""Messy data made on purpose: at the stated rates, deterministic, and only where intended."""
import csv

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from dcn.corrupt import CONDITIONS, JUNK, NoisyLabels, coerce, damage
from dcn.run import FIELDS, done_pairs, upgrade_header


def frame(n=2000):
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "a": rng.normal(size=n), "b": rng.integers(0, 9, size=n).astype(float),
        "c": rng.choice(["x", "y", "z"], size=n), "target": rng.integers(0, 2, size=n),
    })


@pytest.mark.parametrize("condition,rate", [("missing_10", 0.10), ("missing_30", 0.30)])
def test_missing_cells_at_the_stated_rate_and_never_the_target(condition, rate):
    clean = frame()
    out = damage(clean, condition, numeric=["a", "b"])
    for col in ["a", "b", "c"]:
        assert abs(out[col].isna().mean() - rate) < 0.03, col
    assert out.target.equals(clean.target)
    assert damage(clean, condition, numeric=["a", "b"]).equals(out), "seeded: the same damage every time"


def test_junk_goes_only_into_numeric_columns():
    clean = frame()
    out = damage(clean, "dirty_5", numeric=["a", "b"])
    junk = out.a.map(lambda v: isinstance(v, str))
    assert abs(junk.mean() - 0.05) < 0.02
    assert set(out.a[junk]) <= set(JUNK)
    assert out.c.equals(clean.c), "a categorical column is left alone"
    assert out.target.equals(clean.target)


def test_coercion_turns_junk_into_missing_and_categories_into_text():
    out = coerce(pd.DataFrame({"a": [1.5, "?", "n/a", 2], "c": [1, "x", np.nan, 2.0]}), ["a"], ["c"])
    assert out.a.isna().tolist() == [False, True, True, False]
    assert out.a.dtype.kind == "f"
    assert out.c.tolist()[:2] == ["1", "x"] and pd.isna(out.c.iloc[2])


class Recorder(DummyClassifier):
    """Remembers the labels it was fitted on."""

    def fit(self, X, y, sample_weight=None):
        self.seen_ = np.asarray(y).copy()
        return super().fit(X, y, sample_weight)


def test_label_noise_is_only_in_what_the_model_learns_from():
    X = np.arange(2000).reshape(-1, 1).astype(float)
    y = np.zeros(2000, dtype=int)
    y[::2] = 1
    noisy = NoisyLabels(Recorder(), rate=0.1).fit(X, y)
    changed = (noisy.estimator_.seen_ != y).mean()
    # 10% of labels are replaced by another row's; on balanced labels about
    # half of those replacements land on the other class.
    assert 0.03 < changed < 0.08
    assert noisy.classes_.tolist() == [0, 1]
    assert y[::2].tolist() == [1] * 1000, "the caller's labels are not modified"
    assert (noisy.predict(X) == noisy.estimator_.predict(X)).all()


def test_label_noise_hurts_a_model_scored_on_clean_labels():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(1500, 3))
    y = (X[:, 0] > 0).astype(int)
    clean = (LogisticRegression().fit(X[:1000], y[:1000]).predict(X[1000:]) == y[1000:]).mean()
    noisy = (NoisyLabels(LogisticRegression(), rate=0.9).fit(X[:1000], y[:1000]).predict(X[1000:]) == y[1000:]).mean()
    assert noisy < clean


def test_the_runner_keeps_each_condition_apart(tmp_path):
    path = tmp_path / "messy.csv"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow({"dataset": "d", "model": "m", "seed": 0, "condition": "missing_10"})
    done = done_pairs(path)
    assert ("d", "m", "0", "missing_10") in done
    assert ("d", "m", "0", "clean") not in done
    assert set(CONDITIONS) == {"missing_10", "missing_30", "dirty_5", "labels_10"}


def test_an_old_results_file_without_conditions_reads_as_clean(tmp_path):
    path = tmp_path / "old.csv"
    path.write_text("dataset,model,seed\nd,m,0\n")
    assert done_pairs(path) == {("d", "m", "0", "clean")}


def test_an_older_results_file_is_upgraded_before_rows_are_appended(tmp_path):
    path = tmp_path / "full.csv"
    old_fields = [f for f in FIELDS if f != "condition"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=old_fields)
        writer.writeheader()
        writer.writerow({f: "x" for f in old_fields} | {"dataset": "d", "score": "0.5"})
    upgrade_header(path)
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == FIELDS
    assert rows[0]["score"] == "0.5" and rows[0]["condition"] == ""
    before = path.read_text()
    upgrade_header(path)
    assert path.read_text() == before, "a current file is left alone"
