"""The problems the checks benchmark plants are what it says they are."""
import numpy as np
import pandas as pd

from dcn.checks import COLUMN, MAX_ROWS, plant


def frame(n=300, classes=True, seed=0):
    rng = np.random.default_rng(seed)
    out = pd.DataFrame({"a": rng.normal(size=n).round(4), "b": rng.integers(0, 5, n)})
    out["target"] = np.where(out.a > 0, "class_1", "class_0") if classes else (out.a * 3 + rng.normal(size=n)).round(4)
    return out


def test_a_leak_is_the_target_renamed_or_rescaled():
    f = frame()
    leaked = plant(f, "leak", "classification", "x")
    assert leaked.groupby(COLUMN).target.nunique().max() == 1, "each renamed label is one class"
    reg = frame(classes=False)
    leaked = plant(reg, "leak", "regression", "x")
    assert np.allclose(leaked[COLUMN].astype(float), reg.target * 1.7 + 3)


def test_a_noisy_leak_changes_that_share_of_rows():
    reg = frame(n=1000, classes=False)
    leaked = plant(reg, "leak_5", "regression", "x")
    changed = ~np.isclose(leaked[COLUMN].astype(float), reg.target * 1.7 + 3)
    assert 0.03 <= changed.mean() <= 0.05


def test_ids_are_unique_and_copies_stay_within_what_the_page_reads():
    f = frame()
    assert plant(f, "id_run", "classification", "x")[COLUMN].sort_values().tolist() == list(range(1, 301))
    assert plant(f, "id_code", "classification", "x")[COLUMN].nunique() == 300
    copied = plant(f, "copies_2", "classification", "x")
    assert len(copied) == 306 and copied.duplicated().sum() >= 6
    full = frame(n=MAX_ROWS)
    assert len(plant(full, "copies_2", "classification", "x")) == MAX_ROWS


def test_planting_is_the_same_every_time():
    f = frame()
    assert plant(f, "leak_1", "classification", "x").equals(plant(f, "leak_1", "classification", "x"))
