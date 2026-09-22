"""Dataset loading. The parts that need the network are marked and skipped
unless DCN_NETWORK=1, so CI stays offline and fast."""
from __future__ import annotations

import os

import pandas as pd
import pytest

from dcn import datasets
from dcn.csvread import parse_csv
from dcn.profile import profile_data, signature

needs_network = pytest.mark.skipif(
    os.environ.get("DCN_NETWORK") != "1", reason="set DCN_NETWORK=1 to download datasets"
)

METADATA = """
dataset: example
task: classification
target:
  type: binary
features:
  - name: age
    type: continuous
    description: age
  - name: workclass
    type: categorical
    code: >
      'Private' = 4
  - name: 'weird name'
    type: binary
"""


def test_metadata_types_are_read_without_a_yaml_library(monkeypatch):
    monkeypatch.setattr(datasets, "_fetch", lambda url, cache, binary=False: METADATA.encode())
    assert datasets._metadata_types("example") == {
        "age": "continuous", "workclass": "categorical", "weird name": "binary",
    }


def test_missing_metadata_is_not_fatal(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("404")
    monkeypatch.setattr(datasets, "_fetch", boom)
    assert datasets._metadata_types("example") == {}


def test_a_dataset_round_trips_through_the_csv_reader():
    frame = pd.DataFrame({"num": [1.5, 2.5, 3.5], "cat": ["a_1", "a_2", "a_1"], "target": ["class_0", "class_1", "class_0"]})
    ds = datasets.Dataset("toy", "classification", frame, ["cat"], True)
    parsed = parse_csv(ds.to_csv_text())
    assert parsed.head == ["num", "cat", "target"]
    assert len(parsed.body) == 3
    profile = profile_data(parsed.head, parsed.body)
    assert [c.numeric for c in profile.columns] == [True, False, False]


@needs_network
def test_the_candidate_list_covers_both_tasks():
    counts = datasets.candidates().groupby("task").size().to_dict()
    assert counts["classification"] > 50 and counts["regression"] > 50


@needs_network
def test_categorical_columns_are_restored_from_metadata():
    ds = datasets.load("analcatdata_lawsuit")
    assert ds.task == "classification"
    assert ds.types_known and ds.categorical
    sig = signature(profile_data(*(lambda p: (p.head, p.body))(parse_csv(ds.to_csv_text()))), "target", "A21")
    assert sig["codes"][0] == "A11"
    assert sig["codes"][2] in ("A32", "A38")  # labels survived the integer encoding
