"""The Python profiler must measure exactly what the site's JavaScript does.

Runs bench/tools/js_profile.mjs over every fixture and compares, column by
column and target by target. If the two implementations ever drift apart, the
benchmark would be scoring signatures the site never shows, so this fails.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from dcn.csvread import parse_csv
from dcn.profile import class_balance, measure_drift, measured_axes, profile_data, signature

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = (
    sorted((ROOT / "tests" / "fixtures").glob("*.csv"))
    + [ROOT / "site" / "sample.csv"]
    + sorted((Path(__file__).parent / "data").glob("*.csv"))  # awkward cases built for this test
)
TOL = 1e-6


@pytest.fixture(scope="session")
def js_results():
    proc = subprocess.run(
        ["node", str(ROOT / "bench" / "tools" / "js_profile.mjs"), *map(str, FIXTURES)],
        capture_output=True, text=True, check=True, cwd=ROOT,
    )
    return json.loads(proc.stdout)


def close(a, b):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) < TOL


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_parsing_and_profile_match_javascript(path: Path, js_results):
    js = js_results[path.name]
    parsed = parse_csv(path.read_text())

    assert parsed.delimiter == js["delimiter"]
    assert parsed.decimal_comma == js["decimalComma"]
    assert parsed.truncated == js["truncated"]
    assert parsed.head == js["head"]

    profile = profile_data(parsed.head, parsed.body)
    assert profile.n == js["rows"]
    assert len(profile.columns) == len(js["columns"])

    for col, exp in zip(profile.columns, js["columns"]):
        assert col.name == exp["name"]
        assert (col.numeric, col.date_like, col.text_like) == (exp["numeric"], exp["dateLike"], exp["textLike"]), col.name
        assert col.uniq == exp["uniq"], col.name
        assert close(col.missing, exp["missing"]), col.name
        assert close(col.dirty_rate, exp["dirtyRate"]), col.name
        assert close(col.zero_rate, exp["zeroRate"]), col.name

    for key, exp in js["perTarget"].items():
        target = None if key == "__none__" else key
        axes = measured_axes(profile, target)
        assert [axes["a3"], axes["a4"], axes["a6"]] == exp["axes"], key
        assert close(axes["sparsity"], exp["sparsity"]), key
        assert close(axes["miss"], exp["miss"]), key
        assert close(axes["noise"], exp["noise"]), key

        balance = class_balance(profile, target)
        if exp["balance"] is None:
            assert balance is None, key
        else:
            assert balance["code"] == exp["balance"]["code"], key
            assert close(balance["majority_share"], exp["balance"]["majorityShare"]), key
            assert balance["levels"] == exp["balance"]["levels"], key

        drift = measure_drift(profile, target)
        if exp["drift"] is None:
            assert drift is None, key
        else:
            assert drift["code"] == exp["drift"]["code"], key
            assert drift["column"] == exp["drift"]["column"], key
            assert drift["scope"] == exp["drift"]["scope"], key
            assert close(drift["psi"], exp["drift"]["psi"]), key

        assert " ".join(signature(profile, target, "A21")["codes"]) == exp["signature"], key
