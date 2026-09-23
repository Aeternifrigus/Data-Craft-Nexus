"""The Python profiler must measure exactly what the site's JavaScript does.

Runs bench/tools/js_profile.mjs over every fixture and compares, column by
column and target by target. If the two implementations ever drift apart, the
benchmark would be scoring signatures the site never shows, so this fails.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pandas as pd

import pytest

from dcn.csvread import parse_csv
from dcn.meta import FEATURES as META_FEATURES, meta_features
from dcn.profile import class_balance, measure_drift, measured_axes, profile_data, signature
from dcn.recommend import caution, load_taxonomy, rank_drifts, rank_models, rank_pipelines

TAXONOMY = load_taxonomy()

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

        # The meta-features place a dataset among the benchmark datasets, and
        # the neighbour order ranks by them, so they have to agree as well.
        meta = meta_features(parsed.head, parsed.body, target)
        for name in META_FEATURES:
            assert abs(meta[name] - exp["meta"][name]) < 1e-4, (key, name)

        # The ranking has to match too, or the benchmark would score advice
        # the site never gave.
        exp_shape = exp["shape"]
        for run_key, expected in exp["rankings"].items():
            order, task = run_key.split("|")
            sig = signature(profile, target, order)
            flags = f" +{' '.join(sig['flags'])}" if sig["flags"] else ""
            assert " ".join(sig["codes"]) + flags == expected["signature"], (key, run_key)

            assert (sig["rows"], sig["features"]) == (exp_shape["rows"], exp_shape["features"]), key

            models = rank_models(TAXONOMY, sig, task, meta=meta)
            got = [f"{m['c']}:{m['score']}/{m['of']}" + ("!" if caution(m, sig) else "")
                   + ("" if m["evidence_score"] is None else f"@{m['evidence_score']:.4f}")
                   for m in models.items]
            assert got == expected["models"], (key, run_key)
            assert models.tied == expected["tied"], (key, run_key)
            assert models.candidates == expected["candidates"], (key, run_key)
            assert [f"{m['c']}:{m['why']}" for m in models.ruled_out] == expected["ruledOut"], (key, run_key)
            assert [d["c"] for d in rank_drifts(TAXONOMY, sig).items] == expected["drifts"], (key, run_key)
            assert [f"{p['c']}:{p['score']}" for p in rank_pipelines(TAXONOMY, sig, task).items] == expected["pipelines"], (key, run_key)


@pytest.fixture(scope="session")
def knn_ranking(tmp_path_factory):
    """The neighbour order fitted on the committed 40-dataset run, whichever order ships."""
    from dcn.learn import build_frame, export
    results = pd.read_csv(ROOT / "bench" / "results" / "after-fixes.csv")
    meta = pd.read_csv(ROOT / "bench" / "results" / "meta.csv")
    path = tmp_path_factory.mktemp("ranking") / "ranking.json"
    frame = build_frame(results, TAXONOMY)
    return path, export(frame, TAXONOMY, path, "prior_knn", meta)


@pytest.fixture(scope="session")
def js_knn_results(knn_ranking):
    proc = subprocess.run(
        ["node", str(ROOT / "bench" / "tools" / "js_profile.mjs"), *map(str, FIXTURES)],
        capture_output=True, text=True, check=True, cwd=ROOT,
        env={**os.environ, "DCN_RANKING": str(knn_ranking[0])},
    )
    return json.loads(proc.stdout)


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_the_neighbour_order_matches_javascript(path: Path, js_knn_results, knn_ranking):
    """The order that depends on the dataset has to depend on it identically on both sides."""
    ranking = knn_ranking[1]
    assert ranking["chosen"] == "prior_knn"
    js = js_knn_results[path.name]
    parsed = parse_csv(path.read_text())
    profile = profile_data(parsed.head, parsed.body)
    for key, exp in js["perTarget"].items():
        target = None if key == "__none__" else key
        meta = meta_features(parsed.head, parsed.body, target)
        for run_key, expected in exp["rankings"].items():
            order, task = run_key.split("|")
            if task not in ("number", "category"):
                continue
            sig = signature(profile, target, order)
            models = rank_models(TAXONOMY, sig, task, ranking=ranking, meta=meta)
            got = [f"{m['c']}:{m['score']}/{m['of']}" + ("!" if caution(m, sig) else "")
                   + ("" if m["evidence_score"] is None else f"@{m['evidence_score']:.4f}")
                   for m in models.items]
            assert got == expected["models"], (key, run_key)


def test_the_neighbours_actually_move_the_scores(knn_ranking):
    """Otherwise the agreement above would hold trivially."""
    ranking = knn_ranking[1]
    path = ROOT / "tests" / "fixtures" / "numeric.csv"
    parsed = parse_csv(path.read_text())
    profile = profile_data(parsed.head, parsed.body)
    target = parsed.head[-1]
    meta = meta_features(parsed.head, parsed.body, target)
    sig = signature(profile, target, "A21")
    knn = rank_models(TAXONOMY, sig, "number", limit=99, ranking=ranking, meta=meta)
    plain = rank_models(TAXONOMY, sig, "number", limit=99, ranking={**ranking, "chosen": "prior"}, meta=meta)
    knn_scores = {m["c"]: m["evidence_score"] for m in knn.items}
    plain_scores = {m["c"]: m["evidence_score"] for m in plain.items}
    assert knn_scores.keys() == plain_scores.keys()
    assert any(abs(knn_scores[c] - plain_scores[c]) > 1e-4 for c in knn_scores if knn_scores[c] is not None)
    # Without meta-features there are no neighbours, and the order is the prior.
    fallback = rank_models(TAXONOMY, sig, "number", limit=99, ranking=ranking, meta=None)
    assert [m["evidence_score"] for m in fallback.items] == [m["evidence_score"] for m in plain.items]
