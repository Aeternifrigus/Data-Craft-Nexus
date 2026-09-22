"""Write the reference links into site/taxonomy/*.json.

Two kinds of link, neither of them an encyclopaedia entry:
  read  where to read about it: the library that implements it, the paper that
        introduced it, or the project's repository (tools/links_table.py)
  docs  the documentation of the implementation this project actually runs,
        derived from the estimator in dcn/models.py rather than typed by hand

Run after changing either, then check them with `node tools/check_links.mjs`.
"""
from __future__ import annotations

import json
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from links_table import READ  # noqa: E402

from dcn.models import BY_CODE  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TAXONOMY = ROOT / "site" / "taxonomy"

LIBRARY_DOCS = {
    "xgboost": "https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.{cls}",
    "lightgbm": "https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.{cls}.html",
}


def read_link(entry: tuple[str, str]) -> dict:
    label, url = entry
    return {"label": label, "url": url}


def docs_for(code: str) -> dict | None:
    """Where to read about the implementation the benchmark runs."""
    spec = BY_CODE.get(code)
    if spec is None:
        return None
    estimator = spec.build("classification" if "classification" in spec.tasks else "regression")
    cls = type(estimator)
    module, name = cls.__module__, cls.__name__
    root = module.split(".")[0]
    if root in LIBRARY_DOCS:
        return {"label": f"{name} in the {root} docs", "url": LIBRARY_DOCS[root].format(cls=name)}
    if root == "sklearn":
        if name == "Pipeline":  # a composed model has no single page
            return None
        public = ".".join(module.split(".")[:2])  # sklearn.ensemble._forest -> sklearn.ensemble
        return {
            "label": f"{name} in the scikit-learn docs",
            "url": f"https://scikit-learn.org/stable/modules/generated/{public}.{name}.html",
        }
    return None


def load(name: str) -> dict:
    return json.loads((TAXONOMY / f"{name}.json").read_text())


def save(name: str, data: dict) -> None:
    (TAXONOMY / f"{name}.json").write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")


def main() -> int:
    counts = {"read": 0, "docs": 0, "unlinked": []}

    def attach(entry: dict, code: str) -> None:
        link = READ.get(code, "missing")
        entry.pop("ref", None)
        if link and link != "missing":
            entry["ref"] = read_link(link)
            counts["read"] += 1

        doc = docs_for(code)
        entry.pop("docs", None)
        if doc:
            entry["docs"] = doc
            counts["docs"] += 1

        # A code with neither kind of link, and no decision that it should
        # have none, is an oversight worth printing.
        if "ref" not in entry and "docs" not in entry and link != None:  # noqa: E711
            counts["unlinked"].append(code)

    axes = load("axes")
    for code, entry in axes["codes"].items():
        attach(entry, code)
    save("axes", axes)

    math = load("math")
    for code, entry in math["formulas"].items():
        attach(entry, code)
    save("math", math)

    models = load("models")
    for entry in models["models"]:
        attach(entry, entry["c"])
    save("models", models)

    drift = load("drift")
    for entry in drift["checkers"]:
        attach(entry, entry["c"])
    save("drift", drift)

    pipelines = load("pipelines")
    for entry in pipelines["pipelines"]:
        attach(entry, entry["c"])
    save("pipelines", pipelines)

    print(f"read links: {counts['read']}, docs links: {counts['docs']}")
    if counts["unlinked"]:
        print(f"no decision recorded for: {', '.join(sorted(counts['unlinked']))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
