"""The profiler, ported from site/js/profile.js.

Same measurements, same thresholds. tests/test_agreement.py runs both
implementations over every fixture and fails if they disagree.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .dates import date_like_values
from .stats import PSI_SHIFT, is_number, js_number, psi_categorical, psi_numeric

MISSING_WORDS = {"", "na", "n/a", "null", "nan", "none", "-"}


def is_missing(value) -> bool:
    return value is None or str(value).strip().lower() in MISSING_WORDS


@dataclass
class Column:
    name: str
    index: int
    numeric: bool
    date_like: bool
    text_like: bool
    uniq: int
    values: list[str]
    missing: float
    dirty_rate: float
    zero_rate: float


@dataclass
class Profile:
    n: int
    feat: int
    columns: list[Column]
    date_cols: list[Column] = field(default_factory=list)


def profile_data(head: list[str], body: list[list[str]]) -> Profile:
    n = len(body)
    columns = []
    for i, name in enumerate(head):
        raw = [(r[i] if i < len(r) else "") for r in body]
        non_empty = [v for v in raw if not is_missing(v)]
        nums = [v for v in non_empty if v != "" and is_number(v)]
        numeric = len(non_empty) > 0 and len(nums) / len(non_empty) > 0.9
        uniq = len(set(non_empty))
        date_like = (
            not numeric
            and len(non_empty) > 0
            and date_like_values(non_empty[:40]) / min(40, len(non_empty)) > 0.8
        )
        zeros = sum(1 for v in nums if js_number(v) == 0)
        avg_len = (sum(len(v) for v in non_empty) / len(non_empty)) if non_empty else 0
        text_like = (
            not numeric
            and not date_like
            and len(non_empty) > 0
            and (avg_len > 25 or uniq / len(non_empty) > 0.7)
            and uniq > min(20, len(non_empty) * 0.5)
        )

        dirty = 0
        numeric_share = (len(nums) / len(non_empty)) if non_empty else 0
        if numeric_share > 0.5:
            dirty = len(non_empty) - len(nums)
        elif not date_like and not text_like:
            canon: dict[str, set[str]] = {}
            for v in non_empty:
                canon.setdefault(v.strip().lower(), set()).add(v)
            for variants in canon.values():
                if len(variants) > 1:
                    dirty += len(variants) - 1

        columns.append(
            Column(
                name=name,
                index=i,
                numeric=numeric,
                date_like=date_like,
                text_like=text_like,
                uniq=uniq,
                values=raw,
                missing=(n - len(non_empty)) / n if n else 0,
                dirty_rate=dirty / len(non_empty) if non_empty else 0,
                zero_rate=zeros / len(nums) if nums else 0,
            )
        )

    return Profile(n=n, feat=len(columns), columns=columns, date_cols=[c for c in columns if c.date_like])


def measured_axes(profile: Profile, target: str | None) -> dict:
    features = [c for c in profile.columns if c.name != target]
    cols = features or profile.columns
    n = profile.n

    numeric_cols = sum(1 for c in cols if c.numeric and not c.date_like)
    cat_cols = sum(1 for c in cols if not c.numeric and not c.date_like and not c.text_like)
    text_cols = sum(1 for c in cols if c.text_like)

    kinds = sum(1 for k in (numeric_cols > 0, cat_cols > 0, text_cols > 0) if k)
    a3 = "A38"
    if kinds == 1:
        a3 = "A31" if numeric_cols else ("A32" if cat_cols else "A34")

    feat = len(cols)
    sparsity = sum(max(c.missing, c.zero_rate) for c in cols) / feat
    high_dim = feat > n / 2 or (feat >= 20 and feat > n / 10)
    a4 = "A43" if sparsity > 0.5 else ("A42" if high_dim else "A41")

    miss = sum(c.missing for c in cols) / feat
    noise = sum(c.dirty_rate for c in cols) / feat
    a6 = "A64" if noise > 0.01 else ("A62" if miss > 0.001 else "A61")

    return {"a3": a3, "a4": a4, "a6": a6, "feat": feat, "sparsity": sparsity, "miss": miss, "noise": noise}


def class_balance(profile: Profile, target: str | None):
    col = next((c for c in profile.columns if c.name == target), None)
    if col is None or col.numeric or col.date_like or col.text_like:
        return None
    if col.uniq <= 1 or col.uniq > 20 or col.uniq > profile.n / 2:
        return None
    counts: dict[str, int] = {}
    for v in col.values:
        if not is_missing(v):
            counts[v] = counts.get(v, 0) + 1
    vals = sorted(counts.values(), reverse=True)
    total = sum(vals)
    if len(vals) < 2 or total == 0:
        return None
    share = vals[0] / total
    return {"code": "A52" if share > 0.75 else "A51", "majority_share": share, "levels": len(vals)}


def measure_drift(profile: Profile, target: str | None):
    half = profile.n // 2
    if half < 20:
        return None

    def scan(cols):
        worst = None
        for col in cols:
            first, second = col.values[:half], col.values[half:]
            psi = psi_numeric(first, second) if col.numeric else psi_categorical(first, second)
            if psi is None:
                continue
            if worst is None or psi > worst["psi"]:
                worst = {"psi": psi, "column": col.name}
        return worst

    usable = [c for c in profile.columns if not c.date_like and not c.text_like]
    worst, scope = scan([c for c in usable if c.name != target]), "features"
    if worst is None:
        worst, scope = scan([c for c in usable if c.name == target]), "target"
    if worst is None:
        return None
    worst["scope"] = scope
    worst["code"] = "A54" if worst["psi"] > PSI_SHIFT else "A53"
    return worst


def signature(profile: Profile, target: str | None, order: str) -> dict:
    axes = measured_axes(profile, target)
    balance = class_balance(profile, target)
    drift = measure_drift(profile, target)

    a5 = balance["code"] if balance else (drift["code"] if drift else "A53")
    flags = [drift["code"]] if (balance and drift) else []

    return {
        "codes": ["A11" if target else "A12", order, axes["a3"], axes["a4"], a5, axes["a6"]],
        "flags": flags,
        "balance": balance,
        "drift": drift,
    }


def match_codes(sig: dict) -> list[str]:
    return [*sig["codes"], *sig["flags"]]
