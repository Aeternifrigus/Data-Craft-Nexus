"""Statistics helpers, ported from site/js/stats.py's JavaScript twin."""
from __future__ import annotations

import math
import re

PSI_SHIFT = 0.25


# What Number() accepts: decimal (with exponent), hex, octal, binary, and
# Infinity. Python's float() is more generous (it takes "1_000" and non-ASCII
# digits), so the shape is checked before converting.
_DECIMAL = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$", re.ASCII)
_RADIX = re.compile(r"^[+-]?0([xX][0-9a-fA-F]+|[oO][0-7]+|[bB][01]+)$", re.ASCII)


def js_number(value: str) -> float:
    """Number(value) as JavaScript computes it, for the values a CSV holds."""
    s = value.strip()
    if s == "":
        return 0.0
    if s in ("Infinity", "+Infinity"):
        return math.inf
    if s == "-Infinity":
        return -math.inf
    if _DECIMAL.match(s):
        return float(s)
    if _RADIX.match(s):
        sign = -1.0 if s[0] == "-" else 1.0
        return sign * float(int(s.lstrip("+-"), 0))
    return math.nan


def is_number(value: str) -> bool:
    return not math.isnan(js_number(value))


def to_numbers(values) -> list[float]:
    out = []
    for v in values:
        if v is None or v == "":
            continue
        n = js_number(v)
        if math.isfinite(n):
            out.append(n)
    return out


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return math.nan
    pos = (len(sorted_values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def _psi_from_counts(ref_counts, cur_counts, ref_n, cur_n) -> float:
    floor = 0.0001
    psi = 0.0
    for a_count, b_count in zip(ref_counts, cur_counts):
        a = max(a_count / ref_n, floor)
        b = max(b_count / cur_n, floor)
        psi += (b - a) * math.log(b / a)
    return psi


def psi_numeric(reference, current, bins: int = 10):
    ref = sorted(to_numbers(reference))
    cur = to_numbers(current)
    if len(ref) < bins * 2 or len(cur) < bins * 2:
        return None

    edges = [quantile(ref, i / bins) for i in range(1, bins)]
    unique = list(dict.fromkeys(edges))
    if len(unique) < 2:
        return None

    def bucket(v: float) -> int:
        i = 0
        while i < len(unique) and v > unique[i]:
            i += 1
        return i

    def count(xs):
        c = [0] * (len(unique) + 1)
        for v in xs:
            c[bucket(v)] += 1
        return c

    return _psi_from_counts(count(ref), count(cur), len(ref), len(cur))


def psi_categorical(reference, current):
    ref = [v for v in reference if v not in (None, "")]
    cur = [v for v in current if v not in (None, "")]
    if len(ref) < 20 or len(cur) < 20:
        return None
    levels = list(dict.fromkeys([*ref, *cur]))
    if len(levels) < 2 or len(levels) > 50:
        return None
    count = lambda xs: [sum(1 for v in xs if v == l) for l in levels]  # noqa: E731
    return _psi_from_counts(count(ref), count(cur), len(ref), len(cur))
