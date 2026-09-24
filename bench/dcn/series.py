"""The shape of a series, ported from site/js/series.js.

How often it is zero, how strongly it repeats, how strongly it trends: the
page keys its forecasting evidence on these, so the benchmark measures them
the same way, in the same order of operations. tests/test_series.py runs the
page's JavaScript on the same inputs and compares.
"""
from __future__ import annotations

import math
import re

INTERMITTENT_ADI = 1.32   # Syntetos, Boylan and Croston (2005)
LUMPY_CV2 = 0.49
STRENGTH = 0.5            # fixed before the forecasting benchmark ran

PERIODS = [
    ("hourly", 0.035, 0.05, 24),
    ("daily", 0.9, 1.1, 7),
    ("weekly", 6, 8, 52),
    ("monthly", 27, 32, 12),
    ("quarterly", 88, 93, 4),
    ("yearly", 360, 370, 1),
]

CELLS = {
    "lumpy": "intermittent, with sizes that vary a lot",
    "intermittent": "intermittent, with steady sizes",
    "seasonal-trend": "seasonal and trending",
    "seasonal": "seasonal, without much trend",
    "trend": "trending, without much season",
    "level": "neither seasonal nor trending",
}


def days_from_civil(y: int, m: int, d: int) -> int:
    y -= 1 if m <= 2 else 0
    era = y // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


ISO = re.compile(r"^(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?(?:[T ](\d{1,2}):(\d{2})(?::(\d{2}))?)?")
TAIL = re.compile(r"^[.\dZ:+-]*$")
SLASH = re.compile(r"^(\d{4})/(\d{1,2})/(\d{1,2})$")
DOTTED = re.compile(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$")


def day_number(text) -> float:
    s = str(text).strip()
    h = mi = se = 0
    m = ISO.match(s)
    if m and (m.end() == len(s) or TAIL.match(s[m.end():])):
        y, mo, d = int(m[1]), int(m[2] or 1), int(m[3] or 1)
        h, mi, se = int(m[4] or 0), int(m[5] or 0), int(m[6] or 0)
    elif (m := SLASH.match(s)):
        y, mo, d = int(m[1]), int(m[2]), int(m[3])
    elif (m := DOTTED.match(s)):
        y, mo, d = int(m[3]), int(m[2]), int(m[1])
    else:
        return math.nan
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return math.nan
    return days_from_civil(y, mo, d) + (h * 3600 + mi * 60 + se) / 86400


def period_from_days(days: list[float]) -> tuple[str, int] | None:
    known = [d for d in days if math.isfinite(d)]
    if len(known) < 3 or len(known) < len(days) * 0.9:
        return None
    steps = sorted(abs(b - a) for a, b in zip(known, known[1:]) if abs(b - a) > 0)
    if not steps:
        return None
    mid = len(steps) >> 1
    median = steps[mid] if len(steps) % 2 else (steps[mid - 1] + steps[mid]) / 2
    for name, lo, hi, period in PERIODS:
        if lo <= median <= hi:
            if name == "daily" and sum(abs(s - 3) < 0.01 for s in steps) > len(steps) * 0.1:
                return ("business-daily", 5)
            return (name, period)
    return None


def _mean(xs):
    total = 0.0
    for x in xs:
        total += x
    return total / len(xs)


def _variance(xs):
    m = _mean(xs)
    total = 0.0
    for x in xs:
        total += (x - m) * (x - m)
    return total / len(xs)


def centred_average(y: list[float], window: int) -> list[float]:
    n = len(y)
    out = [math.nan] * n
    half = window // 2
    for t in range(half, n - half):
        if window % 2:
            total = 0.0
            for k in range(-half, half + 1):
                total += y[t + k]
            out[t] = total / window
        else:
            total = 0.5 * y[t - half] + 0.5 * y[t + half]
            for k in range(-half + 1, half):
                total += y[t + k]
            out[t] = total / window
    return out


def strengths(y: list[float], period: int) -> tuple[float, float]:
    """(seasonal, trend) strength from a classical additive decomposition."""
    n = len(y)
    seasonal = period > 1 and n >= 2 * period
    window = period if seasonal else 3
    if n < window + 2:
        return 0.0, 0.0
    trend = centred_average(y, window)
    idx = [t for t in range(n) if math.isfinite(trend[t])]
    season = [0.0] * n
    if seasonal:
        sums, counts = [0.0] * period, [0] * period
        for t in idx:
            sums[t % period] += y[t] - trend[t]
            counts[t % period] += 1
        index = [s / c if c else 0.0 for s, c in zip(sums, counts)]
        centre = _mean(index)
        season = [index[t % period] - centre for t in range(n)]
    rem = [y[t] - trend[t] - season[t] for t in idx]
    detrended = [y[t] - trend[t] for t in idx]
    deseasoned = [y[t] - season[t] for t in idx]

    def strength(whole):
        v = _variance(whole)
        return max(0.0, 1 - _variance(rem) / v) if v > 0 else 0.0

    return (strength(detrended) if seasonal else 0.0), strength(deseasoned)


def series_features(values, period: int | None = None) -> dict:
    y = [float(v) for v in values if math.isfinite(float(v))]
    n = len(y)
    non_negative = all(v >= 0 for v in y)
    sizes = [v for v in y if v > 0]
    adi = n / len(sizes) if sizes else math.inf
    cv2 = 0.0
    if len(sizes) > 1:
        m = _mean(sizes)
        cv2 = _variance(sizes) / (m * m)
    seasonal, trend = strengths(y, period or 1) if n else (0.0, 0.0)
    intermittent = n > 0 and non_negative and adi >= INTERMITTENT_ADI
    return {
        "n": n, "period": period, "zero_share": 1 - len(sizes) / n if n else 0.0, "non_negative": non_negative,
        "adi": adi, "cv2": cv2, "intermittent": intermittent, "lumpy": intermittent and cv2 >= LUMPY_CV2,
        "seasonal_strength": seasonal, "trend_strength": trend,
    }


def series_cell(f: dict) -> str:
    if f["intermittent"]:
        return "lumpy" if f["lumpy"] else "intermittent"
    s, t = f["seasonal_strength"] >= STRENGTH, f["trend_strength"] >= STRENGTH
    return "seasonal-trend" if s and t else "seasonal" if s else "trend" if t else "level"
