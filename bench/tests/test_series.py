"""The series the forecasting evidence is keyed on, measured the same way here and in the page.

site/js/series.js decides which kind of series an upload is (intermittent,
seasonal, trending...) and the benchmark files its evidence under the same
kinds, so the two must agree exactly. This runs the page's JavaScript on the
same inputs as the port and compares.
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

from dcn.series import day_number, period_from_days, series_cell, series_features

ROOT = Path(__file__).resolve().parents[2]


def js(cases: list[dict]) -> list[dict]:
    proc = subprocess.run(["node", str(ROOT / "bench" / "tools" / "js_series.mjs")], input=json.dumps(cases),
                          capture_output=True, text=True, check=True, cwd=ROOT)
    return json.loads(proc.stdout)


def shapes(seed: int = 3) -> list[tuple[list[float], int | None]]:
    rng = np.random.default_rng(seed)
    t = np.arange(120)
    out = [
        (list(10 + 5 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.5, 120)), 12),          # seasonal
        (list(0.4 * t + rng.normal(0, 2, 120)), 12),                                          # trend
        (list(0.3 * t + 8 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 1, 120)), 7),         # both
        (list(rng.normal(0, 1, 120)), 12),                                                     # level
        (list(np.cumsum(rng.normal(0, 1, 200))), None),                                       # random walk
        (list(rng.poisson(3, 60) * (rng.random(60) < 0.3)), 12),                             # intermittent
        (list(np.round(rng.lognormal(2, 1.2, 80)) * (rng.random(80) < 0.25)), 4),           # lumpy
        (list(rng.normal(5, 1, 30)), 24),                                                      # too short for its season
        ([0.0] * 20, 12), ([3.0, 3.0, 3.0], None), ([], None), ([1.0, -2.0, 0.0, 0.0, 5.0], 1),
    ]
    return [([float(v) for v in y], p) for y, p in out]


def close(a, b) -> bool:
    if a is None or b is None:
        return a is b or (a is None and isinstance(b, float) and math.isinf(b)) \
            or (b is None and isinstance(a, float) and math.isinf(a))
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def test_every_feature_and_kind_agrees_with_the_page():
    cases = shapes()
    theirs = js([{"values": y, "period": p} for y, p in cases])
    for (y, p), page in zip(cases, theirs):
        ours = series_features(y, p)
        for ours_key, page_key in [("n", "n"), ("zero_share", "zeroShare"), ("adi", "adi"), ("cv2", "cv2"),
                                   ("intermittent", "intermittent"), ("lumpy", "lumpy"),
                                   ("seasonal_strength", "seasonalStrength"), ("trend_strength", "trendStrength")]:
            assert close(ours[ours_key], page[page_key]), f"{ours_key}: {ours[ours_key]} vs {page[page_key]} ({p})"
        assert series_cell(ours) == page["cell"]


@pytest.mark.parametrize("dates,expected", [
    ([f"2024-{m:02d}-01" for m in range(1, 13)], ("monthly", 12)),
    ([f"2024-{m:02d}" for m in range(1, 13)], ("monthly", 12)),
    ([f"{d:02d}.01.2025" for d in range(1, 29)], ("daily", 7)),
    ([f"2025/01/{d:02d}" for d in range(1, 29)], ("daily", 7)),
    ([f"2025-01-01 {h:02d}:00" for h in range(24)], ("hourly", 24)),
    ([str(y) for y in range(1990, 2020)], ("yearly", 1)),
    ([f"2024-{m:02d}-01" for m in (1, 4, 7, 10)] * 1, ("quarterly", 4)),
])
def test_the_period_follows_the_step_between_dates(dates, expected):
    ours = period_from_days([day_number(d) for d in dates])
    page = js([{"dates": dates}])[0]["period"]
    assert ours == expected
    assert (page["name"], page["period"]) == expected


def test_weekdays_only_make_a_five_row_week():
    import datetime as dt
    days = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(60)]
    dates = [d.isoformat() for d in days if d.weekday() < 5]
    assert period_from_days([day_number(d) for d in dates]) == ("business-daily", 5)
    assert js([{"dates": dates}])[0]["period"] == {"name": "business-daily", "period": 5}


def test_unreadable_dates_give_no_period():
    dates = ["Jan 4", "Feb 4", "Mar 4", "Apr 4"]
    assert period_from_days([day_number(d) for d in dates]) is None
    assert js([{"dates": dates}])[0]["period"] is None


def test_day_numbers_agree_with_the_calendar():
    import datetime as dt
    for d in (dt.date(1970, 1, 1), dt.date(2000, 2, 29), dt.date(2024, 12, 31), dt.date(1899, 3, 1)):
        assert day_number(d.isoformat()) == (d - dt.date(1970, 1, 1)).days


def test_the_kinds_mean_what_they_say():
    cells = [series_cell(series_features(y, p)) for y, p in shapes()[:7]]
    assert cells == ["seasonal", "trend", "seasonal-trend", "level", "trend", "intermittent", "lumpy"]
