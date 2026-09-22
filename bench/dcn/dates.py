"""Date detection matching what Date.parse accepts in a browser.

The profiler calls a column a date column when more than 80% of its first 40
values parse as dates. JavaScript uses Date.parse for that, so this module
accepts the same shapes: ISO timestamps, slash and dash dates, and the
written-month forms.
"""
from __future__ import annotations

import re

# Longest first: the alternation must not match "jan" inside "january" and
# then fail on the rest.
MONTHS = (
    "january|february|march|april|august|september|october|november|december|"
    "june|july|"
    "jan|feb|mar|apr|may|jun|jul|aug|sept|sep|oct|nov|dec"
)

PATTERNS = [
    # 2025, 2025-01, 2025-01-04, 2025-01-04T10:30:00.500Z, 2025-01-04 10:30
    re.compile(r"(?a)^\d{4}(-\d{1,2}(-\d{1,2})?)?([T ]\d{1,2}:\d{2}(:\d{2})?(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$"),
    # 2025/01/04, 01/04/2025
    re.compile(r"(?a)^\d{1,4}/\d{1,2}/\d{1,4}([T ]\d{1,2}:\d{2}(:\d{2})?)?$"),
    # 04.01.2025
    re.compile(r"(?a)^\d{1,2}\.\d{1,2}\.\d{2,4}$"),
    # Jan 4, 2025 / January 4 2025 / Jan 4
    re.compile(rf"(?a)^({MONTHS})\.?\s+\d{{1,2}}(,)?(\s*\d{{4}})?$", re.I),
    # Jan 2025
    re.compile(rf"(?a)^({MONTHS})\.?\s+\d{{4}}$", re.I),
    # 4 Jan 2025 / 04-Jan-25
    re.compile(rf"(?a)^\d{{1,2}}[\s-]({MONTHS})\.?[\s-]\d{{2,4}}$", re.I),
    # Sat, 04 Jan 2025 10:00:00 GMT
    re.compile(rf"(?a)^[a-z]{{3}},?\s+\d{{1,2}}\s+({MONTHS})\s+\d{{4}}\b.*$", re.I),
]


def is_date_like(value: str) -> bool:
    s = value.strip()
    if not s:
        return False
    return any(p.match(s) for p in PATTERNS)


def date_like_values(values) -> int:
    return sum(1 for v in values if is_date_like(v))
