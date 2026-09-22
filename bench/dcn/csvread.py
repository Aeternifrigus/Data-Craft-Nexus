"""CSV reading, ported from site/js/csv.js.

The port exists so the benchmark scores the same signatures the site shows.
tests/test_agreement.py checks the two implementations against each other on
every fixture, so any drift between them fails the build.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_ROWS = 5000
DELIMITERS = [",", ";", "\t", "|"]
DECIMAL_COMMA = re.compile(r"^[+-]?(\d+|\d{1,3}(\.\d{3})+),\d+$")


@dataclass
class Parsed:
    head: list[str] = field(default_factory=list)
    body: list[list[str]] = field(default_factory=list)
    delimiter: str = ","
    truncated: bool = False
    decimal_comma: bool = False


def decode_bytes(data: bytes) -> tuple[str, str]:
    """UTF-8 if it decodes cleanly, else Windows-1250, else Windows-1252."""
    try:
        return data.decode("utf-8").lstrip("﻿"), "UTF-8"
    except UnicodeDecodeError:
        for encoding in ("cp1250", "cp1252"):
            try:
                return data.decode(encoding), {"cp1250": "windows-1250", "cp1252": "windows-1252"}[encoding]
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace"), "UTF-8 (with invalid bytes)"


def tokenize(text: str, delimiter: str, limit: float = float("inf")) -> tuple[list[list[str]], bool]:
    rows: list[list[str]] = []
    row: list[str] = []
    field_text = ""
    quoted = False
    field_was_quoted = False
    i, n = 0, len(text)

    def end_field() -> None:
        nonlocal field_text, field_was_quoted
        row.append(field_text if field_was_quoted else field_text.strip())
        field_text, field_was_quoted = "", False

    def end_row() -> None:
        nonlocal row
        end_field()
        if not (len(row) == 1 and row[0] == ""):
            rows.append(row)
        row = []

    while i < n and len(rows) < limit:
        ch = text[i]
        if quoted:
            if ch == '"':
                if i + 1 < n and text[i + 1] == '"':
                    field_text += '"'
                    i += 2
                    continue
                quoted = False
                i += 1
                continue
            field_text += ch
            i += 1
            continue
        if ch == '"' and field_text.strip() == "":
            quoted, field_was_quoted, field_text = True, True, ""
            i += 1
            continue
        if ch == delimiter:
            end_field()
            i += 1
            continue
        if ch in "\r\n":
            end_row()
            i += 2 if (ch == "\r" and i + 1 < n and text[i + 1] == "\n") else 1
            continue
        field_text += ch
        i += 1

    if len(rows) < limit and (field_text != "" or row or field_was_quoted):
        end_row()
    truncated = i < n and text[i:].strip() != ""
    return rows, truncated


def detect_delimiter(text: str) -> str:
    sample = text[: 64 * 1024]
    best, best_score = ",", -1
    for d in DELIMITERS:
        rows, _ = tokenize(sample, d, 30)
        if not rows:
            continue
        width = len(rows[0])
        if width < 2:
            continue
        checked = rows[:-1] if len(rows) > 2 else rows
        consistent = sum(1 for r in checked if len(r) == width) / len(checked)
        score = consistent * 1000 + width
        if score > best_score:
            best, best_score = d, score
    return best


def normalize_decimal(value: str) -> str:
    if not DECIMAL_COMMA.match(value):
        return value
    return re.sub(r"\.(?=\d{3}(\.|,))", "", value).replace(",", ".")


def clean_header(head: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for i, name in enumerate(head):
        base = f"column_{i + 1}" if name == "" else name
        if base not in seen:
            seen[base] = 1
            out.append(base)
            continue
        k = seen[base] + 1
        while f"{base}_{k}" in seen:
            k += 1
        seen[base] = k
        unique = f"{base}_{k}"
        seen[unique] = 1
        out.append(unique)
    return out


def parse_csv(text: str, delimiter: str | None = None) -> Parsed:
    clean = text.lstrip("﻿")
    delimiter = delimiter or detect_delimiter(clean)
    rows, truncated = tokenize(clean, delimiter, MAX_ROWS + 1)
    if not rows:
        return Parsed(delimiter=delimiter)

    head = clean_header(rows[0])
    body = rows[1:]
    decimal_comma = False
    if delimiter != ",":
        converted = []
        for r in body:
            new_row = []
            for v in r:
                out = normalize_decimal(v)
                decimal_comma = decimal_comma or out != v
                new_row.append(out)
            converted.append(new_row)
        body = converted
    return Parsed(head, body, delimiter, truncated, decimal_comma)
