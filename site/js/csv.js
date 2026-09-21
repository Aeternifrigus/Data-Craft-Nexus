// CSV reading.
//
// Handles what real exports throw at it:
// - comma, semicolon, tab or pipe delimiters, detected from the file
//   (Polish and most EU Excel exports use semicolons)
// - quoted fields with delimiters, line breaks and "" escaped quotes inside
// - UTF-8 with or without a byte order mark, and Windows-1250 files
// - decimal commas (12,50 or 1.234,56) in files that aren't comma-separated
// - blank or repeated column names
//
// Reads the header plus at most MAX_ROWS data rows.

export const MAX_ROWS = 5000;
export const DELIMITERS = [',', ';', '\t', '|'];

// Turns file bytes into text. Tries strict UTF-8 first, then Windows-1250
// (the default ANSI code page for Polish Windows), then Windows-1252.
export function decodeBytes(bytes) {
  const data = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  try {
    return { text: new TextDecoder('utf-8', { fatal: true }).decode(data).replace(/^﻿/, ''), encoding: 'UTF-8' };
  } catch {
    for (const encoding of ['windows-1250', 'windows-1252']) {
      try { return { text: new TextDecoder(encoding).decode(data), encoding }; } catch { /* not supported, try next */ }
    }
    return { text: new TextDecoder('utf-8').decode(data), encoding: 'UTF-8 (with invalid bytes)' };
  }
}

// Splits text into rows of fields, following RFC 4180 quoting.
// Stops after `limit` rows. Blank lines are skipped.
export function tokenize(text, delimiter, limit = Infinity) {
  const rows = [];
  let row = [], field = '', quoted = false, fieldWasQuoted = false;
  let i = 0;
  const n = text.length;

  const endField = () => {
    row.push(fieldWasQuoted ? field : field.trim());
    field = ''; fieldWasQuoted = false;
  };
  const endRow = () => {
    endField();
    if (!(row.length === 1 && row[0] === '')) rows.push(row);
    row = [];
  };

  while (i < n && rows.length < limit) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"') {
        if (text[i + 1] === '"') { field += '"'; i += 2; continue; }
        quoted = false; i++; continue;
      }
      field += ch; i++; continue;
    }
    if (ch === '"' && field.trim() === '') { quoted = true; fieldWasQuoted = true; field = ''; i++; continue; }
    if (ch === delimiter) { endField(); i++; continue; }
    if (ch === '\r' || ch === '\n') {
      endRow();
      i += (ch === '\r' && text[i + 1] === '\n') ? 2 : 1;
      continue;
    }
    field += ch; i++;
  }
  if (rows.length < limit && (field !== '' || row.length || fieldWasQuoted)) endRow();

  const truncated = i < n && text.slice(i).trim() !== '';
  return { rows, truncated };
}

// Picks the delimiter that splits the first rows into the same number of
// fields (more than one). Falls back to a comma.
export function detectDelimiter(text) {
  const sample = text.slice(0, 64 * 1024);
  let best = { delimiter: ',', score: -1 };
  for (const d of DELIMITERS) {
    const { rows } = tokenize(sample, d, 30);
    if (rows.length === 0) continue;
    const width = rows[0].length;
    if (width < 2) continue;
    // The last sampled row may be cut off by the 64 KB window, so ignore it.
    const checked = rows.length > 2 ? rows.slice(0, -1) : rows;
    const consistent = checked.filter(r => r.length === width).length / checked.length;
    const score = consistent * 1000 + width;
    if (score > best.score) best = { delimiter: d, score };
  }
  return best.delimiter;
}

const DECIMAL_COMMA = /^[+-]?(\d+|\d{1,3}(\.\d{3})+),\d+$/;

// 12,50 -> 12.50 and 1.234,56 -> 1234.56. Only used when the delimiter isn't
// a comma, since in a comma-separated file a comma can't sit inside a number.
export function normalizeDecimal(value) {
  if (!DECIMAL_COMMA.test(value)) return value;
  return value.replace(/\.(?=\d{3}(\.|,))/g, '').replace(',', '.');
}

// Gives blank headers a name and makes repeated ones unique.
export function cleanHeader(head) {
  const seen = new Map();
  return head.map((name, i) => {
    let base = name === '' ? `column_${i + 1}` : name;
    if (!seen.has(base)) { seen.set(base, 1); return base; }
    let k = seen.get(base) + 1;
    while (seen.has(`${base}_${k}`)) k++;
    seen.set(base, k);
    const unique = `${base}_${k}`;
    seen.set(unique, 1);
    return unique;
  });
}

export function parseCSV(text, options = {}) {
  const clean = text.replace(/^﻿/, '');
  const delimiter = options.delimiter ?? detectDelimiter(clean);
  const { rows, truncated } = tokenize(clean, delimiter, MAX_ROWS + 1);
  if (rows.length === 0) return { head: [], body: [], delimiter, truncated: false, decimalComma: false };

  const head = cleanHeader(rows[0]);
  let body = rows.slice(1);
  let decimalComma = false;
  if (delimiter !== ',') {
    body = body.map(r => r.map(v => {
      const out = normalizeDecimal(v);
      if (out !== v) decimalComma = true;
      return out;
    }));
  }
  return { head, body, delimiter, truncated, decimalComma };
}

export function delimiterName(d) {
  return { ',': 'comma', ';': 'semicolon', '\t': 'tab', '|': 'pipe' }[d] ?? JSON.stringify(d);
}
