// CSV reading. Reads the header plus at most 5000 data rows.

export const MAX_ROWS = 5000;

export function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/).filter(l => l.length);
  const head = splitLine(lines[0]);
  const body = lines.slice(1, MAX_ROWS + 1).map(splitLine);
  return { head, body };
}

export function splitLine(l) {
  const out = []; let cur = '', q = false;
  for (let i = 0; i < l.length; i++) {
    const ch = l[i];
    if (ch === '"') { q = !q; continue; }
    if (ch === ',' && !q) { out.push(cur.trim()); cur = ''; continue; }
    cur += ch;
  }
  out.push(cur.trim());
  return out;
}
