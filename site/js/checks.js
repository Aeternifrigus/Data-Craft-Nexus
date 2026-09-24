// "Before you trust a score": what can make any model look better than it is.
//
// Choosing among good models moves a score by about a point on the
// benchmark. These move it by more, and no choice of model fixes them:
//
//   leak        one column predicts the target almost perfectly on its own,
//               usually because it was recorded after the outcome
//   id          a column with a different value in every row, like an order
//               number, which a model can memorise
//   duplicates  more repeated rows than chance would produce, so a random
//               split puts copies on both sides of it
//   time        a date column, with rows declared independent: a random split
//               lets the model learn from rows later than the ones it is
//               tested on
//
// Pure functions: no DOM access. The thresholds were set before they were
// measured on the benchmark (bench/dcn/checks.py) and were not tuned to it.
// Two rules changed after that first measurement, the need for a measurement
// column in the repeat check and a rank correlation beside the leak check;
// bench/README.md ("Before you trust a score") says why and what it cost.

import { isMissing } from './profile.js';

// One column, used alone on rows it was not fitted on, gets at least this
// much right: balanced accuracy for a category, R² for a number.
export const LEAK_SCORE = 0.99;
// Fewer usable rows than this and one column's score says too little.
export const LEAK_MIN_ROWS = 30;
// A column is ID-like when this share of its values are distinct.
export const ID_DISTINCT = 0.98;
export const ID_MIN_ROWS = 20;
// Repeated rows are flagged when they are at least this share of the file,
// several times what chance would give if the columns were independent, and
// the table has a measurement: a numeric column with at least this many
// distinct values. A table of categories alone repeats rows naturally
// (identical voting records, the same answers to the same questions), and a
// first version that flagged those fired on 33 of the 195 clean benchmark
// datasets, mostly for that reason.
export const DUPLICATE_SHARE = 0.01;
export const DUPLICATE_OVER_CHANCE = 3;
export const DUPLICATE_MEASURED = 50;
// Numeric columns are cut into at most this many bins for the leak check.
const BINS = 32;

const ID_NAME = /(^|[^a-z])(id|uuid|guid|idx|index|key|no|nr|num|number)$|^(id|uuid|guid)([^a-z]|$)/i;
const ID_CAMEL = /[a-z](Id|ID|Key|No|Nr|Num|Number)$/;

// Does this column look like a row identifier? `rows` are the rows to judge
// it on: the first of each set of exact copies, so a file with repeated
// records does not hide its ID column.
export function idLike(col, n, rows = null) {
  if (col.dateLike || n < ID_MIN_ROWS) return false;
  const present = (rows ? rows.map(i => col.values[i]) : col.values).filter(v => !isMissing(v));
  if (present.length < ID_MIN_ROWS || new Set(present).size < ID_DISTINCT * present.length) return false;
  const named = ID_NAME.test(col.name.trim()) || ID_CAMEL.test(col.name.trim());
  if (col.numeric) {
    const nums = present.map(Number);
    if (!nums.every(Number.isInteger)) return false;
    // A run like 1..n (a row number) or an integer column named like an ID.
    const span = Math.max(...nums) - Math.min(...nums) + 1;
    return named || span <= 1.1 * present.length;
  }
  // Short codes without spaces ("SHP-0042"), not free text.
  const avgLen = present.reduce((s, v) => s + v.length, 0) / present.length;
  const spaced = present.filter(v => /\s/.test(v.trim())).length / present.length;
  return named || (avgLen <= 32 && spaced <= 0.2);
}

// The target as the check needs it: labels for a category, numbers otherwise.
function targetValues(profile, target, task) {
  const col = profile.columns.find(c => c.name === target);
  if (!col) return null;
  const asNumber = task === 'number' || (task !== 'category' && col.numeric);
  const values = col.values.map(v => {
    if (isMissing(v)) return null;
    if (!asNumber) return String(v).trim();
    const x = Number(v);
    return Number.isFinite(x) ? x : null;
  });
  return { values, kind: asNumber ? 'number' : 'category' };
}

// A feature's value as a key: numbers cut into quantile bins fitted on the
// training half, anything else as trimmed text.
function keyer(col, train) {
  if (!col.numeric) return (i) => isMissing(col.values[i]) ? '∅' : String(col.values[i]).trim();
  const xs = train.map(i => Number(col.values[i])).filter(Number.isFinite).sort((a, b) => a - b);
  const distinct = [...new Set(xs)];
  const edges = distinct.length <= BINS ? distinct
    : [...new Set(Array.from({ length: BINS - 1 }, (_, k) => xs[Math.floor((k + 1) * xs.length / BINS)]))];
  return (i) => {
    const x = Number(col.values[i]);
    if (isMissing(col.values[i]) || !Number.isFinite(x)) return '∅';
    let lo = 0, hi = edges.length;          // the number of edges at or below x
    while (lo < hi) { const mid = (lo + hi) >> 1; if (edges[mid] <= x) lo = mid + 1; else hi = mid; }
    return 'b' + lo;
  };
}

// Average ranks, ties sharing the mean of the ranks they span.
function ranks(xs) {
  const order = xs.map((x, i) => [x, i]).sort((a, b) => a[0] - b[0]);
  const r = new Array(xs.length);
  for (let k = 0; k < order.length;) {
    let j = k;
    while (j + 1 < order.length && order[j + 1][0] === order[k][0]) j += 1;
    for (let m = k; m <= j; m++) r[order[m][1]] = (k + j) / 2;
    k = j + 1;
  }
  return r;
}

// Rank correlation between a numeric column and a numeric target, over the
// rows where both are numbers: a column that is the target on another scale
// (price in another currency, a log of it) scores 1 however skewed it is.
export function rankCorrelation(col, target, rows) {
  const pairs = rows.filter(i => !isMissing(col.values[i]) && Number.isFinite(Number(col.values[i])))
    .map(i => [Number(col.values[i]), target.values[i]]);
  if (pairs.length < LEAK_MIN_ROWS) return null;
  const rx = ranks(pairs.map(p => p[0])), ry = ranks(pairs.map(p => p[1]));
  const m = (rx.length - 1) / 2;
  let sxy = 0, sxx = 0, syy = 0;
  for (let k = 0; k < rx.length; k++) {
    sxy += (rx[k] - m) * (ry[k] - m); sxx += (rx[k] - m) ** 2; syy += (ry[k] - m) ** 2;
  }
  return sxx && syy ? Math.abs(sxy / Math.sqrt(sxx * syy)) : null;
}

// How well one column predicts the target on rows it was not fitted on.
// Two halves (alternate rows), each predicted from the other: the majority
// label or the mean target for the feature's value, the overall one for a
// value not seen. Balanced accuracy for a category, R² for a number.
export function singleColumnScore(col, target, rows) {
  const halves = [rows.filter((_, k) => k % 2 === 0), rows.filter((_, k) => k % 2 === 1)];
  const predicted = new Map();
  for (const [fit, apply] of [[halves[0], halves[1]], [halves[1], halves[0]]]) {
    const key = keyer(col, fit);
    const groups = new Map();
    for (const i of fit) {
      const k = key(i);
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k).push(target.values[i]);
    }
    const summarise = target.kind === 'number'
      ? (ys) => ys.reduce((a, b) => a + b, 0) / ys.length
      : (ys) => { const c = new Map(); for (const y of ys) c.set(y, (c.get(y) || 0) + 1);
        return [...c.entries()].sort((a, b) => b[1] - a[1] || String(a[0]).localeCompare(String(b[0])))[0][0]; };
    const fallback = summarise(fit.map(i => target.values[i]));
    const fitted = new Map([...groups].map(([k, ys]) => [k, summarise(ys)]));
    for (const i of apply) predicted.set(i, fitted.has(key(i)) ? fitted.get(key(i)) : fallback);
  }

  if (target.kind === 'number') {
    const ys = rows.map(i => target.values[i]);
    const mean = ys.reduce((a, b) => a + b, 0) / ys.length;
    const sst = ys.reduce((s, y) => s + (y - mean) ** 2, 0);
    if (sst === 0) return null;
    const sse = rows.reduce((s, i) => s + (target.values[i] - predicted.get(i)) ** 2, 0);
    return 1 - sse / sst;
  }
  const byClass = new Map();
  for (const i of rows) {
    const y = target.values[i];
    if (!byClass.has(y)) byClass.set(y, [0, 0]);
    const c = byClass.get(y);
    c[0] += 1; if (predicted.get(i) === y) c[1] += 1;
  }
  if (byClass.size < 2) return null;
  return [...byClass.values()].reduce((s, [n, hit]) => s + hit / n, 0) / byClass.size;
}

// Rows that repeat an earlier row exactly, against what chance would give if
// every column were independent of the others.
export function repeatedRows(profile, target) {
  const n = profile.n;
  const cols = profile.columns;
  const seen = new Set();
  let copies = 0;
  const byInputs = new Map();
  for (let i = 0; i < n; i++) {
    const row = JSON.stringify(cols.map(c => c.values[i]));
    if (seen.has(row)) copies += 1; else seen.add(row);
    if (target) {
      const inputs = JSON.stringify(cols.filter(c => c.name !== target).map(c => c.values[i]));
      if (!byInputs.has(inputs)) byInputs.set(inputs, new Map());
      const ys = byInputs.get(inputs);
      const y = cols.find(c => c.name === target).values[i];
      ys.set(y, (ys.get(y) || 0) + 1);
    }
  }
  // Chance that two different rows match: the product over columns of the
  // chance two different rows match on that column.
  let match = n > 1 ? 1 : 0;
  for (const c of cols) {
    const counts = new Map();
    for (const v of c.values) counts.set(v, (counts.get(v) || 0) + 1);
    let pairs = 0;
    for (const k of counts.values()) pairs += k * (k - 1);
    match *= n > 1 ? pairs / (n * (n - 1)) : 0;
  }
  const expected = Math.min(n - 1, (n * (n - 1) / 2) * match);
  // Rows whose exact inputs appear again with a different target.
  let conflicting = 0;
  for (const ys of byInputs.values()) if (ys.size > 1) for (const k of ys.values()) conflicting += k;
  return { copies, share: n ? copies / n : 0, expected, conflicting };
}

// All four checks. `decl` is the intake answers: target, task, order.
export function runChecks(profile, decl) {
  const target = decl.target && decl.target !== '__none__' ? decl.target : null;
  const n = profile.n;
  const flags = [];
  const ran = [];

  // ID-like columns first: the leak check skips them, because a value seen
  // once says nothing about rows it was not fitted on.
  const firsts = [];
  const seenRows = new Set();
  for (let i = 0; i < n; i++) {
    const row = JSON.stringify(profile.columns.map(c => c.values[i]));
    if (!seenRows.has(row)) { seenRows.add(row); firsts.push(i); }
  }
  const ids = profile.columns.filter(c => c.name !== target && idLike(c, n, firsts));
  ran.push('id');
  for (const c of ids) {
    flags.push({ kind: 'id', column: c.name,
      title: `${c.name} looks like an ID`,
      text: `It has a different value in ${c.uniq === firsts.length ? 'every row' : 'almost every row'}, like an order number. A model can memorise it and score well on rows it has seen, and it says nothing about new ones.`,
      fix: 'Leave it out of the features; the script this page writes already does. If it counts time instead (a year, a day number), keep it and answer that row order matters, so the split follows it.' });
  }

  // One column that predicts the target by itself.
  let leakRows = 0;
  let best = null;
  const t = target ? targetValues(profile, target, decl.task) : null;
  if (t) {
    const rows = [];
    for (let i = 0; i < n; i++) if (t.values[i] !== null) rows.push(i);
    leakRows = rows.length;
    if (rows.length >= LEAK_MIN_ROWS) {
      ran.push('leak');
      const idNames = new Set(ids.map(c => c.name));
      const all = profile.columns
        .filter(c => c.name !== target && !idNames.has(c.name) && !c.dateLike && !c.textLike)
        .map(c => {
          const fitted = singleColumnScore(c, t, rows);
          const rank = t.kind === 'number' && c.numeric ? rankCorrelation(c, t, rows) : null;
          return rank !== null && rank > (fitted ?? -Infinity)
            ? { col: c, score: rank, measure: 'a rank correlation of' }
            : { col: c, score: fitted, measure: t.kind === 'number' ? 'R²' : 'balanced accuracy' };
        })
        .filter(s => s.score !== null)
        .sort((a, b) => b.score - a.score);
      best = all[0] ? { column: all[0].col.name, score: all[0].score } : null;
      for (const s of all.filter(x => x.score >= LEAK_SCORE).slice(0, 3)) {
        flags.push({ kind: 'leak', column: s.col.name, score: s.score,
          title: `${s.col.name} predicts ${target} almost perfectly on its own`,
          text: `Used alone, on rows it was not fitted on, it reaches ${s.measure} ${s.score.toFixed(3)}. Real predictors are rarely this good by themselves. This usually means it was recorded after the outcome, or computed from it, and it will not be there when you predict.`,
          fix: `Check when ${s.col.name} becomes known. If it is only known after ${target} is, leave it out.` });
      }
    }
  }

  // Rows repeated more than chance.
  const rep = repeatedRows(profile, target);
  const measured = profile.columns.some(c => c.name !== target && c.numeric && c.uniq >= DUPLICATE_MEASURED);
  if (measured) ran.push('duplicates');
  if (measured && rep.copies >= 2 && rep.share >= DUPLICATE_SHARE && rep.copies > DUPLICATE_OVER_CHANCE * rep.expected) {
    const conflict = target && rep.conflicting
      ? ` ${rep.conflicting.toLocaleString('en-US')} rows share their inputs with a row whose ${target} is different, which caps how well any model can do.`
      : '';
    flags.push({ kind: 'duplicates', copies: rep.copies, share: rep.share,
      title: `${rep.copies.toLocaleString('en-US')} rows are exact copies of other rows`,
      text: `That is ${(100 * rep.share).toFixed(1)}% of the file, more than chance would give. A random split puts a copy in the test set whose twin the model trained on, so the score is higher than it will be on new rows.${conflict}`,
      fix: 'Remove the copies, or keep each set of copies in the same fold when you split.' });
  }

  // Dates, with rows declared independent.
  const dates = profile.columns.filter(c => c.dateLike && c.name !== target);
  if (target) ran.push('time');
  if (target && dates.length && decl.order === 'A21') {
    const col = dates[0].name;
    flags.push({ kind: 'time', column: col,
      title: `${col} holds dates, and the rows were declared independent`,
      text: 'If the model will be used on rows that come later, a random split lets it learn from rows after the ones it is tested on. The score will be higher than what you see once it runs.',
      fix: 'Answer "Yes, it is a sequence" to row order, and the script splits by time instead.' });
  }

  return { flags, ran, rows: n, leakRows, best, repeats: rep };
}

// What the checks looked at, in words, for when nothing was found.
export function checksSummary(result, target) {
  const parts = [];
  if (result.ran.includes('leak')) parts.push(`no column predicts ${target} on its own`);
  else if (target) parts.push(`too few rows with ${target} (${result.leakRows}) to test whether one column predicts it on its own`);
  parts.push('no column looks like a row ID');
  parts.push(result.ran.includes('duplicates') ? 'no more repeated rows than chance'
    : 'repeated rows not judged, since without a numeric measurement column identical rows can be natural');
  if (result.ran.includes('time')) parts.push('no dates split at random');
  return parts;
}
