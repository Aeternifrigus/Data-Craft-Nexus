// "Before you trust a score": the checks fire on what they are for, and stay
// quiet on ordinary data.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseCSV } from '../site/js/csv.js';
import { profileData, signature } from '../site/js/profile.js';
import { checksSummary, idLike, repeatedRows, runChecks, LEAK_SCORE } from '../site/js/checks.js';
import { columnRoles, pythonScript } from '../site/js/export.js';
import { rankDrifts, rankModels, rankPipelines } from '../site/js/recommend.js';
import { readingMarkdown } from '../site/js/report.js';
import { loadTaxonomyFromDisk, readFixture } from './helpers.js';

const profileOf = (csv) => { const { head, body } = parseCSV(csv); return profileData(head, body); };
const kinds = (result) => result.flags.map(f => `${f.kind}:${f.column ?? ''}`);

// A small deterministic generator, so every run sees the same table.
function rng(seed) {
  let s = seed >>> 0;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 2 ** 32; };
}

// 400 shipments: an honest predictor (distance), noise, a category, and a
// label that depends on them only partly.
function shipments({ leak = false, copyTarget = false, withId = false, repeats = 0, dates = false } = {}) {
  const r = rng(7);
  const head = ['distance_km', 'weight_kg', 'carrier', 'delayed'];
  if (leak) head.splice(3, 0, 'days_late');
  if (copyTarget) head.splice(3, 0, 'status_after');
  if (withId) head.unshift('shipment_id');
  if (dates) head.unshift('ship_date');
  const rows = [];
  for (let i = 0; i < 400; i++) {
    const distance = Math.round(500 + r() * 9000);
    const weight = Math.round(100 + r() * 2000);
    const carrier = ['A', 'B', 'C'][Math.floor(r() * 3)];
    const delayed = r() < 0.2 + distance / 20000 ? 'yes' : 'no';
    const row = [distance, weight, carrier];
    if (leak) row.push(delayed === 'yes' ? 1 + Math.floor(r() * 9) : 0);   // known only once it arrived
    if (copyTarget) row.push(delayed === 'yes' ? 'LATE' : 'ON_TIME');
    row.push(delayed);
    if (withId) row.unshift(10001 + i);
    if (dates) row.unshift(`2025-${String(1 + (i % 12)).padStart(2, '0')}-${String(1 + (i % 28)).padStart(2, '0')}`);
    rows.push(row);
  }
  for (let k = 0; k < repeats; k++) rows.push([...rows[k * 3]]);
  return [head, ...rows].map(r => r.join(',')).join('\n') + '\n';
}

const decl = { target: 'delayed', task: 'category', order: 'A21' };

test('an ordinary table raises nothing, and says what it checked', () => {
  const result = runChecks(profileOf(shipments()), decl);
  assert.deepEqual(result.flags, []);
  assert.deepEqual(result.ran, ['id', 'leak', 'duplicates', 'time']);
  assert.match(checksSummary(result, 'delayed').join(', '), /no column predicts delayed on its own/);
});

test('a column recorded after the outcome is caught, and so is a relabelled copy of the target', () => {
  const leak = runChecks(profileOf(shipments({ leak: true })), decl);
  assert.deepEqual(kinds(leak), ['leak:days_late']);
  assert.ok(leak.flags[0].score >= LEAK_SCORE);
  assert.match(leak.flags[0].text, /balanced accuracy 1\.000/);
  const copy = runChecks(profileOf(shipments({ copyTarget: true })), decl);
  assert.deepEqual(kinds(copy), ['leak:status_after']);
});

test('a strong but honest predictor of a number is not a leak', () => {
  const r = rng(3);
  const rows = Array.from({ length: 500 }, () => {
    const carat = 0.2 + r() * 2;
    return [carat.toFixed(2), Math.round(3000 * carat + 900 * (r() - 0.5) * 2)];   // R² around 0.9
  });
  const csv = ['carat,price', ...rows.map(x => x.join(','))].join('\n');
  assert.deepEqual(runChecks(profileOf(csv), { target: 'price', task: 'number', order: 'A21' }).flags, []);
  const exact = ['carat,price,price_eur', ...rows.map(([c, p]) => `${c},${p},${(p * 0.92).toFixed(2)}`)].join('\n');
  assert.deepEqual(kinds(runChecks(profileOf(exact), { target: 'price', task: 'number', order: 'A21' })),
    ['leak:price_eur'], 'the same number in another currency is the target');
});

test('row numbers and codes look like IDs; measurements and free text do not', () => {
  const n = 60;
  const col = (name, values) => profileOf(`${name},y\n${values.map((v, i) => `${v},${i % 2}`).join('\n')}\n`).columns[0];
  assert.ok(idLike(col('row', Array.from({ length: n }, (_, i) => i + 1)), n), 'a run 1..n');
  assert.ok(idLike(col('customer_id', Array.from({ length: n }, (_, i) => 90000 + i * 37)), n), 'named like an ID');
  assert.ok(idLike(col('customerId', Array.from({ length: n }, (_, i) => 5 + i * 991)), n), 'camelCase ID');
  assert.ok(idLike(col('ref', Array.from({ length: n }, (_, i) => `SHP-${1000 + i}`)), n), 'short codes');
  assert.ok(!idLike(col('income', Array.from({ length: n }, (_, i) => 4000 + i * 311)), n), 'a sparse integer measure');
  assert.ok(!idLike(col('weight', Array.from({ length: n }, (_, i) => (i * 1.37).toFixed(2))), n), 'decimals');
  assert.ok(!idLike(col('review', Array.from({ length: n }, (_, i) => `good value number ${i}`)), n), 'free text');
  assert.ok(!idLike(col('carrier', Array.from({ length: n }, (_, i) => ['A', 'B'][i % 2])), n), 'a category');
});

test('an ID column is flagged, skipped by the leak check, and left out of the script', () => {
  const profile = profileOf(shipments({ withId: true }));
  const result = runChecks(profile, decl);
  assert.deepEqual(kinds(result), ['id:shipment_id']);
  const roles = columnRoles(profile, 'delayed', ['shipment_id']);
  assert.ok(![...roles.numeric, ...roles.categorical].includes('shipment_id'));
  const script = pythonScript({ fileName: 'x.csv', columns: profile.columns.map(c => c.name), target: 'delayed',
    task: 'category', ordered: false, ...roles, shortlist: ['TR2'], leftOut: ['shipment_id'], date: '2026-01-01' });
  assert.match(script, /# Left out of the features: shipment_id\./);
  assert.doesNotMatch(script.split('NUMERIC =')[1].split('\n')[0], /shipment_id/);
});

test('an ID column is still found when some rows are copied', () => {
  const result = runChecks(profileOf(shipments({ withId: true, repeats: 20 })), decl);
  assert.deepEqual(kinds(result), ['id:shipment_id', 'duplicates:']);
});

test('repeated rows are flagged against chance, not on their own', () => {
  const copies = runChecks(profileOf(shipments({ repeats: 20 })), decl);
  assert.deepEqual(kinds(copies), ['duplicates:']);
  assert.match(copies.flags[0].title, /^20 rows are exact copies/);
  // Three categories of three values each: repeats are what chance gives.
  const small = ['a,b,c', ...Array.from({ length: 200 }, (_, i) => `${i % 3},${(i >> 1) % 3},${(i >> 2) % 3}`)].join('\n');
  const rep = repeatedRows(profileOf(small), null);
  assert.ok(rep.copies > 150 && rep.expected >= rep.copies / 3);
  assert.deepEqual(runChecks(profileOf(small), { target: null, task: 'category', order: 'A21' }).flags, []);
  assert.deepEqual(runChecks(profileOf(readFixture('sparse.csv')), { target: 'target', task: 'category', order: 'A21' })
    .flags.filter(f => f.kind === 'duplicates'), [], 'mostly-zero rows repeat by chance');
});

test('the same inputs with different targets are counted', () => {
  const csv = ['x,z,y', ...Array.from({ length: 100 }, (_, i) => `${i * 7.1},${i * 3.3},${i % 2}`),
    '0,0,1', '7.1,3.3,1', '7.1,3.3,0'].join('\n');
  const rep = repeatedRows(profileOf(csv), 'y');
  assert.equal(rep.copies, 1, 'one row repeats an earlier one exactly');
  assert.equal(rep.conflicting, 5, 'two input pairs appear with both labels');
});

test('dates with rows declared independent are flagged; declared as a sequence they are not', () => {
  const csv = shipments({ dates: true });
  assert.deepEqual(kinds(runChecks(profileOf(csv), decl)), ['time:ship_date']);
  assert.deepEqual(runChecks(profileOf(csv), { ...decl, order: 'A22' }).flags, []);
});

test('too few rows to judge a leak is said, not hidden', () => {
  const result = runChecks(profileOf(readFixture('sample.csv')), { target: 'delayed', task: 'category', order: 'A22' });
  assert.ok(!result.ran.includes('leak'));
  assert.match(checksSummary(result, 'delayed')[0], /too few rows with delayed \(20\)/);
});

test('the reading opens with the checks', () => {
  const T = loadTaxonomyFromDisk();
  const profile = profileOf(shipments({ leak: true, withId: true }));
  const sig = signature(profile, decl);
  const checks = runChecks(profile, decl);
  const text = readingMarkdown({ T, sig, task: 'category', fileName: 'x.csv', date: '2026-09-24',
    models: rankModels(T, sig, 'category'), drifts: rankDrifts(T, sig), pipelines: rankPipelines(T, sig, 'category'),
    checks: { flags: checks.flags, clear: checksSummary(checks, 'delayed') } });
  assert.match(text, /## Before you trust a score\n\n- \*\*shipment_id looks like an ID\.\*\*/);
  assert.match(text, /- \*\*days_late predicts delayed almost perfectly on its own\.\*\*/);
  assert.ok(text.indexOf('## Before you trust a score') < text.indexOf('## Signature'));
});

test('the fixtures stay quiet except where they should not', () => {
  const cases = [['balanced.csv', 'label', 'category'], ['numeric.csv', 'y', 'number'],
    ['imbalanced.csv', 'churn', 'category'], ['categorical.csv', 'size', 'category'], ['missing.csv', 't', 'number']];
  for (const [file, target, task] of cases) {
    assert.deepEqual(kinds(runChecks(profileOf(readFixture(file)), { target, task, order: 'A21' })), [], file);
  }
  assert.deepEqual(kinds(runChecks(profileOf(readFixture('quoted.csv')), { target: 'score', task: 'number', order: 'A21' })),
    ['id:name'], 'item0, item1, ... is a row label');
});
