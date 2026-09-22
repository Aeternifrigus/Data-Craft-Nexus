import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseCSV } from '../site/js/csv.js';
import { profileData, measuredAxes, classBalance, measureDrift, signature } from '../site/js/profile.js';
import { psiNumeric, psiCategorical } from '../site/js/stats.js';
import { readFixture } from './helpers.js';

const load = (name) => {
  const { head, body } = parseCSV(readFixture(name));
  return { profile: profileData(head, body), head, body };
};

test('a small table with few columns is not high-dimensional', () => {
  const { profile } = load('sample.csv'); // 20 rows, 10 columns
  assert.equal(measuredAxes(profile, 'delayed').a4, 'A41');
});

test('many columns relative to rows is high-dimensional', () => {
  const { profile } = load('highdim.csv'); // 60 rows, 41 columns
  assert.equal(measuredAxes(profile, 'out').a4, 'A42');
});

test('the target column is left out of the feature axes', () => {
  const csv = 'x1,x2,label\n1,2,yes\n3,4,no\n5,6,yes\n7,8,no\n';
  const { head, body } = parseCSV(csv);
  const profile = profileData(head, body);
  assert.equal(measuredAxes(profile, 'label').a3, 'A31', 'features alone are numeric');
  assert.equal(measuredAxes(profile, null).a3, 'A38', 'with the label counted in, the table is mixed');
});

test('missing values give A62, and text in a numeric column gives A64', () => {
  const rows = (mk) => 'a,b\n' + Array.from({ length: 50 }, (_, i) => mk(i)).join('\n') + '\n';
  const complete = load2(rows(i => `${i},${i * 2}`));
  assert.equal(measuredAxes(complete, null).a6, 'A61');

  const missing = load2(rows(i => (i % 10 === 0 ? `,${i}` : `${i},${i}`)));
  assert.equal(measuredAxes(missing, null).a6, 'A62');

  const dirty = load2(rows(i => (i % 10 === 0 ? `n/d,${i}` : `${i},${i}`)));
  assert.equal(measuredAxes(dirty, null).a6, 'A64', 'unparseable values are noise, not missingness');
});

test('labels differing only by case or padding count as noise', () => {
  const values = Array.from({ length: 60 }, (_, i) => (i % 12 === 0 ? ' gdynia ' : 'Gdynia'));
  const csv = 'port\n' + values.join('\n') + '\n';
  assert.equal(measuredAxes(load2(csv), null).a6, 'A64');
});

function load2(csv) {
  const { head, body } = parseCSV(csv);
  return profileData(head, body);
}

test('class balance is measured from the target column', () => {
  const imbalanced = load('imbalanced.csv');
  assert.equal(classBalance(imbalanced.profile, 'churn').code, 'A52');
  const balanced = load('balanced.csv');
  assert.equal(classBalance(balanced.profile, 'label').code, 'A51');
  assert.equal(classBalance(balanced.profile, 'f1'), null, 'a numeric target has no class balance');
});

test('PSI is zero for identical samples and large for a shifted one', () => {
  const a = Array.from({ length: 400 }, (_, i) => String(i % 40));
  assert.ok(psiNumeric(a, a) < 1e-9);
  const shifted = a.map(v => String(Number(v) + 60));
  assert.ok(psiNumeric(a, shifted) > 0.25);
  assert.equal(psiNumeric(a.slice(0, 5), a.slice(0, 5)), null, 'too few rows to judge');
  const cats = Array.from({ length: 200 }, (_, i) => (i % 2 ? 'x' : 'y'));
  assert.ok(psiCategorical(cats, cats) < 1e-9);
  assert.ok(psiCategorical(cats, cats.map(() => 'x')) > 0.25);
});

test('drift across the file is measured, not assumed from the order answer', () => {
  const drifting = 'v\n' + Array.from({ length: 300 }, (_, i) => i).join('\n') + '\n';
  const stable = 'v\n' + Array.from({ length: 300 }, (_, i) => i % 50).join('\n') + '\n';
  assert.equal(measureDrift(load2(drifting), null).code, 'A54');
  assert.equal(measureDrift(load2(stable), null).code, 'A53');

  // Saying "it is a sequence" no longer makes the data non-stationary by itself.
  const sig = signature(load2(stable), { target: null, task: 'forecast', order: 'A22' });
  assert.equal(sig.codes[4], 'A53');
});

test('with only a date and a target column, the target drift is reported', () => {
  const rows = Array.from({ length: 200 }, (_, i) => `2024-01-01,${i}`).join('\n');
  const { head, body } = parseCSV('date,value\n' + rows + '\n');
  const drift = measureDrift(profileData(head, body), 'value');
  assert.equal(drift.scope, 'target');
  assert.equal(drift.code, 'A54');
});

test('a short file reports no drift measurement rather than guessing', () => {
  assert.equal(measureDrift(load('sample.csv').profile, 'delayed'), null);
});

test('when both are measured, balance leads and drift is carried alongside', () => {
  const { profile } = load('imbalanced.csv');
  const sig = signature(profile, { target: 'churn', task: 'category', order: 'A21' });
  assert.equal(sig.codes[4], 'A52');
  assert.ok(['A53', 'A54'].includes(sig.flags[0]));
  assert.equal(sig.codes[0], 'A11');
});
