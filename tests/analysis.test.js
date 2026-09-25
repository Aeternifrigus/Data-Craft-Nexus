import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseCSV } from '../site/js/csv.js';
import { profileData, signature } from '../site/js/profile.js';
import { analyse, scriptPlan, takeHomeScript } from '../site/js/analysis.js';
import { loadTaxonomyFromDisk, readFixture } from './helpers.js';

const T = loadTaxonomyFromDisk();

function read(name, decl) {
  const { head, body } = parseCSV(readFixture(name));
  const profile = profileData(head, body);
  return { profile, sig: signature(profile, decl) };
}

test('a reading of the sample shipment table has every part the page draws', () => {
  const decl = { target: 'delayed', task: 'category', order: 'A21', mode: 'batch', labels: 'delayed', stage: null };
  const { profile, sig } = read('sample.csv', decl);
  const a = analyse(T, sig, 'category', profile);
  assert.ok(a.checks.ran.includes('id'));
  assert.ok(a.models.items.length > 0);
  assert.ok(a.drifts.items.length > 0);
  assert.equal(a.home.task, 'category');
  assert.deepEqual(a.takeHome.codes, a.models.items.map(m => m.c));
  // The sample has a date column and the rows were declared independent.
  assert.ok(a.checks.flags.some(f => f.kind === 'time' && f.column === 'ship_date'));
});

test('ID columns the checks find are left out of the script', () => {
  const decl = { target: 'delayed', task: 'category', order: 'A22', mode: 'batch', labels: 'delayed', stage: null };
  const csv = readFixture('sample.csv').split('\n').filter(Boolean)
    .map((line, i) => (i === 0 ? `shipment_id,${line}` : `SHP-${String(i).padStart(4, '0')},${line}`)).join('\n');
  const { head, body } = parseCSV(csv);
  const profile = profileData(head, body);
  const sig = signature(profile, decl);
  const a = analyse(T, sig, 'category', profile);
  assert.deepEqual(a.leftOut, ['shipment_id']);
  const plan = scriptPlan(a.home, a.takeHome.codes);
  assert.ok(plan.runnable);
  const script = takeHomeScript({ sig, home: a.home, profile, codes: a.takeHome.codes, leftOut: a.leftOut, cost: a.cost });
  const line = (name) => script.split('\n').find(l => l.startsWith(`${name} = `));
  assert.match(line('COLUMNS'), /"shipment_id"/);
  assert.doesNotMatch(line('NUMERIC'), /shipment_id/);
  assert.doesNotMatch(line('CATEGORICAL'), /shipment_id/);
});

test('there is no script without a target, and the reason is given', () => {
  const decl = { target: '__none__', task: 'group', order: 'A21', mode: 'batch', labels: 'delayed', stage: null };
  const { profile, sig } = read('sample.csv', decl);
  const a = analyse(T, sig, 'group', profile);
  assert.equal(a.home.task, null);
  assert.ok(a.home.why);
  assert.equal(scriptPlan(a.home, a.takeHome.codes).runnable, false);
});

test('a future number in time order always has a script, even when no card can run', () => {
  assert.equal(scriptPlan({ task: 'number', forecast: true }, []).runnable, true);
  assert.equal(scriptPlan({ task: 'number' }, []).runnable, false);
});
