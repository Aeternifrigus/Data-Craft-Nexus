// A messy upload is told what the same kind of damage did on the benchmark.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { MESSY_CONDITIONS, messyConditionFor, messyNote } from '../site/js/evidence.js';
import { loadTaxonomyFromDisk } from './helpers.js';

const sig = (a6, flags = []) => ({ codes: ['A11', 'A21', 'A31', 'A41', 'A51', a6], flags });

test('the damaged run quoted is the kind of mess the file has, at the nearest amount', () => {
  assert.equal(messyConditionFor(sig('A61')), null);
  assert.equal(messyConditionFor(sig('A62')), 'missing_10');
  assert.equal(messyConditionFor(sig('A62'), 0.04), 'missing_10');
  assert.equal(messyConditionFor(sig('A62'), 0.35), 'missing_30');
  assert.equal(messyConditionFor(sig('A64')), 'dirty_5');
  assert.equal(messyConditionFor(sig('A61', ['A64'])), 'dirty_5');
});

test('the note quotes the benchmark, and says nothing for a clean file or an unbenchmarked task', () => {
  const T = { EVIDENCE: { messy: { conditions: { missing_10: { tasks: { classification: {
    datasets: 20, median_loss: 0.0264, first_regret: 0.0199, first_regret_clean: 0.0163 } } } } } } };
  assert.equal(messyNote(T, sig('A62'), 'category', 0.1),
    ' Your file has missing values. On 20 benchmark classification datasets with 10% of their cells blanked, scores '
    + 'fell by a median of 0.026, and the model shown first landed 0.020 below the best, against 0.016 on the same '
    + 'datasets clean.');
  assert.equal(messyNote(T, sig('A61'), 'category'), '');
  assert.equal(messyNote(T, sig('A62'), 'forecast'), '');
  assert.equal(messyNote({ EVIDENCE: null }, sig('A62'), 'category'), '');
});

test('every damaged run the page can quote is published for both tasks', () => {
  const messy = loadTaxonomyFromDisk().EVIDENCE.messy;
  if (!messy) return;
  for (const condition of Object.keys(MESSY_CONDITIONS)) {
    for (const task of ['classification', 'regression']) {
      const entry = messy.conditions[condition]?.tasks?.[task];
      assert.ok(entry, `${condition} ${task} missing`);
      assert.ok(entry.datasets > 0 && Object.keys(entry.models).length > 0, `${condition} ${task} empty`);
    }
  }
});
