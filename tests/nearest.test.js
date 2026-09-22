// "Datasets like yours": measuring an upload on the same ruler as the
// benchmark datasets, and comparing it with them.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { parseCSV } from '../site/js/csv.js';
import { profileData } from '../site/js/profile.js';
import { distance, metaFeatures, nearestDatasets, performanceOn, winnerAmong } from '../site/js/nearest.js';
import { loadTaxonomyFromDisk } from './helpers.js';

const T = loadTaxonomyFromDisk();
const ROOT = new URL('../', import.meta.url);

function measure(path, target) {
  const { head, body } = parseCSV(fs.readFileSync(new URL(path, ROOT), 'utf8'));
  return metaFeatures(profileData(head, body), target);
}

test('the benchmark datasets carry the features an upload is measured on', () => {
  assert.ok(T.EVIDENCE.meta_features.length >= 6);
  assert.ok(T.EVIDENCE.meta_scale, 'no scale to measure distance with');
  const withMeta = T.EVIDENCE.datasets.filter(d => d.meta);
  assert.equal(withMeta.length, T.EVIDENCE.datasets.length, 'every dataset needs its features');
  for (const name of T.EVIDENCE.meta_features) {
    assert.ok(name in withMeta[0].meta, `${name} missing from a dataset`);
    assert.ok(T.EVIDENCE.meta_scale[name].std > 0, `${name} has no spread`);
  }
});

test('an upload is measured the same way the benchmark datasets were', () => {
  const meta = measure('site/sample.csv', 'transit_days');
  assert.ok(Math.abs(meta.log_rows - Math.log10(20)) < 1e-9);
  assert.equal(meta.log_features, Math.log10(9), 'the target is not a feature');
  assert.ok(meta.numeric_share > 0 && meta.numeric_share < 1, 'the sample is a mixed table');
  assert.ok(meta.categorical_share > 0);
  for (const name of T.EVIDENCE.meta_features) assert.ok(name in meta, `${name} not computed for an upload`);
});

test('different files land in different places, which the old plot could not do', () => {
  // The old plot used axes 1 to 3: every labelled table with independent rows
  // sat at the same point regardless of size or content.
  const a = measure('tests/fixtures/numeric.csv', 'y');
  const b = measure('tests/fixtures/highdim.csv', 'out');
  const c = measure('tests/fixtures/balanced.csv', 'label');
  const names = T.EVIDENCE.meta_features, scale = T.EVIDENCE.meta_scale;
  assert.ok(distance(a, b, names, scale) > 0.3, 'a wide table and a tall one must not coincide');
  assert.ok(distance(a, c, names, scale) > 0.1);
  assert.equal(distance(a, a, names, scale), 0);
});

test('neighbours come back nearest first, and only from the right task', () => {
  const meta = measure('tests/fixtures/imbalanced.csv', 'churn');
  const neighbours = nearestDatasets(T, meta, 'category', 5);
  assert.equal(neighbours.length, 5);
  assert.ok(neighbours.every(d => d.task === 'classification'));
  for (let i = 1; i < neighbours.length; i++) {
    assert.ok(neighbours[i - 1].distance <= neighbours[i].distance);
  }
  const regression = nearestDatasets(T, meta, 'number', 5);
  assert.ok(regression.every(d => d.task === 'regression'));
});

test('a task the benchmark never covered has no neighbours to offer', () => {
  const meta = measure('site/sample.csv', 'transit_days');
  assert.deepEqual(nearestDatasets(T, meta, 'forecast'), []);
  assert.deepEqual(nearestDatasets(T, meta, 'anomaly'), []);
});

test('how a model did on the neighbours is counted from their recorded scores', () => {
  const meta = measure('tests/fixtures/imbalanced.csv', 'churn');
  const neighbours = nearestDatasets(T, meta, 'category', 6);
  const forest = performanceOn(neighbours, 'TR2');
  assert.ok(forest.datasets > 0 && forest.datasets <= 6);
  assert.ok(forest.wins >= 0 && forest.wins <= forest.datasets);
  assert.ok(forest.medianGap >= 0, 'a gap below the winner cannot be negative');

  // A model that never ran on those datasets makes no claim.
  assert.equal(performanceOn(neighbours, 'NN9'), null);
  assert.equal(performanceOn([], 'TR2'), null);

  const winner = winnerAmong(neighbours);
  assert.ok(winner.wins >= 1 && winner.of === neighbours.length);
  assert.ok(neighbours.some(d => d.best.model === winner.model));
});
