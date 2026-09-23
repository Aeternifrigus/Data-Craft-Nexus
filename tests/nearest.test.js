// "Datasets like yours": measuring an upload on the same ruler as the
// benchmark datasets, and comparing it with them.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { parseCSV } from '../site/js/csv.js';
import { profileData, signature } from '../site/js/profile.js';
import { coverageNotes, distance, metaFeatures, nearestDatasets, performanceOn, quantile, winnerAmong }
  from '../site/js/nearest.js';
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

test('what the benchmark covered can be recomputed from the datasets on the page', () => {
  const ev = T.EVIDENCE;
  for (const [task, cov] of Object.entries(ev.coverage)) {
    const rows = ev.datasets.filter(d => d.task === task && d.meta);
    assert.equal(cov.datasets, rows.length, task);
    assert.equal(cov.min_rows, Math.min(...rows.map(d => d.rows)), `${task}: smallest dataset`);
    // A dataset's own family does not count as its neighbour.
    const nearest = rows.map(d => Math.min(...rows.filter(o => o.family !== d.family)
      .map(o => distance(d.meta, o.meta, ev.meta_features, ev.meta_scale))));
    assert.ok(Math.abs(quantile(nearest, 0.95) - cov.nearest_p95) < 1e-3,
      `${task}: page says ${cov.nearest_p95}, datasets give ${quantile(nearest, 0.95)}`);
  }
});

test('a tiny upload is told it is outside what was tested', () => {
  const meta = measure('site/sample.csv', 'delayed');
  const neighbours = nearestDatasets(T, meta, 'category');
  const notes = coverageNotes(T, meta, 20, 'category', neighbours);
  const rows = notes.find(n => n.kind === 'rows');
  assert.ok(rows, 'twenty rows is below the smallest benchmark dataset');
  assert.match(rows.text, /20 rows/);
  assert.match(rows.text, new RegExp(String(T.EVIDENCE.coverage.classification.min_rows)));
});

test('an upload like a benchmark dataset gets no warning', () => {
  const d = T.EVIDENCE.datasets.find(x => x.task === 'regression' && x.meta);
  const neighbours = nearestDatasets(T, d.meta, 'number');
  assert.equal(neighbours[0].distance, 0, 'it is its own nearest neighbour');
  assert.deepEqual(coverageNotes(T, d.meta, d.rows, 'number', neighbours), []);
});

test('an upload far from everything is told so, and a task without a benchmark says nothing', () => {
  const d = T.EVIDENCE.datasets.find(x => x.task === 'classification' && x.meta);
  const far = { ...d.meta, log_features: d.meta.log_features + 10, missing_share: 0.9 };
  const notes = coverageNotes(T, far, d.rows, 'category', nearestDatasets(T, far, 'category'));
  assert.deepEqual(notes.map(n => n.kind), ['far']);
  assert.deepEqual(coverageNotes(T, far, 5, 'group', []), []);
});

test('quantile matches numpy on a small case', () => {
  assert.equal(quantile([1, 2, 3, 4], 0.5), 2.5);
  assert.ok(Math.abs(quantile([0, 10], 0.95) - 9.5) < 1e-12);
});

test('an upload is placed without its target, as the benchmark datasets are', () => {
  // results.js measures the upload with sig.target. signature() did not
  // return one, so the target was counted as a feature and the class balance
  // of every upload read as zero.
  const { head, body } = parseCSV(fs.readFileSync(new URL('tests/fixtures/imbalanced.csv', ROOT), 'utf8'));
  const profile = profileData(head, body);
  const sig = signature(profile, { target: 'churn', task: 'category', order: 'A21' });
  assert.equal(sig.target, 'churn');
  const meta = metaFeatures(profile, sig.target ?? null);
  assert.equal(meta.log_features, Math.log10(head.length - 1));
  assert.ok(meta.majority_share > 0.5, 'an imbalanced target has a majority class');
  assert.equal(signature(profile, { target: '__none__', task: 'group', order: 'A21' }).target, null);
});
