// The learned order: what it uses, what it refuses to claim, and that it
// matches the weights committed in site/taxonomy/ranking.json.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_PRIOR, blend, hasLearnedRanking, learnedScore, metaDistance, metaVector, neighboursOf, rankingFeatures,
  rankingNeighbours, rankingProvenance } from '../site/js/ranking.js';
import { rankModels } from '../site/js/recommend.js';
import { loadTaxonomyFromDisk } from './helpers.js';

const T = loadTaxonomyFromDisk();
const sig = (codes, extra = {}) => ({ codes, flags: [], rows: 1000, features: 10, ...extra });
const NUMERIC = sig(['A11', 'A21', 'A31', 'A41', 'A53', 'A61']);

test('the weights are present, and say what they were fitted on', () => {
  assert.ok(T.RANKING, 'site/taxonomy/ranking.json is missing');
  assert.ok(T.RANKING.trained_on.datasets >= 20);
  assert.ok(['prior', 'prior_fit', 'prior_knn'].includes(T.RANKING.chosen));
  assert.deepEqual(Object.keys(T.RANKING.tasks).sort(), ['category', 'number']);
});

test('features come from the measured signature and the shape', () => {
  const f = rankingFeatures(sig(['A11', 'A21', 'A38', 'A42', 'A52', 'A62'], { rows: 100, features: 50 }));
  assert.equal(f.is_A38, 1);
  assert.equal(f.is_A31, 0);
  assert.equal(f.log_rows, 2);
  assert.ok(Math.abs(f.features_per_row - Math.log10(0.5)) < 1e-9);
});

test('a model the benchmark never ran gets no score rather than a made-up one', () => {
  const unseen = T.MODELS.find(m => m.c === 'TSM1');   // ARIMA: never benchmarked
  assert.equal(learnedScore(T, unseen, 'number', rankingFeatures(NUMERIC)), null);
  const seen = T.MODELS.find(m => m.c === 'EN2');
  assert.ok(learnedScore(T, seen, 'number', rankingFeatures(NUMERIC)) > DEFAULT_PRIOR);
});

test('tasks outside the benchmark have no learned ranking', () => {
  assert.equal(hasLearnedRanking(T, 'number'), true);
  assert.equal(hasLearnedRanking(T, 'category'), true);
  for (const task of ['forecast', 'survival', 'group', 'anomaly', 'compress', 'generate']) {
    assert.equal(hasLearnedRanking(T, task), false, task);
    assert.equal(rankingProvenance(T, task), null, task);
  }
});

test('the order matches the committed weights, best first', () => {
  for (const task of ['number', 'category']) {
    const items = rankModels(T, NUMERIC, task, 99).items;
    const scored = items.filter(m => m.evidenceScore != null);
    const unscored = items.filter(m => m.evidenceScore == null);
    for (let i = 1; i < scored.length; i++) {
      assert.ok(scored[i - 1].evidenceScore >= scored[i].evidenceScore, `${task}: out of order`);
    }
    assert.ok(items.indexOf(unscored[0] ?? items[items.length - 1]) >= scored.length - 1,
      `${task}: unbenchmarked models must sit below benchmarked ones`);
    const prior = T.RANKING.tasks[task].prior;
    assert.ok(Math.abs(scored[0].evidenceScore - prior[scored[0].c]) < 1e-9);
  }
});

test('boosting and the tree ensembles lead, which is what the benchmark found', () => {
  const top3 = rankModels(T, NUMERIC, 'number', 3).items.map(m => m.c);
  assert.ok(top3.includes('EN2'), 'gradient boosting was the strongest regressor in the run');
  assert.ok(!top3.includes('LM1'), 'plain linear regression used to be first and was worth 0.3 R² less');
});

// A small neighbour block, built by hand so the arithmetic can be checked.
const BLOCK = {
  k: 2, strength: 4, bandwidth: 1, features: ['a', 'b'], scale: { a: 1, b: 2 },
  datasets: [
    { dataset: 'near', meta: [0, 0], ranks: { X: 1.0, Y: 0.0 } },
    { dataset: 'mid', meta: [1, 0], ranks: { X: 0.5 } },
    { dataset: 'far', meta: [5, 5], ranks: { X: 0.0, Y: 1.0 } },
  ],
};

test('distance is scaled by each feature\'s spread, at stored precision', () => {
  assert.deepEqual(metaVector({ a: 0.123456, b: 2 }, ['a', 'b']), [0.1235, 2]);
  assert.ok(Math.abs(metaDistance([0, 0], [1, 2], ['a', 'b'], { a: 1, b: 2 }) - 1) < 1e-12);
});

test('the k nearest datasets come back nearest first, weighted by closeness', () => {
  const found = neighboursOf(BLOCK, { a: 0, b: 0 });
  assert.deepEqual(found.map(n => n.d.dataset), ['near', 'mid'], 'k = 2 leaves the far one out');
  assert.equal(found[0].w, 1, 'distance 0 weighs fully');
  assert.ok(Math.abs(found[1].w - Math.exp(-0.5)) < 1e-12, 'mid is sqrt(1/2) away');
});

test('the blend pulls the prior toward what the model did nearby, and no further than it earned', () => {
  const found = neighboursOf(BLOCK, { a: 0, b: 0 });
  const w = Math.exp(-0.5);
  assert.ok(Math.abs(blend(0.5, 'X', found, 4) - (4 * 0.5 + 1 * 1.0 + w * 0.5) / (4 + 1 + w)) < 1e-12);
  // Y never ran on "mid", so only "near" speaks for it.
  assert.ok(Math.abs(blend(0.5, 'Y', found, 4) - (4 * 0.5 + 0) / (4 + 1)) < 1e-12);
  // A model no neighbour saw keeps its prior.
  assert.equal(blend(0.7, 'Z', found, 4), 0.7);
});

test('neighbours are only used when the order in use is the neighbour order', () => {
  const task = 'number';
  const withBlock = { ...T, RANKING: { ...T.RANKING, chosen: 'prior_knn',
    tasks: { ...T.RANKING.tasks, [task]: { ...T.RANKING.tasks[task], neighbours: BLOCK } } } };
  assert.ok(rankingNeighbours(withBlock, task, { a: 0, b: 0 }));
  assert.equal(rankingNeighbours(withBlock, task, null), null, 'no meta-features, no neighbours');
  assert.equal(rankingNeighbours({ ...withBlock, RANKING: { ...withBlock.RANKING, chosen: 'prior' } }, task,
    { a: 0, b: 0 }), null);
  const model = { c: 'X' };
  const prior = { ...withBlock, RANKING: { ...withBlock.RANKING, tasks: { [task]: { prior: { X: 0.5 }, neighbours: BLOCK } } } };
  const found = rankingNeighbours(prior, task, { a: 0, b: 0 });
  assert.ok(learnedScore(prior, model, task, {}, found) > 0.5, 'X did well on its neighbours');
  assert.equal(learnedScore(prior, model, task, {}, null), 0.5);
});
