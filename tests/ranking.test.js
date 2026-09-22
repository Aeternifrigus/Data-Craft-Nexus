// The learned order: what it uses, what it refuses to claim, and that it
// matches the weights committed in site/taxonomy/ranking.json.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_PRIOR, hasLearnedRanking, learnedScore, rankingFeatures, rankingProvenance } from '../site/js/ranking.js';
import { rankModels } from '../site/js/recommend.js';
import { loadTaxonomyFromDisk } from './helpers.js';

const T = loadTaxonomyFromDisk();
const sig = (codes, extra = {}) => ({ codes, flags: [], rows: 1000, features: 10, ...extra });
const NUMERIC = sig(['A11', 'A21', 'A31', 'A41', 'A53', 'A61']);

test('the weights are present, and say what they were fitted on', () => {
  assert.ok(T.RANKING, 'site/taxonomy/ranking.json is missing');
  assert.ok(T.RANKING.trained_on.datasets >= 20);
  assert.ok(['prior', 'prior_fit'].includes(T.RANKING.chosen));
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
