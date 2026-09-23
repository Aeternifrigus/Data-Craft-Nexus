import { test } from 'node:test';
import assert from 'node:assert/strict';
import { caution, conflict, leadRecommendation, rankModels, rankPipelines } from '../site/js/recommend.js';
import { leadSentence } from '../site/js/evidence.js';
import { loadTaxonomyFromDisk } from './helpers.js';

const T = loadTaxonomyFromDisk();
const sig = (codes, flags = []) => ({ codes, flags });

const LABELLED_NUMERIC_IID = sig(['A11', 'A21', 'A31', 'A41', 'A53', 'A61']);
const LABELLED_NUMERIC_SEQ = sig(['A11', 'A22', 'A31', 'A41', 'A54', 'A61']);
const UNLABELLED_NUMERIC = sig(['A12', 'A21', 'A31', 'A41', 'A53', 'A61']);
const MIXED_TABLE = sig(['A11', 'A21', 'A38', 'A41', 'A52', 'A62']);

const names = (r) => r.items.map(m => m.n);

test('survival models are no longer offered as forecasters', () => {
  const forecast = rankModels(T, LABELLED_NUMERIC_SEQ, 'forecast');
  assert.ok(!names(forecast).includes('Kaplan-Meier Estimator'));
  assert.deepEqual(names(forecast).slice(0, 3), ['ARIMA', 'Exponential Smoothing (Holt-Winters)', 'Prophet']);
  const survival = rankModels(T, LABELLED_NUMERIC_IID, 'survival');
  assert.deepEqual(names(survival), ['Cox Proportional Hazards', 'Kaplan-Meier Estimator']);
});

test('forecasting on unordered rows returns nothing, and says why', () => {
  const r = rankModels(T, LABELLED_NUMERIC_IID, 'forecast');
  assert.equal(r.items.length, 0);
  assert.ok(r.ruledOut.length > 0);
  assert.match(r.ruledOut.map(m => m.why).join(' '), /A22/);
});

test('time series models are ruled out when rows are independent', () => {
  const arima = T.MODELS.find(m => m.n === 'ARIMA');
  assert.match(conflict(arima, LABELLED_NUMERIC_IID), /A22/);
  assert.equal(conflict(arima, LABELLED_NUMERIC_SEQ), null);
});

test('text and image models are ruled out on a numeric table', () => {
  for (const name of ['RNN', 'LSTM']) {
    const m = T.MODELS.find(x => x.n === name);
    assert.match(conflict(m, LABELLED_NUMERIC_SEQ), /A34/);
  }
});

test('models for numbers or categories still fit a mixed table', () => {
  const rf = T.MODELS.find(m => m.n === 'Random Forest');
  const logistic = T.MODELS.find(m => m.n === 'Logistic Regression');
  assert.equal(conflict(rf, MIXED_TABLE), null);
  assert.equal(conflict(logistic, MIXED_TABLE), null);
});

test('a model for numbers fits a table of categories once they are encoded, and says so', () => {
  const categorical = sig(['A11', 'A21', 'A32', 'A41', 'A51', 'A61']);
  const logistic = T.MODELS.find(m => m.c === 'LM2');
  assert.equal(conflict(logistic, categorical), null, 'the benchmark one-hot encoded categories, and these models won');
  assert.match(caution(logistic, categorical), /one-hot encode/);
  assert.equal(caution(logistic, LABELLED_NUMERIC_IID), null);
  const arima = T.MODELS.find(m => m.n === 'ARIMA');
  const orderedCategories = sig(['A11', 'A22', 'A32', 'A41', 'A51', 'A61']);
  assert.match(conflict(arima, orderedCategories), /A31/, 'encoding categories does not make a numeric series');
});

test('supervised models are ruled out without a target, and unsupervised ones with one', () => {
  const rf = T.MODELS.find(m => m.n === 'Random Forest');
  const kmeans = T.MODELS.find(m => m.n === 'K-Means');
  assert.match(conflict(rf, UNLABELLED_NUMERIC), /supervised/);
  assert.match(conflict(kmeans, LABELLED_NUMERIC_IID), /unsupervised/);
});

test('reinforcement learning models are never recommended', () => {
  const rl = T.MODELS.filter(m => m.p === 'RL');
  assert.ok(rl.length > 0);
  for (const m of rl) assert.match(conflict(m, LABELLED_NUMERIC_IID), /environment/);
});

test('GAN and VAE are offered for generating examples, not for classification', () => {
  assert.deepEqual(names(rankModels(T, UNLABELLED_NUMERIC, 'category')), [],
    'unlabelled data used to be answered with GAN and VAE');
  // GAN is tagged spatial (A23), so it needs image-shaped data; VAE assumes
  // independent examples and is usable either way.
  assert.deepEqual(names(rankModels(T, sig(['A12', 'A23', 'A35', 'A41', 'A53', 'A61']), 'generate')), ['GAN', 'VAE']);
  assert.deepEqual(names(rankModels(T, sig(['A12', 'A21', 'A35', 'A41', 'A53', 'A61']), 'generate')), ['VAE']);
  // They are image models, so they don't turn up for a numeric table.
  assert.deepEqual(names(rankModels(T, UNLABELLED_NUMERIC, 'generate')), []);
});

test('every model except the reinforcement learning ones can be recommended somewhere', () => {
  const sigs = [];
  for (const supervision of ['A11', 'A12']) {
    for (const structure of ['A21', 'A22', 'A23', 'A24', 'A25', 'A26']) {
      for (const modality of ['A31', 'A32', 'A33', 'A34', 'A35', 'A36', 'A37', 'A38']) {
        sigs.push(sig([supervision, structure, modality, 'A41', 'A53', 'A61']));
      }
    }
  }
  const reachable = new Set();
  for (const s of sigs) {
    for (const task of T.TASKS.map(t => t.id)) {
      for (const m of rankModels(T, s, task, 99).items) reachable.add(m.c);
    }
  }
  const unreachable = T.MODELS.filter(m => m.p !== 'RL' && !reachable.has(m.c)).map(m => `${m.c} ${m.n}`);
  assert.deepEqual(unreachable, []);
});

test('a model is judged on the features it eats, not on its target', () => {
  // The modality code used to mean the target on classifiers (Logistic
  // Regression was A32 because its output is a class), so the conflict rule
  // ruled it out on numeric tables. The benchmark caught it: it would have
  // won on three datasets it was never offered for.
  const logistic = T.MODELS.find(m => m.n === 'Logistic Regression');
  const boosting = T.MODELS.find(m => m.n === 'Gradient Boosting (GBM)');
  const tree = T.MODELS.find(m => m.n === 'Decision Tree');
  assert.equal(conflict(logistic, LABELLED_NUMERIC_IID), null, 'logistic regression eats numbers');
  assert.equal(conflict(boosting, MIXED_TABLE), null, 'boosting handles a mixed table');
  assert.equal(conflict(tree, MIXED_TABLE), null, 'so does a decision tree');
  assert.equal(conflict(boosting, sig(['A11', 'A21', 'A32', 'A41', 'A51', 'A61'])), null,
    'and a table of categories');
});

test('models with a regressor are offered for a numeric target', () => {
  // AdaBoost and the SVMs were tagged classification-only.
  const numeric = rankModels(T, LABELLED_NUMERIC_IID, 'number', 99).items.map(m => m.n);
  for (const name of ['AdaBoost', 'SVM (linear)', 'Kernel SVM (RBF/Poly)']) {
    assert.ok(numeric.includes(name), `${name} should be available for a number`);
  }
});

test('a benchmarked task is ordered by evidence, and ties disappear', () => {
  const r = rankModels(T, LABELLED_NUMERIC_IID, 'number');
  assert.equal(r.rankedBy, 'evidence');
  assert.equal(r.tied, 1, 'measured scores separate models that coordinates could not');
  assert.ok(r.items[0].evidenceScore > r.items[1].evidenceScore);
  assert.equal(r.items[0].of, 3, 'the coordinate count is still reported');
});

test('a task the benchmark never covered falls back to coordinates, and says so', () => {
  const r = rankModels(T, sig(['A11', 'A22', 'A31', 'A41', 'A54', 'A61']), 'forecast');
  assert.equal(r.rankedBy, 'coordinates');
  assert.ok(r.items.every(m => m.evidenceScore == null));
  assert.ok(r.tied >= 1);
});

test('a pipeline earns no points from a wildcard code', () => {
  const r = rankPipelines(T, LABELLED_NUMERIC_IID, 'number', 99);
  const batch = r.items.find(p => p.c === 'PL-T1'); // data: A22, A3x, A6x
  assert.deepEqual(batch.hits, []);
  assert.equal(batch.of, 1);
  assert.equal(batch.score, 1, 'only the task match counts');
});

test('tuned boosting goes first only where the benchmark measured it, and says why', () => {
  const table = sig(['A11', 'A21', 'A38', 'A41', 'A51', 'A61']);
  assert.equal(T.RANKING.lead.led, true, 'the committed ranking puts tuned boosting first');
  const lead = leadRecommendation(T, table, 'category');
  assert.equal(lead.c, 'BASE-HGB-TUNED');
  assert.ok(leadRecommendation(T, table, 'number'));
  assert.equal(leadRecommendation(T, table, 'forecast'), null, 'not benchmarked for forecasting');
  assert.equal(leadRecommendation(T, sig(['A12', 'A21', 'A31', 'A41', 'A53', 'A61']), 'cluster'), null, 'needs labels');
  assert.equal(leadRecommendation(T, sig(['A11', 'A21', 'A34', 'A41', 'A51', 'A61']), 'category'), null, 'no text was benchmarked');
  assert.equal(leadRecommendation({ RANKING: { lead: { led: false } } }, table, 'category'), null);
  assert.match(leadSentence(T, lead), /beat the order's first pick on 49 of 78 independent classification units/);
  assert.match(leadSentence(T, lead), /by more than luck on classification/);
});
