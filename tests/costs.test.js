// "What does a wrong answer cost?": the answer picks what the take-home
// script scores by. bench/tests/test_export.py runs the scripts it produces.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseCSV } from '../site/js/csv.js';
import { profileData } from '../site/js/profile.js';
import { MISS_RATIO, MISS_RATIO_MAX, benchCost, costOptions, costSentence, costTask, costWarning, resolveCost,
  twoClasses } from '../site/js/costs.js';
import { pythonScript } from '../site/js/export.js';
import { readFixture } from './helpers.js';

const column = (file, name) => {
  const { head, body } = parseCSV(readFixture(file));
  return profileData(head, body).columns.find(c => c.name === name);
};
const ids = (options) => options.map(o => o.id);
const script = (task, cost) => pythonScript({
  fileName: 'x.csv', columns: ['a', 'y'], target: 'y', task, ordered: false, numeric: ['a'], categorical: [],
  shortlist: ['TR1'], cost, date: '2026-01-01',
});

test('left alone, the script scores by the benchmark\'s own score', () => {
  assert.equal(benchCost('category').scoring, 'balanced_accuracy');
  assert.equal(benchCost('number').scoring, 'r2');
  assert.deepEqual(resolveCost('category', null, column('balanced.csv', 'label')).id, 'balanced');
  const plain = script('category', null);
  assert.match(plain, /^SCORING = "balanced_accuracy"$/m);
  assert.match(plain, /^METRIC = "balanced accuracy"$/m);
  assert.match(script('number', null), /^SCORING = "r2"$/m);
  // Choosing the benchmark's score by hand is the same script.
  assert.equal(script('category', resolveCost('category', { id: 'balanced' }, column('balanced.csv', 'label'))), plain);
  assert.doesNotMatch(plain, /threshold_report|POSITIVE/);
});

test('the choices follow the kind of answer the script checks', () => {
  assert.equal(costTask('category', null), 'category');
  assert.equal(costTask('number', null), 'number');
  // A future value is checked as a number, or as a category when the target is text (export.js takeHomeTask).
  assert.equal(costTask('forecast', { numeric: true }), 'number');
  assert.equal(costTask('forecast', { numeric: false }), 'category');
  assert.equal(costTask('group', { numeric: true }), null);
  assert.deepEqual(ids(costOptions('number', column('numeric.csv', 'y'))), ['squared', 'absolute', 'percent']);
});

test('a miss can be priced only on a yes or no target, and the rare class is the case', () => {
  const churn = column('imbalanced.csv', 'churn');
  assert.deepEqual(twoClasses(churn), { rare: 'yes', rareCount: 35, common: 'no', commonCount: 365 });
  assert.ok(ids(costOptions('category', churn)).includes('miss'));
  assert.match(costOptions('category', churn).find(o => o.id === 'miss').label, /Missing a "yes"/);
  assert.ok(!ids(costOptions('category', column('balanced.csv', 'label'))).includes('miss'), 'three classes');
  // Missing words do not count as a class, and padding does not make a new one.
  assert.deepEqual(twoClasses({ values: ['a', ' a', 'b', 'NA', '', 'a'] }),
    { rare: 'b', rareCount: 1, common: 'a', commonCount: 3 });
});

test('a priced miss carries its price, bounded, and the threshold it implies', () => {
  const churn = column('imbalanced.csv', 'churn');
  const cost = resolveCost('category', { id: 'miss', ratio: 9 }, churn);
  assert.equal(cost.positive, 'yes');
  assert.equal(cost.ratio, 9);
  assert.equal(cost.threshold, 0.1);
  assert.equal(resolveCost('category', { id: 'miss' }, churn).ratio, MISS_RATIO);
  assert.equal(resolveCost('category', { id: 'miss', ratio: 0.5 }, churn).ratio, MISS_RATIO, 'below 1 is not a miss costing more');
  assert.equal(resolveCost('category', { id: 'miss', ratio: 1e9 }, churn).ratio, MISS_RATIO_MAX);
  // A miss on a target that is not yes or no falls back to the benchmark's score.
  assert.equal(resolveCost('category', { id: 'miss' }, column('balanced.csv', 'label')).id, 'balanced');

  const text = script('category', cost);
  assert.match(text, /^POSITIVE = "yes"$/m);
  assert.match(text, /^MISS_COST = 9$/m);
  assert.match(text, /^SCORING = cost_score$/m);
  assert.match(text, /y = \(frame\[TARGET\]\.astype\(str\)\.str\.strip\(\) == POSITIVE\)/);
  assert.match(text, /^    threshold_report\(results, frame\)$/m);
  assert.match(text, /You said a missed "yes" costs 9 false alarms/);
});

test('each other answer names its score in the script', () => {
  const label = column('balanced.csv', 'label');
  const churn = column('imbalanced.csv', 'churn');
  const y = column('numeric.csv', 'y');
  const cases = [
    ['category', 'rows', label, 'accuracy'],
    ['category', 'rank', churn, 'roc_auc'],
    ['category', 'rank', label, 'roc_auc_ovr'],
    ['category', 'chances', label, 'neg_log_loss'],
    ['number', 'absolute', y, 'neg_mean_absolute_error'],
    ['number', 'percent', y, 'neg_mean_absolute_percentage_error'],
  ];
  for (const [task, id, col, scoring] of cases) {
    const cost = resolveCost(task, { id }, col);
    const text = script(task, cost);
    assert.match(text, new RegExp(`^SCORING = "${scoring}"$`, 'm'), `${task} ${id}`);
    assert.match(text, /It scores by .*, from what you said a wrong answer costs/, 'the docstring says so');
    if (!cost.higher) assert.match(text, /shown negative: closer to zero is better/);
  }
});

test('a cost set for the other kind of answer is not used', () => {
  const cost = resolveCost('number', { id: 'absolute' }, column('numeric.csv', 'y'));
  assert.match(script('category', cost), /^SCORING = "balanced_accuracy"$/m);
});

test('the page says what the script scores by, and that the order above is the benchmark\'s', () => {
  const churn = column('imbalanced.csv', 'churn');
  assert.equal(costSentence(resolveCost('category', null, churn)),
    'It scores by balanced accuracy, the benchmark\'s own score.');
  const miss = costSentence(resolveCost('category', { id: 'miss', ratio: 20 }, churn));
  assert.match(miss, /because you said a missed "yes" costs 20 false alarms/);
  assert.match(miss, /The order above is the benchmark's, which scored by balanced accuracy/);
  assert.match(costSentence(resolveCost('number', { id: 'absolute' }, column('numeric.csv', 'y'))),
    /because you said every unit of error costs the same\. .* scored by R squared/);
  assert.equal(costSentence(null), '');
});

test('a percentage score on a target with zeros is warned about, and so is accuracy on a rare class', () => {
  assert.match(costWarning('number', 'percent', { name: 'units', zeroRate: 0.2 }), /20% of the values of units are zero/);
  assert.equal(costWarning('number', 'percent', { name: 'units', zeroRate: 0 }), '');
  assert.match(costWarning('category', 'rows', column('imbalanced.csv', 'churn')), /Always answering "no" already gets 91%/);
  assert.equal(costWarning('category', 'rows', column('balanced.csv', 'label')), '');
});
