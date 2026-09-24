// "Save this reading": the page's findings as Markdown, without the file's rows.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseCSV } from '../site/js/csv.js';
import { profileData, signature } from '../site/js/profile.js';
import { leadRecommendation, rankDrifts, rankModels, rankPipelines } from '../site/js/recommend.js';
import { leadSentence } from '../site/js/evidence.js';
import { readingFileName, readingMarkdown } from '../site/js/report.js';
import { resolveCost } from '../site/js/costs.js';
import { loadTaxonomyFromDisk, readFixture } from './helpers.js';

const T = loadTaxonomyFromDisk();

function reading(file, target, task, answer = null) {
  const { head, body } = parseCSV(readFixture(file));
  const profile = profileData(head, body);
  const sig = signature(profile, { target, task, order: 'A21' });
  const lead = leadRecommendation(T, sig, task);
  return {
    body,
    text: readingMarkdown({
      T, sig, task, fileName: file, date: '2026-09-24', build: 'abc1234', lead,
      leadText: lead ? leadSentence(T, lead) : '',
      models: rankModels(T, sig, task), drifts: rankDrifts(T, sig), pipelines: rankPipelines(T, sig, task),
      cost: resolveCost(task, answer, profile.columns.find(c => c.name === target)),
      notes: { models: 'Ordered by evidence.' }, sentences: { EN4: 'On 94 classification datasets it was best 9 times.' },
    }),
  };
}

test('a reading says what was measured and what fits, with its provenance', () => {
  const { text } = reading('imbalanced.csv', 'churn', 'category');
  assert.match(text, /^# Data Craft Nexus reading: imbalanced\.csv\n/);
  assert.match(text, /Read on 2026-09-24 by the page built from commit abc1234\./);
  assert.match(text, /Target: `churn`\. Predicting: a category\./);
  assert.match(text, /## Signature\n\n`A11 A21 /);
  assert.match(text, /\| A11 \| Labeled: Every example has a known target\. \|/);
  assert.match(text, /\*\*Start here: Histogram Gradient Boosting, tuned\*\* \(`BASE-HGB-TUNED`\)\. Leave-one-dataset-out/);
  assert.match(text, /1\. \*\*LightGBM\*\* \(`EN4`\), evidence \d\.\d\d, \d of \d coordinates matched\.\n   Measured: On 94/);
  assert.match(text, /## Drift checkers\n/);
  assert.match(text, /## Pipelines\n/);
});

test('a reading carries no rows of the file', () => {
  const { text, body } = reading('imbalanced.csv', 'churn', 'category');
  const cities = new Set(body.map(r => r[1]));
  for (const city of cities) assert.ok(!text.includes(city), `${city} from the file leaked into the reading`);
  for (const row of body.slice(0, 20)) assert.ok(!text.includes(row.join(',')));
});

test('names and pipes in a file cannot break the document', () => {
  assert.equal(readingFileName('Q3 | shipments (final).csv'), 'dcn_reading_Q3_shipments_final_.md');
  assert.equal(readingFileName(''), 'dcn_reading_data.md');
  const { text } = reading('balanced.csv', 'label', 'category');
  for (const row of text.split('\n').filter(l => l.startsWith('| A'))) {
    assert.equal(row.split(/(?<!\\)\|/).length, 4, `a table row split into the wrong number of cells: ${row}`);
  }
});

test('a reading says what the script scores by, and why', () => {
  assert.match(reading('imbalanced.csv', 'churn', 'category').text, /It scores by balanced accuracy, the benchmark's own score\./);
  const { text } = reading('imbalanced.csv', 'churn', 'category', { id: 'miss', ratio: 10 });
  assert.match(text, /It scores by cost per row, because you said a missed "yes" costs 10 false alarms\./);
});
