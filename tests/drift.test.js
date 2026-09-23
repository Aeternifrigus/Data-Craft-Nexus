// Drift checkers are ordered by what the drift benchmark measured
// (bench/dcn/drift.py), and the numbers on the page are recomputed here from
// the committed run, so the page cannot disagree with it.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { driftCodes, driftMeasure, driftUnmeasured, rankDrifts } from '../site/js/recommend.js';
import { DRIFT_SCENARIOS, driftBlindSpots, driftSentence } from '../site/js/evidence.js';
import { loadTaxonomyFromDisk } from './helpers.js';

const T = loadTaxonomyFromDisk();
const DRIFT = T.EVIDENCE.drift;
const sig = (codes, ops = {}) => ({ codes, flags: [], rows: 500, features: 8, ops });

function readRuns() {
  const [head, ...lines] = fs.readFileSync(new URL('../bench/results/drift.csv', import.meta.url), 'utf8')
    .trim().split('\n');
  const cols = head.split(',');
  return lines.map(line => Object.fromEntries(line.split(',').map((v, i) => [cols[i], v])));
}

test('every checker in the taxonomy was measured, or says why not', () => {
  for (const d of T.DRIFTS) {
    assert.ok(driftMeasure(T, d.c) || driftUnmeasured(T, d.c), `${d.c} has neither a measurement nor a reason`);
    assert.ok(!(driftMeasure(T, d.c) && driftUnmeasured(T, d.c)), `${d.c} is both measured and not`);
  }
});

test('the published rates are the committed run, each dataset counted once', () => {
  const runs = readRuns();
  const scenarios = ['none', ...Object.keys(DRIFT_SCENARIOS)];
  for (const [code, published] of Object.entries(DRIFT.checkers)) {
    // Rate per dataset over its repetitions, then the mean over datasets.
    const perDataset = new Map();
    for (const r of runs.filter(r => r.checker === code)) {
      const key = r.dataset;
      if (!perDataset.has(key)) perDataset.set(key, Object.fromEntries(scenarios.map(s => [s, []])));
      perDataset.get(key)[r.scenario].push(Number(r.fired));
    }
    const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
    const rate = (s) => mean([...perDataset.values()].map(d => mean(d[s])));
    assert.equal(perDataset.size, DRIFT.datasets.length, code);
    assert.ok(Math.abs(rate('none') - published.false_alarm) < 5e-4, `${code} false alarms`);
    for (const s of Object.keys(DRIFT_SCENARIOS)) {
      assert.ok(Math.abs(rate(s) - published.caught[s]) < 5e-4, `${code} ${s}`);
    }
    const net = mean([...perDataset.values()].map(d =>
      mean(Object.keys(DRIFT_SCENARIOS).map(s => mean(d[s]))) - mean(d.none)));
    assert.ok(Math.abs(net - published.net) < 5e-4, `${code} net`);
  }
});

test('checkers are shown best first by net score, the unmeasured after them', () => {
  const r = rankDrifts(T, sig(['A11', 'A21', 'A31', 'A41', 'A51', 'A61']), 99);
  assert.equal(r.rankedBy, 'evidence');
  const nets = r.items.map(d => d.evidenceScore);
  const measured = nets.filter(n => n != null);
  assert.deepEqual(measured, [...measured].sort((a, b) => b - a));
  const firstUnmeasured = nets.indexOf(null);
  if (firstUnmeasured >= 0) assert.ok(nets.slice(firstUnmeasured).every(n => n == null));
  assert.equal(r.items[0].c, Object.keys(DRIFT.checkers)[0], 'the best measured checker that fits comes first');
});

test('a mixed table is offered tests for its numeric and categorical columns', () => {
  const mixed = rankDrifts(T, sig(['A11', 'A21', 'A38', 'A41', 'A51', 'A61'], { mode: 'batch', labels: 'none' }), 99);
  const codes = mixed.items.map(d => d.c);
  for (const c of ['DR-M1', 'DR-M2', 'DR-M3']) assert.ok(codes.includes(c), `${c} missing for a mixed table`);
  assert.deepEqual(driftCodes(['A31']), ['A31']);
  assert.deepEqual(driftCodes(['A38']).sort(), ['A31', 'A32', 'A38']);
});

test('error drift needs labels, like every checker that watches errors', () => {
  const r = rankDrifts(T, sig(['A11', 'A21', 'A31', 'A41', 'A51', 'A61'], { mode: 'batch', labels: 'none' }), 99);
  assert.ok(!r.items.some(d => d.c === 'DR-M8'));
  assert.match(r.ruledOut.find(d => d.c === 'DR-M8').why, /labels/);
});

test('a card says what was caught, and what was not seen at all', () => {
  const m = { false_alarm: 0.02, caught_mean: 0.4,
    caught: { shift: 0.9, scale: 0.6, correlation: 0.05, selection: 1, label_shift: 0.3, concept: 0.02 } };
  assert.deepEqual(driftBlindSpots(m), ['correlation', 'concept']);
  assert.equal(driftSentence(m, 18),
    'On 18 datasets it caught 40% of the injected drift, and fired on 2% of the windows where nothing had '
    + 'changed. It caught a broken correlation or labels that mean something else no more often than it fired on nothing.');
  const deaf = { false_alarm: 0, caught_mean: 0, caught: Object.fromEntries(Object.keys(DRIFT_SCENARIOS).map(s => [s, 0])) };
  assert.match(driftSentence(deaf, 18), /caught no kind of drift more often than it fired on nothing/);
});
