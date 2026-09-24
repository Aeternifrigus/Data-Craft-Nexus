// A future value is ordered by the forecasting benchmark (bench/dcn/forecast.py
// and forecast_learn.py): per kind of series only when that passed the rule
// fixed before the run, one order for every series otherwise. The published
// shares are recomputed from the committed run.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { rankModels } from '../site/js/recommend.js';
import { parseCSV } from '../site/js/csv.js';
import { describeSeries, forecastMetric, forecastNote, forecastOrder, forecasterSentence } from '../site/js/forecasting.js';
import { forecastDecisionSentence } from '../site/js/evidence.js';
import { loadTaxonomyFromDisk } from './helpers.js';

const T = loadTaxonomyFromDisk();
const F = T.FORECAST;
const sig = (series) => ({ codes: ['A11', 'A22', 'A31', 'A41', 'A53', 'A61'], flags: [], rows: 200, features: 1, ops: {}, series });
const seasonal = (n = 120) => Array.from({ length: n }, (_, t) => 100 + 10 * Math.sin((2 * Math.PI * t) / 12) + (t % 5) * 0.1);
const demand = (n = 60) => Array.from({ length: n }, (_, t) => (t % 4 === 0 ? 3 + (t % 3) : 0));
const series = (values, period) => describeSeries({ series: { values, period, step: 'monthly' }, season: period }, 'r2');

// A stand-in benchmark where the kinds disagree, to check which one the page reads.
const fake = (chosen) => ({
  ...T,
  FORECAST: {
    ...F, chosen,
    prior: { r2: { TSM1: 0.4, TSM2: 0.7, TSM4: 0.5, TSM5: 0.45 }, mae: { TSM1: 0.4, TSM2: 0.5, TSM4: 0.7, TSM5: 0.6 } },
    kind_prior: { r2: { intermittent: { TSM1: 0.3, TSM2: 0.4, TSM4: 0.8, TSM5: 0.6 } },
      mae: { intermittent: { TSM1: 0.3, TSM2: 0.4, TSM4: 0.6, TSM5: 0.8 } } },
  },
});

test('the page measures the series it forecasts the way the benchmark did', () => {
  assert.equal(series(seasonal(), 12).kind, 'seasonal');
  assert.equal(series(demand(), 12).kind, 'intermittent');
  assert.equal(forecastMetric({ scoring: 'neg_mean_absolute_error' }), 'mae');
  assert.equal(forecastMetric(null), 'r2');
});

test('the kind decides the order only when it passed the rule', () => {
  const s = series(demand(), 12);
  const byKind = rankModels(fake('kind'), sig(s), 'forecast');
  assert.equal(byKind.items[0].c, 'TSM4', 'intermittent demand: Croston first, when the kind earned it');
  assert.equal(byKind.forecast.by, 'kind');
  const fixed = rankModels(fake('fixed'), sig(s), 'forecast');
  assert.equal(fixed.items[0].c, 'TSM2', 'the same series gets the fixed order when the kind did not earn it');
  assert.equal(fixed.forecast.by, 'fixed');
  assert.equal(fixed.rankedBy, 'evidence');
  assert.match(forecastNote(fake('fixed'), fixed.forecast), /did not beat one order for every series/);
  assert.match(forecastNote(fake('kind'), byKind.forecast), /kept for series that are intermittent, with steady sizes, like yours/);
});

test('the score the user cares about picks which order is read', () => {
  const s = { ...series(demand(), 12), metric: 'mae' };
  assert.equal(rankModels(fake('kind'), sig(s), 'forecast').items[0].c, 'TSM5');
  assert.equal(forecastOrder(fake('fixed'), s).prior.TSM4, 0.7);
});

test('forecasters the benchmark did not run sit below the ones it did', () => {
  const out = rankModels(T, sig(series(seasonal(), 12)), 'forecast', 99);
  const measured = out.items.map(m => m.evidenceScore != null);
  assert.deepEqual(measured, [...measured].sort((a, b) => b - a), 'no unmeasured forecaster above a measured one');
  assert.deepEqual(out.items.filter(m => m.evidenceScore != null).map(m => m.c).sort(), ['TSM1', 'TSM2', 'TSM4', 'TSM5']);
});

test('demand forecasters are ruled out for a series that goes negative', () => {
  const s = series(seasonal().map(v => v - 100), 12);
  const out = rankModels(T, sig(s), 'forecast', 99);
  assert.ok(!out.items.some(m => ['TSM4', 'TSM5'].includes(m.c)));
  assert.match(out.ruledOut.find(m => m.c === 'TSM4').why, /never negative/);
});

// Read with the page's own CSV reader: some series names hold quoted commas.
function readRuns() {
  const { head, body, truncated } = parseCSV(fs.readFileSync(new URL('../bench/results/forecast.csv', import.meta.url), 'utf8'));
  assert.ok(!truncated, 'the run fits in what the reader reads');
  return body.map(cells => Object.fromEntries(head.map((c, i) => [c, cells[i]])));
}

test('the published shares are the committed run', () => {
  const runs = readRuns().filter(r => r.status === 'ok');
  const bySeries = new Map();
  for (const r of runs) {
    const key = `${r.dataset}|${r.series}`;
    if (!bySeries.has(key)) bySeries.set(key, { cell: r.cell, scores: {} });
    bySeries.get(key).scores[r.method] = Number(r.r2);
  }
  for (const [kind, entry] of Object.entries(F.per_kind)) {
    const rows = [...bySeries.values()].filter(s => s.cell === kind)
      .map(s => Object.fromEntries(['TSM1', 'TSM2', 'TSM4', 'TSM5'].filter(c => c in s.scores).map(c => [c, s.scores[c]])))
      .filter(s => Object.keys(s).length >= 2);
    assert.equal(rows.length, entry.series, kind);
    for (const [code, share] of Object.entries(entry.best_share)) {
      const wins = rows.filter(s => Object.entries(s).reduce((a, b) => (b[1] > a[1] ? b : a))[0] === code).length;
      assert.ok(Math.abs(wins / rows.length - share) < 0.0006, `${kind} ${code}: ${wins / rows.length} vs ${share}`);
    }
  }
});

test('every sentence says what was measured, and where', () => {
  const kind = Object.keys(F.per_kind)[0];
  const code = Object.keys(F.per_kind[kind].best_share)[0];
  assert.match(forecasterSentence(T, kind, code), /benchmark series that were .* it was the best of the four forecasters on \d+%/);
  assert.equal(forecasterSentence(T, 'no-such-kind', code), '');
  assert.match(forecastDecisionSentence(F), F.decision.replaced ? /beat one order/ : /did not beat one order/);
});

test('the other columns do not rule a forecaster out: it reads only the past of the target', () => {
  const categorical = { ...sig(series(seasonal(), 12)), codes: ['A11', 'A22', 'A32', 'A41', 'A53', 'A61'] };
  const out = rankModels(T, categorical, 'forecast', 99);
  assert.ok(out.items.some(m => m.c === 'TSM2'), 'a text column beside the dates does not rule smoothing out');
  const unmeasured = { ...categorical, series: undefined };
  assert.ok(!rankModels(T, unmeasured, 'forecast', 99).items.some(m => m.c === 'TSM2'),
    'without a measured numeric target the old rule stands');
});
