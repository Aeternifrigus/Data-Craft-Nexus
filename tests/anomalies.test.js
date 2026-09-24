// "Unusual records" is ordered by the anomaly benchmark (bench/dcn/anomaly.py
// and anomaly_learn.py): per width of table only when that passed the rule
// fixed before the run, one order for every table otherwise. The published
// shares are recomputed from the committed run.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { rankModels } from '../site/js/recommend.js';
import { parseCSV } from '../site/js/csv.js';
import { anomalyKind, anomalyNote, anomalyOrder, detectorSentence } from '../site/js/anomalies.js';
import { loadTaxonomyFromDisk } from './helpers.js';

const T = loadTaxonomyFromDisk();
const A = T.ANOMALY;
const sig = (features) => ({ codes: ['A12', 'A21', 'A31', 'A41', 'A53', 'A61'], flags: [], rows: 1000, features, ops: {} });
const fake = (chosen) => ({
  ...T,
  ANOMALY: { ...A, chosen,
    prior: { auc: { TR4: 0.7, SV3: 0.4, IB3: 0.5, IB4: 0.65, PR6: 0.6, CL2: 0.2 } },
    kind_prior: { auc: { high: { TR4: 0.5, SV3: 0.4, IB3: 0.3, IB4: 0.35, PR6: 0.8, CL2: 0.2 } } } },
});

test('the width of a table is read the way the benchmark cut it', () => {
  assert.equal(anomalyKind(10), 'low');
  assert.equal(anomalyKind(11), 'mid');
  assert.equal(anomalyKind(50), 'mid');
  assert.equal(anomalyKind(51), 'high');
  assert.equal(anomalyKind(null), null);
});

test('the width decides the order only when it passed the rule', () => {
  assert.equal(rankModels(fake('kind'), sig(120), 'anomaly').items[0].c, 'PR6');
  assert.equal(rankModels(fake('fixed'), sig(120), 'anomaly').items[0].c, 'TR4');
  assert.equal(rankModels(fake('kind'), sig(5), 'anomaly').items[0].c, 'TR4', 'no order kept for that width: the fixed one');
  assert.equal(anomalyOrder(fake('fixed'), sig(120)).by, 'fixed');
});

test('every detector the benchmark ran is ordered by it, and the rest come after', () => {
  const out = rankModels(T, sig(20), 'anomaly', 99);
  assert.equal(out.rankedBy, 'evidence');
  const measured = out.items.map(m => m.evidenceScore != null);
  assert.deepEqual(measured, [...measured].sort((a, b) => b - a));
  assert.deepEqual(out.items.filter(m => m.evidenceScore != null).map(m => m.c).sort(), ['CL2', 'IB3', 'IB4', 'PR6', 'SV3', 'TR4']);
  assert.match(anomalyNote(T, out.anomaly), /benchmark tables with known anomalies/);
});

test('the published shares are the committed run', () => {
  const { head, body, truncated } = parseCSV(fs.readFileSync(new URL('../bench/results/anomaly.csv', import.meta.url), 'utf8'));
  assert.ok(!truncated);
  const rows = body.map(cells => Object.fromEntries(head.map((c, i) => [c, cells[i]]))).filter(r => r.status === 'ok');
  const tables = new Map();
  for (const r of rows) {
    if (!tables.has(r.dataset)) tables.set(r.dataset, { kind: r.kind, auc: {} });
    tables.get(r.dataset).auc[r.detector] = Number(r.auc);
  }
  const order = ['TR4', 'SV3', 'IB3', 'IB4', 'PR6', 'CL2'];
  for (const [kind, entry] of Object.entries(A.per_kind)) {
    const here = [...tables.values()].filter(t => t.kind === kind && Object.keys(t.auc).length >= 2);
    assert.equal(here.length, entry.datasets, kind);
    for (const [code, share] of Object.entries(entry.best_share)) {
      const wins = here.filter(t => order.filter(c => c in t.auc).reduce((a, c) => (t.auc[c] > t.auc[a] ? c : a)) === code).length;
      assert.ok(Math.abs(wins / here.length - share) < 0.0006, `${kind} ${code}`);
    }
  }
});

test('the sentences say what was measured', () => {
  const kind = Object.keys(A.per_kind)[0];
  assert.match(detectorSentence(T, kind, 'TR4'), /benchmark tables that were .* best of the six detectors on \d+% .* median ROC AUC of \d\.\d\d/);
});

test('robust covariance is ruled out when a table has too few rows for its columns', () => {
  const narrow = { ...sig(60), rows: 100 };
  const out = rankModels(T, narrow, 'anomaly', 99);
  assert.ok(!out.items.some(m => m.c === 'PR6'));
  assert.match(out.ruledOut.find(m => m.c === 'PR6').why, /twice as many rows as columns/);
  assert.ok(rankModels(T, { ...sig(60), rows: 1000 }, 'anomaly', 99).items.some(m => m.c === 'PR6'));
});
