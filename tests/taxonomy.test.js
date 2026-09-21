// Every cross-reference in the taxonomy must point at something that exists.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { loadTaxonomyFromDisk } from './helpers.js';

const T = loadTaxonomyFromDisk();
const taskIds = new Set(T.TASKS.map(t => t.id));

function missing(refs, exists) {
  return refs.filter(r => !exists(r));
}

test('codes are unique within each collection', () => {
  for (const [name, list] of [['models', T.MODELS], ['drift checkers', T.DRIFTS], ['pipelines', T.PIPELINES]]) {
    const seen = new Set();
    const dupes = list.map(x => x.c).filter(c => seen.has(c) || !seen.add(c));
    assert.deepEqual(dupes, [], `duplicate ${name} codes`);
  }
});

test('every data code a model or drift checker uses is defined', () => {
  const bad = [
    ...T.MODELS.flatMap(m => missing(m.data, d => d in T.CODES).map(d => `${m.c} -> ${d}`)),
    ...T.DRIFTS.flatMap(d => missing(d.fits, f => f in T.CODES).map(f => `${d.c} -> ${f}`)),
  ];
  assert.deepEqual(bad, []);
});

test('every math reference is defined', () => {
  const bad = [...T.MODELS, ...T.DRIFTS]
    .flatMap(x => missing(x.math, m => m in T.MATH).map(m => `${x.c} -> ${m}`));
  assert.deepEqual(bad, []);
});

test('every math formula belongs to a known domain', () => {
  const bad = Object.entries(T.MATH).filter(([, m]) => !(m.domain in T.MATH_DOMAINS)).map(([c]) => c);
  assert.deepEqual(bad, []);
});

test('every model has a known domain, paradigm and task', () => {
  const bad = T.MODELS.flatMap(m => [
    ...(m.dom in T.MODEL_DOMAINS ? [] : [`${m.c} domain ${m.dom}`]),
    ...(['SL', 'USL', 'SSL', 'RL'].includes(m.p) ? [] : [`${m.c} paradigm ${m.p}`]),
    ...missing(m.task, t => taskIds.has(t)).map(t => `${m.c} task ${t}`),
  ]);
  assert.deepEqual(bad, []);
});

// A pipeline can also list whole pipelines as steps (PL-T10 retraining re-runs PL-T1, PL-T3...).
test('every pipeline step is a defined stage or pipeline', () => {
  const pipelineCodes = new Set(T.PIPELINES.map(p => p.c));
  const bad = T.PIPELINES.flatMap(p =>
    missing(p.stages, s => s in T.STAGES || (pipelineCodes.has(s) && s !== p.c)).map(s => `${p.c} -> ${s}`));
  assert.deepEqual(bad, []);
});
