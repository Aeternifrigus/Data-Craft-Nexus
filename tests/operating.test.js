// Drift checkers and pipelines are chosen by how the thing will run, not only
// by what the data looks like. A detector that watches an error stream is
// useless when labels never arrive; one that compares two batches has nothing
// to compare in a stream; a RAG pipeline is noise when you are building the
// ingest.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { PIPELINE_STAGES, operatingConflict, pipelineConflict, rankDrifts, rankPipelines } from '../site/js/recommend.js';
import { loadTaxonomyFromDisk } from './helpers.js';

const T = loadTaxonomyFromDisk();
const sig = (ops) => ({
  codes: ['A11', 'A21', 'A38', 'A41', 'A52', 'A62'], flags: [], rows: 500, features: 8, ops,
});

const codes = (r) => r.items.map(x => x.c);
const domainsOf = (r) => [...new Set(r.items.map(x => x.domain ?? x.p))];

test('every drift checker says what it needs to run', () => {
  for (const checker of T.DRIFTS) {
    assert.ok(checker.needs, `${checker.c} has no operating requirements`);
    assert.ok(checker.needs.modes?.length, `${checker.c} does not say which modes it supports`);
    assert.ok(checker.needs.labels?.length, `${checker.c} does not say which labels it needs`);
    assert.ok(checker.needs.note?.length > 10, `${checker.c} needs a reason worth showing`);
  }
});

test('without labels, nothing that watches the error rate is offered', () => {
  const r = rankDrifts(T, sig({ mode: 'streaming', labels: 'none' }), 99);
  assert.ok(codes(r).length > 0, 'distribution tests still work without labels');
  assert.ok(!domainsOf(r).includes('DR-C'), 'error-rate detectors need labels');
  assert.ok(r.ruledOut.length > 0);
  assert.match(r.ruledOut.map(d => d.why).join(' '), /error stream|labels/i);
});

test('a streaming job is not offered batch-only detectors, and the reverse', () => {
  const streaming = rankDrifts(T, sig({ mode: 'streaming', labels: 'delayed' }), 99);
  assert.ok(!codes(streaming).includes('DR-CV1'), 'adversarial validation needs two batches in hand');

  const batch = rankDrifts(T, sig({ mode: 'batch', labels: 'delayed' }), 99);
  assert.ok(!domainsOf(batch).includes('DR-C'), 'streaming detectors need a stream');
  assert.ok(codes(batch).includes('DR-CV1'));
});

test('unanswered questions filter nothing', () => {
  const all = rankDrifts(T, sig({}), 99);
  const none = rankDrifts(T, sig({ mode: null, labels: null }), 99);
  assert.equal(all.ruledOut.length, 0);
  assert.deepEqual(codes(all), codes(none));
});

test('the pipeline list follows the part of the work being built', () => {
  for (const [stage, { domains }] of Object.entries(PIPELINE_STAGES)) {
    const r = rankPipelines(T, sig({ stage }), 'category', 99);
    assert.ok(r.items.length > 0, `${stage} has no pipelines`);
    for (const p of r.items) {
      assert.ok(domains.includes(p.p), `${stage} should not offer ${p.c} (${p.p})`);
    }
  }
  const everything = rankPipelines(T, sig({ stage: null }), 'category', 99);
  assert.equal(everything.items.length, T.PIPELINES.length);
  assert.equal(everything.ruledOut.length, 0);
});

test('what was excluded comes with a reason, for every exclusion', () => {
  const drifts = rankDrifts(T, sig({ mode: 'batch', labels: 'none' }), 99);
  const pipelines = rankPipelines(T, sig({ stage: 'llm', mode: 'batch', labels: 'none' }), 'category', 99);
  for (const entry of [...drifts.ruledOut, ...pipelines.ruledOut]) {
    assert.ok(entry.why && entry.why.length > 10, `${entry.c} was dropped without a reason`);
    assert.ok(entry.n, `${entry.c} was dropped without a name`);
  }
});

test('operatingConflict only speaks when it has both a requirement and an answer', () => {
  const checker = T.DRIFTS.find(d => d.c === 'DR-C1');
  assert.equal(operatingConflict(checker, {}), null, 'no answers, no filtering');
  assert.equal(operatingConflict({ c: 'X' }, { labels: 'none' }), null, 'no requirements, no filtering');
  assert.ok(operatingConflict(checker, { labels: 'none' }));
  assert.equal(pipelineConflict(T.PIPELINES[0], {}), null);
});
