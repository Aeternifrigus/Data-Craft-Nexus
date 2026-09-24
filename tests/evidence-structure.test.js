// The evidence tab is in numbered parts with subsections a contents list can
// jump to, and every part that has mathematics behind it shows it.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { anomalyDecisionSentence, confirmationSentence, evidenceParts } from '../site/js/evidence.js';
import { loadTaxonomyFromDisk } from './helpers.js';

const T = loadTaxonomyFromDisk();

test('the evidence is in numbered parts, each with subsections and a mathematical part where there is one', () => {
  const parts = evidenceParts(T);
  assert.deepEqual(parts.map(p => p.id), ['tables', 'forecast', 'anomaly', 'drift', 'checks', 'reference']);
  for (const p of parts) {
    assert.ok(p.subs.length >= 2, p.id);
    for (const s of p.subs) assert.match(s.anchor, /^ev-[a-z]+-[a-z]+$/);
    if (p.id !== 'reference') assert.ok(p.subs.some(s => s.id === 'math' && /ev-math-item/.test(s.html)), `${p.id} has its derivations`);
  }
  const anchors = parts.flatMap(p => p.subs.map(s => s.anchor));
  assert.equal(new Set(anchors).size, anchors.length, 'every subsection can be jumped to');
});

test('every derivation the evidence shows is in the reference, with a formula and an argument', () => {
  const html = evidenceParts(T).flatMap(p => p.subs.map(s => s.html)).join('');
  const codes = [...html.matchAll(/data-math="([A-Z]+\d+)"/g)].map(m => m[1]);
  assert.ok(codes.length >= 20, `${codes.length} derivations shown`);
  for (const code of codes) {
    assert.ok(T.MATH[code]?.f && T.MATH[code]?.mech, code);
  }
});

test('each verdict is stated the way the rule decided it', () => {
  assert.match(anomalyDecisionSentence(T.ANOMALY), T.ANOMALY.decision.replaced ? /beat one order/ : /did not beat one order/);
  const c = T.FORECAST.confirmation;
  assert.match(confirmationSentence(T.FORECAST), new RegExp(`${c.series} intermittent series from ${c.units} collections`));
  assert.match(confirmationSentence(T.FORECAST), c.decision.replaced ? /replaces the fixed one/ : /the fixed order stays/);
});

