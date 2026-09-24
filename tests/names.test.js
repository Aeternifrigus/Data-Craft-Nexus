// Plain names first: every code the page shows has a name to lead with, and
// the name still opens the code's definition.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { codeTag, nameChip, nameOf, plainFlowchart, plainReason } from '../site/js/names.js';
import { loadTaxonomyFromDisk } from './helpers.js';

const T = loadTaxonomyFromDisk();

test('every code a card or the drawer shows has a plain name', () => {
  const pipelineCodes = new Set(T.PIPELINES.map(p => p.c));
  for (const m of T.MODELS) {
    for (const d of m.data) assert.ok(nameOf(T, 'code', d), `${m.c}: ${d} has no name`);
    for (const x of m.math) assert.ok(nameOf(T, 'math', x), `${m.c}: ${x} has no name`);
    assert.equal(nameOf(T, 'model', m.c), m.n);
  }
  for (const d of T.DRIFTS) for (const x of d.math) assert.ok(nameOf(T, 'math', x), `${d.c}: ${x} has no name`);
  for (const p of T.PIPELINES) {
    for (const s of p.stages) {
      assert.ok(nameOf(T, pipelineCodes.has(s) ? 'pipeline' : 'stage', s), `${p.c}: ${s} has no name`);
    }
  }
});

test('a chip reads as its name, and keeps the code for hover and for the drawer', () => {
  const chip = nameChip(T, 'code', 'A11', { cls: ' hit', lower: true });
  assert.equal(chip, '<span class="chip hit" data-code="A11" title="A11: click for the definition">labeled</span>');
  assert.match(nameChip(T, 'math', Object.keys(T.MATH)[0]), /^<span class="chip" data-math="[^"]+" title="[^"]+">[^<]+<\/span>$/);
  // A code the taxonomy does not know shows as itself, escaped.
  assert.equal(nameChip(T, 'stage', '<x>'), '<span class="chip" data-stage="&lt;x&gt;" title="&lt;x&gt;: click for the definition">&lt;x&gt;</span>');
  assert.equal(codeTag('model', 'TR2'), '<span class="rec-code" data-model="TR2" title="TR2: click for the definition">TR2</span>');
});

test('flowcharts name their boxes without the stage codes', () => {
  for (const p of T.PIPELINES.filter(x => x.flowchart)) {
    const plain = plainFlowchart(p.flowchart);
    assert.doesNotMatch(plain, /PL-S\d/, p.c);
    // Only the codes go: every box keeps a label, and the arrows are untouched.
    assert.doesNotMatch(plain, /\[""\]|\{""\}|\[\]|\{\}/, p.c);
    assert.equal(plain.split('\n').length, p.flowchart.split('\n').length, p.c);
  }
  assert.equal(plainFlowchart('A["PL-S9 Train"] --> B{"PL-S12 Gate"}'), 'A["Train"] --> B{"Gate"}');
});

test('a reason written in axis codes reads in words', () => {
  assert.equal(plainReason(T, 'needs A22 data (yours is A21)'), 'needs sequential data (yours is independent)');
  assert.equal(plainReason(T, 'needs A99 data'), 'needs A99 data', 'an unknown code stays');
  assert.equal(plainReason(T, 'needs an environment to interact with, not a table'),
    'needs an environment to interact with, not a table');
});
