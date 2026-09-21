import { test } from 'node:test';
import assert from 'node:assert/strict';
import { esc } from '../site/js/html.js';

test('escapes the characters that matter in HTML text and attributes', () => {
  assert.equal(esc(`<img src=x onerror="alert('1')">&`), '&lt;img src=x onerror=&quot;alert(&#39;1&#39;)&quot;&gt;&amp;');
});

test('formulas with comparison signs survive', () => {
  assert.equal(esc('P(t≤T<t+Δt|T≥t)'), 'P(t≤T&lt;t+Δt|T≥t)');
});

test('null and undefined become empty text, numbers become strings', () => {
  assert.equal(esc(null), '');
  assert.equal(esc(undefined), '');
  assert.equal(esc(42), '42');
});
