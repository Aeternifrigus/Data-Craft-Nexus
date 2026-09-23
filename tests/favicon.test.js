// The favicon is inlined into the page, so it survives being opened from
// disk. site/favicon.svg is the source of truth; these tests fail if the
// inlined copy has drifted from it, or if it stops being self-contained.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { faviconDataUri, faviconTag } from '../scripts/favicon.mjs';
import { buildPage } from '../scripts/build.mjs';

const root = new URL('../', import.meta.url);
const html = fs.readFileSync(new URL('site/index.html', root), 'utf8');
const svg = fs.readFileSync(new URL('site/favicon.svg', root), 'utf8');

test('the icon in the page is the one in site/favicon.svg', () => {
  assert.ok(html.includes(faviconTag()), 'site/favicon.svg changed: run `node scripts/favicon.mjs` and paste the tag');
});

test('the icon carries no reference to a file the page may not have', () => {
  const uri = faviconDataUri();
  assert.match(uri, /^data:image\/svg\+xml,/);
  assert.doesNotMatch(uri, /[<>"]/, 'unencoded characters would end the attribute early');
  assert.doesNotMatch(svg, /href|xlink|<image|url\(/i, 'the icon must not load anything');
});

test('the built page carries the icon too', async () => {
  const page = await buildPage();
  assert.ok(page.includes(faviconTag()));
  assert.ok(page.includes('name="theme-color"'));
});

test('the icon is drawn on the site palette', () => {
  const css = fs.readFileSync(new URL('site/css/style.css', root), 'utf8');
  const colours = [...svg.matchAll(/#[0-9A-F]{6}/g)].map(m => m[0]);
  assert.ok(colours.length >= 6);
  for (const colour of new Set(colours)) {
    assert.ok(css.includes(colour), `${colour} is not one of the site's colours`);
  }
});
