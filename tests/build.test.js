// The built page must be self-contained and carry exactly the taxonomy in site/.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { buildPage } from '../scripts/build.mjs';
import { TAXONOMY_FILES } from '../site/js/taxonomy.js';

const page = await buildPage();
const root = new URL('../', import.meta.url);

test('the built page loads no local files and no modules', () => {
  assert.doesNotMatch(page, /type="module"/);
  assert.doesNotMatch(page, /href="css\//);
  assert.doesNotMatch(page, /src="js\//);
  assert.doesNotMatch(page, /dev-only/);
});

test('the embedded data matches site/taxonomy and site/sample.csv', () => {
  const m = page.match(/<script id="dcn-data" type="application\/json">([\s\S]*?)<\/script>/);
  assert.ok(m, 'embedded data tag is missing');
  const data = JSON.parse(m[1]);
  for (const name of TAXONOMY_FILES) {
    const onDisk = JSON.parse(fs.readFileSync(new URL(`site/taxonomy/${name}.json`, root), 'utf8'));
    assert.deepEqual(data.taxonomy[name], onDisk, `${name}.json differs`);
  }
  assert.equal(data.sample, fs.readFileSync(new URL('site/sample.csv', root), 'utf8'));
});

test('nothing inside the page closes a script tag early', () => {
  const scripts = [...page.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/g)];
  // 2 CDN scripts + the data + the code.
  assert.equal(scripts.length, 4);
});

test('dist/index.html is up to date with site/', () => {
  const committed = fs.readFileSync(new URL('dist/index.html', root), 'utf8');
  assert.ok(committed === page, 'dist/index.html is stale: run `npm run build`');
});
