// The built page must be self-contained and carry exactly the taxonomy in site/.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { COPIED, buildPage, buildStamp } from '../scripts/build.mjs';
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

test('the share image a link preview asks for is published beside the page', () => {
  const m = page.match(/<meta property="og:image" content="https:\/\/aeternifrigus\.github\.io\/Data-Craft-Nexus\/([^"]+)">/);
  assert.ok(m, 'og:image is missing or points somewhere other than the live site');
  assert.ok(COPIED.includes(m[1]), `the build does not copy ${m[1]} into dist/`);
  const source = fs.readFileSync(new URL(`site/${m[1]}`, root));
  const built = fs.readFileSync(new URL(`dist/${m[1]}`, root));
  assert.ok(source.equals(built), `dist/${m[1]} is stale: run \`npm run build\``);
  // PNG header: width and height, which the page also declares.
  assert.equal(source.readUInt32BE(16), Number(page.match(/og:image:width" content="(\d+)"/)[1]));
  assert.equal(source.readUInt32BE(20), Number(page.match(/og:image:height" content="(\d+)"/)[1]));
});

test('the deployed page names its commit out of sight, for the deploy check', async () => {
  const stamp = buildStamp('fe43b93230cfacd748c351169c0c39e41df0b20b', new Date('2026-09-24T10:00:00Z'));
  assert.deepEqual(stamp, { sha: 'fe43b93', date: '2026-09-24' });
  const stamped = await buildPage({ stamp });
  assert.match(stamped, /<meta name="dcn-build" content="fe43b93 2026-09-24">/);
  assert.doesNotMatch(stamped, /Built from commit/, 'nothing about the build shows on the page');
  assert.doesNotMatch(page, /<meta name="dcn-build"/, 'the committed page carries no stamp');
  assert.equal(buildStamp(undefined), null);
  assert.throws(() => buildStamp('main"><script>'), /commit hash/);
});
