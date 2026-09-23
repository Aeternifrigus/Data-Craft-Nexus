// Builds dist/index.html: one self-contained page with the styles, the
// bundled code, the taxonomy and the sample table inside it. It works when
// opened straight from disk, which the ES modules in site/ can't (browsers
// block module scripts on file:// addresses).
//
// Plotly, Mermaid and the fonts still come from their CDNs, as before.
//
//   npm run build            writes dist/index.html
//   npm run build -- --check fails if dist/index.html is out of date
//
// With DCN_BUILD set to a commit hash (the Pages workflow sets it), the page
// names that commit in a meta tag and in the footer, so anyone can see which
// version is live. The committed dist/ is built without it.

import fs from 'node:fs';
import { pathToFileURL } from 'node:url';
import { build } from 'esbuild';
import { TAXONOMY_FILES } from '../site/js/taxonomy.js';

const root = new URL('../', import.meta.url);
const read = (p) => fs.readFileSync(new URL(p, root), 'utf8');

// "abc1234 2026-09-24" from a commit hash, or null. Anything else is refused:
// it goes into the page.
export function buildStamp(sha, date = new Date()) {
  if (!sha) return null;
  if (!/^[0-9a-f]{7,40}$/.test(sha)) throw new Error(`build: DCN_BUILD must be a commit hash, got ${sha}`);
  return { sha: sha.slice(0, 7), date: date.toISOString().slice(0, 10) };
}

export async function buildPage({ stamp = null } = {}) {
  const bundle = await build({
    entryPoints: [new URL('site/js/app.js', root).pathname],
    bundle: true,
    format: 'iife',
    target: ['es2019', 'safari13', 'chrome80', 'firefox78'],
    write: false,
    legalComments: 'none',
  });
  const js = bundle.outputFiles[0].text;

  const data = {
    taxonomy: Object.fromEntries(TAXONOMY_FILES.map(n => [n, JSON.parse(read(`site/taxonomy/${n}.json`))])),
    sample: read('site/sample.csv'),
  };
  // < keeps "</script>" and "<!--" inside strings from closing the tag early.
  const json = JSON.stringify(data).replace(/</g, '\\u003c');

  const css = read('site/css/style.css');
  let html = read('site/index.html');

  const replace = (from, to) => {
    if (!html.includes(from)) throw new Error(`build: site/index.html no longer contains ${from}`);
    html = html.replace(from, () => to);
  };
  replace('<link rel="stylesheet" href="css/style.css">', `<style>\n${css}</style>`);
  html = html.replace(/<!-- dev-only -->[\s\S]*?<!-- \/dev-only -->\n?/, '');
  replace('<!-- build-stamp -->', stamp
    ? `<div class="build">Built from commit ${stamp.sha} on ${stamp.date}.</div>` : '');
  if (stamp) replace('<meta name="theme-color"', `<meta name="dcn-build" content="${stamp.sha} ${stamp.date}">\n<meta name="theme-color"`);
  replace('<script type="module" src="js/app.js"></script>',
    `<script id="dcn-data" type="application/json">${json}</script>\n` +
    `<script>\n${js.replace(/<\/script/gi, '<\\/script')}</script>`);

  // The note goes after the doctype: nothing should come before it.
  return html.replace('<!DOCTYPE html>\n', '<!DOCTYPE html>\n<!-- Built from site/ by scripts/build.mjs. Edit the files in site/, then run npm run build. -->\n');
}

const isMain = process.argv[1] && pathToFileURL(process.argv[1]).href === import.meta.url;
if (isMain) {
  const page = await buildPage({ stamp: buildStamp(process.env.DCN_BUILD) });
  const out = new URL('dist/index.html', root);
  if (process.argv.includes('--check')) {
    const current = fs.existsSync(out) ? fs.readFileSync(out, 'utf8') : '';
    if (current !== page) {
      console.error('dist/index.html is out of date. Run `npm run build` and commit the result.');
      process.exit(1);
    }
    console.log('dist/index.html is up to date.');
  } else {
    fs.mkdirSync(new URL('dist/', root), { recursive: true });
    fs.writeFileSync(out, page);
    console.log(`wrote dist/index.html (${(page.length / 1024).toFixed(0)} KB)`);
  }
}
