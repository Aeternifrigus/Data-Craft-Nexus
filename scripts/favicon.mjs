// The favicon is inlined into site/index.html as a data URI rather than
// linked as a file, because the built page in dist/ has to work when opened
// straight from disk: a <link href="favicon.svg"> next to a file:// page
// resolves to wherever the page was saved, and usually to nothing.
//
// site/favicon.svg stays the source of truth, and tests/favicon.test.js
// fails if the copy inlined in index.html has drifted from it.
//
//   node scripts/favicon.mjs     prints the tag to paste into site/index.html

import fs from 'node:fs';
import { pathToFileURL } from 'node:url';

const root = new URL('../', import.meta.url);

// Percent-encode only what would end an attribute or start a comment. Leaving
// the rest readable keeps the tag diffable, which base64 would not.
export function faviconDataUri() {
  const svg = fs.readFileSync(new URL('site/favicon.svg', root), 'utf8').trim();
  const encoded = svg
    .replace(/%/g, '%25')
    .replace(/#/g, '%23')
    .replace(/"/g, '%22')
    .replace(/</g, '%3C')
    .replace(/>/g, '%3E')
    .replace(/\n/g, '');
  return `data:image/svg+xml,${encoded}`;
}

export function faviconTag() {
  return `<link rel="icon" href="${faviconDataUri()}">`;
}

if (process.argv[1] && pathToFileURL(process.argv[1]).href === import.meta.url) {
  console.log(faviconTag());
}
