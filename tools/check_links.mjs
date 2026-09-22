// Checks that every reference link in the taxonomy actually resolves.
// Needs the network, so it runs in CI rather than in the unit tests:
//   node tools/check_links.mjs
import fs from 'node:fs';

const files = ['axes', 'math', 'models', 'drift', 'pipelines']
  .map(n => JSON.parse(fs.readFileSync(new URL(`../site/taxonomy/${n}.json`, import.meta.url))));

const links = new Map();
const collect = (code, entry) => {
  for (const link of [entry.ref, entry.docs].filter(Boolean)) {
    if (!links.has(link.url)) links.set(link.url, []);
    links.get(link.url).push(code);
  }
};
Object.entries(files[0].codes).forEach(([c, v]) => collect(c, v));
Object.entries(files[1].formulas).forEach(([c, v]) => collect(c, v));
files[2].models.forEach(m => collect(m.c, m));
files[3].checkers.forEach(d => collect(d.c, d));
files[4].pipelines.forEach(p => collect(p.c, p));

console.log(`checking ${links.size} distinct links`);
const failures = [];
const urls = [...links.keys()];
const CONCURRENCY = 8;

async function check(url) {
  try {
    // A missing Wikipedia article answers 404, so the status is enough.
    const res = await fetch(url, { redirect: 'follow', headers: { 'user-agent': 'data-craft-nexus link check' } });
    if (!res.ok) failures.push(`${res.status} ${url} (${links.get(url).join(', ')})`);
  } catch (err) {
    failures.push(`ERR ${url} ${err.message} (${links.get(url).join(', ')})`);
  }
}

for (let i = 0; i < urls.length; i += CONCURRENCY) {
  await Promise.all(urls.slice(i, i + CONCURRENCY).map(check));
  process.stdout.write('.');
}
console.log();

if (failures.length) {
  console.error(`\n${failures.length} broken link(s):`);
  for (const f of failures.sort()) console.error('  ' + f);
  process.exit(1);
}
console.log('all links resolve');
