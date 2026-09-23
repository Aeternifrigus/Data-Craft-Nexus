// Checks that every reference link in the taxonomy actually resolves.
// Needs the network, so it runs in CI rather than in the unit tests:
//   node tools/check_links.mjs
// How a failure is told apart from a network hiccup is in tools/links.mjs.
import fs from 'node:fs';
import { checkLinks } from './links.mjs';

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
let done = 0, retried = 0;
const failures = await checkLinks([...links.keys()], {
  onProgress: (r) => {
    if (r.tries > 1) retried++;
    if (++done % 10 === 0) process.stdout.write('.');
  },
});
console.log();
if (retried) console.log(`${retried} link(s) needed more than one try`);

const where = (url) => `(${links.get(url).join(', ')})`;
const broken = failures.filter(f => f.status != null);
const unreachable = failures.filter(f => f.status == null);
if (broken.length) {
  console.error(`\n${broken.length} broken link(s), the server answered with an error:`);
  for (const f of broken.sort((a, b) => a.url.localeCompare(b.url))) console.error(`  ${f.status} ${f.url} ${where(f.url)}`);
}
if (unreachable.length) {
  console.error(`\n${unreachable.length} link(s) could not be reached after every retry:`);
  for (const f of unreachable.sort((a, b) => a.url.localeCompare(b.url))) console.error(`  ERR ${f.url} ${f.error} ${where(f.url)}`);
}
if (failures.length) process.exit(1);
console.log('all links resolve');
