import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { assembleTaxonomy, TAXONOMY_FILES } from '../site/js/taxonomy.js';

const root = new URL('../', import.meta.url);

export function loadTaxonomyFromDisk() {
  const files = Object.fromEntries(TAXONOMY_FILES.map(name => [
    name, JSON.parse(fs.readFileSync(new URL(`site/taxonomy/${name}.json`, root), 'utf8')),
  ]));
  return assembleTaxonomy(files);
}

export const fixturesDir = fileURLToPath(new URL('tests/fixtures/', root));

export function readFixture(name) {
  const path = name === 'sample.csv' ? new URL('site/sample.csv', root) : new URL(`tests/fixtures/${name}`, root);
  return fs.readFileSync(path, 'utf8');
}

export function fixtureNames() {
  return ['sample.csv', ...fs.readdirSync(fixturesDir).filter(f => f.endsWith('.csv')).sort()];
}
