// Every reference link has to be a real, safe URL, and every concept that
// should have one must have one. Whether the page exists is checked by
// tools/check_links.mjs, which needs the network.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { loadTaxonomyFromDisk } from './helpers.js';

const T = loadTaxonomyFromDisk();

const entries = [
  ...Object.entries(T.CODES).map(([c, v]) => ({ code: c, ...v })),
  ...Object.entries(T.MATH).map(([c, v]) => ({ code: c, ...v })),
  ...T.MODELS.map(m => ({ code: m.c, ...m })),
  ...T.DRIFTS.map(d => ({ code: d.c, ...d })),
  ...T.PIPELINES.map(p => ({ code: p.c, ...p })),
];

// Documentation, papers and repositories: things you would open while
// working. No encyclopaedia entries.
const ALLOWED_HOSTS = [
  'scikit-learn.org', 'pytorch.org', 'pytorch-geometric.readthedocs.io', 'numpy.org', 'docs.scipy.org',
  'www.statsmodels.org', 'arxiv.org', 'www.nature.com', 'huggingface.co', 'github.com', 'rasbt.github.io',
  'riverml.xyz', 'lifelines.readthedocs.io', 'hmmlearn.readthedocs.io', 'sklearn-crfsuite.readthedocs.io',
  'umap-learn.readthedocs.io', 'xgboost.readthedocs.io', 'lightgbm.readthedocs.io', 'imbalanced-learn.org',
  'librosa.org', 'networkx.org', 'pandas.pydata.org', 'geopandas.org', 'gymnasium.farama.org',
  'stable-baselines3.readthedocs.io', 'optuna.readthedocs.io', 'mlflow.org', 'airflow.apache.org',
];

test('every link is https and points at an allowed reference site', () => {
  const bad = [];
  for (const entry of entries) {
    for (const link of [entry.ref, entry.docs].filter(Boolean)) {
      const url = new URL(link.url);
      if (url.protocol !== 'https:') bad.push(`${entry.code} is not https: ${link.url}`);
      if (!ALLOWED_HOSTS.includes(url.host)) bad.push(`${entry.code} -> ${url.host}`);
      if (!link.label) bad.push(`${entry.code} has a link with no label`);
      if (/\s/.test(link.url)) bad.push(`${entry.code} url contains a space`);
    }
  }
  assert.deepEqual(bad, []);
});

test('nothing links to an encyclopaedia', () => {
  const encyclopaedic = entries.filter(e =>
    [e.ref, e.docs].filter(Boolean).some(l => /wikipedia|britannica|wikiwand/i.test(l.url)));
  assert.deepEqual(encyclopaedic.map(e => e.code), []);
});

test('everything that should point somewhere does', () => {
  // Three entries are deliberately unlinked: "low-dimensional", "dense" and
  // "complete" describe a dataset, they are not techniques to read about.
  const missing = entries.filter(e => !e.ref && !e.docs).map(e => e.code).sort();
  assert.deepEqual(missing, ['A41', 'A44', 'A61']);
});

test('every model points somewhere, and the ones the benchmark runs link to their implementation', () => {
  assert.deepEqual(T.MODELS.filter(m => !m.ref && !m.docs).map(m => m.c), []);
  const withDocs = T.MODELS.filter(m => m.docs).map(m => m.c);
  assert.ok(withDocs.length >= 20, `only ${withDocs.length} models link to docs`);
  for (const m of T.MODELS) {
    if (!m.docs) continue;
    assert.match(m.docs.url, /^https:\/\/(scikit-learn\.org|xgboost\.readthedocs\.io|lightgbm\.readthedocs\.io)/);
  }
});
