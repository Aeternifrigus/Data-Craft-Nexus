// "Datasets like yours."
//
// The plot used to place a dataset by axes 1 to 3 alone, so every labelled
// table with independent rows landed on the same point: three different files
// could sit on top of each other at (0,0,0) against ten invented reference
// points. This measures a dataset the way the benchmark datasets were
// measured, and compares it with them.
//
// The features and the scale come from bench/dcn/meta.py via evidence.json,
// so an upload is judged on the same ruler as the datasets it is compared to.

import { toNumbers } from './stats.js';

// Mirrors meta_features() in bench/dcn/meta.py.
export function metaFeatures(profile, target) {
  const features = profile.columns.filter(c => c.name !== target);
  const n = Math.max(features.length, 1);
  const numeric = features.filter(c => c.numeric && !c.dateLike).length;
  const textOrDate = features.filter(c => c.textLike || c.dateLike).length;

  const targetColumn = profile.columns.find(c => c.name === target);
  let majority = 0;
  if (targetColumn && !targetColumn.numeric) {
    const counts = new Map();
    for (const value of targetColumn.values) {
      if (value !== '') counts.set(value, (counts.get(value) ?? 0) + 1);
    }
    const total = [...counts.values()].reduce((a, b) => a + b, 0);
    if (total) majority = Math.max(...counts.values()) / total;
  }

  return {
    log_rows: Math.log10(Math.max(profile.n, 1)),
    log_features: Math.log10(n),
    numeric_share: numeric / n,
    categorical_share: (n - numeric - textOrDate) / n,
    missing_share: features.reduce((s, c) => s + c.missing, 0) / n,
    noise_share: features.reduce((s, c) => s + c.dirtyRate, 0) / n,
    majority_share: majority,
    drift_psi: 0,   // filled in from the measured drift, which the profile already has
  };
}

// Distance in units of how much each feature varies across the benchmark, so
// a column count and a missingness rate count for the same amount.
export function distance(a, b, names, scale) {
  let total = 0;
  for (const name of names) {
    const spread = scale?.[name]?.std || 1;
    const diff = ((a[name] ?? 0) - (b[name] ?? 0)) / spread;
    total += diff * diff;
  }
  return Math.sqrt(total / names.length);
}

// The benchmark datasets closest to this one, nearest first.
export function nearestDatasets(T, meta, task, limit = 6) {
  const ev = T.EVIDENCE;
  if (!ev?.meta_features || !ev.meta_scale) return [];
  const benchTask = task === 'category' ? 'classification' : task === 'number' ? 'regression' : null;
  // Only classification and regression were benchmarked. For anything else
  // there is nothing comparable to show, and inventing neighbours from the
  // wrong task would be worse than showing none.
  if (!benchTask) return [];

  return ev.datasets
    .filter(d => d.meta && d.task === benchTask)
    .map(d => ({ ...d, distance: distance(meta, d.meta, ev.meta_features, ev.meta_scale) }))
    .sort((a, b) => a.distance - b.distance)
    .slice(0, limit);
}

// What a model did on those neighbours: how often it was the best choice
// there, and how far below the best it typically landed.
export function performanceOn(neighbours, code) {
  const seen = neighbours.filter(d => d.scores && d.scores[code] != null);
  if (!seen.length) return null;
  let wins = 0;
  const gaps = [];
  for (const d of seen) {
    const best = Math.max(...Object.values(d.scores));
    if (Math.abs(d.scores[code] - best) < 1e-9) wins++;
    gaps.push(best - d.scores[code]);
  }
  gaps.sort((a, b) => a - b);
  const median = gaps.length % 2
    ? gaps[(gaps.length - 1) / 2]
    : (gaps[gaps.length / 2 - 1] + gaps[gaps.length / 2]) / 2;
  return { datasets: seen.length, wins, medianGap: median };
}

// Which model won most often among the neighbours, for the summary line.
export function winnerAmong(neighbours) {
  const wins = new Map();
  for (const d of neighbours) {
    if (!d.scores || !Object.keys(d.scores).length) continue;
    const best = Object.entries(d.scores).sort((a, b) => b[1] - a[1])[0][0];
    wins.set(best, (wins.get(best) ?? 0) + 1);
  }
  const ranked = [...wins.entries()].sort((a, b) => b[1] - a[1]);
  return ranked.length ? { model: ranked[0][0], wins: ranked[0][1], of: neighbours.length } : null;
}

export { toNumbers };
