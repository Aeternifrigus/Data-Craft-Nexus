// The order models are shown in.
//
// Counting matched coordinates ranked models by how describable they were:
// Linear Regression came first on 17 of 20 regression datasets and cost up to
// 0.7 R². This orders them by what they were actually worth on the benchmark,
// using weights fitted in bench/dcn/learn.py and judged leave-one-dataset-out.
//
// The base is a prior per model: its average percentile rank among the models
// that ran on the same dataset, shrunk toward the middle so a model seen on
// few datasets is not trusted too far. Interactions between the dataset's
// features and the model's family (`weights`) and a blend with the nearest
// benchmark datasets (`neighbours`) are the two richer orders learn.py can
// choose; each is read here, so switching needs no code change.
//
// A task the benchmark never covered (grouping, anomalies, forecasting) has no
// weights, and the ranking falls back to counting coordinates. The page says
// which of the two it used.
//
// When ranking.json's "chosen" is "prior_knn", the prior is blended with what
// each model was worth on the benchmark datasets nearest to the upload:
// (S * prior + sum w * rank) / (S + sum w) over the k nearest, with
// w = exp(-(distance / h)^2). learn.py fixes S, k and h before evaluating, and
// only ships this order if it beats the plain prior by more than luck.
// Mirrors bench/dcn/ranking.py; the agreement test holds the two together.

export const DEFAULT_PRIOR = 0.5;

// Everything the weights can use, computed from what the page already measured.
export function rankingFeatures(sig) {
  const codes = new Set(sig.codes);
  const rows = Math.max(sig.rows ?? 0, 1);
  const features = Math.max(sig.features ?? 0, 1);
  const out = {
    log_rows: Math.log10(rows),
    log_features: Math.log10(features),
    features_per_row: Math.log10(features / rows),
  };
  for (const code of ['A31', 'A32', 'A38', 'A41', 'A42', 'A43', 'A51', 'A52', 'A53', 'A54', 'A61', 'A62', 'A64']) {
    out[`is_${code}`] = codes.has(code) ? 1 : 0;
  }
  return out;
}

// A dataset's meta-features in a fixed order, at the four decimals the
// benchmark stores, so the page and the benchmark measure the same distance.
export function metaVector(meta, names) {
  return names.map(name => Math.round((meta?.[name] ?? 0) * 1e4) / 1e4);
}

export function metaDistance(x, y, names, scale) {
  let total = 0;
  for (let i = 0; i < names.length; i++) {
    const spread = scale?.[names[i]] || 1;
    const diff = (x[i] - y[i]) / spread;
    total += diff * diff;
  }
  return Math.sqrt(total / names.length);
}

// The k nearest benchmark datasets and their weights, nearest first.
export function neighboursOf(block, meta) {
  const x = metaVector(meta, block.features);
  return block.datasets
    .map(d => ({ d, dist: metaDistance(x, d.meta, block.features, block.scale) }))
    .sort((a, b) => a.dist - b.dist || (a.d.dataset < b.d.dataset ? -1 : a.d.dataset > b.d.dataset ? 1 : 0))
    .slice(0, block.k)
    .map(({ d, dist }) => ({ d, w: Math.exp(-((dist / block.bandwidth) ** 2)) }));
}

// Neighbours for the order in use, or null when it does not use them.
export function rankingNeighbours(T, task, meta) {
  const block = T.RANKING?.tasks?.[task]?.neighbours;
  if (!block || !meta || T.RANKING.chosen !== 'prior_knn') return null;
  return neighboursOf(block, meta);
}

export function blend(prior, code, neighbours, strength) {
  let total = 0, weight = 0;
  for (const { d, w } of neighbours) {
    const rank = d.ranks[code];
    if (rank == null) continue;
    total += w * rank;
    weight += w;
  }
  return (strength * prior + total) / (strength + weight);
}

export function hasLearnedRanking(T, task) {
  return Boolean(T.RANKING?.tasks?.[task]);
}

// A model's learned score, or null when this task was never benchmarked.
export function learnedScore(T, model, task, features, neighbours = null) {
  const ranking = T.RANKING?.tasks?.[task];
  if (!ranking) return null;
  const prior = ranking.prior[model.c];
  if (prior == null) return null;   // never benchmarked: no claim to make
  if (neighbours) return blend(prior, model.c, neighbours, ranking.neighbours.strength);
  let score = prior;
  const family = T.RANKING.families?.[model.c];
  for (const [name, value] of Object.entries(features)) {
    score += (ranking.weights?.[`${family}|${name}`] ?? 0) * value;
  }
  return score;
}

// How many datasets the weights were fitted on, for the page to state.
export function rankingProvenance(T, task) {
  const ranking = T.RANKING?.tasks?.[task];
  if (!ranking) return null;
  return {
    datasets: ranking.datasets,
    trainedOn: T.RANKING.trained_on,
    chosen: T.RANKING.chosen,
  };
}
