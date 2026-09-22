// The order models are shown in.
//
// Counting matched coordinates ranked models by how describable they were:
// Linear Regression came first on 17 of 20 regression datasets and cost up to
// 0.7 R². This orders them by what they were actually worth on the benchmark,
// using weights fitted in bench/dcn/learn.py and judged leave-one-dataset-out.
//
// The weights are a prior per model: its average percentile rank among the
// models that ran on the same dataset, shrunk toward the middle so a model
// seen on few datasets is not trusted too far. Interactions between the
// dataset's features and the model's family were fitted too and did not beat
// the plain prior on 20 datasets per task, so they are off until the benchmark
// is larger. `weights` is read here so turning them on needs no code change.
//
// A task the benchmark never covered (grouping, anomalies, forecasting) has no
// weights, and the ranking falls back to counting coordinates. The page says
// which of the two it used.

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

export function hasLearnedRanking(T, task) {
  return Boolean(T.RANKING?.tasks?.[task]);
}

// A model's learned score, or null when this task was never benchmarked.
export function learnedScore(T, model, task, features) {
  const ranking = T.RANKING?.tasks?.[task];
  if (!ranking) return null;
  const prior = ranking.prior[model.c];
  if (prior == null) return null;   // never benchmarked: no claim to make
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
