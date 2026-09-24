// The order anomaly detectors are shown in, and what the anomaly benchmark
// (bench/dcn/anomaly.py, anomaly_learn.py) measured for them. Pure functions.
//
// The benchmark fitted every detector without labels on ADBench's tables and
// scored it against their labels. It kept two orders: one for every table,
// and one per width of table (narrow, middling, wide), because distances
// concentrate as columns are added (math AD6). The per-width order is used
// only if it passed the rule fixed before the run (anomaly.json "chosen").

export const LOW_COLUMNS = 10;
export const HIGH_COLUMNS = 50;

export function anomalyKind(columns) {
  if (columns == null) return null;
  return columns <= LOW_COLUMNS ? 'low' : columns > HIGH_COLUMNS ? 'high' : 'mid';
}

const DETECTOR_NAMES = {
  TR4: 'Isolation Forest', SV3: 'One-Class SVM', IB3: 'Local Outlier Factor', IB4: 'k-NN distance',
  PR6: 'robust covariance', CL2: 'DBSCAN',
};

// The prior to order detectors by, or null when there is no benchmark.
export function anomalyOrder(T, sig) {
  const A = T.ANOMALY;
  if (!A) return null;
  const kind = anomalyKind(sig?.features);
  const byKind = A.chosen === 'kind' && kind && A.kind_prior?.auc?.[kind];
  return { prior: byKind || A.prior.auc, by: byKind ? 'kind' : 'fixed', kind, metric: 'auc' };
}

export const widthLabel = (T, kind) => T.ANOMALY?.kinds?.[kind] ?? kind;

export function detectorSentence(T, kind, code) {
  const entry = T.ANOMALY?.per_kind?.[kind];
  const share = entry?.best_share?.[code];
  if (share == null) return '';
  const auc = entry.median_auc?.[code];
  return `On the ${entry.datasets} benchmark tables that were ${widthLabel(T, kind)}, it was the best of the six detectors on ${
    Math.round(share * 100)}% of them, with a median ROC AUC of ${auc.toFixed(2)} (0.5 is a guess).`;
}

// The note above the cards: what ordered them, and what the width did.
export function anomalyNote(T, order) {
  const A = T.ANOMALY;
  if (!A || !order) return '';
  const tally = Object.entries(A.decision?.metrics ?? {}).map(([m, d]) =>
    `${m === 'ap' ? 'average precision' : 'ROC AUC'}: better on ${d.wins} of ${d.units} sources, worse on ${d.losses}`).join('; ');
  const label = order.kind ? widthLabel(T, order.kind) : null;
  if (order.by === 'kind') {
    return ` Ordered by what each detector was worth on the benchmark tables that were ${label}, like yours: fitted without labels on ${
      A.datasets} tables with known anomalies from ${A.units} independent sources, and scored against the labels afterwards. Keeping an order per width beat one order for every table by more than luck (${tally}).`;
  }
  const top = (prior) => Object.entries(prior ?? {}).reduce((a, b) => (b[1] > a[1] ? b : a), ['', -Infinity])[0];
  const kindTop = order.kind ? top(A.kind_prior?.auc?.[order.kind]) : '';
  const fixedTop = top(A.prior?.auc);
  const entry = order.kind ? A.per_kind?.[order.kind] : null;
  const would = kindTop && kindTop !== fixedTop && entry
    ? ` On the ${entry.datasets} benchmark tables that were ${label}, ${DETECTOR_NAMES[kindTop] ?? kindTop} would come first by width, but the width order did not earn its place overall.`
    : '';
  return ` Ordered by what each detector was worth on ${A.datasets} benchmark tables with known anomalies from ${A.units} independent sources, fitted without labels and scored against them afterwards by ROC AUC.${
    label ? ` Yours is ${label}; k` : ' K'}eeping a separate order for each width of table did not beat one order for every table by more than luck (${tally}), so the order is the same for every table.${would}`;
}
