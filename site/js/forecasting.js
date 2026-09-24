// The order forecasting cards are shown in, and what the forecasting
// benchmark (bench/dcn/forecast.py, forecast_learn.py) measured for them.
// Pure functions, no DOM.
//
// The benchmark ran the take-home script's own forecasters on real series,
// sorted each series into a kind the way series.js sorts an upload, and fitted
// two orders: one for every series, and one per kind. The per-kind order is
// used only if it passed the rule fixed before the run (forecast.json
// "chosen"); otherwise every series gets the same order, and the page says the
// kind did not earn it.

import { CELLS, seriesCell, seriesFeatures } from './series.js';

// The score the order is read for: mean absolute error when every unit of a
// miss costs the same, the page's default (R²) otherwise.
export function forecastMetric(cost) {
  return cost?.scoring === 'neg_mean_absolute_error' ? 'mae' : 'r2';
}

// What the page knows about the series it is forecasting.
export function describeSeries(setup, metric = 'r2') {
  const s = setup?.series;
  if (!s || !s.values.length) return null;
  const features = seriesFeatures(s.values, s.period);
  return { features, kind: seriesCell(features), period: s.period, season: setup.season ?? null, step: s.step, metric };
}

export const kindLabel = (kind) => CELLS[kind] ?? kind;

// The prior to order forecasters by, or null when there is no benchmark.
export function forecastOrder(T, series) {
  const F = T.FORECAST;
  if (!F) return null;
  const metric = F.prior?.[series?.metric] ? series.metric : 'r2';
  const kind = series?.kind ?? null;
  const byKind = F.chosen === 'kind' && kind && F.kind_prior?.[metric]?.[kind];
  return { prior: byKind || F.prior[metric], by: byKind ? 'kind' : 'fixed', kind, metric };
}

// What the benchmark saw on series of this kind: how many, how often a
// forecaster was the best, and how often doing nothing beat every one.
export function kindEvidence(T, kind) {
  return T.FORECAST?.per_kind?.[kind] ?? null;
}

export function forecasterSentence(T, kind, code) {
  const entry = kindEvidence(T, kind);
  const share = entry?.best_share?.[code];
  if (share == null) return '';
  const pct = Math.round(share * 100);
  return `On the ${entry.series} benchmark series that were ${kindLabel(kind)}, it was the best of the four forecasters `
    + `on ${pct}% of them.`;
}

// The note above the cards: what ordered them, and how often nothing won.
export function forecastNote(T, order) {
  const F = T.FORECAST;
  if (!F || !order) return '';
  const metricName = order.metric === 'mae' ? 'mean absolute error' : 'R²';
  const entry = order.kind ? kindEvidence(T, order.kind) : null;
  const decision = F.decision?.metrics ?? {};
  const tally = Object.entries(decision).map(([m, d]) =>
    `${m === 'mae' ? 'mean absolute error' : 'R²'}: better on ${d.wins} of ${d.units} collections, worse on ${d.losses}`).join('; ');
  const how = order.by === 'kind'
    ? ` Ordered by what each forecaster was worth on the ${F.series} benchmark series, kept for series that are ${
      kindLabel(order.kind)}, like yours, by ${metricName}. Reading the kind of series earned that: it picked better forecasters than one order for every series, by more than luck (${tally}).`
    : ` Ordered by what each forecaster was worth on ${F.series} benchmark series from ${F.units} independent collections, by ${
      metricName}. ${order.kind ? `Yours reads as ${kindLabel(order.kind)}, but k` : 'K'}eeping a separate order for each kind of series did not beat one order for every series by more than luck (${tally}), so the order is the same for every series.`;
  // When the kind would have changed the first pick, say so, and why it did not.
  const top = (prior) => Object.entries(prior ?? {}).reduce((a, b) => (b[1] > a[1] ? b : a), ['', -Infinity])[0];
  const kindTop = order.kind ? top(F.kind_prior?.[order.metric]?.[order.kind]) : '';
  const fixedTop = top(F.prior?.[order.metric]);
  const changed = Object.values(decision).map(d => d.wins + d.losses);
  const wouldHave = order.by === 'fixed' && kindTop && kindTop !== fixedTop && entry
    ? ` On the ${entry.series} benchmark series that were ${kindLabel(order.kind)}, ${FORECASTER_NAMES[kindTop] ?? kindTop} was the best forecaster on ${
      Math.round((entry.best_share?.[kindTop] ?? 0) * 100)}% and would come first if the order were kept by kind. ${confirmed(F, order.metric, metricName, decision, changed)} The script runs all four on your file, which settles it for your data.`
    : '';
  const nothing = entry && entry.nothing_beat_every_model != null
    ? ` On those ${entry.series} series of your kind, doing nothing (the last value, the value a season earlier, or the average) beat every forecaster on ${Math.round(entry.nothing_beat_every_model * 100)}% of them, which is why the script scores it too.`
    : '';
  return how + wouldHave + nothing;
}

// How the kind order fared, in the first run and, when there is one, the
// confirmation run on new collections.
function confirmed(F, metric, metricName, decision, changed) {
  const first = `In the first run, by ${metricName}, it was better on ${decision[metric]?.wins ?? 0} collections and worse on ${
    decision[metric]?.losses ?? 0}, but it changed the pick in only ${Math.max(...changed)} of the ${F.units}, too few for the rule to call the gain more than luck.`;
  const c = F.confirmation;
  const d = c?.decision?.metrics?.[metric];
  if (!d) return first;
  return `${first} A second run, declared first, froze both orders and tried them on ${c.series} intermittent series from ${c.units} collections the first had never seen: by ${
    metricName} the kind order was better on ${d.wins} and worse on ${d.losses}, with median regret ${d.median_regret.kind.toFixed(3)} against ${
    d.median_regret.fixed.toFixed(3)}, but p = ${d.p_holm.toFixed(2)} after correcting for two scores, above the 0.05 the rule set${
    c.decision.replaced ? '' : ', so the order stays'}.`;
}

const FORECASTER_NAMES = { TSM1: 'ARIMA', TSM2: 'exponential smoothing', TSM4: "Croston's method (SBA)", TSM5: 'TSB' };
