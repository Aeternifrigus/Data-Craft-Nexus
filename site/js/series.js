// The shape of a series: how often it is zero, how strongly it repeats, how
// strongly it trends. Pure functions, no DOM.
//
// These decide which forecaster the page shows first, so the benchmark
// measures the same things the same way (bench/dcn/series.py, held to this
// file by bench/tests/test_series.py). Every threshold below was fixed before
// the forecasting benchmark ran, and none was tuned on its results.

import { isMissing } from './profile.js';

// Syntetos, Boylan and Croston (2005): demand that arrives on fewer than one
// period in 1.32 is intermittent, and sizes that vary with CV² of 0.49 or more
// make it lumpy. Their cut-offs, not ours.
export const INTERMITTENT_ADI = 1.32;
export const LUMPY_CV2 = 0.49;
// Seasonal and trend strength run from 0 (none) to 1 (all of the variation).
// Half is the cut, fixed in advance.
export const STRENGTH = 0.5;

// Days since 1970-01-01 of a civil date (Hinnant's algorithm), in whole-number
// arithmetic so the page and the benchmark agree to the last digit.
export function daysFromCivil(y, m, d) {
  y -= m <= 2 ? 1 : 0;
  const era = Math.floor(y / 400);
  const yoe = y - era * 400;
  const doy = Math.floor((153 * (m + (m > 2 ? -3 : 9)) + 2) / 5) + d - 1;
  const doe = yoe * 365 + Math.floor(yoe / 4) - Math.floor(yoe / 100) + doy;
  return era * 146097 + doe - 719468;
}

// A date as a day number (with the time as a fraction), for the shapes that
// can be read without guessing: 2025, 2025-01, 2025-01-04, 2025-01-04 10:30,
// 2025/01/04 and 04.01.2025. Anything else is NaN, and gives no period.
export function dayNumber(text) {
  const s = String(text).trim();
  let m = s.match(/^(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?(?:[T ](\d{1,2}):(\d{2})(?::(\d{2}))?)?/);
  let y, mo, d, h = 0, mi = 0, se = 0;
  if (m && (m[0].length === s.length || /^[.\dZ:+-]*$/.test(s.slice(m[0].length)))) {
    [y, mo, d] = [Number(m[1]), Number(m[2] ?? 1), Number(m[3] ?? 1)];
    [h, mi, se] = [Number(m[4] ?? 0), Number(m[5] ?? 0), Number(m[6] ?? 0)];
  } else if ((m = s.match(/^(\d{4})\/(\d{1,2})\/(\d{1,2})$/))) {
    [y, mo, d] = [Number(m[1]), Number(m[2]), Number(m[3])];
  } else if ((m = s.match(/^(\d{1,2})\.(\d{1,2})\.(\d{4})$/))) {
    [y, mo, d] = [Number(m[3]), Number(m[2]), Number(m[1])];
  } else {
    return NaN;
  }
  if (mo < 1 || mo > 12 || d < 1 || d > 31) return NaN;
  return daysFromCivil(y, mo, d) + (h * 3600 + mi * 60 + se) / 86400;
}

// How many rows make one cycle, from the typical step between dates: a day
// repeats weekly, a month yearly, and so on. Null when the step is none of
// those, or when too few dates could be read.
export const PERIODS = [
  { name: 'hourly', lo: 0.035, hi: 0.05, period: 24 },
  { name: 'daily', lo: 0.9, hi: 1.1, period: 7 },
  { name: 'weekly', lo: 6, hi: 8, period: 52 },
  { name: 'monthly', lo: 27, hi: 32, period: 12 },
  { name: 'quarterly', lo: 88, hi: 93, period: 4 },
  { name: 'yearly', lo: 360, hi: 370, period: 1 },
];

export function periodFromDays(days) {
  const known = days.filter(Number.isFinite);
  if (known.length < 3 || known.length < days.length * 0.9) return null;
  const steps = [];
  for (let i = 1; i < known.length; i++) {
    const step = Math.abs(known[i] - known[i - 1]);
    if (step > 0) steps.push(step);
  }
  if (!steps.length) return null;
  steps.sort((a, b) => a - b);
  const mid = steps.length >> 1;
  const median = steps.length % 2 ? steps[mid] : (steps[mid - 1] + steps[mid]) / 2;
  const hit = PERIODS.find(p => median >= p.lo && median <= p.hi);
  if (!hit) return null;
  // Weekdays only: the week is five rows, not seven.
  if (hit.name === 'daily' && steps.filter(s => Math.abs(s - 3) < 0.01).length > steps.length * 0.1) {
    return { name: 'business-daily', period: 5 };
  }
  return { name: hit.name, period: hit.period };
}

// The target as numbers, oldest first, as the take-home script reads it: rows
// that run newest first by the date column are turned around.
export function seriesOf(profile, target, dateColumn = null) {
  const col = profile.columns.find(c => c.name === target);
  if (!col) return null;
  const dates = dateColumn ? profile.columns.find(c => c.name === dateColumn) : null;
  const rows = col.values.map((v, i) => ({ v, d: dates ? dayNumber(dates.values[i]) : NaN }))
    .filter(r => !isMissing(r.v) && r.v !== '' && Number.isFinite(Number(r.v)));
  let reversed = false;
  if (dates) {
    const steps = [];
    for (let i = 1; i < rows.length; i++) {
      const s = rows[i].d - rows[i - 1].d;
      if (Number.isFinite(s) && s !== 0) steps.push(s);
    }
    if (steps.length && steps.filter(s => s < 0).length > steps.length * 0.9) {
      rows.reverse();
      reversed = true;
    }
  }
  const found = dates ? periodFromDays(rows.map(r => r.d)) : null;
  return { values: rows.map(r => Number(r.v)), period: found?.period ?? null, step: found?.name ?? null, reversed };
}

const mean = (xs) => xs.reduce((s, x) => s + x, 0) / xs.length;
function variance(xs) {
  const m = mean(xs);
  return xs.reduce((s, x) => s + (x - m) * (x - m), 0) / xs.length;
}

// A centred moving average: of `window` points when it is odd, and the usual
// 2 x window average (half weight at both ends) when it is even. NaN where the
// window runs off either end.
export function centredAverage(y, window) {
  const n = y.length;
  const out = new Array(n).fill(NaN);
  const half = Math.floor(window / 2);
  for (let t = half; t < n - half; t++) {
    let total = 0;
    if (window % 2) {
      for (let k = -half; k <= half; k++) total += y[t + k];
      out[t] = total / window;
    } else {
      total = 0.5 * y[t - half] + 0.5 * y[t + half];
      for (let k = -half + 1; k <= half - 1; k++) total += y[t + k];
      out[t] = total / window;
    }
  }
  return out;
}

// Classical additive decomposition, y = trend + season + remainder, and the
// strength of each part: 1 - Var(remainder) / Var(part + remainder), floored
// at 0 (Wang, Smith and Hyndman, 2006). A season needs a period and two full
// cycles; without one the trend is a three-point average.
export function strengths(y, period) {
  const n = y.length;
  const seasonal = period > 1 && n >= 2 * period;
  const window = seasonal ? period : 3;
  if (n < window + 2) return { seasonal: 0, trend: 0 };
  const trend = centredAverage(y, window);
  const idx = [];
  for (let t = 0; t < n; t++) if (Number.isFinite(trend[t])) idx.push(t);
  let season = new Array(n).fill(0);
  if (seasonal) {
    const sums = new Array(period).fill(0), counts = new Array(period).fill(0);
    for (const t of idx) { sums[t % period] += y[t] - trend[t]; counts[t % period] += 1; }
    const index = sums.map((s, j) => (counts[j] ? s / counts[j] : 0));
    const centre = mean(index);
    season = y.map((_, t) => index[t % period] - centre);
  }
  const rem = idx.map(t => y[t] - trend[t] - season[t]);
  const detrended = idx.map(t => y[t] - trend[t]);
  const deseasoned = idx.map(t => y[t] - season[t]);
  const strength = (whole) => {
    const v = variance(whole);
    return v > 0 ? Math.max(0, 1 - variance(rem) / v) : 0;
  };
  return { seasonal: seasonal ? strength(detrended) : 0, trend: strength(deseasoned) };
}

// Everything the forecasting evidence is keyed on.
export function seriesFeatures(values, period = null) {
  const y = values.filter(Number.isFinite);
  const n = y.length;
  const nonNegative = y.every(v => v >= 0);
  const sizes = y.filter(v => v > 0);
  const adi = sizes.length ? n / sizes.length : Infinity;
  let cv2 = 0;
  if (sizes.length > 1) {
    const m = mean(sizes);
    cv2 = variance(sizes) / (m * m);
  }
  const { seasonal, trend } = n ? strengths(y, period ?? 1) : { seasonal: 0, trend: 0 };
  const intermittent = n > 0 && nonNegative && adi >= INTERMITTENT_ADI;
  return {
    n, period: period ?? null, zeroShare: n ? 1 - sizes.length / n : 0, nonNegative,
    adi, cv2, intermittent, lumpy: intermittent && cv2 >= LUMPY_CV2,
    seasonalStrength: seasonal, trendStrength: trend,
  };
}

// The six kinds of series the evidence is kept for. Intermittent demand comes
// first: a strong season or trend in a series of mostly zeros is not what a
// seasonal or trend model expects.
export const CELLS = {
  lumpy: 'intermittent, with sizes that vary a lot',
  intermittent: 'intermittent, with steady sizes',
  'seasonal-trend': 'seasonal and trending',
  seasonal: 'seasonal, without much trend',
  trend: 'trending, without much season',
  level: 'neither seasonal nor trending',
};

export function seriesCell(f) {
  if (f.intermittent) return f.lumpy ? 'lumpy' : 'intermittent';
  const s = f.seasonalStrength >= STRENGTH, t = f.trendStrength >= STRENGTH;
  return s && t ? 'seasonal-trend' : s ? 'seasonal' : t ? 'trend' : 'level';
}

const CYCLE = {
  hourly: 'the dates step by an hour, so a day is 24 rows',
  daily: 'the dates step by a day, so a week is 7 rows',
  'business-daily': 'the dates step by a weekday, so a week is 5 rows',
  weekly: 'the dates step by a week, so a year is 52 rows',
  monthly: 'the dates step by a month, so a year is 12 rows',
  quarterly: 'the dates step by a quarter, so a year is 4 rows',
};

// What the take-home script needs to forecast a target: the date column that
// orders the rows, and the season, when the dates give one and the series
// holds two full cycles of it.
export function forecastSetup(profile, target) {
  const dateColumn = profile.dateCols[0]?.name ?? null;
  const series = seriesOf(profile, target, dateColumn);
  const period = series?.period;
  const season = period > 1 && series.values.length >= 2 * period ? period : null;
  return { dateColumn, season, seasonNote: season ? CYCLE[series.step] : null, series };
}
