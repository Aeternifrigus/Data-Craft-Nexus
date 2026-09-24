// Measures the data axes. Pure functions: no DOM access.
//
// Axis 3 (modality), 4 (scale) and 6 (quality) are measured from the feature
// columns, which means the target column is left out: it is what you predict,
// not something you feed the model.
//
// Axis 5 is measured too. For a categorical target it reports class balance.
// Otherwise it reports whether the file drifts from its first half to its
// second, using the PSI cutoff the taxonomy records in DR-M2. Whichever of the
// two isn't the headline is reported alongside as an extra code.
//
// A63 (missing not at random) is never reported: whether a gap depends on the
// value that is missing cannot be decided from the file alone.

import { psiNumeric, psiCategorical, PSI_SHIFT } from './stats.js';
import { countDateLike } from './dates.js';

const MISSING_WORDS = new Set(['', 'na', 'n/a', 'null', 'nan', 'none', '-']);
export const isMissing = (v) => v == null || MISSING_WORDS.has(String(v).trim().toLowerCase());

export function profileData(head, body) {
  const n = body.length;
  const columns = head.map((name, i) => {
    const raw = body.map(r => r[i] === undefined ? '' : r[i]);
    const nonEmpty = raw.filter(v => !isMissing(v));
    const nums = nonEmpty.filter(v => !isNaN(Number(v)) && v !== '');
    const numeric = nonEmpty.length > 0 && nums.length / nonEmpty.length > 0.9;
    const uniq = new Set(nonEmpty).size;
    const dateLike = !numeric && nonEmpty.length > 0 &&
      countDateLike(nonEmpty.slice(0, 40)) / Math.min(40, nonEmpty.length) > 0.8;
    const zeros = nums.filter(v => Number(v) === 0).length;
    const avgLen = nonEmpty.length ? nonEmpty.reduce((s, v) => s + v.length, 0) / nonEmpty.length : 0;
    const textLike = !numeric && !dateLike && nonEmpty.length > 0 &&
      (avgLen > 25 || uniq / nonEmpty.length > 0.7) && uniq > Math.min(20, nonEmpty.length * 0.5);

    // Values that look wrong rather than absent: text in a numeric column, or
    // labels that differ only by case or padding ("Gdynia" and "gdynia ").
    let dirty = 0;
    const numericShare = nonEmpty.length ? nums.length / nonEmpty.length : 0;
    if (numericShare > 0.5) {
      // A column of numbers with a few unparseable entries ("n/d", "approx 5").
      dirty = nonEmpty.length - nums.length;
    } else if (!dateLike && !textLike) {
      const canon = new Map();
      for (const v of nonEmpty) {
        const key = v.trim().toLowerCase();
        if (!canon.has(key)) canon.set(key, new Set());
        canon.get(key).add(v);
      }
      for (const [, variants] of canon) if (variants.size > 1) dirty += variants.size - 1;
    }

    return {
      name, index: i, numeric, dateLike, textLike, uniq, values: raw,
      missing: n ? (n - nonEmpty.length) / n : 0,
      dirtyRate: nonEmpty.length ? dirty / nonEmpty.length : 0,
      zeroRate: nums.length ? zeros / nums.length : 0,
    };
  });

  return { n, feat: columns.length, columns, dateCols: columns.filter(c => c.dateLike) };
}

// A column with more than this share of values that look wrong is dirty.
export const DIRTY_COLUMN = 0.01;

// Axes 3, 4 and 6 over the feature columns (everything except the target).
export function measuredAxes(profile, target) {
  const features = profile.columns.filter(c => c.name !== target);
  const cols = features.length ? features : profile.columns;
  const n = profile.n;

  const numericCols = cols.filter(c => c.numeric && !c.dateLike).length;
  const catCols = cols.filter(c => !c.numeric && !c.dateLike && !c.textLike).length;
  const textCols = cols.filter(c => c.textLike).length;

  const kinds = [numericCols > 0, catCols > 0, textCols > 0].filter(Boolean).length;
  let a3 = 'A38';
  if (kinds === 1) {
    if (numericCols > 0) a3 = 'A31';
    else if (catCols > 0) a3 = 'A32';
    else if (textCols > 0) a3 = 'A34';
  }

  const feat = cols.length;
  const sparsity = cols.reduce((s, c) => s + Math.max(c.missing, c.zeroRate), 0) / feat;
  // High-dimensional means many features relative to rows. A small table with
  // a handful of columns is not high-dimensional, however few rows it has.
  const highDim = feat > n / 2 || (feat >= 20 && feat > n / 10);
  const a4 = sparsity > 0.5 ? 'A43' : (highDim ? 'A42' : 'A41');

  const miss = cols.reduce((s, c) => s + c.missing, 0) / feat;
  const noise = cols.reduce((s, c) => s + c.dirtyRate, 0) / feat;
  // Judged per column as well as on average: junk in the few numeric columns
  // of a wide categorical table disappears into an average over all of them.
  // On the benchmark's damaged files, 4 of 36 tables with junk went unflagged
  // that way.
  const worst = cols.reduce((m, c) => Math.max(m, c.dirtyRate), 0);
  const a6 = noise > 0.01 || worst > DIRTY_COLUMN ? 'A64' : (miss > 0.001 ? 'A62' : 'A61');

  return { a3, a4, a6, feat, numericCols, catCols, textCols, sparsity, miss, noise, highDim };
}

// Class balance for a categorical target, or null when it doesn't apply.
export function classBalance(profile, target) {
  const col = profile.columns.find(c => c.name === target);
  // Dates and near-unique columns are not class labels, however few rows there are.
  if (!col || col.numeric || col.dateLike || col.textLike) return null;
  if (col.uniq <= 1 || col.uniq > 20 || col.uniq > profile.n / 2) return null;
  const counts = new Map();
  for (const v of col.values) if (!isMissing(v)) counts.set(v, (counts.get(v) || 0) + 1);
  const vals = [...counts.values()].sort((a, b) => b - a);
  const total = vals.reduce((a, b) => a + b, 0);
  if (vals.length < 2 || total === 0) return null;
  const share = vals[0] / total;
  return { code: share > 0.75 ? 'A52' : 'A51', majorityShare: share, levels: vals.length };
}

// Does the file drift from its first half to its second? Measured per feature
// column with PSI; the worst column decides.
export function measureDrift(profile, target) {
  const half = Math.floor(profile.n / 2);
  if (half < 20) return null;

  const scan = (cols) => {
    let worst = null;
    for (const col of cols) {
      const first = col.values.slice(0, half), second = col.values.slice(half);
      const psi = col.numeric ? psiNumeric(first, second) : psiCategorical(first, second);
      if (psi == null || !Number.isFinite(psi)) continue;
      if (!worst || psi > worst.psi) worst = { psi, column: col.name };
    }
    return worst;
  };

  const usable = profile.columns.filter(c => !c.dateLike && !c.textLike);
  // Features first. With nothing but a date and a target, the target's own
  // drift is still worth reporting: that is label drift.
  let worst = scan(usable.filter(c => c.name !== target));
  let scope = 'features';
  if (!worst) { worst = scan(usable.filter(c => c.name === target)); scope = 'target'; }
  if (!worst) return null;
  return { ...worst, scope, code: worst.psi > PSI_SHIFT ? 'A54' : 'A53' };
}

// The six codes, plus any measured code that didn't fit in a slot.
// `decl` holds the user's answers: {target, task, order}.
export function signature(profile, decl) {
  const target = decl.target === '__none__' ? null : decl.target;
  const { a3, a4, a6 } = measuredAxes(profile, target);
  const balance = classBalance(profile, target);
  const drift = measureDrift(profile, target);

  const a5 = balance ? balance.code : (drift ? drift.code : 'A53');
  const flags = [];
  if (balance && drift) flags.push(drift.code);

  return {
    codes: [target ? 'A11' : 'A12', decl.order, a3, a4, a5, a6],
    // The resolved target, so what is measured later (the meta-features that
    // place an upload among the benchmark datasets) leaves it out too.
    target,
    flags,
    balance,
    drift,
    measured: { a3, a4, a6 },
    rows: profile.n,
    features: profile.columns.filter(c => c.name !== target).length,
    // How it will run, from the intake questions. Decides which drift
    // checkers and pipelines are usable at all.
    // How it will run, and which part of the work is being built: the answers
    // that decide which drift checkers and pipelines can be used at all.
    ops: { mode: decl.mode ?? null, labels: decl.labels ?? null, stage: decl.stage ?? null },
  };
}

// Everything the ranking matches against: the six codes plus the extra ones.
export function matchCodes(sig) {
  return [...sig.codes, ...sig.flags];
}
