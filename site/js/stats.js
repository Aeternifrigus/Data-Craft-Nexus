// Small statistics helpers used by the profiler. No dependencies, no DOM.

export function toNumbers(values) {
  const out = [];
  for (const v of values) {
    if (v === '' || v == null) continue;
    const n = Number(v);
    if (Number.isFinite(n)) out.push(n);
  }
  return out;
}

export function quantile(sorted, q) {
  if (sorted.length === 0) return NaN;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

// Population Stability Index between a reference and a current sample.
// Bin edges come from the reference deciles, as in DR-M2. Empty bins are
// floored so the log ratio stays finite.
export function psiNumeric(reference, current, bins = 10) {
  const ref = toNumbers(reference).sort((a, b) => a - b);
  const cur = toNumbers(current);
  if (ref.length < bins * 2 || cur.length < bins * 2) return null;

  const edges = [];
  for (let i = 1; i < bins; i++) edges.push(quantile(ref, i / bins));
  const unique = [...new Set(edges)];
  if (unique.length < 2) return null; // a near-constant column tells us nothing

  const bucket = (v) => {
    let i = 0;
    while (i < unique.length && v > unique[i]) i++;
    return i;
  };
  const count = (xs) => {
    const c = new Array(unique.length + 1).fill(0);
    for (const v of xs) c[bucket(v)]++;
    return c;
  };
  return psiFromCounts(count(ref), count(cur), ref.length, cur.length);
}

// PSI for categorical columns: one bucket per category seen in either sample.
export function psiCategorical(reference, current) {
  const clean = (xs) => xs.filter(v => v !== '' && v != null);
  const ref = clean(reference), cur = clean(current);
  if (ref.length < 20 || cur.length < 20) return null;
  const levels = [...new Set([...ref, ...cur])];
  if (levels.length < 2 || levels.length > 50) return null;
  const count = (xs) => levels.map(l => xs.filter(v => v === l).length);
  return psiFromCounts(count(ref), count(cur), ref.length, cur.length);
}

function psiFromCounts(refCounts, curCounts, refN, curN) {
  const floor = 0.0001;
  let psi = 0;
  for (let i = 0; i < refCounts.length; i++) {
    const a = Math.max(refCounts[i] / refN, floor);
    const b = Math.max(curCounts[i] / curN, floor);
    psi += (b - a) * Math.log(b / a);
  }
  return psi;
}

// PSI cutoffs are the credit-risk conventions the taxonomy records in DR-M2.
export const PSI_SHIFT = 0.25;
