// "What does a wrong answer cost?": the answer picks what the take-home
// script scores models by. Pure functions: no DOM access.
//
// The benchmark scored every model one way per task: balanced accuracy for a
// category, R squared for a number. Those stay the default, and the order the
// page shows is always the benchmark's. Another answer changes what the
// script scores by, so on your file the script's order is the one that counts
// for what you said a mistake costs.
//
// For a yes or no target, a miss and a false alarm can be priced: the script
// then scores each model by its cost per row, and reports the threshold on
// the model's chances that cost least on your file.

import { isMissing } from './profile.js';

// The starting answer to "a missed case costs how many false alarms".
export const MISS_RATIO = 5;
export const MISS_RATIO_MAX = 1000;

const OPTIONS = {
  category: [
    { id: 'balanced', bench: true, label: 'Every class matters the same, however rare',
      scoring: 'balanced_accuracy', metric: 'balanced accuracy', higher: true,
      says: 'The benchmark\'s own score: the share of each class it gets right, averaged over the classes, so a '
        + 'rare class counts as much as a common one.' },
    { id: 'rows', label: 'Every row counts the same',
      scoring: 'accuracy', metric: 'accuracy', higher: true,
      says: 'The share of rows it gets right. When one class is rare, always guessing the common one already '
        + 'scores well, so read it beside how rare the rare class is.' },
    { id: 'miss', binary: true, label: 'Missing a rare case costs more than a false alarm',
      scoring: null, metric: 'cost per row', higher: false,
      says: 'Each model is scored by what its mistakes cost per row: a missed case at the price you set, a false '
        + 'alarm at 1. The script also finds the threshold on the model\'s chances that cost least on your file.' },
    { id: 'rank', label: 'I will work through the riskiest rows first',
      scoring: 'roc_auc', metric: 'ROC AUC', higher: true,
      says: 'How often the model ranks a real case above one that is not, from its chances: 1 is a perfect order, '
        + '0.5 a coin toss. What matters is the order, not where the line is drawn.' },
    { id: 'chances', label: 'I will use the chances themselves',
      scoring: 'neg_log_loss', metric: 'log loss', higher: false,
      says: 'How far the chances are from what happened, punishing a confident wrong answer most. Use it when a '
        + '30% chance has to come true about 30% of the time.' },
  ],
  number: [
    { id: 'squared', bench: true, label: 'One big miss is worse than several small ones',
      scoring: 'r2', metric: 'R squared', higher: true,
      says: 'The benchmark\'s own score, built on squared error, so one big miss outweighs many small ones. 1 is '
        + 'perfect, 0 is no better than always guessing the average.' },
    { id: 'absolute', label: 'Every unit of error costs the same',
      scoring: 'neg_mean_absolute_error', metric: 'mean absolute error', higher: false,
      says: 'The average size of a miss, in the target\'s own units. A big miss counts for its size and no more.' },
    { id: 'percent', label: 'A miss matters relative to the true value',
      scoring: 'neg_mean_absolute_percentage_error', metric: 'mean absolute percentage error', higher: false,
      says: 'The average miss as a share of the true value, written as a fraction: 0.12 is 12%. Off by 10 on 20 '
        + 'counts far more than off by 10 on 1,000.' },
  ],
};

// Which scores apply: the script's own task for this answer (the same choice
// export.js takeHomeTask makes), or null when the script does not cover it.
export function costTask(task, column) {
  if (task === 'category' || task === 'number') return task;
  if (task === 'forecast') return column && !column.numeric ? 'category' : 'number';
  return null;
}

// The two values of a yes or no target, the rarer first; null otherwise.
// A tie puts the value that sorts last first ("yes" before "no", 1 before 0).
export function twoClasses(column) {
  if (!column) return null;
  const counts = new Map();
  for (const v of column.values ?? []) {
    if (isMissing(v)) continue;
    const key = String(v).trim();
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  if (counts.size !== 2) return null;
  const [a, b] = [...counts.entries()].sort((x, y) => x[1] - y[1] || (x[0] < y[0] ? 1 : x[0] > y[0] ? -1 : 0));
  return { rare: a[0], rareCount: a[1], common: b[0], commonCount: b[1] };
}

// The choices for an answer, with the rare class named where there is one.
export function costOptions(kind, column) {
  const pair = kind === 'category' ? twoClasses(column) : null;
  return (OPTIONS[kind] ?? [])
    .filter(o => !o.binary || pair)
    .map(o => (o.id === 'miss' ? { ...o, label: `Missing a "${pair.rare}" costs more than a false alarm` } : o));
}

export const benchCost = (kind) => OPTIONS[kind]?.find(o => o.bench) ?? null;

// A caution about a choice on this target, or ''.
export function costWarning(kind, id, column) {
  if (kind === 'number' && id === 'percent' && column?.zeroRate > 0) {
    const share = column.zeroRate * 100;
    return `${share < 1 ? 'Under 1%' : `${Math.round(share)}%`} of the values of ${column.name} are zero, and a `
      + 'miss on a true zero is an infinite share of it. The score will be meaningless unless those rows go.';
  }
  if (kind === 'category' && id === 'rows') {
    const pair = twoClasses(column);
    if (pair) {
      const share = pair.commonCount / (pair.rareCount + pair.commonCount);
      if (share >= 0.8) {
        return `Always answering "${pair.common}" already gets ${Math.round(share * 100)}% of rows right here.`;
      }
    }
  }
  return '';
}

// What the script scores by, from the page's answer. `answer` is null (the
// benchmark's score) or { id, ratio }. Every field the script needs is here.
export function resolveCost(kind, answer, column) {
  const options = costOptions(kind, column);
  const chosen = options.find(o => o.id === answer?.id) ?? benchCost(kind);
  if (!chosen) return null;
  const out = { kind, id: chosen.id, bench: !!chosen.bench, label: chosen.label, metric: chosen.metric,
    higher: chosen.higher, scoring: chosen.scoring, says: chosen.says };
  if (chosen.id === 'rank' && !twoClasses(column)) out.scoring = 'roc_auc_ovr';
  if (chosen.id === 'miss') {
    const pair = twoClasses(column);
    const ratio = Number(answer?.ratio);
    out.ratio = Number.isFinite(ratio) && ratio >= 1 ? Math.min(ratio, MISS_RATIO_MAX) : MISS_RATIO;
    out.positive = pair.rare;
    out.threshold = 1 / (1 + out.ratio);
  }
  return out;
}

// One line for the page and the saved reading: what the script scores by, and
// why. `benchOrder` says whether the cards above are in the benchmark's order;
// they are not for a future value, which the benchmark never measured.
export function costSentence(cost, benchOrder = true) {
  if (!cost) return '';
  if (cost.bench) {
    return benchOrder ? `It scores by ${cost.metric}, the benchmark's own score.`
      : `It scores by ${cost.metric}, the page's default for ${cost.kind === 'category' ? 'a category' : 'a number'}.`;
  }
  const said = cost.id === 'miss'
    ? `a missed "${cost.positive}" costs ${fmtRatio(cost.ratio)} false alarms`
    : cost.label.charAt(0).toLowerCase() + cost.label.slice(1);
  const why = `It scores by ${cost.metric}, because you said ${said}.`;
  if (!benchOrder) return why;
  return `${why} The order above is the benchmark's, which scored by ${benchCost(cost.kind).metric}, `
    + 'so on your file the script\'s order may differ, and for your costs it is the one that counts.';
}

export const fmtRatio = (r) => (Number.isInteger(r) ? String(r) : String(Math.round(r * 100) / 100));
