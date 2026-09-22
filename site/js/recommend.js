// Ranks models, drift checkers and pipelines against a measured signature.
// Pure functions: `T` is the taxonomy, `sig` comes from profile.js.
//
// Two stages:
//   1. Rule out what cannot work here. A model that assumes independent rows
//      is wrong on sequential data, a text model is wrong on a numeric table,
//      and a supervised model is wrong without labels.
//   2. Rank what is left by how many of its coordinates the data matches.
//
// Stage 2 leaves large ties, and the order inside a tie carries no meaning:
// it is the order the models happen to sit in the taxonomy. The counts
// returned here (`tied`) exist so the page can say so instead of implying a
// ranking it cannot justify. Ranking within a tie is what the benchmark is
// for.

import { matchCodes } from './profile.js';
import { hasLearnedRanking, learnedScore, rankingFeatures } from './ranking.js';

const STRUCTURE = ['A21', 'A22', 'A23', 'A24', 'A25', 'A26'];
const MODALITY = ['A31', 'A32', 'A33', 'A34', 'A35', 'A36', 'A37', 'A38'];
// A mixed table holds numbers and categories, so models for those still fit.
const MIXED_OK = ['A31', 'A32', 'A33', 'A38'];

// Reinforcement learning needs an environment to act in, not a table, so those
// models are reference material only.
const PARADIGMS = { A11: ['SL'], A12: ['USL', 'SSL'] };

const codeOf = (model, family) => model.data.find(d => family.includes(d));

export function paradigmOf(sig) {
  return sig.codes[0] === 'A11' ? 'SL' : 'USL';
}

export function paradigmLabel(p) {
  return p === 'SL' ? 'supervised' : p === 'USL' ? 'unsupervised' : 'self/semi-supervised';
}

// Why a model can't be used here, or null when it can.
export function conflict(model, sig) {
  const [supervision, structure, modality] = sig.codes;

  if (!(PARADIGMS[supervision] || []).includes(model.p)) {
    return model.p === 'RL'
      ? 'needs an environment to interact with, not a table'
      : `needs ${paradigmLabel(model.p)} data`;
  }

  // A model built for a shape the data doesn't have cannot be used: a
  // sequence model has no order to work with on independent rows, and a graph
  // or image model has no structure to read.
  const wants = codeOf(model, STRUCTURE);
  if (wants && wants !== 'A21' && wants !== structure) {
    return `needs ${wants} data (yours is ${structure})`;
  }

  const wantsModality = codeOf(model, MODALITY);
  if (wantsModality && wantsModality !== modality && wantsModality !== 'A38') {
    if (!(modality === 'A38' && MIXED_OK.includes(wantsModality))) {
      return `needs ${wantsModality} data (yours is ${modality})`;
    }
  }
  return null;
}

// How the thing will run decides which drift checkers and pipelines can be
// used at all. A detector that watches the error stream is useless if labels
// never arrive, and one that compares two batches is useless in a stream.
// `ops` comes from the two questions on the intake: {mode, labels}.
export function operatingConflict(entry, ops) {
  const needs = entry.needs;
  if (!needs || !ops) return null;
  if (needs.modes && ops.mode && !needs.modes.includes(ops.mode)) {
    return `${needs.note || 'is built for the other shape'}, so it does not fit ${ops.mode === 'streaming' ? 'a stream' : 'batch scoring'}`;
  }
  if (needs.labels && ops.labels && !needs.labels.includes(ops.labels)) {
    return needs.note || `needs labels, and yours ${ops.labels === 'none' ? 'never arrive' : 'arrive later'}`;
  }
  return null;
}

// Usable, but with a caveat worth printing on the card.
export function caution(model, sig) {
  const [, structure] = sig.codes;
  if (structure === 'A22' && codeOf(model, STRUCTURE) === 'A21') {
    return 'assumes rows are independent: split by time, not at random, and build lag features yourself';
  }
  if (sig.flags.includes('A54') || sig.codes[4] === 'A54') {
    return 'your data shifts across the file, so hold out the most recent rows and watch for drift after deployment';
  }
  return null;
}

function scored(list, codes, limit, T, sig, task) {
  // The learned order when the benchmark covered this task, coordinates
  // otherwise. Coordinates stay on every card either way: they say what the
  // data has in common with the model, which is worth reading even when it is
  // not what decides the order.
  const learned = T && sig && task && hasLearnedRanking(T, task);
  const features = learned ? rankingFeatures(sig) : null;

  const ranked = list
    .map(x => {
      const own = x.data || x.fits;
      const hits = own.filter(d => codes.includes(d));
      const evidence = learned ? learnedScore(T, x, task, features) : null;
      return { ...x, hits, score: hits.length, of: own.length, evidenceScore: evidence };
    })
    .sort((a, b) => {
      if (learned) {
        // A model the benchmark never ran sits below every model it did,
        // rather than being given a number it has not earned.
        const left = a.evidenceScore, right = b.evidenceScore;
        if (left != null && right == null) return -1;
        if (left == null && right != null) return 1;
        if (left != null && right != null && Math.abs(left - right) > 1e-9) return right - left;
      }
      return b.score - a.score;
    });

  const top = ranked.length ? ranked[0] : null;
  const tied = top == null ? 0 : ranked.filter(x => (learned && top.evidenceScore != null)
    ? x.evidenceScore != null && Math.abs(x.evidenceScore - top.evidenceScore) < 1e-9
    : x.evidenceScore == null && x.score === top.score).length;

  return {
    items: ranked.slice(0, limit),
    candidates: ranked.length,
    tied,
    topScore: top ? top.score : 0,
    rankedBy: learned ? 'evidence' : 'coordinates',
  };
}

export function rankModels(T, sig, task, limit = 4) {
  const codes = matchCodes(sig);
  const usable = T.MODELS.filter(m => m.task.includes(task) && !conflict(m, sig));
  const out = scored(usable, codes, limit, T, sig, task);
  out.items = out.items.map(m => ({ ...m, caution: caution(m, sig) }));
  out.ruledOut = T.MODELS.filter(m => m.task.includes(task) && conflict(m, sig))
    .map(m => ({ c: m.c, n: m.n, why: conflict(m, sig) }));
  return out;
}

export function rankDrifts(T, sig, limit = 4) {
  const codes = matchCodes(sig);
  const matching = T.DRIFTS.filter(d => d.fits.some(f => codes.includes(f)));
  const usable = matching.filter(d => !operatingConflict(d, sig.ops));
  const out = scored(usable, codes, limit);   // no benchmark for drift checkers yet
  out.ruledOut = matching.filter(d => operatingConflict(d, sig.ops))
    .map(d => ({ c: d.c, n: d.n, why: operatingConflict(d, sig.ops) }));
  return out;
}

export function rankPipelines(T, sig, task, limit = 3) {
  const codes = matchCodes(sig);
  const ranked = T.PIPELINES
    .filter(p => !operatingConflict(p, sig.ops))
    .map(p => {
      // Codes like "A3x" mean "any modality": they fit everything, so they
      // say nothing about this dataset and don't earn a point.
      const specific = p.data.filter(d => !d.includes('x'));
      const hits = specific.filter(d => codes.includes(d));
      const taskHit = p.task.includes('any') || p.task.includes(task);
      return { ...p, hits, of: specific.length, score: hits.length + (taskHit ? 1 : 0) };
    })
    .sort((a, b) => b.score - a.score);
  const top = ranked.length ? ranked[0].score : 0;
  return {
    items: ranked.slice(0, limit),
    candidates: ranked.length,
    tied: ranked.filter(x => x.score === top).length,
    topScore: top,
    ruledOut: T.PIPELINES.filter(p => operatingConflict(p, sig.ops))
      .map(p => ({ c: p.c, n: p.n, why: operatingConflict(p, sig.ops) })),
  };
}

// Position of a signature on the 3D plot (axes 1 to 3).
export function plotCoords(sig) {
  const [supervision, structure, modality] = sig.codes;
  return {
    x: { A11: 0, A12: 1, A14: 2, A15: 3 }[supervision] ?? 0,
    y: { A21: 0, A22: 1, A23: 2, A24: 3, A25: 4, A26: 5 }[structure] ?? 0,
    z: { A31: 0, A32: 1, A33: 2, A34: 3, A35: 4, A36: 5, A37: 6, A38: 7 }[modality] ?? 0,
  };
}
