// Ranks models, drift checkers and pipelines against a signature.
// Pure functions: `T` is the taxonomy from taxonomy.js, `sig` the six codes.

export function paradigmOf(sig) {
  return (sig[0] === 'A11') ? 'SL' : (sig[0] === 'A12' ? 'USL' : 'SSL');
}

export function paradigmLabel(p) {
  return p === 'SL' ? 'supervised' : p === 'USL' ? 'unsupervised' : 'self/semi-supervised';
}

export function rankModels(T, sig, task, limit = 4) {
  const paradigm = paradigmOf(sig);
  return T.MODELS
    .filter(m => m.task.includes(task))
    .filter(m => m.p === paradigm)
    .map(m => {
      const hits = m.data.filter(d => sig.includes(d));
      return { ...m, hits, score: hits.length };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
}

export function rankDrifts(T, sig, limit = 4) {
  return T.DRIFTS
    .map(d => {
      const hits = d.fits.filter(f => sig.includes(f));
      return { ...d, hits, score: hits.length };
    })
    .filter(d => d.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
}

export function rankPipelines(T, sig, task, limit = 3) {
  return T.PIPELINES
    .map(p => {
      const hits = p.data.filter(d => sig.includes(d) || d.includes('x'));
      const taskHit = p.task.includes('any') || p.task.includes(task);
      return { ...p, hits, score: hits.length + (taskHit ? 1 : 0) };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
}

// Position of a signature on the 3D plot (axes 1 to 3).
export function plotCoords(sig) {
  return {
    x: { A11: 0, A12: 1, A14: 2, A15: 3 }[sig[0]] ?? 0,
    y: { A21: 0, A22: 1, A23: 2, A24: 3, A25: 4, A26: 5 }[sig[1]] ?? 0,
    z: { A31: 0, A32: 1, A33: 2, A34: 3, A35: 4, A36: 5, A37: 6, A38: 7 }[sig[2]] ?? 0,
  };
}
