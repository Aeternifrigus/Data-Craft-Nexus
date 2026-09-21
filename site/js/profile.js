// Measures axes 3 (modality), 4 (scale) and 6 (quality) from the data,
// and derives axis 5 (distribution) from the data plus the user's answers.
// Pure functions: no DOM access.

export function profileData(head, body) {
  const n = body.length;
  const columns = head.map((name, i) => {
    const raw = body.map(r => r[i] === undefined ? '' : r[i]);
    const nonEmpty = raw.filter(v => v !== '' && v.toLowerCase() !== 'na' && v.toLowerCase() !== 'null' && v.toLowerCase() !== 'nan');
    const nums = nonEmpty.filter(v => v !== '' && !isNaN(Number(v)));
    const numeric = nonEmpty.length > 0 && nums.length / nonEmpty.length > 0.9;
    const uniq = new Set(nonEmpty).size;
    const dateLike = !numeric && nonEmpty.length > 0 &&
      nonEmpty.slice(0, 40).filter(v => !isNaN(Date.parse(v))).length / Math.min(40, nonEmpty.length) > 0.8;
    const zeros = nums.filter(v => Number(v) === 0).length;
    const avgLen = nonEmpty.length ? nonEmpty.reduce((s, v) => s + v.length, 0) / nonEmpty.length : 0;
    const textLike = !numeric && !dateLike && nonEmpty.length > 0 &&
      (avgLen > 25 || uniq / nonEmpty.length > 0.7) && uniq > Math.min(20, nonEmpty.length * 0.5);
    return { name, numeric, dateLike, textLike, uniq,
      missing: (n - nonEmpty.length) / n,
      zeroRate: nums.length ? zeros / nums.length : 0 };
  });

  const feat = columns.length;
  const numericCols = columns.filter(c => c.numeric && !c.dateLike).length;
  const catCols = columns.filter(c => !c.numeric && !c.dateLike && !c.textLike).length;
  const textCols = columns.filter(c => c.textLike).length;
  const dateCols = columns.filter(c => c.dateLike);

  const kinds = [numericCols > 0, catCols > 0, textCols > 0].filter(Boolean).length;
  let a3 = 'A38';
  if (kinds === 1) {
    if (numericCols > 0) a3 = 'A31';
    else if (catCols > 0) a3 = 'A32';
    else if (textCols > 0) a3 = 'A34';
  }

  const sparsity = columns.reduce((s, c) => s + Math.max(c.missing, c.zeroRate), 0) / feat;
  let a4;
  if (sparsity > 0.5) a4 = 'A43';
  else if (feat > n / 10) a4 = 'A42';
  else a4 = 'A41';

  const miss = columns.reduce((s, c) => s + c.missing, 0) / feat;
  let a6 = 'A61';
  if (miss > 0.25) a6 = 'A64';
  else if (miss > 0.001) a6 = 'A62';

  return { n, feat, columns, numericCols, catCols, dateCols, miss, sparsity, a3, a4, a6 };
}

// Axis 5. `decl` holds the user's answers: {target, task, order}.
export function axis5(profile, rows, decl) {
  if (decl.order === 'A22') return 'A54';
  if (!decl.target || decl.target === '__none__') return 'A53';
  const col = profile.columns.find(c => c.name === decl.target);
  if (!col) return 'A53';
  if (!col.numeric && col.uniq > 1 && col.uniq <= 20) {
    const idx = profile.columns.indexOf(col);
    const counts = {};
    rows.forEach(r => { const v = r[idx]; if (v !== '') counts[v] = (counts[v] || 0) + 1; });
    const vals = Object.values(counts).sort((a, b) => b - a);
    if (vals.length > 1 && vals[0] / vals.reduce((a, b) => a + b, 0) > 0.75) return 'A52';
    return 'A51';
  }
  return 'A53';
}

// The six-code signature, in axis order.
export function signature(profile, rows, decl) {
  const a5 = axis5(profile, rows, decl);
  return [decl.target === '__none__' ? 'A12' : 'A11', decl.order, profile.a3, profile.a4, a5, profile.a6];
}
