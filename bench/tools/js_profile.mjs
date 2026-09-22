// Prints what the site's JavaScript profiler measures for each CSV given on
// the command line, as JSON. The Python port is checked against this.
import fs from 'node:fs';
import { parseCSV } from '../../site/js/csv.js';
import { profileData, measuredAxes, classBalance, measureDrift, signature } from '../../site/js/profile.js';

const round = (x) => (x == null ? null : Math.round(x * 1e6) / 1e6);
const out = {};

for (const path of process.argv.slice(2)) {
  const parsed = parseCSV(fs.readFileSync(path, 'utf8'));
  const profile = profileData(parsed.head, parsed.body);
  const targets = [...parsed.head, null];
  const entry = {
    delimiter: parsed.delimiter,
    decimalComma: parsed.decimalComma,
    truncated: parsed.truncated,
    head: parsed.head,
    rows: profile.n,
    columns: profile.columns.map(c => ({
      name: c.name, numeric: c.numeric, dateLike: c.dateLike, textLike: c.textLike,
      uniq: c.uniq, missing: round(c.missing), dirtyRate: round(c.dirtyRate), zeroRate: round(c.zeroRate),
    })),
    perTarget: {},
  };
  for (const target of targets) {
    const key = target ?? '__none__';
    const axes = measuredAxes(profile, target);
    const balance = classBalance(profile, target);
    const drift = measureDrift(profile, target);
    entry.perTarget[key] = {
      axes: [axes.a3, axes.a4, axes.a6],
      sparsity: round(axes.sparsity), miss: round(axes.miss), noise: round(axes.noise),
      balance: balance && { code: balance.code, majorityShare: round(balance.majorityShare), levels: balance.levels },
      drift: drift && { code: drift.code, psi: round(drift.psi), column: drift.column, scope: drift.scope },
      signature: signature(profile, { target: target ?? '__none__', task: 'number', order: 'A21' }).codes.join(' '),
    };
  }
  out[path.split('/').pop()] = entry;
}

process.stdout.write(JSON.stringify(out, null, 1));
