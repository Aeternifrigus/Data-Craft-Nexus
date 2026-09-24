// Prints the Python script the site generates, for the test that runs it.
//   node bench/tools/js_script.mjs <csv> <target> <category|number> <A21|A22> <codes...>
// DCN_COST sets the answer to "What does a wrong answer cost?", as the page
// would: an option id (site/js/costs.js), or miss:<ratio>. Unset, the
// benchmark's own score.
import fs from 'node:fs';
import path from 'node:path';
import { parseCSV } from '../../site/js/csv.js';
import { profileData } from '../../site/js/profile.js';
import { columnRoles, pythonScript } from '../../site/js/export.js';
import { runChecks } from '../../site/js/checks.js';
import { resolveCost } from '../../site/js/costs.js';

const [file, target, task, order, ...codes] = process.argv.slice(2);
const parsed = parseCSV(fs.readFileSync(file, 'utf8'));
const profile = profileData(parsed.head, parsed.body);
const leftOut = runChecks(profile, { target, task, order }).flags.filter(f => f.kind === 'id').map(f => f.column);
const [id, ratio] = (process.env.DCN_COST ?? '').split(':');
const cost = id ? resolveCost(task, { id, ratio: ratio == null ? undefined : Number(ratio) },
  profile.columns.find(c => c.name === target)) : null;
if (id && cost?.id !== id) throw new Error(`"${id}" is not an answer the page offers for this target`);
process.stdout.write(pythonScript({
  fileName: path.basename(file),
  read: { sep: parsed.delimiter, encoding: 'utf-8', decimalComma: parsed.decimalComma },
  columns: parsed.head, target, task, ordered: order === 'A22', ...columnRoles(profile, target, leftOut), shortlist: codes, leftOut,
  cost, date: '2026-01-01',
}));
