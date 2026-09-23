// Prints the Python script the site generates, for the test that runs it.
//   node bench/tools/js_script.mjs <csv> <target> <category|number> <A21|A22> <codes...>
import fs from 'node:fs';
import path from 'node:path';
import { parseCSV } from '../../site/js/csv.js';
import { profileData } from '../../site/js/profile.js';
import { columnRoles, pythonScript } from '../../site/js/export.js';

const [file, target, task, order, ...codes] = process.argv.slice(2);
const parsed = parseCSV(fs.readFileSync(file, 'utf8'));
const profile = profileData(parsed.head, parsed.body);
process.stdout.write(pythonScript({
  fileName: path.basename(file),
  read: { sep: parsed.delimiter, encoding: 'utf-8', decimalComma: parsed.decimalComma },
  columns: parsed.head, target, task, ordered: order === 'A22', ...columnRoles(profile, target), shortlist: codes,
  date: '2026-01-01',
}));
