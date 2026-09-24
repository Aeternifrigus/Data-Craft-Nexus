// Runs the page's "Before you trust a score" checks on CSV files, for the
// benchmark that measures them (dcn/checks.py).
//   node bench/tools/js_checks.mjs < manifest.json
// The manifest is a list of {path, target, task}; prints one JSON line per file.
import fs from 'node:fs';
import { parseCSV } from '../../site/js/csv.js';
import { profileData } from '../../site/js/profile.js';
import { runChecks } from '../../site/js/checks.js';

const manifest = JSON.parse(fs.readFileSync(0, 'utf8'));
for (const { path, target, task } of manifest) {
  const parsed = parseCSV(fs.readFileSync(path, 'utf8'));
  const profile = profileData(parsed.head, parsed.body);
  const result = runChecks(profile, { target, task, order: 'A21' });
  process.stdout.write(JSON.stringify({
    path, rows: result.rows, ran: result.ran, best: result.best,
    flags: result.flags.map(f => ({ kind: f.kind, column: f.column ?? null, score: f.score ?? null })),
  }) + '\n');
}
