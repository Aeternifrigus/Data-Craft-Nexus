// What the MCP server's tools do, without the protocol around them, so they
// can be tested on their own (tests/mcp.test.js).
//
// Every answer comes from the page's own code in site/js: the file is decoded,
// parsed, profiled and checked by the same functions the page runs in
// the browser, on the same first 5,000 rows. Nothing here decides anything the
// page does not; it only reads the file from disk and says the result in words
// a coding agent can act on.

import fs from 'node:fs';
import path from 'node:path';
import { analyse } from '../site/js/analysis.js';
import { checksSummary } from '../site/js/checks.js';
import { decodeBytes, delimiterName, MAX_ROWS, parseCSV } from '../site/js/csv.js';
import { checkSentence } from '../site/js/evidence.js';
import { measuredAxes, profileData, signature } from '../site/js/profile.js';
import { assembleTaxonomy, TAXONOMY_FILES } from '../site/js/taxonomy.js';

// Enough for the 5,000 rows the page reads from any sensible table. A bigger
// file is read up to here and cut at the last complete line.
export const MAX_BYTES = 64 * 1024 * 1024;

// The same answers the page asks for, with the page's defaults.
export const TASKS = ['category', 'number', 'forecast', 'survival', 'group', 'anomaly', 'compress', 'generate'];

// A mistake in what the agent asked for, reported back to it as a tool error
// rather than thrown at the protocol layer.
export class InputError extends Error {}

let taxonomy;
export function loadTaxonomy() {
  if (!taxonomy) {
    const dir = new URL('../site/taxonomy/', import.meta.url);
    taxonomy = assembleTaxonomy(Object.fromEntries(TAXONOMY_FILES.map(name => [
      name, JSON.parse(fs.readFileSync(new URL(`${name}.json`, dir), 'utf8')),
    ])));
  }
  return taxonomy;
}

function readBytes(file) {
  let stat;
  try {
    stat = fs.statSync(file);
  } catch {
    throw new InputError(`No file at ${file}. Pass an absolute path, or the contents as csv.`);
  }
  if (!stat.isFile()) throw new InputError(`${file} is not a file.`);
  const size = Math.min(stat.size, MAX_BYTES);
  const bytes = Buffer.alloc(size);
  const fd = fs.openSync(file, 'r');
  try {
    fs.readSync(fd, bytes, 0, size, 0);
  } finally {
    fs.closeSync(fd);
  }
  if (stat.size <= MAX_BYTES) return { bytes, cut: false };
  const lastLine = bytes.lastIndexOf(0x0a);
  return { bytes: lastLine > 0 ? bytes.subarray(0, lastLine + 1) : bytes, cut: true };
}

// Reads a table the way the page does: decoded (UTF-8, else Windows-1250 or
// -1252), delimiter detected, decimal commas read as numbers, at most the
// first 5,000 rows. `source` is what the take-home script needs to read the
// whole file the same way.
export function readTable({ path: file, csv }) {
  if (!file === !csv) throw new InputError('Pass exactly one of path (a file on this machine) or csv (its contents).');
  let text, encoding, fileName, bytesCut = false;
  if (file) {
    const { bytes, cut } = readBytes(file);
    ({ text, encoding } = decodeBytes(bytes));
    fileName = path.basename(file);
    bytesCut = cut;
  } else {
    text = csv.replace(/^﻿/, '');
    encoding = 'UTF-8';
    fileName = 'data.csv';
  }
  const parsed = parseCSV(text);
  if (!parsed.head.length || !parsed.body.length) {
    throw new InputError('No rows found: the file needs a header row and at least one data row.');
  }
  const profile = profileData(parsed.head, parsed.body);
  const read = {
    sep: parsed.delimiter,
    encoding: /1250/.test(encoding) ? 'cp1250' : /1252/.test(encoding) ? 'cp1252' : 'utf-8',
    decimalComma: parsed.decimalComma,
  };
  const notes = [`${delimiterName(parsed.delimiter)}-separated`];
  if (encoding !== 'UTF-8') notes.push(`${encoding} text`);
  if (parsed.decimalComma) notes.push('decimal commas read as numbers');
  if (parsed.truncated || bytesCut) notes.push(`only the first ${MAX_ROWS.toLocaleString('en-US')} rows are used, as on the page`);
  return {
    profile,
    source: { fileName, path: file ?? null, columns: parsed.head, read },
    file: {
      name: fileName, rows: profile.n, columns: profile.feat, truncated: parsed.truncated || bytesCut,
      delimiter: delimiterName(parsed.delimiter), encoding, decimal_comma: parsed.decimalComma, notes,
    },
  };
}

const columnKind = (c) => (c.numeric ? 'number' : c.dateLike ? 'date' : c.textLike ? 'text' : 'category');

function resolveTarget(profile, target) {
  if (target == null) return null;
  if (profile.columns.some(c => c.name === target)) return target;
  const loose = profile.columns.find(c => c.name.trim().toLowerCase() === String(target).trim().toLowerCase());
  if (loose) return loose.name;
  throw new InputError(`There is no column called "${target}". The columns are: ${
    profile.columns.map(c => c.name).join(', ')}.`);
}

// The kind of answer, when the agent did not say: a number when the target
// is numeric with more than 20 values, a category otherwise. Said in the
// reply, so it can be corrected.
function resolveTask(profile, target, task) {
  if (task) return { task, inferred: null };
  if (!target) return { task: 'group', inferred: 'no target was given, so this is read as looking for natural groupings' };
  const col = profile.columns.find(c => c.name === target);
  const task2 = col.numeric && col.uniq > 20 ? 'number' : 'category';
  return {
    task: task2,
    inferred: `task not given; read as ${task2} because ${target} is ${col.numeric ? 'numeric' : 'not numeric'} with ${
      col.uniq} distinct values. Pass task to override.`,
  };
}

// The intake answers, as the page would hold them.
function declare(profile, args) {
  const target = resolveTarget(profile, args.target ?? null);
  const { task, inferred } = resolveTask(profile, target, args.task ?? null);
  const decl = {
    target: target ?? '__none__', task, order: args.rows_in_time_order ? 'A22' : 'A21',
    mode: args.runs_as ?? 'batch', labels: args.labels_arrive ?? 'delayed', stage: args.stage ?? null,
  };
  return { decl, target, task, inferred };
}

// The page's advice on two flags points at its own buttons and script. An
// agent has neither, so it gets the same advice in terms of these tools.
const AGENT_FIX = {
  id: (f) => `Leave ${f.column} out of the features. If it counts time instead (a year, a day number), keep it and `
    + 'pass rows_in_time_order: true, so the split follows it.',
  time: () => 'If the model will be used on rows that come later, split by time, not at random, and pass '
    + 'rows_in_time_order: true. If the rows really are independent, ignore this.',
};

function flagOut(T, f) {
  return {
    kind: f.kind, column: f.column ?? null, score: f.score ?? null, title: f.title, text: f.text,
    fix: AGENT_FIX[f.kind] ? AGENT_FIX[f.kind](f) : f.fix, measured: checkSentence(T, f.kind) || null,
  };
}

function header(file) {
  return `Read ${file.rows.toLocaleString('en-US')} rows and ${file.columns} columns from ${file.name} (${file.notes.join(', ')}).`;
}

// ── check_dataset ────────────────────────────────────────────────────────

export function checkDataset(args) {
  const T = loadTaxonomy();
  const { profile, file } = readTable(args);
  const { decl, target, task, inferred } = declare(profile, args);
  const sig = signature(profile, decl);
  const { checks } = analyse(T, sig, task, profile);
  const flags = checks.flags.map(f => flagOut(T, f));
  const clear = flags.length ? [] : checksSummary(checks, target);

  const lines = [header(file)];
  if (inferred) lines.push(`Note: ${inferred}`);
  if (!flags.length) {
    lines.push('', `Nothing found: ${clear.join(', ')}.`,
      'These are what most often make a validation score look better than it will be on new data.');
  } else {
    lines.push('', `${flags.length} ${flags.length === 1 ? 'problem' : 'problems'} that would make any model's score look `
      + 'better than it will be on new data. No choice of model fixes these; settle them before trusting a score.');
    flags.forEach((f, i) => {
      lines.push('', `${i + 1}. [${f.kind}] ${f.title}`, `   ${f.text}`, `   Fix: ${f.fix}`);
      if (f.measured) lines.push(`   How well this check works: ${f.measured}`);
    });
  }
  if (checks.best && !flags.some(f => f.kind === 'leak')) {
    lines.push('', `Strongest single column: ${checks.best.column} (${checks.best.score.toFixed(3)} on held-out rows, `
      + 'below the 0.99 that would flag it).');
  }
  return {
    text: lines.join('\n'),
    data: { file, target, task, task_inferred: Boolean(inferred), flags, clear, checks_run: checks.ran,
      best_single_column: checks.best },
  };
}

// ── profile_dataset ──────────────────────────────────────────────────────

export function profileDataset(args) {
  const T = loadTaxonomy();
  const { profile, file } = readTable(args);
  const { decl, target } = declare(profile, { ...args, task: args.task ?? 'group' });
  const sig = signature(profile, decl);
  const axes = measuredAxes(profile, target);
  const signatureOut = sig.codes.map((code, i) => ({
    axis: T.AXES[i]?.label ?? `axis ${i + 1}`, code, name: T.CODES[code]?.name ?? code, note: T.CODES[code]?.note ?? '',
    from: i < 2 ? 'your answers' : 'measured',
  }));
  const extra = sig.flags.map(code => ({ code, name: T.CODES[code]?.name ?? code }));
  const columns = profile.columns.map(c => ({
    name: c.name, kind: columnKind(c), distinct: c.uniq,
    missing_share: Number(c.missing.toFixed(4)), odd_values_share: Number(c.dirtyRate.toFixed(4)),
    ...(c.name === target ? { target: true } : {}),
  }));

  const lines = [header(file), '', 'Signature (axes 1 and 2 come from the answers, 3 to 6 are measured):'];
  for (const s of signatureOut) lines.push(`  ${s.code} ${s.axis}: ${s.name}. ${s.note}`);
  if (extra.length) lines.push(`  also measured: ${extra.map(e => `${e.code} ${e.name}`).join(', ')}`);
  if (sig.balance) lines.push(`Largest class: ${(100 * sig.balance.majorityShare).toFixed(1)}% of ${target}, over ${sig.balance.levels} classes.`);
  if (sig.drift) {
    lines.push(`Worst shift between the first and second half of the file: PSI ${sig.drift.psi.toFixed(2)} on ${
      sig.drift.column}${sig.drift.scope === 'target' ? ' (the target itself)' : ''}, ${sig.drift.psi > 0.25 ? 'above' : 'below'} the 0.25 cutoff.`);
  }
  lines.push(`Features: ${axes.numericCols} numeric, ${axes.catCols} categorical, ${axes.textCols} text; `
    + `${(100 * axes.miss).toFixed(1)}% missing on average.`);
  lines.push('', 'Columns:');
  for (const c of columns) {
    lines.push(`  ${c.name}: ${c.kind}, ${c.distinct} distinct${c.missing_share ? `, ${(100 * c.missing_share).toFixed(1)}% missing` : ''}${
      c.odd_values_share > 0.01 ? `, ${(100 * c.odd_values_share).toFixed(1)}% odd values` : ''}${c.target ? ' (target)' : ''}`);
  }
  return {
    text: lines.join('\n'),
    data: { file, target, signature: signatureOut, also_measured: extra, balance: sig.balance, drift: sig.drift,
      measured: { numeric_columns: axes.numericCols, categorical_columns: axes.catCols, text_columns: axes.textCols,
        missing_share: axes.miss, odd_values_share: axes.noise, sparsity: axes.sparsity, high_dimensional: axes.highDim },
      columns },
  };
}
