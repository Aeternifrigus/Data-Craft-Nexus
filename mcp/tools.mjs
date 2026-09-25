// What the MCP server's tools do, without the protocol around them, so they
// can be tested on their own (tests/mcp.test.js).
//
// Every answer comes from the page's own code in site/js: the file is decoded,
// parsed, profiled, checked and ranked by the same functions the page runs in
// the browser, on the same first 5,000 rows. Nothing here decides anything the
// page does not; it only reads the file from disk and says the result in words
// a coding agent can act on.

import fs from 'node:fs';
import path from 'node:path';
import { analyse, scriptPlan, takeHomeScript } from '../site/js/analysis.js';
import { anomalyNote, detectorSentence } from '../site/js/anomalies.js';
import { checksSummary } from '../site/js/checks.js';
import { costSentence } from '../site/js/costs.js';
import { decodeBytes, delimiterName, MAX_ROWS, parseCSV } from '../site/js/csv.js';
import { checkSentence, driftSentence, evidenceFor, evidenceSentence, leadSentence } from '../site/js/evidence.js';
import { forecasterSentence, forecastNote } from '../site/js/forecasting.js';
import { nameOf, plainReason } from '../site/js/names.js';
import { measuredAxes, profileData, signature } from '../site/js/profile.js';
import { assembleTaxonomy, TAXONOMY_FILES } from '../site/js/taxonomy.js';

// Enough for the 5,000 rows the page reads from any sensible table. A bigger
// file is read up to here and cut at the last complete line.
export const MAX_BYTES = 64 * 1024 * 1024;

// The same answers the page asks for, with the page's defaults.
export const TASKS = ['category', 'number', 'forecast', 'survival', 'group', 'anomaly', 'compress', 'generate'];
export const RUNS_AS = ['batch', 'streaming'];
export const LABELS_ARRIVE = ['immediate', 'delayed', 'none'];
export const STAGES = ['data', 'train', 'ship', 'llm'];

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
    file = path.resolve(file);
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
  id: (f) => `Leave ${f.column} out of the features; take_home_script already does. If it counts time instead (a year, `
    + 'a day number), keep it and pass rows_in_time_order: true, so the split follows it.',
  time: () => 'If the model will be used on rows that come later, split by time, not at random: pass '
    + 'rows_in_time_order: true and take_home_script splits by time. If the rows really are independent, ignore this.',
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

// ── recommend_models ─────────────────────────────────────────────────────

function orderNote(T, models) {
  if (models.forecast) return forecastNote(T, models.forecast).trim();
  if (models.anomaly) return anomalyNote(T, models.anomaly).trim();
  if (models.rankedBy === 'evidence') return 'Ordered by what each model was worth on the benchmark datasets, not by how many coordinates it matches.';
  return 'Ordered by coordinates matched: the benchmark has not covered this task, so there is nothing measured to rank them by.';
}

function modelOut(T, m, task, models) {
  const measured = models.forecast?.kind ? forecasterSentence(T, models.forecast.kind, m.c)
    : models.anomaly?.kind ? detectorSentence(T, models.anomaly.kind, m.c)
    : evidenceSentence(evidenceFor(T, m.c, task));
  return {
    code: m.c, name: m.n, evidence_score: m.evidenceScore ?? null, coordinates_matched: `${m.score} of ${m.of}`,
    how: m.mech, measured: measured || null, caution: m.caution || null, fails_when: m.fail || null,
  };
}

export function recommendModels(args) {
  const T = loadTaxonomy();
  const { profile, file } = readTable(args);
  const { decl, target, task, inferred } = declare(profile, args);
  const sig = signature(profile, decl);
  const a = analyse(T, sig, task, profile);
  const plan = scriptPlan(a.home, a.takeHome.codes);
  const models = a.models.items.map(m => modelOut(T, m, task, a.models));
  const lead = a.lead ? { code: a.lead.c, name: a.lead.n, measured: leadSentence(T, a.lead) } : null;
  const ruledOut = a.models.ruledOut.map(m => ({ code: m.c, name: m.n, why: plainReason(T, m.why) }));
  const drifts = a.drifts.items.map(d => ({
    code: d.c, name: d.n, how: d.mech, threshold: d.thr, fails_when: d.fail,
    measured: d.measure ? driftSentence(d.measure, T.EVIDENCE?.drift?.datasets?.length ?? 0) : null,
  }));
  const driftRuledOut = a.drifts.ruledOut.map(d => ({ code: d.c, name: d.n, why: d.why }));
  // A pipeline's stages can be pipelines themselves, as on the page.
  const stageName = (code) => nameOf(T, 'stage', code) ?? nameOf(T, 'pipeline', code) ?? code;
  const pipelines = a.pipelines.items.map(p => ({
    code: p.c, name: p.n, how: p.mech, stages: p.stages.map(stageName), fails_when: p.fail,
  }));
  const flags = a.checks.flags.map(f => flagOut(T, f));

  const lines = [header(file)];
  if (inferred) lines.push(`Note: ${inferred}`);
  if (flags.length) {
    lines.push('', `First: ${flags.length} ${flags.length === 1 ? 'problem' : 'problems'} would make any score here look better than it is `
      + `(${flags.map(f => f.title).join('; ')}). Call check_dataset for what to do about ${flags.length === 1 ? 'it' : 'them'}.`);
  }
  if (lead) lines.push('', `Start with ${lead.name}. ${lead.measured}`);
  if (models.length) {
    lines.push('', lead ? 'Then these, in order.' : 'In order.', orderNote(T, a.models));
    models.forEach((m, i) => {
      lines.push(`${i + 1}. ${m.name} (${m.code})${m.measured ? `. ${m.measured}` : ''}`);
      if (m.caution) lines.push(`   Caution: ${m.caution}`);
      if (m.fails_when) lines.push(`   Fails when: ${m.fails_when}`);
    });
  } else {
    lines.push('', ruledOut.length ? 'No model fits: every model that could give this kind of answer is ruled out by the data.'
      : 'No model in the taxonomy gives this kind of answer.');
  }
  if (ruledOut.length) {
    lines.push('', 'Ruled out by the data:', ...ruledOut.slice(0, 8).map(m => `  ${m.name} (${m.code}): ${m.why}`));
    if (ruledOut.length > 8) lines.push(`  and ${ruledOut.length - 8} more`);
  }
  if (drifts.length) {
    lines.push('', `Drift checkers for data that runs ${decl.mode === 'streaming' ? 'as a stream' : 'in batches'} with labels arriving ${
      { immediate: 'straight away', delayed: 'later', none: 'never' }[decl.labels]}:`);
    drifts.forEach((d, i) => lines.push(`${i + 1}. ${d.name} (${d.code})${d.measured ? `. ${d.measured}` : ''}`));
  }
  if (pipelines.length) {
    lines.push('', `Pipelines${decl.stage ? ` for this stage (${decl.stage})` : ''}, by fit to the data and the task:`);
    pipelines.forEach((pl, i) => lines.push(`${i + 1}. ${pl.name} (${pl.code}): ${pl.stages.join(', ')}`));
  }
  lines.push('', plan.runnable
    ? `take_home_script writes a Python script that runs ${lead && !a.home.forecast ? 'tuned boosting and ' : ''}${
      plan.run.length} of these on the whole file with the benchmark's cross-validation, next to a do-nothing baseline. `
      + `${costSentence(a.cost, !a.home.framed)}`
    : `No take-home script for this: ${a.home.why ?? 'none of these models can run on a table.'}`);

  return {
    text: lines.join('\n'),
    data: { file, target, task, task_inferred: Boolean(inferred), checks: flags, start_with: lead, models,
      ordered_by: orderNote(T, a.models), ruled_out: ruledOut, drift_checkers: drifts, drift_ruled_out: driftRuledOut, pipelines,
      take_home: { available: plan.runnable, runs: plan.run, not_runnable: plan.skipped, why_not: plan.runnable ? null : a.home.why ?? null } },
  };
}

// ── take_home_script ─────────────────────────────────────────────────────

export function takeHome(args) {
  const T = loadTaxonomy();
  const { profile, file, source } = readTable(args);
  const { decl, target, task, inferred } = declare(profile, args);
  if (!target) throw new InputError('The script checks models that predict a column. Pass target, the column to predict.');
  const sig = signature(profile, decl);
  const a = analyse(T, sig, task, profile);
  const plan = scriptPlan(a.home, a.takeHome.codes);
  if (!plan.runnable) {
    throw new InputError(`No script for this: ${a.home.why ?? 'none of the recommended models can run on a table.'}`);
  }
  const script = takeHomeScript({
    sig, home: a.home, profile, source, codes: a.takeHome.codes, leftOut: a.leftOut, cost: a.cost,
  });
  const needs = a.home.forecast ? 'pandas, scikit-learn and statsmodels' : 'pandas and scikit-learn';
  const where = source.path ? JSON.stringify(source.path) : 'path/to/your.csv';
  const lines = [header(file)];
  if (inferred) lines.push(`Note: ${inferred}`);
  lines.push(
    `Save the script below as dcn_shortlist.py and run: python dcn_shortlist.py ${where}`,
    `It needs ${needs}.${file.truncated ? ` It reads every row of the file, not only the first ${
      MAX_ROWS.toLocaleString('en-US')} the checks used.` : ''}`,
  );
  if (a.leftOut.length) lines.push(`It leaves out ${a.leftOut.join(', ')}, which the checks found look like IDs.`);
  if (a.home.framed) lines.push(`${target} is a category, so the script predicts it from the other columns, split by time.`);
  lines.push('', '```python', script.trimEnd(), '```');
  return {
    text: lines.join('\n'),
    data: { file, target, task: a.home.task, forecast: Boolean(a.home.forecast), runs: plan.run,
      not_runnable: plan.skipped, left_out: a.leftOut, needs, script },
  };
}
