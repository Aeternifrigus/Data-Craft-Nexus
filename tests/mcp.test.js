// The MCP server (mcp/): its tools on tables with known problems planted in
// them, agreement with the page's own analysis, and the protocol end to end.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { createServer } from '../mcp/server.mjs';
import { checkDataset, InputError, profileDataset, readTable } from '../mcp/tools.mjs';
import { analyse } from '../site/js/analysis.js';
import { parseCSV } from '../site/js/csv.js';
import { profileData, signature } from '../site/js/profile.js';
import { fixturesDir, loadTaxonomyFromDisk } from './helpers.js';

const T = loadTaxonomyFromDisk();
const root = fileURLToPath(new URL('../', import.meta.url));
const sample = path.join(root, 'site/sample.csv');

// A seeded generator, so every run plants the same table.
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Shipments with an honest, noisy relation between the features and `late`,
// plus whichever problems are asked for.
function shipments({ n = 400, leak = false, idCode = false, rowNumber = false, copies = 0, date = false, seed = 7 } = {}) {
  const r = rng(seed);
  const carriers = ['Maersk', 'MSC', 'CMA CGM', 'Hapag'];
  const rows = [];
  for (let i = 0; i < n; i++) {
    const weight = Math.round((200 + 1800 * r()) * 10) / 10;
    const distance = Math.round(500 + 9000 * r());
    const carrier = carriers[Math.floor(r() * carriers.length)];
    const priority = r() < 0.3 ? 'high' : 'low';
    const risk = 0.15 + 0.3 * (distance / 9500) + (carrier === 'MSC' ? 0.15 : 0) - (priority === 'high' ? 0.1 : 0);
    const late = r() < risk ? 'yes' : 'no';
    rows.push({ weight, distance, carrier, priority, late });
  }
  const order = rows.map((_, i) => i).sort(() => r() - 0.5);
  rows.forEach((row, i) => {
    if (leak) row.refund_issued = row.late === 'yes' ? 'Y' : 'N';
    if (idCode) row.order_ref = `ORD-${(100000 + order[i] * 7).toString(36).toUpperCase()}`;
    if (rowNumber) row.row_no = order[i] + 1;
    if (date) row.booked_on = new Date(Date.UTC(2025, 0, 1) + i * 86400000).toISOString().slice(0, 10);
  });
  for (let k = 0; k < copies; k++) rows.push({ ...rows[Math.floor(r() * n)] });
  const head = Object.keys(rows[0]);
  return [head.join(','), ...rows.map(row => head.map(h => row[h]).join(','))].join('\n') + '\n';
}

function writeTemp(name, text) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'dcn-mcp-'));
  const file = path.join(dir, name);
  fs.writeFileSync(file, text);
  return file;
}

const kinds = (result) => result.data.flags.map(f => `${f.kind}:${f.column ?? ''}`).sort();

// ── the checks, on planted problems ──────────────────────────────────────

test('a clean table raises nothing, and says what was checked', () => {
  const out = checkDataset({ csv: shipments(), target: 'late', task: 'category' });
  assert.deepEqual(out.data.flags, []);
  assert.match(out.text, /^Nothing found: no column predicts late on its own/m);
  assert.ok(out.data.best_single_column.score < 0.99);
});

test('the target under another name is flagged as a leak, with what to do', () => {
  const out = checkDataset({ csv: shipments({ leak: true }), target: 'late', task: 'category' });
  assert.deepEqual(kinds(out), ['leak:refund_issued']);
  const [flag] = out.data.flags;
  assert.ok(flag.score >= 0.99);
  assert.match(flag.fix, /Check when refund_issued becomes known/);
  assert.match(flag.measured, /benchmark datasets/);
});

test('row numbers and order codes are flagged as IDs, with advice an agent can follow', () => {
  const out = checkDataset({ csv: shipments({ idCode: true, rowNumber: true }), target: 'late', task: 'category' });
  assert.deepEqual(kinds(out), ['id:order_ref', 'id:row_no']);
  assert.match(out.data.flags[0].fix, /^Leave order_ref out of the features/);
});

test('copied rows are flagged when there are more than chance allows', () => {
  const out = checkDataset({ csv: shipments({ copies: 12 }), target: 'late', task: 'category' });
  assert.deepEqual(kinds(out), ['duplicates:']);
  assert.match(out.data.flags[0].title, /^12 rows are exact copies/);
});

test('a date column is flagged while rows are declared independent, and not once they are in time order', () => {
  const csv = shipments({ date: true });
  assert.deepEqual(kinds(checkDataset({ csv, target: 'late', task: 'category' })), ['time:booked_on']);
  assert.deepEqual(kinds(checkDataset({ csv, target: 'late', task: 'category', rows_in_time_order: true })), []);
});

test('every problem at once is every flag at once', () => {
  const csv = shipments({ leak: true, idCode: true, copies: 12, date: true });
  const out = checkDataset({ csv, target: 'late', task: 'category' });
  assert.deepEqual(kinds(out), ['duplicates:', 'id:order_ref', 'leak:refund_issued', 'time:booked_on']);
  assert.match(out.text, /^4 problems/m);
});

// ── reading files ────────────────────────────────────────────────────────

test('a path and the same contents give the same answer', () => {
  const csv = shipments({ leak: true, idCode: true });
  const file = writeTemp('orders.csv', csv);
  const byPath = checkDataset({ path: file, target: 'late', task: 'category' });
  const byText = checkDataset({ csv, target: 'late', task: 'category' });
  assert.deepEqual(byPath.data.flags, byText.data.flags);
  assert.equal(byPath.data.file.name, 'orders.csv');
});

test('a semicolon file with decimal commas is read the way the page reads it', () => {
  const { file, source } = readTable({ path: path.join(fixturesDir, 'semicolon.csv') });
  assert.equal(file.delimiter, 'semicolon');
  assert.equal(source.read.sep, ';');
  assert.equal(source.read.decimalComma, file.decimal_comma);
});

test('only the first 5,000 rows are used, and the reply says so', () => {
  const out = checkDataset({ csv: shipments({ n: 5200 }), target: 'late', task: 'category' });
  assert.equal(out.data.file.rows, 5000);
  assert.equal(out.data.file.truncated, true);
  assert.match(out.text, /only the first 5,000 rows are used/);
});

test('mistakes in the request come back as InputError with something to act on', () => {
  assert.throws(() => checkDataset({ target: 'late' }), InputError);
  assert.throws(() => checkDataset({ path: sample, csv: 'a,b\n1,2\n', target: 'a' }), InputError);
  assert.throws(() => checkDataset({ path: '/no/such/file.csv', target: 'late' }), /No file at \/no\/such\/file.csv/);
  assert.throws(() => checkDataset({ path: sample, target: 'nope' }), /The columns are: ship_date, origin_port/);
  assert.throws(() => checkDataset({ csv: 'only_a_header\n', target: 'only_a_header' }), /No rows found/);
});

test('a target named with the wrong case or spacing still resolves', () => {
  assert.equal(checkDataset({ path: sample, target: ' Delayed ', task: 'category' }).data.target, 'delayed');
});

test('a missing task is inferred from the target, and the reply says how', () => {
  const cat = checkDataset({ path: sample, target: 'delayed' });
  assert.equal(cat.data.task, 'category');
  assert.match(cat.text, /task not given; read as category because delayed is numeric with 2 distinct values/);
  const num = checkDataset({ csv: shipments(), target: 'distance' });
  assert.equal(num.data.task, 'number');
});

// ── agreement with the page ──────────────────────────────────────────────

function pageReading(file, decl) {
  const { head, body } = parseCSV(fs.readFileSync(file, 'utf8'));
  const profile = profileData(head, body);
  const sig = signature(profile, decl);
  return { profile, sig, a: analyse(T, sig, decl.task, profile) };
}

test('profile_dataset gives the signature the page shows', () => {
  const decl = { target: 'delayed', task: 'category', order: 'A22', mode: 'batch', labels: 'delayed', stage: null };
  const { sig } = pageReading(sample, decl);
  const out = profileDataset({ path: sample, target: 'delayed', rows_in_time_order: true });
  assert.deepEqual(out.data.signature.map(s => s.code), sig.codes);
  assert.equal(out.data.columns.find(c => c.name === 'ship_date').kind, 'date');
  assert.equal(out.data.columns.find(c => c.name === 'delayed').target, true);
});

// ── the protocol ─────────────────────────────────────────────────────────

async function connected() {
  const [clientSide, serverSide] = InMemoryTransport.createLinkedPair();
  const client = new Client({ name: 'test', version: '0' });
  await Promise.all([createServer().connect(serverSide), client.connect(clientSide)]);
  return client;
}

test('the server lists its tools, all read-only', async () => {
  const client = await connected();
  const { tools } = await client.listTools();
  assert.deepEqual(tools.map(t => t.name).sort(), ['check_dataset', 'profile_dataset']);
  for (const t of tools) {
    assert.equal(t.annotations.readOnlyHint, true, t.name);
    assert.ok(t.description.length > 100, t.name);
  }
  assert.match(client.getInstructions(), /call check_dataset/);
  await client.close();
});

test('a tool call returns words and the same content as data', async () => {
  const client = await connected();
  const res = await client.callTool({ name: 'check_dataset',
    arguments: { csv: shipments({ leak: true }), target: 'late', task: 'category' } });
  assert.equal(res.isError, undefined);
  assert.match(res.content[0].text, /refund_issued predicts late almost perfectly on its own/);
  assert.equal(res.structuredContent.flags[0].column, 'refund_issued');
  await client.close();
});

test('a bad request is a tool error the agent can read, not a protocol failure', async () => {
  const client = await connected();
  const res = await client.callTool({ name: 'check_dataset', arguments: { path: sample, target: 'nope' } });
  assert.equal(res.isError, true);
  assert.match(res.content[0].text, /There is no column called "nope"/);
  const bad = await client.callTool({ name: 'check_dataset', arguments: { path: sample, target: 'delayed', task: 'regression' } });
  assert.equal(bad.isError, true);
  await client.close();
});

test('the command starts the server over stdio', async () => {
  const client = new Client({ name: 'test', version: '0' });
  await client.connect(new StdioClientTransport({
    command: process.execPath, args: [path.join(root, 'mcp/dcn-mcp.mjs')], stderr: 'pipe',
  }));
  const { name } = client.getServerVersion();
  assert.equal(name, 'data-craft-nexus');
  const res = await client.callTool({ name: 'profile_dataset', arguments: { path: sample } });
  assert.match(res.content[0].text, /^Read 20 rows and 10 columns from sample.csv/);
  await client.close();
});
