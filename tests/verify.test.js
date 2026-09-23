// "Run it here": the messages between the page and its Python worker, checked
// with a stand-in worker. The Python side is run with CPython in
// bench/tests/test_export.py; Pyodide itself needs a browser and its CDN.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  MAX_BYTES, PYODIDE_URL, PYODIDE_VERSION, PY_RUNNER, WORKER_SOURCE, packagesFor, verdict, verifyInBrowser,
} from '../site/js/verify.js';

// A worker that answers the way the real one does.
function fakeWorker(script) {
  return class {
    constructor() { this.terminated = false; fakeWorker.last = this; }
    postMessage(data) {
      this.received = data;
      queueMicrotask(() => script.forEach(msg => !this.terminated && this.onmessage({ data: msg })));
    }
    terminate() { this.terminated = true; }
  };
}

test('the worker is sent the script, the file, the runner and what to load', async () => {
  const Worker = fakeWorker([{ type: 'done', results: [] }]);
  await verifyInBrowser({ script: 'S', csv: 'a,b\n1,2\n', needs: ['xgboost'], WorkerImpl: Worker });
  const sent = fakeWorker.last.received;
  assert.equal(sent.script, 'S');
  assert.equal(sent.csv, 'a,b\n1,2\n');
  assert.equal(sent.runner, PY_RUNNER);
  assert.equal(sent.indexURL, PYODIDE_URL);
  assert.deepEqual(sent.packages, ['numpy', 'pandas', 'scikit-learn', 'xgboost']);
  assert.ok(fakeWorker.last.terminated, 'the worker is stopped when it is done');
});

test('rows arrive as each model finishes, then the whole result', async () => {
  const row = (code, score) => ({ code, name: code, status: 'ok', score, spread: 0.01, seconds: 1 });
  const Worker = fakeWorker([
    { type: 'status', text: 'Loading' },
    { type: 'row', row: row('TR1', 0.8) },
    { type: 'row', row: row('BASE-HGB-TUNED', 0.9) },
    { type: 'done', results: [row('TR1', 0.8), row('BASE-HGB-TUNED', 0.9)] },
  ]);
  const statuses = [], rows = [];
  const results = await verifyInBrowser({ script: '', csv: 'x\n1\n', WorkerImpl: Worker,
    onStatus: (t) => statuses.push(t), onRow: (r) => rows.push(r.code) });
  assert.deepEqual(statuses, ['Loading']);
  assert.deepEqual(rows, ['TR1', 'BASE-HGB-TUNED']);
  assert.equal(results.length, 2);
});

test('a failure in Python comes back as its last line, and the worker is stopped', async () => {
  const traceback = 'Traceback (most recent call last):\n  File "dcn_shortlist.py", line 3\nKeyError: \'label\'\n';
  const Worker = fakeWorker([{ type: 'error', text: traceback }]);
  await assert.rejects(verifyInBrowser({ script: '', csv: 'x\n1\n', WorkerImpl: Worker }),
    (err) => err.message === "KeyError: 'label'");
  assert.ok(fakeWorker.last.terminated);
});

test('a file too large for a browser tab is refused before anything downloads', async () => {
  let started = false;
  const Worker = class { constructor() { started = true; } };
  await assert.rejects(verifyInBrowser({ script: '', csv: 'x'.repeat(MAX_BYTES + 1), WorkerImpl: Worker }), /20 MB/);
  assert.equal(started, false);
});

test('only the libraries the shortlist needs are loaded', () => {
  assert.deepEqual(packagesFor([]), ['numpy', 'pandas', 'scikit-learn']);
  assert.deepEqual(packagesFor(['lightgbm', 'xgboost', 'lightgbm']), ['numpy', 'pandas', 'scikit-learn', 'lightgbm', 'xgboost']);
  assert.deepEqual(packagesFor(['torch']), ['numpy', 'pandas', 'scikit-learn'], 'nothing outside the list is fetched');
});

test('Python is pinned to one release on the CDN, fetched only by the worker', () => {
  assert.match(PYODIDE_VERSION, /^\d+\.\d+\.\d+$/);
  assert.equal(PYODIDE_URL, `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`);
  assert.match(WORKER_SOURCE, /await import\(indexURL \+ 'pyodide\.mjs'\)/, 'Pyodide 314 only loads in a module worker');
  assert.doesNotMatch(WORKER_SOURCE, /importScripts/);
  assert.doesNotThrow(() => new Function(WORKER_SOURCE), 'the worker source parses');
});

test('the runner uses the script\'s own load() and evaluate(), as the download does', () => {
  assert.match(PY_RUNNER, /namespace\["load"\]\(io\.StringIO\(CSV\)\)/);
  assert.match(PY_RUNNER, /namespace\["evaluate"\]\(frame/);
  assert.match(PY_RUNNER, /\n[^\n]*json\.dumps\(\[plain\(row\) for row in results\]\)\n$/, 'the last line is the value returned');
});

test('the verdict says where the page\'s first pick landed', () => {
  const r = (code, score, status = 'ok') => ({ code, name: code, status, score });
  assert.match(verdict([r('A', 0.9), r('B', 0.8)], 'A'), /A, was the best of 2/);
  assert.match(verdict([r('A', 0.8), r('B', 0.9), r('C', 0.7)], 'A'), /came 2nd of 3 on your file, 0\.1000 below B/);
  assert.match(verdict([r('A', null, 'error'), r('B', 0.9)], 'A'), /did not finish here/);
  assert.match(verdict([r('B', 0.9)], 'NN9'), /is not a model the script can run/);
  assert.equal(verdict([r('A', null, 'error')], 'A'), 'No model finished on this file.');
  assert.match(verdict([r('A', 0.5), r('B', 0.5), r('C', 0.4)], 'A'), /A, was the best of 3 on your file, level with B\./);
  assert.match(verdict([r('A', 0.8), r('B', 0.9), r('C', 0.8)], 'A'), /came 2nd of 3 on your file, level with C,/);
});
