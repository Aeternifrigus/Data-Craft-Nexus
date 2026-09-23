// "Run it here": the take-home script, run on the uploaded file inside the
// browser, so the shortlist can be checked without installing anything.
//
// Python comes from Pyodide, which the page downloads from cdn.jsdelivr.net
// only when asked, into a Web Worker so the page stays responsive. The file is
// not sent anywhere: the worker gets the page's own decoded copy.
//
// What runs is exactly the script "Download the script" saves (export.js),
// through its own load() and evaluate(). tests/verify.test.js checks the
// messages with a stand-in worker, and bench/tests/test_export.py runs
// PY_RUNNER on generated scripts with CPython.

export const PYODIDE_VERSION = '314.0.7';
export const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

// Above this, a browser tab is the wrong place to fit ten models.
export const MAX_BYTES = 20 * 1024 * 1024;

// The Python the worker runs once the packages are in. SCRIPT and CSV are set
// from JavaScript; report_row passes each finished model back as it finishes.
export const PY_RUNNER = `
import io
import json
import math


def plain(row):
    """JSON without NaN, which JSON.parse refuses."""
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in row.items()}


namespace = {"__name__": "dcn_shortlist"}
exec(compile(SCRIPT, "dcn_shortlist.py", "exec"), namespace)
frame = namespace["load"](io.StringIO(CSV))
results = namespace["evaluate"](frame, progress=lambda row: report_row(json.dumps(plain(row))))
json.dumps([plain(row) for row in results])
`;

// The worker, as source, so the single-file page can start it from a blob.
// A module worker: Pyodide 314 refuses to load in a classic one.
export const WORKER_SOURCE = `
let pyodide = null;
self.onmessage = async (event) => {
  const { indexURL, packages, script, csv, runner } = event.data;
  const say = (text) => self.postMessage({ type: 'status', text });
  try {
    if (!pyodide) {
      say('Downloading Python (Pyodide). The first time takes a while.');
      const { loadPyodide } = await import(indexURL + 'pyodide.mjs');
      pyodide = await loadPyodide({ indexURL });
    }
    say('Loading ' + packages.join(', ') + '.');
    await pyodide.loadPackage(packages);
    pyodide.globals.set('SCRIPT', script);
    pyodide.globals.set('CSV', csv);
    pyodide.globals.set('report_row', (text) => self.postMessage({ type: 'row', row: JSON.parse(text) }));
    say('Cross-validating each model on your file.');
    const out = await pyodide.runPythonAsync(runner);
    self.postMessage({ type: 'done', results: JSON.parse(out) });
  } catch (err) {
    self.postMessage({ type: 'error', text: String((err && err.message) || err) });
  }
};
`;

// The packages the script needs: scikit-learn brings scipy and joblib with it,
// and XGBoost or LightGBM only when the shortlist has them.
export function packagesFor(needs = []) {
  const extra = [...new Set(needs.filter(n => n === 'xgboost' || n === 'lightgbm'))].sort();
  return ['numpy', 'pandas', 'scikit-learn', ...extra];
}

// A worker from a blob works from a file on disk as well as from a server.
function startWorker(WorkerImpl) {
  if (WorkerImpl) return { worker: new WorkerImpl(), release: () => {} };
  const url = URL.createObjectURL(new Blob([WORKER_SOURCE], { type: 'text/javascript' }));
  return { worker: new Worker(url, { type: 'module' }), release: () => URL.revokeObjectURL(url) };
}

// Runs the script on the file. Resolves with every model's result, calling
// onRow as each finishes and onStatus as the stages change. WorkerImpl is for
// tests; the page uses a real Worker.
export function verifyInBrowser({ script, csv, needs = [], onStatus = () => {}, onRow = () => {},
  WorkerImpl = null, indexURL = PYODIDE_URL }) {
  return new Promise((resolve, reject) => {
    if (new Blob([csv]).size > MAX_BYTES) {
      reject(new Error('The file is over 20 MB: too large to fit models on in a browser tab. Download the script instead.'));
      return;
    }
    const { worker, release } = startWorker(WorkerImpl);
    const finish = () => { worker.terminate(); release(); };
    worker.onmessage = (event) => {
      const msg = event.data;
      if (msg.type === 'status') onStatus(msg.text);
      else if (msg.type === 'row') onRow(msg.row);
      else if (msg.type === 'done') { finish(); resolve(msg.results); }
      else if (msg.type === 'error') { finish(); reject(new Error(lastLine(msg.text))); }
    };
    worker.onerror = (event) => {
      finish();
      reject(new Error(event.message || 'The Python worker could not start.'));
    };
    worker.postMessage({ indexURL, packages: packagesFor(needs), script, csv, runner: PY_RUNNER });
  });
}

// A Python traceback ends with the line that says what went wrong.
function lastLine(text) {
  const lines = String(text).trim().split('\n').filter(l => l.trim());
  return lines.length ? lines[lines.length - 1].trim() : 'Python stopped without saying why.';
}

// What the results say about the page's own order: where its first pick
// landed among everything that ran, and what beat it. Scores are compared at
// the four decimals shown, so a difference too small to see is a tie.
export function verdict(results, firstCode) {
  const shown = (x) => Math.round(x * 1e4);
  const ok = results.filter(r => r.status === 'ok' && r.score != null).sort((a, b) => b.score - a.score);
  if (!ok.length) return 'No model finished on this file.';
  const best = ok[0];
  const first = ok.find(r => r.code === firstCode);
  if (!first) {
    const why = results.some(r => r.code === firstCode) ? 'did not finish here' : 'is not a model the script can run';
    return `The page's first pick ${why}; the best of what ran was ${best.name} at ${best.score.toFixed(4)}.`;
  }
  const above = ok.filter(r => shown(r.score) > shown(first.score));
  const level = ok.filter(r => r !== first && shown(r.score) === shown(first.score)).map(r => r.name);
  const tie = level.length ? `, level with ${level.join(' and ')}` : '';
  if (!above.length) return `The page's first pick, ${first.name}, was the best of ${ok.length} on your file${tie}.`;
  return `The page's first pick, ${first.name}, came ${ordinal(above.length + 1)} of ${ok.length} on your file${tie}, `
    + `${(best.score - first.score).toFixed(4)} below ${best.name}. Cross-validation on one file has its own luck: `
    + `compare the gap with the spread beside each score.`;
}

function ordinal(n) {
  const suffix = n % 100 >= 11 && n % 100 <= 13 ? 'th' : ({ 1: 'st', 2: 'nd', 3: 'rd' }[n % 10] ?? 'th');
  return `${n}${suffix}`;
}
