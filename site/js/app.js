// Entry point: loads the taxonomy and wires up the page.

import { loadTaxonomy, loadSample } from './taxonomy.js';
import { parseCSV, decodeBytes, delimiterName, MAX_ROWS } from './csv.js';
import { esc } from './html.js';
import { profileData, signature, measuredAxes } from './profile.js';
import { renderResults } from './results.js';
import { initDrawer } from './drawer.js';
import { buildLibrary, initLibrarySearch } from './library.js';
import { buildEvidence } from './evidence.js';

// mode and labels describe how the thing will run once it is built. They
// start at the common case and can be changed; the other three have no
// default because the data cannot imply them.
const state = {
  T: null, rows: [], cols: [], profile: null,
  decl: { target: null, task: null, order: null, mode: 'batch', labels: 'delayed' },
};

const OPERATING = {
  mode: [
    ['batch', 'In batches, on a schedule'],
    ['streaming', 'As a stream, row by row'],
  ],
  labels: [
    ['immediate', 'Straight away'],
    ['delayed', 'Later, days or weeks'],
    ['none', 'Never'],
  ],
};

function buildReadout(T) {
  document.getElementById('readout').innerHTML = T.AXES.map(a => `
    <div class="slot" id="slot-${a.n}">
      <div class="slot-code" id="code-${a.n}">-, -</div>
      <div>
        <div class="slot-val" id="val-${a.n}"></div>
        <div class="slot-axis">${a.n}. ${esc(a.label)}</div>
      </div>
    </div>`).join('');
}

function setSlot(n, code, declared) {
  const s = document.getElementById('slot-' + n);
  document.getElementById('code-' + n).textContent = code;
  document.getElementById('val-' + n).textContent = state.T.CODES[code] ? state.T.CODES[code].name : '';
  s.classList.add('filled');
  if (declared) s.classList.add('declared');
}

// What was detected while reading, shown under the row count.
function readNote(parsed, encoding) {
  const parts = [`${delimiterName(parsed.delimiter)}-separated`];
  if (encoding && encoding !== 'UTF-8') parts.push(`${encoding} text`);
  if (parsed.decimalComma) parts.push('decimal commas read as numbers');
  if (parsed.truncated) parts.push(`only the first ${MAX_ROWS.toLocaleString()} rows are used`);
  return parts.join(' · ');
}

function ingest(text, encoding) {
  const parsed = parseCSV(text);
  const { head, body } = parsed;
  const drop = document.getElementById('drop-msg');
  if (head.length === 0 || body.length === 0) {
    drop.innerHTML = `<p><b>No rows found.</b></p>
      <p style="margin-top:7px;font-size:12px">The file needs a header row and at least one data row.</p>`;
    return;
  }
  state.cols = head; state.rows = body;
  state.profile = profileData(head, body);
  const { profile } = state;
  showMeasuredAxes(null);
  drop.innerHTML = `<p><b>${profile.n.toLocaleString()} rows · ${profile.feat} columns</b> read</p>
    <p style="margin-top:7px;font-size:12px">${esc(readNote(parsed, encoding))}</p>
    <p style="margin-top:4px;font-size:12px">Axes 3, 4 and 6 measured. Three more need your intent.</p>`;
  buildDeclarations();
  document.getElementById('declare').classList.add('on');
  document.getElementById('declare').scrollIntoView({ block: 'start' });
}

// Axes 3, 4 and 6 leave the target column out, so they are measured again
// whenever the target changes.
function showMeasuredAxes(target) {
  const { a3, a4, a6 } = measuredAxes(state.profile, target);
  setSlot(3, a3); setSlot(4, a4); setSlot(6, a6);
}

function buildDeclarations() {
  const { decl, profile, T } = state;
  const t = document.getElementById('q-target');
  t.innerHTML = '';
  const sel = document.createElement('select');
  sel.className = 'opt';
  sel.innerHTML = `<option value="">choose a column…</option>` +
    state.cols.map(c => `<option value="${esc(c)}">${esc(c)}</option>`).join('') +
    `<option value="__none__">nothing, there is no target</option>`;
  sel.addEventListener('change', () => {
    decl.target = sel.value || null;
    setSlot(1, decl.target === '__none__' ? 'A12' : 'A11', true);
    showMeasuredAxes(decl.target === '__none__' ? null : decl.target);
    checkReady();
  });
  t.appendChild(sel);

  const tk = document.getElementById('q-task');
  tk.innerHTML = T.TASKS.map(x => `<button class="opt" data-task="${esc(x.id)}">${esc(x.label)}</button>`).join('');
  tk.querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
    tk.querySelectorAll('button').forEach(o => o.classList.remove('sel'));
    b.classList.add('sel'); decl.task = b.dataset.task; checkReady();
  }));

  for (const [key, options] of Object.entries(OPERATING)) {
    const box = document.getElementById(`q-${key}`);
    box.innerHTML = options.map(([value, label]) =>
      `<button class="opt${decl[key] === value ? ' sel' : ''}" data-value="${esc(value)}">${esc(label)}</button>`).join('');
    box.querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
      box.querySelectorAll('button').forEach(o => o.classList.remove('sel'));
      b.classList.add('sel');
      decl[key] = b.dataset.value;
    }));
  }

  const od = document.getElementById('q-order');
  const hint = profile.dateCols.length ? ` (a date column was found: ${esc(profile.dateCols[0].name)})` : '';
  od.innerHTML =
    `<button class="opt" data-o="A22">Yes, it is a sequence${hint}</button>
     <button class="opt" data-o="A21">No, rows are independent</button>`;
  od.querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
    od.querySelectorAll('button').forEach(o => o.classList.remove('sel'));
    b.classList.add('sel'); decl.order = b.dataset.o;
    setSlot(2, decl.order, true); checkReady();
  }));
}

function checkReady() {
  const { decl } = state;
  document.getElementById('run').disabled = !(decl.target && decl.task && decl.order);
}

function initIntake() {
  const drop = document.getElementById('drop'), fileIn = document.getElementById('file');
  const readFile = f => {
    const fr = new FileReader();
    fr.onload = () => {
      const { text, encoding } = decodeBytes(fr.result);
      ingest(text, encoding);
    };
    fr.readAsArrayBuffer(f);
  };
  drop.addEventListener('click', () => fileIn.click());
  drop.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileIn.click(); } });
  drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', e => {
    e.preventDefault(); drop.classList.remove('over');
    const f = e.dataTransfer.files[0]; if (f) readFile(f);
  });
  fileIn.addEventListener('change', e => {
    const f = e.target.files[0]; if (f) readFile(f);
    fileIn.value = ''; // so choosing the same file again still triggers a read
  });

  document.getElementById('sample').addEventListener('click', async () => {
    ingest(await loadSample(), 'UTF-8');
  });

  document.getElementById('run').addEventListener('click', () => {
    const sig = signature(state.profile, state.decl);
    setSlot(5, sig.codes[4], sig.balance != null);
    showFlags(sig);
    renderResults(state.T, sig, state.decl.task, state.profile);
    document.getElementById('results').classList.add('on');
    document.getElementById('results').scrollIntoView({ block: 'start' });
  });
}

// A measured code that didn't fit in a slot (class balance and drift both
// live on axis 5) is shown under it rather than dropped.
function showFlags(sig) {
  const slot = document.getElementById('val-5');
  const extra = sig.flags.map(f => `${f} ${state.T.CODES[f]?.name ?? ''}`.trim()).join(', ');
  slot.textContent = state.T.CODES[sig.codes[4]]?.name + (extra ? ` · ${extra}` : '');
}

function initNav() {
  document.querySelectorAll('.navlink').forEach(b => b.addEventListener('click', () => {
    document.querySelectorAll('.navlink').forEach(o => o.classList.remove('on'));
    b.classList.add('on');
    document.querySelectorAll('.view').forEach(v => v.classList.remove('on'));
    document.getElementById('view-' + b.dataset.view).classList.add('on');
  }));
}

async function main() {
  try {
    state.T = await loadTaxonomy();
  } catch (err) {
    document.getElementById('drop-msg').innerHTML =
      `<p><b>The taxonomy could not be loaded.</b></p>
       <p style="margin-top:7px;font-size:12px">To open it straight from disk, use <code>dist/index.html</code>. The files in <code>site/</code> need a server: <code>npm run serve</code></p>`;
    throw err;
  }
  buildReadout(state.T);
  initNav();
  initIntake();
  initDrawer(state.T);
  buildLibrary(state.T);
  initLibrarySearch();
  buildEvidence(state.T);
}

main();
