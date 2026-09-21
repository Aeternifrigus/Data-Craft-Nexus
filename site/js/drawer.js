// The side drawer that opens when any code is clicked.

import { esc } from './html.js';
import { paradigmLabel } from './recommend.js';

export function initDrawer(T) {
  const drawer = document.getElementById('drawer');
  const scrim = document.getElementById('scrim');

  function show(html) {
    drawer.innerHTML = `<button class="drawer-x" aria-label="Close">×</button>` + html;
    drawer.querySelector('.drawer-x').addEventListener('click', closeDrawer);
    drawer.classList.add('on'); scrim.classList.add('on');
  }
  function closeDrawer() { drawer.classList.remove('on'); scrim.classList.remove('on'); }

  // `html` is already-built markup; plain text goes through text().
  function field(l, html, warn) {
    return `<div class="d-field"><div class="d-label">${esc(l)}</div><div class="d-text${warn ? ' warn' : ''}">${html}</div></div>`;
  }
  const text = (l, t, warn) => field(l, esc(t), warn);
  const chips = (attr, codes) => codes.map(x => `<span class="chip" ${attr}="${esc(x)}">${esc(x)}</span>`).join(' ');
  const head = (code, name, meta) => `<div class="d-code">${esc(code)}</div><div class="d-name">${esc(name)}</div>
    ${meta ? `<div class="rec-meta" style="margin-bottom:12px">${esc(meta)}</div>` : ''}`;

  function openMath(code) {
    const m = T.MATH[code]; if (!m) return;
    const domain = T.MATH_DOMAINS[m.domain];
    show(`${head(code, m.name, domain?.name)}
    <div class="d-formula">${esc(m.f)}</div>
    ${m.intu ? text('The idea', m.intu) : ''}
    ${text('What it computes', m.mech)}
    ${text('Where it misleads', m.fail, true)}`);
  }
  function openCode(code) {
    const c = T.CODES[code]; if (!c) return;
    const axis = T.AXES.find(a => a.n === c.axis);
    show(`${head(code, c.name, axis && `Axis ${axis.n}, ${axis.label}`)}
    ${text('Definition', c.note)}`);
  }
  function openModel(code) {
    const m = T.MODELS.find(x => x.c === code); if (!m) return;
    const dom = T.MODEL_DOMAINS[m.dom];
    show(`${head(code, m.n, dom && `${dom.name} · ${paradigmLabel(m.p)}`)}
    <p class="rec-metaphor">${esc(m.met)}</p>
    ${text('How it works', m.mech)}
    ${m.dfit ? text('Why these coordinates', m.dfit) : ''}
    ${text('Where it breaks', m.fail, true)}
    ${text('What to use instead', m.alt)}
    ${field('Built on', chips('data-math', m.math))}`);
  }
  function openDrift(code) {
    const d = T.DRIFTS.find(x => x.c === code); if (!d) return;
    const dom = T.DRIFT_DOMAINS[d.domain];
    show(`${head(code, d.n, dom?.name)}
    <p class="rec-metaphor">${esc(d.met)}</p>
    ${text('How it works', d.mech)}
    ${text('What counts as drift', d.thr)}
    ${text('Where it misleads', d.fail, true)}
    ${field('Built on', chips('data-math', d.math))}`);
  }
  function openPipeline(code) {
    const p = T.PIPELINES.find(x => x.c === code); if (!p) return;
    const dom = T.PIPELINE_DOMAINS[p.p];
    show(`${head(code, p.n, dom?.name)}
    <p class="rec-metaphor">${esc(p.met)}</p>
    ${text('How it works', p.mech)}
    ${text('Where it breaks', p.fail, true)}
    ${field('Stages', chips('data-stage', p.stages))}`);
  }
  function openStage(code) {
    const s = T.STAGES[code]; if (!s) return;
    show(`${head(code, s.name)}
    ${text('Definition', s.note)}`);
  }

  document.addEventListener('click', e => {
    const m = e.target.closest('[data-math]');
    const c = e.target.closest('[data-code]');
    const mo = e.target.closest('[data-model]');
    const dr = e.target.closest('[data-drift]');
    const pl = e.target.closest('[data-pipeline]');
    const st = e.target.closest('[data-stage]');
    if (m) openMath(m.dataset.math);
    else if (c) openCode(c.dataset.code);
    else if (mo) openModel(mo.dataset.model);
    else if (dr) openDrift(dr.dataset.drift);
    else if (pl) openPipeline(pl.dataset.pipeline);
    else if (st) openStage(st.dataset.stage);
  });
  scrim.addEventListener('click', closeDrawer);
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeDrawer(); });
}
