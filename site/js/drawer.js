// The side drawer that opens when any code is clicked.

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

  function field(l, t, warn) {
    return `<div class="d-field"><div class="d-label">${l}</div><div class="d-text${warn ? ' warn' : ''}">${t}</div></div>`;
  }

  function openMath(code) {
    const m = T.MATH[code]; if (!m) return;
    const domain = T.MATH_DOMAINS[m.domain];
    show(`<div class="d-code">${code}</div><div class="d-name">${m.name}</div>
    ${domain ? `<div class="rec-meta" style="margin-bottom:12px">${domain.name}</div>` : ''}
    <div class="d-formula">${m.f}</div>
    ${m.intu ? field('The idea', m.intu) : ''}
    ${field('What it computes', m.mech)}
    ${field('Where it misleads', m.fail, true)}`);
  }
  function openCode(code) {
    const c = T.CODES[code]; if (!c) return;
    const axis = T.AXES.find(a => a.n === c.axis);
    show(`<div class="d-code">${code}</div><div class="d-name">${c.name}</div>
    ${axis ? `<div class="rec-meta" style="margin-bottom:12px">Axis ${axis.n}, ${axis.label}</div>` : ''}
    ${field('Definition', c.note)}`);
  }
  function openModel(code) {
    const m = T.MODELS.find(x => x.c === code); if (!m) return;
    const dom = T.MODEL_DOMAINS[m.dom];
    show(`<div class="d-code">${code}</div><div class="d-name">${m.n}</div>
    ${dom ? `<div class="rec-meta" style="margin-bottom:12px">${dom.name} · ${paradigmLabel(m.p)}</div>` : ''}
    <p class="rec-metaphor">${m.met}</p>
    ${field('How it works', m.mech)}
    ${m.dfit ? field('Why these coordinates', m.dfit) : ''}
    ${field('Where it breaks', m.fail, true)}
    ${field('What to use instead', m.alt)}
    ${field('Built on', m.math.map(x => `<span class="chip" data-math="${x}">${x}</span>`).join(' '))}`);
  }
  function openDrift(code) {
    const d = T.DRIFTS.find(x => x.c === code); if (!d) return;
    const dom = T.DRIFT_DOMAINS[d.domain];
    show(`<div class="d-code">${code}</div><div class="d-name">${d.n}</div>
    ${dom ? `<div class="rec-meta" style="margin-bottom:12px">${dom.name}</div>` : ''}
    <p class="rec-metaphor">${d.met}</p>
    ${field('How it works', d.mech)}
    ${field('What counts as drift', d.thr)}
    ${field('Where it misleads', d.fail, true)}
    ${field('Built on', d.math.map(x => `<span class="chip" data-math="${x}">${x}</span>`).join(' '))}`);
  }
  function openPipeline(code) {
    const p = T.PIPELINES.find(x => x.c === code); if (!p) return;
    const dom = T.PIPELINE_DOMAINS[p.p];
    show(`<div class="d-code">${code}</div><div class="d-name">${p.n}</div>
    ${dom ? `<div class="rec-meta" style="margin-bottom:12px">${dom.name}</div>` : ''}
    <p class="rec-metaphor">${p.met}</p>
    ${field('How it works', p.mech)}
    ${field('Where it breaks', p.fail, true)}
    ${field('Stages', p.stages.map(s => `<span class="chip" data-stage="${s}">${s}</span>`).join(' '))}`);
  }
  function openStage(code) {
    const s = T.STAGES[code]; if (!s) return;
    show(`<div class="d-code">${code}</div><div class="d-name">${s.name}</div>
    ${field('Definition', s.note)}`);
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
