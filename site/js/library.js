// "The reference" view: every code in the taxonomy, searchable.

import { esc } from './html.js';

function libCard(code, name, note, attr) {
  return `<button class="lib-card" ${attr}>
      <div class="lib-code">${esc(code)}</div>
      <div class="lib-name">${esc(name)}</div>
      <div class="lib-note">${esc(note)}</div>
    </button>`;
}

export function buildLibrary(T) {
  let html = '';
  html += `<div class="lib-group"><h3>Data axes</h3>
    <p class="lib-sub">The six coordinates every dataset gets scored on.</p>`;
  T.AXES.forEach(a => {
    const codes = Object.keys(T.CODES).filter(c => c.startsWith('A' + a.n));
    if (!codes.length) return;
    html += `<div class="lib-subheading">Axis ${a.n}, ${esc(a.label)}</div>
      <div class="lib-grid">${codes.map(c =>
        libCard(c, T.CODES[c].name, T.CODES[c].note, `data-code="${esc(c)}"`)).join('')}</div>`;
  });
  html += `</div>`;

  html += `<div class="lib-group"><h3>Math</h3>
    <p class="lib-sub">Every formula, organized by domain.</p>`;
  Object.keys(T.MATH_DOMAINS).forEach(dk => {
    const domain = T.MATH_DOMAINS[dk];
    const codes = Object.keys(T.MATH).filter(m => T.MATH[m].domain === dk);
    if (!codes.length) return;
    html += `<div class="lib-subheading">${esc(dk)}, ${esc(domain.name)}</div>
      <p class="lib-sub" style="margin-bottom:8px">${esc(domain.note)}</p>
      <div class="lib-grid">${codes.map(c =>
        libCard(c, T.MATH[c].name, T.MATH[c].f, `data-math="${esc(c)}"`)).join('')}</div>`;
  });
  html += `</div>`;

  html += `<div class="lib-group"><h3>Models</h3>
    <p class="lib-sub">Every model, organized by architecture family.</p>`;
  Object.keys(T.MODEL_DOMAINS).forEach(dk => {
    const domain = T.MODEL_DOMAINS[dk];
    const codes = T.MODELS.filter(m => m.dom === dk);
    if (!codes.length) return;
    html += `<div class="lib-subheading">${esc(domain.name)}</div>
      <p class="lib-sub" style="margin-bottom:8px">${esc(domain.note)}</p>
      <div class="lib-grid">${codes.map(m =>
        libCard(m.c, m.n, m.met, `data-model="${esc(m.c)}"`)).join('')}</div>`;
  });
  html += `</div>`;

  html += `<div class="lib-group"><h3>Drift checkers</h3>
    <p class="lib-sub">Every detector, organized by approach.</p>`;
  Object.keys(T.DRIFT_DOMAINS).forEach(dk => {
    const domain = T.DRIFT_DOMAINS[dk];
    const codes = T.DRIFTS.filter(d => d.domain === dk);
    if (!codes.length) return;
    html += `<div class="lib-subheading">${esc(dk)}, ${esc(domain.name)}</div>
      <p class="lib-sub" style="margin-bottom:8px">${esc(domain.note)}</p>
      <div class="lib-grid">${codes.map(d =>
        libCard(d.c, d.n, d.met, `data-drift="${esc(d.c)}"`)).join('')}</div>`;
  });
  html += `</div>`;

  html += `<div class="lib-group"><h3>Pipelines</h3>
    <p class="lib-sub">Every pipeline type, organized by function.</p>`;
  Object.keys(T.PIPELINE_DOMAINS).forEach(dk => {
    const domain = T.PIPELINE_DOMAINS[dk];
    const codes = T.PIPELINES.filter(p => p.p === dk);
    if (!codes.length) return;
    html += `<div class="lib-subheading">${esc(domain.name)}</div>
      <p class="lib-sub" style="margin-bottom:8px">${esc(domain.note)}</p>
      <div class="lib-grid">${codes.map(p =>
        libCard(p.c, p.n, p.met, `data-pipeline="${esc(p.c)}"`)).join('')}</div>`;
  });
  html += `</div>`;

  html += `<div class="lib-group"><h3>Stage library</h3>
    <p class="lib-sub">Every reusable stage.</p>
    <div class="lib-grid">${Object.keys(T.STAGES).map(s =>
      libCard(s, T.STAGES[s].name, T.STAGES[s].note, `data-stage="${esc(s)}"`)).join('')}</div></div>`;

  document.getElementById('lib-body').innerHTML = html;
}

export function initLibrarySearch() {
  document.getElementById('lib-search').addEventListener('input', e => {
    const q = e.target.value.trim().toLowerCase();
    document.querySelectorAll('#lib-body .lib-card').forEach(card => {
      const hit = card.textContent.toLowerCase().includes(q);
      card.style.display = hit ? '' : 'none';
    });
    document.querySelectorAll('#lib-body .lib-subheading').forEach(h => {
      const group = h.nextElementSibling;
      const nextCards = group ? group.querySelectorAll('.lib-card') : [];
      const any = [...nextCards].some(c => c.style.display !== 'none');
      h.style.display = any ? '' : 'none';
    });
    document.querySelectorAll('#lib-body .lib-group').forEach(g => {
      const any = [...g.querySelectorAll('.lib-card')].some(c => c.style.display !== 'none');
      g.style.display = any ? '' : 'none';
    });
  });
}
