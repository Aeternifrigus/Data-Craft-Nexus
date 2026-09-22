// Renders the recommendation sections and the 3D plot.
// Plotly and mermaid are loaded as globals by index.html.

import { esc } from './html.js';
import { matchCodes } from './profile.js';
import { paradigmOf, paradigmLabel, rankModels, rankDrifts, rankPipelines, plotCoords } from './recommend.js';
import { evidenceFor, evidenceSentence } from './evidence.js';

// A tie means the data can't separate those models. Say so rather than
// letting the order on the page look like a verdict.
function tieNote(result, noun = 'models') {
  if (result.tied > result.items.length) {
    return ` ${result.tied} ${noun} match the data equally well, so the order below is arbitrary: the coordinates can't separate them.`;
  }
  if (result.tied > 1) {
    return ` The top ${result.tied} match the data equally well, so the order between them is arbitrary.`;
  }
  return '';
}

// Where to read more about a recommendation, outside this site.
function refLinks(entry) {
  const rows = [entry.ref, entry.docs].filter(Boolean);
  if (!rows.length) return '';
  return `<div class="mathline">read: ${rows.map(l =>
    `<a href="${esc(l.url)}" target="_blank" rel="noopener noreferrer">${esc(l.label)} &#8599;</a>`).join(' · ')}</div>`;
}

function chip(code, kind, codes) {
  const attr = kind === 'stage' ? 'data-stage' : kind === 'pipeline' ? 'data-pipeline' : kind === 'math' ? 'data-math' : 'data-code';
  const cls = codes ? ` ${codes.includes(code) ? 'hit' : 'miss'}` : '';
  return `<span class="chip${cls}" ${attr}="${esc(code)}">${esc(code)}</span>`;
}

export function renderResults(T, sig, task) {
  const codes = matchCodes(sig);
  const models = rankModels(T, sig, task);
  const taskLabel = T.TASKS.find(t => t.id === task).label.toLowerCase();

  const note = document.getElementById('model-note');
  if (models.items.length) {
    note.textContent = `${models.candidates} of the ${paradigmLabel(paradigmOf(sig))} models can produce ${taskLabel} for data shaped like yours.${tieNote(models)}`;
  } else if (models.ruledOut.length) {
    note.textContent = `No model fits. Every model that could produce ${taskLabel} is ruled out by your data: ${models.ruledOut.slice(0, 3).map(m => `${m.n} ${m.why}`).join('; ')}.`;
  } else {
    note.textContent = `No model in the taxonomy produces ${taskLabel}.`;
  }

  document.getElementById('models').innerHTML = models.items.map(m => `
    <article class="rec">
      <div>
        <div class="rec-code" data-model="${esc(m.c)}">${esc(m.c)}</div>
        <div class="rec-rank">${m.score} of ${m.of} coordinates</div>
      </div>
      <div>
        <div class="rec-name">${esc(m.n)}</div>
        <div class="rec-meta">${esc(T.MODEL_DOMAINS[m.dom]?.name || m.dom)} · ${paradigmLabel(m.p)}</div>
        <p class="rec-metaphor">${esc(m.met)}</p>
        <div class="matchline">
          ${m.data.map(d => `<span class="chip ${codes.includes(d) ? 'hit' : 'miss'}" data-code="${esc(d)}">${esc(d)} ${T.CODES[d] ? esc(T.CODES[d].name.toLowerCase()) : ''}</span>`).join('')}
        </div>
        <p class="rec-body">${esc(m.mech)}</p>
        ${m.caution ? `<p class="rec-body caution">Caution. ${esc(m.caution)}</p>` : ''}
        ${(() => { const e = evidenceFor(T, m.c, task); return e
          ? `<p class="rec-body evidence">Measured. ${esc(evidenceSentence(e))}</p>` : ''; })()}
        <p class="rec-body warn">${esc(m.fail)}</p>
        <div class="mathline">math:
          ${m.math.map(x => chip(x, 'math')).join('')}
        </div>
        ${refLinks(m)}
      </div>
    </article>`).join('');

  // What the data ruled out, and why. This is the part a reader learns from.
  const ruled = document.getElementById('ruled-out');
  if (models.ruledOut.length) {
    const shown = models.ruledOut.slice(0, 8);
    const rest = models.ruledOut.length - shown.length;
    ruled.innerHTML = `<p class="sect-note" style="margin-bottom:12px">Ruled out for this data:</p>` +
      `<ul class="ruled">${shown.map(m =>
        `<li><span class="rec-code" data-model="${esc(m.c)}">${esc(m.c)}</span> ${esc(m.n)}: ${esc(m.why)}</li>`).join('')}` +
      (rest ? `<li>and ${rest} more</li>` : '') + `</ul>`;
  } else {
    ruled.innerHTML = '';
  }

  const drifts = rankDrifts(T, sig);
  const driftMsg = sig.drift
    ? `Worst shift between the first and second half of your file: PSI ${sig.drift.psi.toFixed(2)} on ${sig.drift.column}${sig.drift.scope === 'target' ? ' (the target itself)' : ''}, ${sig.drift.psi > 0.25 ? 'above' : 'below'} the 0.25 cutoff in DR-M2.`
    : 'Drift across the file could not be measured: too few rows, or no column steady enough to compare. These are matched on modality and scale only.';
  document.getElementById('drift-note').textContent = driftMsg + tieNote(drifts, 'drift checkers');

  document.getElementById('drifts').innerHTML = drifts.items.map(d => `
    <article class="rec">
      <div><div class="rec-code" data-drift="${esc(d.c)}">${esc(d.c)}</div></div>
      <div>
        <div class="rec-name">${esc(d.n)}</div>
        <div class="rec-meta">${esc(T.DRIFT_DOMAINS[d.domain]?.name || d.domain)}</div>
        <p class="rec-metaphor">${esc(d.met)}</p>
        <p class="rec-body">${esc(d.mech)}</p>
        <p class="rec-body"><span style="color:var(--sage)">Threshold.</span> ${esc(d.thr)}</p>
        <p class="rec-body warn">${esc(d.fail)}</p>
        <div class="mathline">math:
          ${d.math.map(x => chip(x, 'math')).join('')}
        </div>
        ${refLinks(d)}
      </div>
    </article>`).join('');

  const pipelines = rankPipelines(T, sig, task);
  document.getElementById('pipeline-note').textContent =
    'Pipelines ranked by fit to your data signature and task.' + tieNote(pipelines, 'pipelines');

  const pipelineCodes = new Set(T.PIPELINES.map(p => p.c));
  document.getElementById('pipelines').innerHTML = pipelines.items.map(p => `
    <article class="rec">
      <div><div class="rec-code" data-pipeline="${esc(p.c)}">${esc(p.c)}</div></div>
      <div>
        <div class="rec-name">${esc(p.n)}</div>
        <div class="rec-meta">${esc(T.PIPELINE_DOMAINS[p.p]?.name || p.p)}</div>
        <p class="rec-metaphor">${esc(p.met)}</p>
        <p class="rec-body">${esc(p.mech)}</p>
        <p class="rec-body warn">${esc(p.fail)}</p>
        <div class="mathline">stages:
          ${p.stages.map(s => chip(s, pipelineCodes.has(s) ? 'pipeline' : 'stage')).join('')}
        </div>
        ${refLinks(p)}
        <div class="pipeline-diagram" id="diagram-${esc(p.c)}"></div>
      </div>
    </article>`).join('');

  pipelines.items.forEach(p => {
    const el = document.getElementById('diagram-' + p.c);
    if (el && p.flowchart) {
      mermaid.render('mermaid-' + p.c, p.flowchart).then(({ svg }) => {
        el.innerHTML = svg;
      }).catch(() => {
        el.innerHTML = '<p style="color:var(--clay);font-size:12px">Flowchart unavailable.</p>';
      });
    }
  });

  plotSpace(T, sig);
}

function plotSpace(T, sig) {
  const { x: xi, y: yi, z: zi } = plotCoords(sig);

  const ref = {
    type: 'scatter3d', mode: 'markers', name: 'reference',
    x: T.REFERENCE.map(r => r.x), y: T.REFERENCE.map(r => r.y), z: T.REFERENCE.map(r => r.z),
    text: T.REFERENCE.map(r => r.t), hoverinfo: 'text',
    marker: { size: 6, color: '#7FFF00', opacity: .5 }
  };
  const you = {
    type: 'scatter3d', mode: 'markers+text', name: 'your data',
    x: [xi], y: [yi], z: [zi], text: ['your specimen'], textposition: 'top center',
    textfont: { color: '#E0A458', family: 'JetBrains Mono', size: 11 },
    hoverinfo: 'text',
    marker: { size: 13, color: '#E0A458', symbol: 'diamond',
      line: { color: '#0D1626', width: 2 } }
  };

  const ax = t => ({
    title: { text: t, font: { color: '#7C879B', size: 10, family: 'JetBrains Mono' } },
    gridcolor: '#25344E', zerolinecolor: '#25344E', showbackground: false,
    tickfont: { color: '#7C879B', size: 9, family: 'JetBrains Mono' }
  });

  Plotly.newPlot('plot', [ref, you], {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    margin: { l: 0, r: 0, t: 0, b: 0 },
    showlegend: false,
    scene: {
      xaxis: { ...ax('1 · supervision'), tickmode: 'array', tickvals: [0, 1, 2, 3], ticktext: ['labeled', 'unlabeled', 'weak', 'semi'] },
      yaxis: { ...ax('2 · structure'), tickmode: 'array', tickvals: [0, 1, 3, 5], ticktext: ['i.i.d.', 'sequential', 'graph', 'set'] },
      zaxis: { ...ax('3 · modality'), tickmode: 'array', tickvals: [0, 2, 3, 4, 5, 6, 7], ticktext: ['numeric', 'ordinal', 'text', 'image', 'audio', 'graph', 'mixed'] }
    }
  }, { displayModeBar: false, responsive: true });
}
