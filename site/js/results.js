// Renders the recommendation sections and the 3D plot.
// Plotly and mermaid are loaded as globals by index.html.

import { paradigmOf, paradigmLabel, rankModels, rankDrifts, rankPipelines, plotCoords } from './recommend.js';

export function renderResults(T, sig, task) {
  const paradigm = paradigmOf(sig);
  const scored = rankModels(T, sig, task);

  document.getElementById('model-note').textContent = scored.length
    ? `Filtered to ${paradigmLabel(paradigm)} models that can produce ${T.TASKS.find(t => t.id === task).label.toLowerCase()}.`
    : 'No model matches.';

  document.getElementById('models').innerHTML = scored.map((m, i) => `
    <article class="rec">
      <div>
        <div class="rec-code" data-model="${m.c}">${m.c}</div>
        <div class="rec-rank">${m.score} of 3 axes</div>
      </div>
      <div>
        <div class="rec-name">${m.n}</div>
        <div class="rec-meta">${T.MODEL_DOMAINS[m.dom]?.name || m.dom} · ${paradigmLabel(m.p)}</div>
        <p class="rec-metaphor">${m.met}</p>
        <div class="matchline">
          ${m.data.map(d => `<span class="chip ${sig.includes(d) ? 'hit' : 'miss'}" data-code="${d}">${d} ${T.CODES[d] ? T.CODES[d].name.toLowerCase() : ''}</span>`).join('')}
        </div>
        <p class="rec-body">${m.mech}</p>
        <p class="rec-body warn">${m.fail}</p>
        <div class="mathline">math:
          ${m.math.map(x => `<span class="chip" data-math="${x}">${x}</span>`).join('')}
        </div>
      </div>
    </article>`).join('');

  const dscored = rankDrifts(T, sig);

  document.getElementById('drift-note').textContent =
    `Matched against your modality and scale. ${sig[4] === 'A54'
      ? 'Your data is non-stationary: drift is expected.'
      : 'Ranked by coordinate fit.'}`;

  document.getElementById('drifts').innerHTML = dscored.map(d => `
    <article class="rec">
      <div><div class="rec-code" data-drift="${d.c}">${d.c}</div></div>
      <div>
        <div class="rec-name">${d.n}</div>
        <div class="rec-meta">${T.DRIFT_DOMAINS[d.domain]?.name || d.domain}</div>
        <p class="rec-metaphor">${d.met}</p>
        <p class="rec-body">${d.mech}</p>
        <p class="rec-body"><span style="color:var(--sage)">Threshold.</span> ${d.thr}</p>
        <p class="rec-body warn">${d.fail}</p>
        <div class="mathline">math:
          ${d.math.map(x => `<span class="chip" data-math="${x}">${x}</span>`).join('')}
        </div>
      </div>
    </article>`).join('');

  const pscored = rankPipelines(T, sig, task);

  document.getElementById('pipeline-note').textContent =
    'Pipelines ranked by fit to your data signature and task.';

  document.getElementById('pipelines').innerHTML = pscored.map(p => `
    <article class="rec">
      <div><div class="rec-code" data-pipeline="${p.c}">${p.c}</div></div>
      <div>
        <div class="rec-name">${p.n}</div>
        <div class="rec-meta">${T.PIPELINE_DOMAINS[p.p]?.name || p.p}</div>
        <p class="rec-metaphor">${p.met}</p>
        <p class="rec-body">${p.mech}</p>
        <p class="rec-body warn">${p.fail}</p>
        <div class="mathline">stages:
          ${p.stages.map(s => `<span class="chip" data-stage="${s}">${s}</span>`).join('')}
        </div>
        <div class="pipeline-diagram" id="diagram-${p.c}"></div>
      </div>
    </article>`).join('');

  pscored.forEach(p => {
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
