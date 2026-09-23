// Renders the recommendation sections and the 3D plot.
// Plotly and mermaid are loaded as globals by index.html.

import { esc } from './html.js';
import { matchCodes } from './profile.js';
import { paradigmOf, paradigmLabel, rankModels, rankDrifts, rankPipelines, plotCoords } from './recommend.js';
import { evidenceFor, evidenceSentence } from './evidence.js';
import { rankingProvenance } from './ranking.js';
import { coverageNotes, metaFeatures, nearestDatasets, performanceOn, winnerAmong } from './nearest.js';

// A tie means the data can't separate those models. Say so rather than
// letting the order on the page look like a verdict.
function tieNote(result, noun = 'models') {
  if (result.tied <= 1) return '';
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

export function renderResults(T, sig, task, profile) {
  const codes = matchCodes(sig);

  // Where this dataset sits among the benchmark datasets, measured the same
  // way. The order can depend on it, so it is measured first.
  const meta = profile ? metaFeatures(profile, sig.target ?? null) : null;
  if (meta && sig.drift) meta.drift_psi = Math.min(sig.drift.psi, 5);
  const models = rankModels(T, sig, task, 4, meta);
  const neighbours = meta ? nearestDatasets(T, meta, task) : [];
  const taskLabel = T.TASKS.find(t => t.id === task).label.toLowerCase();

  const provenance = rankingProvenance(T, task);
  const orderedBy = models.rankedBy === 'evidence' && provenance
    ? (provenance.chosen === 'prior_knn'
      ? ` Ordered by what each model was worth on ${provenance.datasets} benchmark datasets, weighted toward the ones most like yours, not by how many coordinates it matches.`
      : ` Ordered by what each model was worth on ${provenance.datasets} benchmark datasets, not by how many coordinates it matches.`)
    : ' Ordered by coordinates matched: the benchmark has not covered this task, so there is nothing measured to rank them by.';

  const note = document.getElementById('model-note');
  if (models.items.length) {
    note.textContent = `${models.candidates} of the ${paradigmLabel(paradigmOf(sig))} models can produce ${taskLabel} for data shaped like yours.${orderedBy}${tieNote(models)}`;
  } else if (models.ruledOut.length) {
    note.textContent = `No model fits. Every model that could produce ${taskLabel} is ruled out by your data: ${models.ruledOut.slice(0, 3).map(m => `${m.n} ${m.why}`).join('; ')}.`;
  } else {
    note.textContent = `No model in the taxonomy produces ${taskLabel}.`;
  }

  document.getElementById('models').innerHTML = models.items.map(m => `
    <article class="rec">
      <div>
        <div class="rec-code" data-model="${esc(m.c)}">${esc(m.c)}</div>
        <div class="rec-rank">${m.evidenceScore == null
          ? `${m.score} of ${m.of} coordinates`
          : `evidence ${m.evidenceScore.toFixed(2)}<span class="rec-rank-sub">${m.score} of ${m.of} coordinates</span>`}</div>
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
        ${(() => { const n = neighbours.length ? performanceOn(neighbours, m.c) : null; return n
          ? `<p class="rec-body evidence">Nearby. On the ${n.datasets} benchmark datasets closest to yours it was best
             ${n.wins} ${n.wins === 1 ? 'time' : 'times'}, typically ${n.medianGap.toFixed(3)} below the winner.</p>` : ''; })()}
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

  renderRuledOut('drift-ruled', drifts.ruledOut, 'drift checkers');

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

  renderRuledOut('pipeline-ruled', pipelines.ruledOut, 'pipelines');

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

  renderCoverage(coverageNotes(T, meta, sig.rows, task, neighbours));
  renderNeighbours(T, neighbours, task);
  plotSpace(T, sig, meta, neighbours);
}

// Outside what was tested: said once, above the map, before any number.
function renderCoverage(notes) {
  const el = document.getElementById('coverage');
  if (!el) return;
  el.hidden = !notes.length;
  el.innerHTML = notes.length
    ? `<p class="coverage-h">Outside what the benchmark tested</p>${notes.map(n =>
      `<p class="coverage-p" data-kind="${esc(n.kind)}">${esc(n.text)}</p>`).join('')}`
    : '';
}

// The benchmark datasets as a map, with this one placed on it.
//
// The plot used to position a dataset by axes 1 to 3, where the first two are
// the same for every labelled table with independent rows: three different
// uploads could land on the same point, scattered among ten invented
// reference datasets. It now uses measured properties that actually differ,
// and the reference points are the benchmark datasets the recommendations
// were tested on.
function plotSpace(T, sig, meta, neighbours) {
  const ev = T.EVIDENCE;
  const points = (ev?.datasets ?? []).filter(d => d.meta);
  const nearest = new Set(neighbours.map(d => d.dataset));
  const note = document.getElementById('plot-note');
  if (note && points.length) {
    note.textContent = `Your data placed among the ${points.length} datasets the recommendations were tested on, by size, ` +
      'shape and how much of the table is numeric. The brighter points are the ones closest to yours, and what won on ' +
      'them is listed below.';
  }

  if (!points.length || !meta) {
    document.getElementById('plot').innerHTML =
      '<p class="sect-note">No benchmark datasets are recorded in this copy, so there is nothing to compare against.</p>';
    return;
  }

  const axes = { x: 'log_rows', y: 'log_features', z: 'numeric_share' };
  const trace = (list, name, color, size, symbol) => ({
    type: 'scatter3d', mode: 'markers', name,
    x: list.map(d => d.meta[axes.x]), y: list.map(d => d.meta[axes.y]), z: list.map(d => d.meta[axes.z]),
    text: list.map(d => `${d.dataset}<br>${d.rows} rows x ${d.features} columns<br>` +
      `won by ${d.best.name} (${d.best.score})`),
    hoverinfo: 'text',
    marker: { size, color, opacity: name === 'other datasets' ? 0.45 : 0.9, symbol },
  });

  const others = points.filter(d => !nearest.has(d.dataset));
  const near = points.filter(d => nearest.has(d.dataset));
  const you = {
    type: 'scatter3d', mode: 'markers+text', name: 'your data',
    x: [meta[axes.x]], y: [meta[axes.y]], z: [meta[axes.z]],
    text: ['your data'], textposition: 'top center',
    textfont: { color: '#E0A458', family: 'JetBrains Mono', size: 11 },
    hovertext: [`your data<br>${sig.rows} rows x ${sig.features} columns`], hoverinfo: 'text',
    marker: { size: 13, color: '#E0A458', symbol: 'diamond', line: { color: '#0D1626', width: 2 } },
  };

  const axis = (title) => ({
    title: { text: title, font: { color: '#7C879B', size: 10, family: 'JetBrains Mono' } },
    gridcolor: '#25344E', zerolinecolor: '#25344E', showbackground: false,
    tickfont: { color: '#7C879B', size: 9, family: 'JetBrains Mono' },
  });

  Plotly.newPlot('plot', [
    trace(others, 'other datasets', '#5CBF00', 5, 'circle'),
    trace(near, 'closest to yours', '#7FFF00', 9, 'circle'),
    you,
  ], {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    margin: { l: 0, r: 0, t: 0, b: 0 }, showlegend: false,
    scene: {
      xaxis: { ...axis('rows (log)') },
      yaxis: { ...axis('columns (log)') },
      zaxis: { ...axis('share of numeric columns'), range: [-0.05, 1.05] },
    },
  }, { displayModeBar: false, responsive: true });
}

// What the answers about how it will run excluded, and why. Saying "these
// eleven cannot work here because labels never arrive" teaches more than
// quietly showing four that can.
function renderRuledOut(id, ruledOut, noun) {
  const el = document.getElementById(id);
  if (!el) return;
  if (!ruledOut || !ruledOut.length) { el.innerHTML = ''; return; }
  const shown = ruledOut.slice(0, 6);
  const rest = ruledOut.length - shown.length;
  el.innerHTML = `<p class="sect-note" style="margin-top:22px">Ruled out by how it will run
      (${ruledOut.length} ${noun}):</p>
    <ul class="ruled">${shown.map(d =>
      `<li><span class="rec-code" data-${noun.startsWith('drift') ? 'drift' : 'pipeline'}="${esc(d.c)}">${esc(d.c)}</span>
        ${esc(d.n)}: ${esc(d.why)}</li>`).join('')}${rest ? `<li>and ${rest} more</li>` : ''}</ul>`;
}

// The neighbours in words, under the plot: what won on datasets like this one.
function renderNeighbours(T, neighbours, task) {
  const el = document.getElementById('neighbours');
  if (!el) return;
  if (!neighbours.length) {
    el.innerHTML = `<p class="sect-note">The benchmark only covers classification and regression, so there are no
      comparable datasets to show for this task yet.</p>`;
    return;
  }

  const winner = winnerAmong(neighbours);
  const names = Object.fromEntries(T.MODELS.map(m => [m.c, m.n]));
  const summary = winner
    ? `On the ${winner.of} benchmark datasets closest to yours, ${esc(names[winner.model] ?? winner.model)} won
       ${winner.wins} of them.`
    : '';

  el.innerHTML = `<p class="sect-note">${summary} Closeness is measured on the same properties for both: size, shape,
      how much of the table is numeric, how much is missing or noisy, and how uneven the target is.</p>
    <table class="ev-table">
      <thead><tr><th>dataset</th><th>shape</th><th>won by</th><th>score</th><th>distance</th></tr></thead>
      <tbody>${neighbours.map(d => `<tr>
        <td>${esc(d.dataset)}<span class="ev-sub">${esc(d.task)}</span></td>
        <td>${d.rows} × ${d.features}</td>
        <td><span class="rec-code" data-model="${esc(d.best.model)}">${esc(d.best.model)}</span> ${esc(d.best.name)}</td>
        <td>${d.best.score.toFixed(3)}</td>
        <td>${d.distance.toFixed(2)}</td>
      </tr>`).join('')}</tbody>
    </table>`;
}


