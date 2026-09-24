// Plain names first. The page leads with what a thing is called ("Labeled",
// "Random Forest", "Train"), and keeps the taxonomy's code ("A11", "TR2",
// "PL-S9") small beside it and on hover: the code is what opens the
// definition, and what a saved reading or the reference view points at.
// Pure: returns markup, touches no DOM.

import { esc } from './html.js';

// The attribute the definition drawer (drawer.js) listens for, per kind.
const ATTR = {
  code: 'data-code', math: 'data-math', stage: 'data-stage', pipeline: 'data-pipeline', model: 'data-model',
  drift: 'data-drift',
};

export function nameOf(T, kind, code) {
  switch (kind) {
    case 'code': return T.CODES[code]?.name ?? null;
    case 'math': return T.MATH[code]?.name ?? null;
    case 'stage': return T.STAGES[code]?.name ?? null;
    case 'pipeline': return T.PIPELINES.find(p => p.c === code)?.n ?? null;
    case 'model': return T.MODELS.find(m => m.c === code)?.n ?? null;
    case 'drift': return T.DRIFTS.find(d => d.c === code)?.n ?? null;
    default: return null;
  }
}

const hover = (code) => `${code}: click for the definition`;

// A chip that reads as its name and opens its definition. `cls` adds classes
// (" hit", " miss"); `lower` writes the name in lower case, inside a sentence-like row.
export function nameChip(T, kind, code, { cls = '', lower = false } = {}) {
  const name = nameOf(T, kind, code) ?? code;
  return `<span class="chip${cls}" ${ATTR[kind]}="${esc(code)}" title="${esc(hover(code))}">${
    esc(lower ? name.toLowerCase() : name)}</span>`;
}

// The code itself, small, after a name: still clickable for the definition.
export function codeTag(kind, code, cls = 'rec-code') {
  return `<span class="${cls}" ${ATTR[kind]}="${esc(code)}" title="${esc(hover(code))}">${esc(code)}</span>`;
}

// A flowchart's boxes without the stage codes in front of their names
// ("PL-S9 Train" reads as "Train"). The stages are listed, with codes, above it.
export const plainFlowchart = (text) => String(text).replace(/(["[{(])PL-S\d+\s+/g, '$1');

// A reason written in axis codes ("needs A22 data (yours is A21)"), in words
// ("needs sequential data (yours is independent)"). Unknown codes stay.
export const plainReason = (T, text) =>
  String(text).replace(/\bA\d\d\b/g, (code) => T.CODES[code]?.name.toLowerCase() ?? code);
