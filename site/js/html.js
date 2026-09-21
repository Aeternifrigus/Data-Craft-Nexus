// Escapes text before it goes into innerHTML. Everything that comes from a
// user's CSV (column names, values) or from the taxonomy goes through esc(),
// so a column called "<img onerror=...>" or a formula like "T<t" is shown as
// text instead of being read as markup.

const ENTITIES = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };

export function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, c => ENTITIES[c]);
}
