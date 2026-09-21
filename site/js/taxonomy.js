// Loads the taxonomy from site/taxonomy/*.json.
// The JSON files are the single source of truth: the site, the tests and
// (later) the Python benchmark all read the same files.

export const TAXONOMY_FILES = ['axes', 'math', 'models', 'drift', 'pipelines', 'instrument'];

// Turns the raw JSON files into the flat shape the rest of the code uses.
export function assembleTaxonomy(files) {
  const { axes, math, models, drift, pipelines, instrument } = files;
  return {
    AXES: axes.axes,
    CODES: axes.codes,
    MATH_DOMAINS: math.domains,
    MATH: math.formulas,
    MODEL_DOMAINS: models.domains,
    MODELS: models.models,
    DRIFT_DOMAINS: drift.domains,
    DRIFTS: drift.checkers,
    PIPELINE_DOMAINS: pipelines.domains,
    PIPELINES: pipelines.pipelines,
    STAGES: pipelines.stages,
    TASKS: instrument.tasks,
    REFERENCE: instrument.reference,
  };
}

// Browser loader. `base` is the URL of the taxonomy directory.
export async function loadTaxonomy(base = new URL('../taxonomy/', import.meta.url)) {
  const entries = await Promise.all(TAXONOMY_FILES.map(async (name) => {
    const res = await fetch(new URL(`${name}.json`, base));
    if (!res.ok) throw new Error(`Could not load taxonomy/${name}.json (HTTP ${res.status})`);
    return [name, await res.json()];
  }));
  return assembleTaxonomy(Object.fromEntries(entries));
}
