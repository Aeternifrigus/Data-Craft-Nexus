# Data Craft Nexus

**Every dataset has coordinates. Read them.**

Data Craft Nexus is a reference instrument for the full machine learning lifecycle. Feed it a CSV snippet, answer three questions about your intent, and it reads six data axes — then prescribes:

- **Models** that fit your data's signature
- **Drift checkers** that watch for decay
- **Pipelines** assembled from a reusable stage library

Every code is clickable, down to the formula underneath.

---

## Live Demo

**[Try it here](https://aeternifrigus.github.io/Data-Craft-Nexus/)**

---

## How It Works

1. **Drop a CSV** — modality, scale, and missingness are measured automatically
2. **Declare your intent** — what to predict, what kind of answer, whether order matters
3. **Read the specimen** — get the full six-axis signature
4. **Prescriptions** — models, drift checkers, and pipelines ranked by fit

---

## The Taxonomy

Data Craft Nexus is built on a complete, interconnected taxonomy:

| Component | Count | Description |
|-----------|-------|-------------|
| Data axes | 6 | Supervision, structure, modality, scale, distribution, quality |
| Math formulas | 80+ | Across 15 domains |
| Models | 60+ | Across 14 architecture families |
| Drift checkers | 28 | Distributional, streaming, multivariate, adversarial, DL-native |
| Pipelines | 14 | ETL, feature store, training, deployment, monitoring, RAG |
| Stages | 31 | Reusable pipeline building blocks |

The full written notes are in [`docs/taxonomy/`](docs/taxonomy): [Data](docs/taxonomy/Data.md), [Math](docs/taxonomy/Math.md), [ML Models](<docs/taxonomy/ML Models.md>), [Model Stacking](<docs/taxonomy/Model Stacking.md>), [Model Drift](<docs/taxonomy/Model Drift.md>) and [ML Pipeline](<docs/taxonomy/ML Pipeline.md>).

---

## Features

- **Six-axis data signature** — every dataset gets graded like a specimen
- **Metaphor-first explanations** — every model has a plain-language metaphor
- **Clickable everything** — any code opens a drawer with formulas and mechanisms
- **3D coordinate plot** — your data plotted against reference datasets
- **Mermaid flowcharts** — every pipeline renders its diagram inline
- **Full library** — search the entire taxonomy
- **Reads real exports**: comma, semicolon, tab or pipe separated; quoted fields; UTF-8 or Windows-1250; decimal commas like `1.234,56`
- **100% client-side** — nothing is uploaded. Everything runs in your browser.

---

## Run Locally

The quickest way: download [`dist/index.html`](dist/index.html) and open it. It's one self-contained file and works straight from disk.

To work on the code:

```bash
git clone https://github.com/Aeternifrigus/Data-Craft-Nexus.git
cd Data-Craft-Nexus
npm install          # only installs esbuild, for the build step
npm run serve        # serves site/ at http://localhost:8000
```

`site/` is the source, split into ES modules. Browsers won't load modules from disk, so `site/index.html` needs a server. After changing anything in `site/`, run `npm run build` to regenerate `dist/index.html` and commit both. CI fails if they're out of sync, and GitHub Pages publishes `dist/`.

## Tests

```bash
npm test             # Node 20+
```

- `tests/csv.test.js`: delimiter detection, quoting, encodings, decimal commas, header clean-up.
- `tests/html.test.js`: escaping of everything that goes into the page.
- `tests/taxonomy.test.js`: every code a model, drift checker or pipeline points at must exist.
- `tests/build.test.js`: the built page is self-contained, carries exactly the taxonomy in `site/`, and is up to date.
- `tests/recommend.snapshot.test.js`: runs every fixture in `tests/fixtures/` through every target, task and order answer, and compares what gets recommended with `tests/snapshots/recommendations.json`. When a change to the profiler or the ranking is intended, run `npm run test:update` and review the snapshot diff in the commit.

## Project Layout

```
dist/index.html        the built single-file page (what GitHub Pages serves)
site/                  the source
  index.html
  css/style.css
  sample.csv           the "load a sample" shipment table
  taxonomy/*.json      axes, math, models, drift checkers, pipelines: the single source of truth
  js/
    taxonomy.js        loads the taxonomy (embedded in dist/, fetched in site/)
    csv.js             CSV reading: delimiters, quoting, encodings, decimal commas
    html.js            escaping
    profile.js         measures axes 3, 4, 6 and derives axis 5 (no DOM)
    recommend.js       ranks models, drift checkers, pipelines (no DOM)
    results.js         renders the recommendations and the 3D plot
    drawer.js          the definition drawer
    library.js         "The reference" view
    app.js             entry point
scripts/build.mjs      bundles site/ into dist/index.html
docs/taxonomy/         the original taxonomy notes
tests/
```

`profile.js` and `recommend.js` don't touch the page, so the same logic runs in the browser, in the tests and, later, in the benchmark.

## Stack

- **HTML/CSS/JS**: plain ES modules, bundled into one file with esbuild
- **Plotly**: 3D coordinate visualization
- **Mermaid**: pipeline flowcharts
- **Google Fonts**: Instrument Serif, JetBrains Mono, Press Start 2P, Mrs Saint Delafield

---

## Credits

By [Aeternifrigus](https://aeternifrigus.netlify.app/)

---

License
MIT — use it, remix it, cite it.
