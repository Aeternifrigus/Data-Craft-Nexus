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

1. **Drop a CSV** — modality, scale, quality and drift are measured automatically
2. **Declare your intent** — what to predict, what kind of answer, whether order matters
3. **Read the specimen** — get the full six-axis signature
4. **Prescriptions** — models, drift checkers, and pipelines that fit, and what your data rules out

### What is measured, and what is asked

Three things cannot be read off a file: what you want to predict, what kind of answer you need, and whether row order carries meaning. Those are asked. Everything else is measured:

| Axis | How it is decided |
|---|---|
| 1 supervision | from whether you named a target column |
| 2 structure | asked: does row order matter |
| 3 modality | measured over the feature columns, with the target left out |
| 4 scale | measured: sparsity, and columns relative to rows |
| 5 distribution | measured: class balance for a categorical target, and drift between the first and second half of the file (PSI, against the 0.25 cutoff in DR-M2) |
| 6 quality | measured: missingness (A62) and noise (A64), meaning text in numeric columns or labels differing only by case or padding |

Three further questions decide which drift checkers and pipelines can be used at all, because those depend on how the thing will run rather than on what the data looks like:

- **How does new data arrive?** A detector that compares two windows has nothing to compare in a stream; a streaming detector has no running error rate in a batch job.
- **Do the true answers arrive later?** Everything that watches the error rate needs labels to come back. Without them you can only watch the data itself move.
- **Which part are you building?** Filters the pipelines to the part of the work you are in, or shows all of them.

Whatever those answers exclude is listed with its reason, the same way models are.

A63 (missing not at random) is never reported. Whether a gap depends on the value that is missing cannot be decided from the file alone.

### How the recommendation is made

First, what cannot work is ruled out, with the reason shown on the page: a sequence model has no order to use on independent rows, a text model has nothing to read in a numeric table, a supervised model has no labels to learn from. A model that assumes independent rows still appears on ordered data, with a caution to split by time rather than at random.

What is left is **ordered by what those models were worth on the benchmark**, not by how many coordinates they match. The coordinate count is still shown on every card, because it says what your data has in common with the model, but it no longer decides the order: it used to, and it put plain Linear Regression first on 17 of 20 regression datasets at a cost of up to 0.7 R².

The weights are a prior per model, fitted in `bench/dcn/learn.py` and judged leave-one-dataset-out, so a dataset never contributes to the weights that rank it:

| median regret, leave-one-dataset-out | classification | regression |
|---|---|---|
| counting matched coordinates (before) | 0.055 | 0.432 |
| **learned from the benchmark (now)** | **0.011** | **0.002** |
| always use boosting | 0.023 | 0.007 |

A model the benchmark never ran is shown below the ones it did, with no score attached, and a task the benchmark never covered (forecasting, grouping, anomalies) still falls back to coordinates. The page says which of the two it used.

### Datasets like yours

The plot used to place a dataset by axes 1 to 3, and the first two are the same for every labelled table with independent rows, so three different uploads could land on the same point among ten invented reference datasets. It now shows the 40 datasets the recommendations were tested on, positioned by measured properties (size, shape, how much of the table is numeric), with your data placed among them and its closest neighbours highlighted.

Underneath, those neighbours are listed with what actually won on each, and every recommendation carries a second line: how often that model was the best choice on the datasets closest to yours, and how far below the winner it typically landed. Closeness is measured on the same eight properties for an upload and for a benchmark dataset, scaled by how much each varies across the benchmark.

### What it was worth on real data

The instrument now publishes its own scoreboard, in **The evidence** tab: 40
real datasets from [PMLB](https://github.com/EpistasisLab/pmlb), every
runnable model fitted on every one of them, five-fold cross-validated, 760
model runs. Each recommendation also carries a line saying how that model did.

| median regret | classification | regression |
|---|---|---|
| the first model shown | 0.055 | 0.432 |
| best of the four shown | 0.015 | 0.176 |
| always use boosting | 0.023 | 0.007 |
| a model picked at random from the eligible ones | 0.060 | 0.240 |

Regret is how far below the best model that ran a choice landed, in balanced
accuracy and in R². The four models shown contain the best available choice
15% of the time on classification and beat reaching for boosting on 55% of
those datasets. On regression boosting wins outright. The single model shown
first is the weakest part, for the reason in the next section.

The numbers on the page are generated from `bench/results/`, and a test
recomputes them from that data on every run, so the page cannot quietly
disagree with the run behind it.

### Not done yet

- **The learned order is a per-model prior, not yet a per-dataset one.** Interactions between a dataset's measured features and a model's family were fitted and did not beat the plain prior on 20 datasets per task. Running all 196 datasets rather than 40 is what would let the order depend on your data rather than on averages.
- **Only classification and regression are benchmarked.** Forecasting, survival, grouping, anomalies, compression and generation still fall back to counting coordinates.
- Image, audio, graph and spatial data cannot be detected from a CSV, so those models are reachable in the reference but never recommended from an upload.
- Separability (A55/A56) and weak or self-supervised labelling (A13 to A15) are not measured yet.

---

## The Taxonomy

Data Craft Nexus is built on a complete, interconnected taxonomy:

| Component | Count | Description |
|-----------|-------|-------------|
| Data axes | 6 | Supervision, structure, modality, scale, distribution, quality |
| Tasks | 8 | A number, a category, a future value, time until an event, groupings, anomalies, a simpler view, new examples |
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
- `tests/profile.test.js`: what each axis measures, including PSI drift and class balance.
- `tests/recommend.test.js`: what gets ruled out and why, and that every model except the reinforcement learning ones is reachable from some dataset.
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
    stats.js           PSI and other small statistics
    html.js            escaping
    profile.js         measures the axes (no DOM)
    recommend.js       rules out and ranks models, drift checkers, pipelines (no DOM)
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
