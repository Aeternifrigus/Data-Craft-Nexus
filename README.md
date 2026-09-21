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

---

## Features

- **Six-axis data signature** — every dataset gets graded like a specimen
- **Metaphor-first explanations** — every model has a plain-language metaphor
- **Clickable everything** — any code opens a drawer with formulas and mechanisms
- **3D coordinate plot** — your data plotted against reference datasets
- **Mermaid flowcharts** — every pipeline renders its diagram inline
- **Full library** — search the entire taxonomy
- **100% client-side** — nothing is uploaded. Everything runs in your browser.

---

## Run Locally

```bash
git clone https://github.com/Aeternifrigus/Data-Craft-Nexus.git
cd Data-Craft-Nexus
npm run serve        # http://localhost:8000
```

The page loads its taxonomy from JSON files, so it needs to be served over HTTP. Opening `site/index.html` straight from disk won't work. Any static server does the job; `npm run serve` just runs `python3 -m http.server`.

## Tests

```bash
npm test             # Node 20+, no dependencies to install
```

- `tests/taxonomy.test.js` checks every cross-reference: each code a model, drift checker or pipeline points at must exist.
- `tests/recommend.snapshot.test.js` runs every fixture in `tests/fixtures/` through every target, task and order answer, and compares what gets recommended with `tests/snapshots/recommendations.json`. When a change to the profiler or the ranking is intended, run `npm run test:update` and review the snapshot diff in the commit.

## Project Layout

```
site/                  what GitHub Pages serves
  index.html
  css/style.css
  sample.csv           the "load a sample" shipment table
  taxonomy/*.json      axes, math, models, drift checkers, pipelines: the single source of truth
  js/
    taxonomy.js        loads and assembles the JSON
    csv.js             CSV parsing
    profile.js         measures axes 3, 4, 6 and derives axis 5 (no DOM)
    recommend.js       ranks models, drift checkers, pipelines (no DOM)
    results.js         renders the recommendations and the 3D plot
    drawer.js          the definition drawer
    library.js         "The reference" view
    app.js             entry point
docs/taxonomy/         the original taxonomy notes
tests/
```

`profile.js` and `recommend.js` don't touch the page, so the same logic runs in the browser, in the tests and, later, in the benchmark.

## Stack

- **HTML/CSS/JS**: plain ES modules, no build step
- **Plotly**: 3D coordinate visualization
- **Mermaid**: pipeline flowcharts
- **Google Fonts**: Instrument Serif, JetBrains Mono, Press Start 2P, Mrs Saint Delafield

---

## Credits

By [Aeternifrigus](https://aeternifrigus.netlify.app/)

---

License
MIT — use it, remix it, cite it.
