# Development

[README](../README.md) · [Docs](README.md) · [How it works](how-it-works.md) · [The evidence](evidence.md) · **Development** · [For coding agents](mcp.md)

## Contents

- [Run locally](#run-locally)
- [Tests](#tests)
- [The benchmark](#the-benchmark)
- [Project layout](#project-layout)
- [Stack](#stack)
- [How the live page is deployed](#how-the-live-page-is-deployed)
- [The tour video](#the-tour-video)

## Run locally

The quickest way: download [`dist/index.html`](../dist/index.html) and open it. It's one self-contained file and works straight from disk.

To work on the code:

```bash
git clone https://github.com/Aeternifrigus/Data-Craft-Nexus.git
cd Data-Craft-Nexus
npm install          # esbuild for the build step, and the MCP SDK for the server in mcp/
npm run serve        # serves site/ at http://localhost:8000
```

`site/` is the source, split into ES modules. Browsers won't load modules from disk, so `site/index.html` needs a server. After changing anything in `site/`, run `npm run build` to regenerate `dist/index.html` and commit both. CI fails if they're out of sync, and GitHub Pages publishes `dist/`.

## Tests

```bash
npm test             # Node 20+
```

- `tests/csv.test.js`: delimiter detection, quoting, encodings, decimal commas, header clean-up.
- `tests/dates.test.js`: which values read as dates, in every common format, and which only look like them.
- `tests/profile.test.js`: what each axis measures, including PSI drift and class balance.
- `tests/recommend.test.js`: what gets ruled out and why, and that every model except the reinforcement learning ones is reachable from some dataset.
- `tests/ranking.test.js`: the learned order, what it uses, what it refuses to claim, and that it matches the weights committed in `site/taxonomy/ranking.json`.
- `tests/nearest.test.js`: "Datasets like yours", measuring an upload on the same ruler as the benchmark datasets and comparing it with them.
- `tests/evidence.test.js`: the numbers the site publishes are the numbers in the committed run, recomputed from `bench/results/`.
- `tests/evidence-structure.test.js`: the evidence tab is in numbered parts with subsections the contents list can jump to, and every part with mathematics behind it shows it.
- `tests/forecasting.test.js`: a future value is ordered by the forecasting benchmark, per kind of series only when that passed the rule fixed before the run, and the published shares are recomputed from the committed run.
- `tests/anomalies.test.js`: unusual records are ordered by the anomaly benchmark, per width of table only when that passed the rule, and the published shares are recomputed from the committed run.
- `tests/drift.test.js`: drift checkers ordered by what they measured, and every published rate recomputed from `bench/results/drift.csv`.
- `tests/operating.test.js`: drift checkers and pipelines are chosen by how the thing will run, not only by what the data looks like.
- `tests/checks.test.js`: "Before you trust a score": what each check catches and leaves alone, and every published rate recomputed from `bench/results/checks.csv`.
- `tests/messy.test.js`: which damaged run a messy upload is told about, and that every one is published.
- `tests/costs.test.js`: "What does a wrong answer cost?": which choices a target gets, the score each one puts in the script, and what the page says about it.
- `tests/export.test.js`: what the generated script carries: the page's reading of the file, the column roles, every runnable model's estimator, and the models it cannot run, named.
- `tests/verify.test.js`: "Run it here": the messages between the page and its Python worker, with a stand-in worker.
- `tests/report.test.js`: what a saved reading says, that no row of the file gets into it, and that odd names cannot break it.
- `tests/names.test.js`: every code a card, chip or drawer shows has a plain name to lead with, and flowcharts and ruled-out reasons read in words.
- `tests/html.test.js`: escaping of everything that goes into the page.
- `tests/taxonomy.test.js`: every code a model, drift checker or pipeline points at must exist.
- `tests/links.test.js`: every reference link is https, points at an allowed documentation or paper site, and every concept that should have one has one.
- `tests/check_links.test.js`: the link checker against a fake network. A connection that resets and then answers passes, a 404 fails on the first answer, no host gets more than two requests at once, and a DOI is checked where it is registered (doi.org), not at a publisher that turns away anything but a browser. `npm run check:links` runs the real check, which needs the internet and runs in CI.
- `tests/favicon.test.js`: the favicon inlined into the page matches `site/favicon.svg` and stays self-contained.
- `tests/build.test.js`: the built page is self-contained, carries exactly the taxonomy in `site/`, and is up to date.
- `tests/analysis.test.js`: a reading worked out without the page: every part the page draws, ID columns left out of the script, and when there is no script and why.
- `tests/mcp.test.js`: the MCP server. Each problem the checks look for is planted in a generated table and found; the signature, order, lead, drift checkers, pipelines and script match the page's for the same file; bad requests come back as tool errors; the protocol runs in memory and over stdio; and the npm package carries every file the server loads.
- `tests/recommend.snapshot.test.js`: runs every fixture in `tests/fixtures/` through every target, task and order answer, and compares what gets recommended with `tests/snapshots/recommendations.json`. When a change to the profiler or the ranking is intended, run `npm run test:update` and review the snapshot diff in the commit.

## The benchmark

Everything the page measured lives in [`bench/`](../bench): a Python copy of the profiler and the recommender, held to the JavaScript by tests, the benchmark runs, and their results. How to run each one, and what each had to show written down before it ran, is in [`bench/README.md`](../bench/README.md).

```bash
pip install -r bench/requirements.txt
cd bench && python -m pytest -q      # the Python side must agree with the JavaScript
```

## Project layout

```
dist/index.html        the built single-file page (what GitHub Pages serves)
site/                  the source
  index.html
  css/style.css
  sample.csv           the "load a sample" shipment table
  taxonomy/*.json      axes, math, models, drift checkers, pipelines, and the benchmark's
                       results the page reads: the single source of truth
  js/
    taxonomy.js        loads the taxonomy (embedded in dist/, fetched in site/)
    csv.js             CSV reading: delimiters, quoting, encodings, decimal commas
    dates.js           which values are dates
    stats.js           PSI and other small statistics
    html.js            escaping
    profile.js         measures the axes (no DOM)
    series.js          the shape of a series: intermittent, seasonal, trending (no DOM)
    recommend.js       rules out and ranks models, drift checkers, pipelines (no DOM)
    analysis.js        a whole reading worked out, for the page and the MCP server (no DOM)
    ranking.js         the learned order, fitted by the benchmark
    forecasting.js     the order of forecasters, and what the forecasting benchmark measured (no DOM)
    anomalies.js       the order of anomaly detectors, and what the anomaly benchmark measured (no DOM)
    nearest.js         meta-features, and the benchmark datasets nearest to yours
    evidence.js        "The evidence" view, and the measured line on each card
    checks.js          "Before you trust a score": leaks, IDs, repeated rows, dates (no DOM)
    costs.js           "What does a wrong answer cost?": what the script scores by (no DOM)
    names.js           plain names for the taxonomy's codes, with the code kept for hover (no DOM)
    export.js          the take-home Python script
    report.js          "Save this reading": the findings as Markdown
    verify.js          "Run it here": that script in a Pyodide worker
    results.js         renders the recommendations and the 3D plot
    drawer.js          the definition drawer
    library.js         "The reference" view
    app.js             entry point
mcp/                   the MCP server for coding agents (docs/mcp.md)
  dcn-mcp.mjs          entry point: the server on stdio
  server.mjs           the four tools and their descriptions
  tools.mjs            reads a file from disk and says a reading in words
server.json            the server's MCP Registry entry
scripts/build.mjs      bundles site/ into dist/index.html
bench/                 the benchmark, in Python: runs, results, and the tests that hold it to the page
docs/                  these pages, the tour video and gif (media/), and the original taxonomy notes (taxonomy/)
tests/
```

`profile.js` and `recommend.js` don't touch the page, so the same logic runs in the browser, in the tests and in the benchmark's Python copy, which the tests hold to agree with it. `analysis.js` puts them together into a reading, which the page draws and the MCP server returns to an agent.

## Stack

- **HTML/CSS/JS**: plain ES modules, bundled into one file with esbuild
- **Plotly**: 3D coordinate visualization
- **Mermaid**: pipeline flowcharts
- **Pyodide**: "Run it here", Python in a Web Worker
- **Google Fonts**: Instrument Serif, JetBrains Mono, Press Start 2P, Mrs Saint Delafield
- **Python** (benchmark only): scikit-learn, XGBoost, LightGBM, statsmodels, river, SciPy

## How the live page is deployed

The live page carries the commit it was built from in a meta tag (`dcn-build`), out of sight. The "Deploy static content to Pages" workflow builds the page, publishes the tour video beside it, deploys it, and then fetches the live page until it carries the new commit, failing if it does not within five minutes, so a deploy that did not take shows up as a failed run under Actions. A browser can also hold on to an older copy for a few minutes; a hard refresh fetches the new one.

## The tour video

[`media/dcn-tour.mp4`](media/dcn-tour.mp4) is the 95-second tour and [`media/dcn-demo.gif`](media/dcn-demo.gif) the short cut of it in the README. Both were recorded from `dist/index.html` in headless Chromium, with the real Plotly, Mermaid and fonts, and captions added for the recording only. GitHub does not play a video committed to a repository inside a README, so the deploy workflow copies the mp4 next to the page, where Pages serves it and the README links to it.
