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

The weights are a prior per model, fitted in `bench/dcn/learn.py` on 195 real datasets and judged leave-one-dataset-out, so a dataset never contributes to the weights that rank it:

| median regret, leave-one-dataset-out (95% interval) | classification (94 datasets, 78 independent) | regression (101 datasets, 35 independent) |
|---|---|---|
| counting matched coordinates (before) | 0.047 (0.035 to 0.056) | 0.205 (0.071 to 0.303) |
| learned from the benchmark (now) | 0.016 (0.012 to 0.021) | 0.012 (0.002 to 0.027) |
| always use boosting | 0.014 (0.011 to 0.022) | 0.021 (0.009 to 0.055) |
| always use boosting, tuned (reference) | 0.011 (0.008 to 0.018) | 0.006 (0.002 to 0.017) |

Datasets generated from one function are not independent. Of the 101 regression datasets, 54 come from Friedman's benchmark functions and 14 from Strogatz's equations: sisters that share a winner. Such a family is held out whole when its members are ranked, and counted once in every number above, so 101 regression datasets are 35 independent units. Counting them one by one made the learned order look better on regression than it is (0.007 instead of 0.012). The same holds for datasets cut from one table: PMLB has the Garvan thyroid records under six names with six targets, three horse colic targets, four feature sets of the same handwritten digits, two exact duplicates, and one generator behind each of led7 and led24, waveform_21 and waveform_40, and the three MONK's problems. Counted once, 94 classification datasets are 78 independent units.

The intervals come from resampling those units. The order in use is also compared, unit by unit, with the two things it claims to beat, and with tuned boosting, a reference it makes no claim to beat but a reader will ask about (Wilcoxon signed-rank, Holm-corrected for the three comparisons):

- **Against counting coordinates it is better by more than luck:** on 57 of 78 classification units, worse on 19 (p = 2 × 10⁻⁶), and on 27 of 35 regression units (p = 3 × 10⁻⁵).
- **Against always using default boosting, it is level:** better on 39 classification units and worse on 35 (p = 0.61), better on 21 of 35 regression units and worse on 14 (p = 0.091). With two comparisons the regression edge was p = 0.046; adding a third comparison to the family, as honesty requires, puts it back inside luck.
- **Against tuned boosting, it loses on classification by more than luck:** better on 24 units, worse on 49 (p = 0.028). On regression they are level: better on 16 of 35 units, worse on 19 (p = 0.55).

**What that means for the advice.** The site recommends model families, at their default settings. Choosing among the top families is worth about as much as reaching for boosting; spending ten configurations tuning boosting is worth more. Tuned boosting beat its own defaults on 62 of 94 classification datasets and 88 of 101 regression datasets, and has the best average rank of anything that ran in both tasks. So whichever model the page puts first, tune it before trusting its score. The tuning budget is small on purpose (scikit-learn's defaults and nine random configurations, chosen by cross-validation inside each training fold) and the details are in [`bench/README.md`](bench/README.md#references).

Two richer orders are built and judged the same way, and either can replace the per-model prior: one adds interactions between the dataset's measured features and each model's family, and one weights each model toward what it did on the benchmark datasets nearest to yours. Which one ships is decided by a rule fixed before the results were seen (`choose()` in `bench/dcn/learn.py`): a richer order replaces the prior only if it is no worse on either task and better by more than luck on at least one. Neither qualified on the full run (interactions: 9 better and 10 worse of 35 regression units; neighbours: 8 and 6), so the order in use is still the prior. The evidence tab reports each decision with its numbers.

A model the benchmark never ran is shown below the ones it did, with no score attached, and a task the benchmark never covered (forecasting, grouping, anomalies) still falls back to coordinates. The page says which of the two it used.

### Datasets like yours

The plot used to place a dataset by axes 1 to 3, and the first two are the same for every labelled table with independent rows, so three different uploads could land on the same point among ten invented reference datasets. It now shows the 195 datasets the recommendations were tested on, positioned by measured properties (size, shape, how much of the table is numeric), with your data placed among them and its closest neighbours highlighted.

Underneath, those neighbours are listed with what actually won on each, and every recommendation carries a second line: how often that model was the best choice on the datasets closest to yours, and how far below the winner it typically landed. Closeness is measured on the same eight properties for an upload and for a benchmark dataset, scaled by how much each varies across the benchmark.

When an upload is outside what the benchmark tested, the page says so above the map, before any number: when it has fewer rows than the smallest benchmark dataset (200), and when its nearest benchmark dataset is further away than 95% of benchmark datasets are from their own nearest neighbour of a different kind (a synthetic dataset's sisters do not count). The bundled 20-row sample triggers both, which is the point: twenty rows is too few for any measured score to mean much. The thresholds are written to `evidence.json` by the benchmark and recomputed from the page's own data by the tests.

### Take it home

Under the recommendations, **Download the script** gives you `dcn_shortlist.py`: the shortlist the page showed, run on your whole file (the page reads at most 5,000 rows) with exactly the benchmark's preprocessing, estimators and cross-validation, and tuned boosting beside it, because on the benchmark a small tuning budget was worth more than the choice among the top models. It reads your file the way the page did (delimiter, encoding, decimal commas, the column names the page showed), splits by time when you said row order matters, and needs only pandas and scikit-learn, plus XGBoost or LightGBM if the shortlist has them.

```
python dcn_shortlist.py shipments.csv

4 models, 5-fold cross-validation, balanced accuracy:

  0.3803 ± 0.0682  EN4             LightGBM  (7.6s)
  0.3740 ± 0.0529  BASE-HGB-TUNED  Histogram Gradient Boosting, tuned (reference)  (63.9s)
  0.3359 ± 0.0355  LM2             Logistic Regression  (0.1s)
  0.3196 ± 0.0361  TR3             Extra Trees  (2.2s)
```

The script is held to the benchmark by `bench/tests/test_export.py`: it is generated with the site's own JavaScript, every estimator in it is compared with the benchmark's registry, and on the test fixtures it produces exactly the benchmark's scores.

### What it was worth on real data

The instrument publishes its own scoreboard, in **The evidence** tab: 195
datasets from [PMLB](https://github.com/EpistasisLab/pmlb), every runnable
model fitted on every one of them, five-fold cross-validated, 5,446 model runs
(40 of the datasets under three cross-validation seeds, to measure how much a
split alone moves things, and tuned boosting on every dataset as a reference). Each recommendation also carries a line saying how
that model did.

| median regret, as recorded | classification | regression |
|---|---|---|
| the first model shown | 0.012 | 0.011 |
| best of the four shown | 0.005 | 0.002 |
| always use boosting | 0.015 | 0.021 |
| a model picked at random from the eligible ones | 0.051 | 0.162 |

Regret is how far below the best model that ran a choice landed, in balanced
accuracy and in R². These are the recommendations as the run recorded them,
ordered by the prior fitted on the earlier 40-dataset run, which had already
seen 40 of these datasets; the leave-one-dataset-out table above is the fairer
test. The four models shown contain the best available choice 31% of the time
on classification and 43% on regression.

The full run also measured how much one split decides. Between cross-validation
seeds a model's score moves by a median of 0.004 on classification and 0.005 on
regression. The best model was the same under all three seeds on only 30% of
classification datasets and 50% of regression ones, and the first pick and
boosting swapped places, depending only on the split, on 10 of 20 classification
datasets and 4 of 20 regression ones. A single dataset's winner is
weak evidence, which is why every comparison here is made across datasets.

The numbers on the page are generated from `bench/results/`, and a test
recomputes them from that data on every run, so the page cannot quietly
disagree with the run behind it.

### Not done yet

- **Recommendations are made at default settings.** Tuned boosting beats the site's first pick on classification (see above). Folding a tuning step into the advice, or recommending "tuned boosting" outright when nothing in the data argues against it, is the change the evidence points to.
- **TabPFN has not been run yet.** The runner supports it; its weights need a Prior Labs login this benchmark's environment could not reach. `bench/README.md` has the three commands to add it on your own machine.
- **The order in use is a per-model prior, not yet a per-dataset one.** Both ways of making it depend on your data (interactions, and weighting toward the nearest benchmark datasets) are built, tested against the JavaScript, and judged leave-one-dataset-out, and neither beat the prior by more than luck on 195 datasets once families count once. With only 35 independent regression units, a per-dataset order needs more collected data to prove itself, not more of the same generators.
- **The prior itself still counts a family's datasets one by one when it is fitted**, so on regression it leans toward what wins on Friedman's functions. Weighting a family once in the fit is the obvious change, and it should be declared before the next run rather than tried after this one.
- **The benchmark found a rule that throws away winners:** models that need numeric features (A31) are ruled out on all-categorical tables (A32), yet with one-hot encoding they won on 9 datasets, by up to 0.13 R² on solar_flare.
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
- `tests/export.test.js`: what the generated script carries: the page's reading of the file, the column roles, every runnable model's estimator, and the models it cannot run, named.
- `tests/check_links.test.js`: the link checker against a fake network. A connection that resets and then answers passes, a 404 fails on the first answer, and no host gets more than two requests at once. `npm run check:links` runs the real check, which needs the internet and runs in CI.
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
