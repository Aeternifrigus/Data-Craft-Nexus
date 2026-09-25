# How it works

[README](../README.md) · [Docs](README.md) · **How it works** · [The evidence](evidence.md) · [Development](development.md) · [For coding agents](mcp.md)

Most tabular ML advice fits in one line: tune gradient boosting. The 195-dataset benchmark behind this site agrees. Data Craft Nexus is for the three things that line leaves out, answered from your own CSV, in your browser, with nothing uploaded:

1. **Is my file the kind of table that advice is for?** It measures column types, size, missing and junk values and drift, asks what a file cannot say (the target, whether row order matters), and rules out what cannot work, with the reason: a text model has nothing to read in a table of numbers, a supervised model has nothing to learn from without labels. Ordered rows get a warning to split by time. And before any score, it checks for what makes every score lie: a column that gives the answer away, an ID column a model can memorise, copied rows on both sides of a split, and dates split at random.
2. **Does the advice hold on my data?** Tuned boosting comes first, then a shortlist, each model with its benchmark record. A generated Python script runs them against each other on your whole file with the benchmark's cross-validation, or runs right in the page.
3. **How will I know when the model stops working?** Drift detectors are filtered by how your data arrives and whether the true answers come back, then ranked by what they caught in a drift benchmark of their own.

## Contents

- [The flow](#the-flow)
- [What is measured, and what is asked](#what-is-measured-and-what-is-asked)
- [How the recommendation is made](#how-the-recommendation-is-made)
- [Before you trust a score](#before-you-trust-a-score)
- [Datasets like yours](#datasets-like-yours)
- [Take it home](#take-it-home)
- [Run it here](#run-it-here)
- [Features](#features)
- [The taxonomy](#the-taxonomy)

## The flow

1. **Drop a CSV**: modality, scale, quality and drift are measured automatically
2. **Declare your intent**: what to predict, what kind of answer, whether order matters, and what a wrong answer costs
3. **Read the specimen**: get the full six-axis signature
4. **Before you trust a score**: one column that predicts the target almost perfectly on its own, ID-like columns, more repeated rows than chance, and dates with rows declared independent. None of these is fixed by choosing a better model
5. **Prescriptions**: models, drift checkers, and pipelines that fit, and what your data rules out
6. **Take it with you**: the shortlist as a Python script for your whole file, run here in the browser, or the whole reading saved as a Markdown file

## What is measured, and what is asked

Three things cannot be read off a file: what you want to predict, what kind of answer you need, and whether row order carries meaning. Those are asked. Everything else is measured:

| Axis | How it is decided |
|---|---|
| 1 supervision | from whether you named a target column |
| 2 structure | asked: does row order matter |
| 3 modality | measured over the feature columns, with the target left out |
| 4 scale | measured: sparsity, and columns relative to rows |
| 5 distribution | measured: class balance for a categorical target, and drift between the first and second half of the file (PSI, against the 0.25 cutoff in DR-M2) |
| 6 quality | measured: missingness (A62) and noise (A64), meaning text in numeric columns or labels differing only by case or padding, judged on each column as well as on the whole table |

Three further questions decide which drift checkers and pipelines can be used at all, because those depend on how the thing will run rather than on what the data looks like:

- **How does new data arrive?** A detector that compares two windows has nothing to compare in a stream; a streaming detector has no running error rate in a batch job.
- **Do the true answers arrive later?** Everything that watches the error rate needs labels to come back. Without them you can only watch the data itself move.
- **Which part are you building?** Filters the pipelines to the part of the work you are in, or shows all of them.

Whatever those answers exclude is listed with its reason, the same way models are. What is left is ordered by what the drift benchmark measured ([The evidence, drift checkers](evidence.md#4-drift-checkers)): the share of injected drift each checker caught, minus how often it fired when nothing had changed.

A63 (missing not at random) is never reported. Whether a gap depends on the value that is missing cannot be decided from the file alone.

## How the recommendation is made

First, what cannot work is ruled out, with the reason shown on the page: a sequence model has no order to use on independent rows, a text model has nothing to read in a numeric table, a supervised model has no labels to learn from. A model that assumes independent rows still appears on ordered data, with a caution to split by time rather than at random, and a model built for numbers still appears on a table of categories, with a caution to one-hot encode them: it used to be ruled out, and on 9 of the benchmark's categorical tables one of those models would have been the best choice. Letting them back changed none of the four models shown on the 26 categorical tables, because the learned order ranks boosting above them there; what changed is that the page stops calling usable models unusable, and that counting coordinates, the order both are compared with, got better (the numbers in [The evidence](evidence.md#the-order-in-use) are with the rule fixed).

What is left is **ordered by what those models were worth on the benchmark**, not by how many coordinates they match. The coordinate count is still shown on every card, because it says what your data has in common with the model, but it no longer decides the order: it used to, and it put plain Linear Regression first on 17 of 20 regression datasets at a cost of up to 0.7 R².

The weights are a prior per model, fitted in `bench/dcn/learn.py` on 195 real datasets and judged leave-one-dataset-out, so a dataset never contributes to the weights that rank it. How that order compares with counting coordinates, with always using boosting and with tuned boosting, family by family and with significance tests, is in [The evidence, tables](evidence.md#1-tables-a-number-or-a-category).

**Tuned boosting comes first.** The page shows tuned boosting first, above the order's picks, on every labelled table predicting a category or a number, because it beat the order's first pick on the benchmark and passes the same rule that decides which learned order ships ([why](evidence.md#what-that-means-for-the-advice)). Whichever family you pick after that, tune it before trusting its score.

Two richer orders are built and judged the same way, and either can replace the per-model prior: one adds interactions between the dataset's measured features and each model's family, and one weights each model toward what it did on the benchmark datasets nearest to yours. Which one ships is decided by a rule fixed before the results were seen (`choose()` in `bench/dcn/learn.py`): a richer order replaces the prior only if it is no worse on either task and better by more than luck on at least one. Neither has qualified so far ([the numbers](evidence.md#richer-orders)).

A model the benchmark never ran is shown below the ones it did, with no score attached. A future value and unusual records are ordered by benchmarks of their own ([forecasting](evidence.md#2-forecasting-a-future-value), [anomalies](evidence.md#3-anomalies-unusual-records)), and a task no benchmark covered (grouping, survival) still falls back to coordinates. The page says which of the two it used.

## Before you trust a score

The first section of the results checks for what makes any score look better than it will be on new data. None of these is fixed by choosing a better model:

- one column that predicts the target almost perfectly on its own
- ID-like columns a model can memorise
- more repeated rows than chance, which can land on both sides of a split
- dates in a table whose rows were declared independent, which a random split mixes

Each flag says what to do about it. How often each check caught a planted problem, and what it flags on clean data, is in [The evidence, the checks](evidence.md#5-what-the-checks-catch).

## Datasets like yours

The plot used to place a dataset by axes 1 to 3, and the first two are the same for every labelled table with independent rows, so three different uploads could land on the same point among ten invented reference datasets. It now shows the 195 datasets the recommendations were tested on, positioned by measured properties (size, shape, how much of the table is numeric), with your data placed among them and its closest neighbours highlighted.

Underneath, those neighbours are listed with what actually won on each, and every recommendation carries a second line: how often that model was the best choice on the datasets closest to yours, and how far below the winner it typically landed. Closeness is measured on the same eight properties for an upload and for a benchmark dataset, scaled by how much each varies across the benchmark.

When an upload is outside what the benchmark tested, the page says so above the map, before any number: when it has fewer rows than the smallest benchmark dataset (200), and when its nearest benchmark dataset is further away than 95% of benchmark datasets are from their own nearest neighbour of a different kind (a synthetic dataset's sisters do not count). The bundled 20-row sample triggers both, which is the point: twenty rows is too few for any measured score to mean much. The thresholds are written to `evidence.json` by the benchmark and recomputed from the page's own data by the tests.

## Take it home

Under the recommendations, **Download the script** gives you `dcn_shortlist.py`: the shortlist the page showed, run on your whole file (the page reads at most 5,000 rows) with exactly the benchmark's preprocessing, estimators and cross-validation, and tuned boosting beside it, because on the benchmark a small tuning budget was worth more than the choice among the top models. It reads your file the way the page did (delimiter, encoding, decimal commas, the column names the page showed), splits by time when you said row order matters, and needs only pandas and scikit-learn, plus XGBoost or LightGBM if the shortlist has them.

```
python dcn_shortlist.py shipments.csv

4 models, 5-fold cross-validation, balanced accuracy:

  0.3803 ± 0.0682  EN4             LightGBM  (7.6s)
  0.3740 ± 0.0529  BASE-HGB-TUNED  Histogram Gradient Boosting, tuned (reference)  (63.9s)
  0.3359 ± 0.0355  LM2             Logistic Regression  (0.1s)
  0.3196 ± 0.0361  TR3             Extra Trees  (2.2s)
```

### Doing nothing is scored too

Beside the models, the script scores a guess that learns nothing: always the most common answer for a category, always the average (or the median, when every unit of a miss costs the same) for a number, and the last known value when rows are in time order. The last line says whether any model beat it. On a small file that matters: on the 20-row sample, LightGBM and tuned boosting cannot make a single split with that few rows, so they predict the average, and the page used to call that a win.

### Which answers get a script

The script is offered when you answer "A number", "A category" or "A future value". **A future number is forecast for real:** each value is predicted from the values before it and never from a later row, one step ahead, in time-ordered folds, by the forecasting models on the cards that can run here (ARIMA, its order chosen by AIC on the training rows; exponential smoothing, with a trend and a season where AIC prefers them; Croston's method with Syntetos and Boylan's correction, and TSB, for intermittent demand), by tuned boosting on the last few changes, and by doing nothing (carrying the last value forward, the value a season earlier, and the average so far). The season comes from the date column: a daily series repeats weekly, a monthly one yearly. Prophet is named and not run: it needs Stan, a compiled backend the page cannot load. The other columns are not used, and when a date column shows the rows run newest first, they are turned around. The cards are in the order the forecasting benchmark measured, and the page says when your kind of series points elsewhere; the run on your file settles it for your data. A future category is predicted from the other columns, split by time, and the page says that too. For any other kind of answer, the section says why there is no script and what to answer instead, rather than leaving a gap.

`bench/tests/test_export.py` checks that no forecast moves when every value from a later row onward is changed, that doing nothing is what it says, and that on a series with a trend and a weekly swing the forecasting models beat carrying the last value forward.

### What a wrong answer costs

This decides what the script scores by. Left alone, it is the benchmark's own score: balanced accuracy for a category, R² for a number. For a category you can instead say every row counts the same (accuracy), that you will work through the riskiest rows first (ROC AUC), or that you will use the chances themselves (log loss). For a number, that every unit of error costs the same (mean absolute error), or that a miss matters relative to the true value (mean absolute percentage error, with a warning when the target has zeros). On a yes or no target you can price a miss, as in "a missed yes costs 10 false alarms". Each model is then scored by its cost per row, flagging a row when its chance is above 1 / (1 + 10), where a miss and a false alarm cost the same if the chances are right. After the scores, the script prints the threshold that cost least on your file, beside that one and the usual 0.5, and says which of them was picked after looking. Tuned boosting is tuned for the same score. The order on the page stays the benchmark's, and the page says so under the buttons: on your file, for your costs, the script's order is the one that counts.

### Held to the benchmark

The script is held to the benchmark by `bench/tests/test_export.py`: it is generated with the site's own JavaScript, every estimator in it is compared with the benchmark's registry, and on the test fixtures it produces exactly the benchmark's scores. Each other score is recomputed by hand, fold by fold, and the reported threshold is checked against every threshold on a grid.

## Run it here

**Run it here** runs the same script without installing anything. The page downloads Python from cdn.jsdelivr.net ([Pyodide](https://pyodide.org) 314.0.7 with pandas and scikit-learn, about 40 MB, plus XGBoost or LightGBM when the shortlist has them, and statsmodels for a forecast), starts it in a Web Worker, and hands it the page's own copy of your file: nothing is uploaded anywhere. Each model's score appears as it finishes, and the page then says where its own first pick landed on your data. What runs is the downloaded script, through its own `load()` and `evaluate()` (`site/js/verify.js`).

It was checked end to end in headless Chromium, with Pyodide 314.0.7's release files served locally in place of the CDN, on two test files: every score matched the same script run with CPython, to the fourth decimal, except XGBoost's on one file (0.4924 against 0.4965), because Pyodide ships XGBoost 2.1.4 and the benchmark ran 3.2.0; with 2.1.4 installed, CPython gives 0.4924 too. That run is also how a real defect was found: Pyodide 314 refuses to load in a classic worker, so the page starts a module worker. `bench/tests/test_export.py` runs the page's Python runner with CPython on every test run, and `tests/verify.test.js` checks the messages with a stand-in worker.

## Features

- **Six-axis data signature**: every dataset gets graded like a specimen
- **Metaphor-first explanations**: every model has a plain-language metaphor
- **Plain names first**: the readout, the cards and the chips say "Labeled", "Random Forest", "Train"; the taxonomy's code sits small beside each name and on hover, and any name or code opens a drawer with formulas and mechanisms
- **3D coordinate plot**: your data plotted against the benchmark's datasets
- **Mermaid flowcharts**: every pipeline renders its diagram inline
- **Full library**: search the entire taxonomy
- **Reads real exports**: comma, semicolon, tab or pipe separated; quoted fields; UTF-8 or Windows-1250; decimal commas like `1.234,56`
- **Save this reading**: the signature, what fits, what was ruled out and why, and every measured line, as a Markdown file to keep or send on. The file's rows are not in it
- **100% client-side**: nothing is uploaded. Everything runs in your browser.

## The taxonomy

Data Craft Nexus is built on a complete, interconnected taxonomy:

| Component | Count | Description |
|-----------|-------|-------------|
| Data axes | 6 | Supervision, structure, modality, scale, distribution, quality |
| Tasks | 8 | A number, a category, a future value, time until an event, groupings, anomalies, a simpler view, new examples |
| Math formulas | 144 | Across 16 domains, with the derivations behind the benchmark results |
| Models | 72 | Across 14 architecture families |
| Drift checkers | 28 | Distributional, streaming, multivariate, adversarial, DL-native |
| Pipelines | 14 | ETL, feature store, training, deployment, monitoring, RAG |
| Stages | 31 | Reusable pipeline building blocks |

The full written notes are in [`taxonomy/`](taxonomy): [Data](taxonomy/Data.md), [Math](taxonomy/Math.md), [ML Models](<taxonomy/ML Models.md>), [Model Stacking](<taxonomy/Model Stacking.md>), [Model Drift](<taxonomy/Model Drift.md>) and [ML Pipeline](<taxonomy/ML Pipeline.md>).
