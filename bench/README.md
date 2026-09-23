# Benchmark

The site says which models fit a dataset. This directory is where that claim
gets tested against real data instead of being asserted.

## Why a Python copy of the profiler

The page measures a dataset in the browser; the benchmark has to measure
thousands of datasets in Python. Two implementations means they can drift
apart, and a benchmark that scores signatures the site never shows would prove
nothing. So `tests/test_agreement.py` runs the site's own JavaScript profiler
(`tools/js_profile.mjs`) over every fixture and compares it with the Python
port, column by column and target by target: parsing, column types, every
axis, class balance, PSI drift. Any disagreement fails.

Keeping them honest turned up one real bug: date detection used to call
`Date.parse`, which is only specified for ISO strings, so Chrome and Safari
disagreed about formats like `04.01.2025` and the same file could be measured
differently in different browsers. Both sides now use the same explicit list
of date shapes (`site/js/dates.js`, `dcn/dates.py`).

## Running

```bash
cd bench
pip install -r requirements.txt
python -m pytest -q
```

## Layout

```
dcn/csvread.py   CSV reading, ported from site/js/csv.js
dcn/stats.py     PSI and JavaScript number semantics
dcn/dates.py     the shared date shapes
dcn/profile.py   the profiler, ported from site/js/profile.js
tools/           the JavaScript side of the agreement test
tests/           the agreement test and its awkward CSVs
```

## Datasets

`dcn/datasets.py` pulls from PMLB (Penn Machine Learning Benchmarks): real
datasets collected from UCI, OpenML and elsewhere, all in one shape. 234 of
them are the right size to run here (between 200 and 20,000 rows and at most
100 columns). Two kinds are left out: the 5 simulated GAMETES genetics
datasets, because the point is data somebody actually collected, and the 33
entries PMLB marks deprecated, which duplicate datasets already in the list.
That leaves 196: 95 classification and 101 regression.

PMLB stores categorical columns as integers, which would make every dataset
look like a table of numbers and would quietly rig the modality axis. Each
dataset's `metadata.yaml` records the real type of every column, so those
columns are turned back into labels before the profiler sees them. Datasets
whose metadata is unfilled are still used, with `types_known=False` recorded
against them so the effect can be checked later.

Downloads are cached under `bench/cache/` (git-ignored).

## Models

`dcn/models.py` maps taxonomy codes to real estimators: 21 of them, plus
XGBoost and LightGBM when installed. Preprocessing follows the profiler's own
column classification, so a model is judged on the data the axes describe.

Models the taxonomy lists but this benchmark cannot run are in `NOT_RUNNABLE`
with a reason each (a CNN needs images, an HMM needs sequences, and so on).
A test fails if a new tabular model appears with neither an estimator nor a
reason, so coverage cannot quietly rot.

`BASE-HGB`, histogram gradient boosting, is kept aside as the "just reach for
boosting" baseline every recommendation will be measured against.

## Running it

```bash
python -m dcn.run --limit 20 --budget 60 --out results/pilot.csv   # fit everything
python -m dcn.analyze --results results/pilot.csv                  # how good is the advice
python -m dcn.compare --before results/pilot.csv \
                      --after results/after-fixes.csv              # what did a change do
python -m dcn.evidence --results results/after-fixes.csv           # publish it to the site
```

The full run, which is what the site now publishes:

```bash
python -m dcn.run --budget 60 --out results/full.csv                        # every dataset, seed 0
python -m dcn.run --limit 20 --seeds 3 --budget 60 --out results/full.csv   # seeds 1 and 2 on 40 of them
python -m dcn.meta --results results/full.csv --out results/meta.csv
python -m dcn.analyze --results results/full.csv
python -m dcn.learn --results results/full.csv
python -m dcn.evidence --results results/full.csv --label "195 PMLB datasets, 5250 model runs" \
    --ranked-by "the per-model prior fitted on the earlier 40-dataset run, which had already seen 40 of these datasets"
```

The runner is resumable, so more seeds can be added later (`--seeds 5`)
without refitting what is there. Run it as one process: two processes on two
cores starve the multithreaded boosting libraries, which then run out of
their time budget for reasons that have nothing to do with the model.

`evidence.py` writes `site/taxonomy/evidence.json`, which is what the site's
**The evidence** tab and the "Measured." line under each recommendation read.
A test in `tests/evidence.test.js` recomputes the published headline from
`results/per_dataset.csv`, so the page and the run cannot drift apart.

`run.py` fits every runnable model on every dataset, including the ones the
site rules out, and appends a row per result so it can be stopped and resumed.
`analyze.py` turns that into regret: how much worse than the best available
model each strategy was.

## What the first runs found

Two runs of the same 40 datasets (20 classification, 20 regression), 760
model runs each, committed in `results/`: `pilot.csv` before the taxonomy
fixes, `after-fixes.csv` after them. Regret is how far a strategy was from
the best of the 23 models run, in balanced accuracy and in R².

| median regret | classification | regression |
|---|---|---|
| the site's first pick, before | 0.024 | 0.312 |
| the site's first pick, after | 0.055 | 0.432 |
| best of the four shown, before | 0.013 | 0.176 |
| best of the four shown, after | **0.015** | 0.176 |
| always use boosting | 0.023 | **0.007** |
| a random eligible model | 0.060 | 0.240 |

The first run found three defects.

1. **The ranking is wrong**, and it is the one still open. Counting matched
   coordinates puts Linear Regression first on 17 of 20 regression datasets,
   because its codes (A11, A21, A31) match a numeric table exactly. On
   nonlinear data that costs 0.3 to 0.7 R². Coordinate matching rewards a
   model for being *describable*, not for being *right*.
2. **Task tags were incomplete**: AdaBoost, both SVMs and CatBoost were
   tagged classification-only though each has a standard regressor. AdaBoost
   beat every recommended model on nine regression datasets, by up to
   0.56 R². Fixed.
3. **The modality code meant two different things**: the target on
   classifiers, the features everywhere else, so Logistic Regression was
   ruled out on numeric tables and boosting on categorical ones. It now
   always means the features a model consumes. Fixed.

Fixing 2 and 3 made the eligible field correct: **no ruled-out model beats
the recommendations any more** (it happened on 17 of 40 datasets before), and
the best of the four shown now beats boosting on 55% of classification
datasets, up from 45%.

It also made the first pick *worse*, which is the useful part. With the pool
corrected, six to eight models tie at the top and the order between them is
the order they sit in the taxonomy: a plain Decision Tree now leads 12
classification datasets. The first recommendation is close to a random draw
from the tied models, and no amount of content fixing changes that. Only a
ranking learned from results will.

Both fixes are now pinned by tests, so neither can come back: the taxonomy
must tag a model for every task its estimator can run, and a classifier must
not be excluded from a numeric table.

`results/delta.csv` has the two runs side by side, one row per dataset. The
first recommendation changed on 20 of 20 classification datasets and got
worse on 12 of them, while the four shown stayed level and the ruled-out
winners went from 17 datasets to none. A correct pool, an arbitrary order.

## What the full run found

Every dataset that fits, 196, of which 195 could be scored: auto_insurance_symboling
has a class with 3 rows and cannot be split five ways. 5,250 model runs, with
40 of the datasets under three cross-validation seeds. Ten fits ran out of
their 60-second budget, all of them scikit-learn's GradientBoosting on large
multiclass tables. (An earlier attempt ran two processes on two cores; the
boosting libraries starved each other, and the baseline timed out on datasets
it normally fits in a second. Every timeout from that period was deleted and
refitted alone.)

Leave-one-dataset-out, with synthetic families held out whole and counted once
(see "A family counts once" below):

| median regret (95% interval) | classification, 94 datasets | regression, 101 datasets, 35 independent |
|---|---|---|
| counting coordinates | 0.046 (0.030 to 0.056) | 0.205 (0.071 to 0.303) |
| learned prior, in use | 0.015 (0.011 to 0.018) | 0.012 (0.002 to 0.027) |
| prior + interactions | 0.015 (0.009 to 0.021) | 0.012 (0.003 to 0.025) |
| prior + neighbours | 0.016 (0.009 to 0.020) | 0.012 (0.001 to 0.025) |
| always boosting | 0.014 (0.011 to 0.021) | 0.021 (0.009 to 0.055) |

- The learned order beats counting coordinates by more than luck on both
  tasks (p = 2e-6 and 2e-5).
- Against always using boosting it is level on classification (better on 47,
  worse on 40, p = 0.52) and ahead on regression (21 of 35 units, p = 0.046),
  which is just under the line and not settled: the interval on the median
  difference touches zero.
- Neither richer order beats the prior, so it stays.
- With 94 datasets, two classifiers need average ranks 2.7 apart to be told
  apart, and 10 of the 18 are within that of the best (LightGBM). On regression
  the 35 units give a critical difference of 5.0, with 10 of 20 tied with the
  best (scikit-learn's GradientBoosting).
- One split decides a lot. Across three seeds the best model stayed the same on
  30% of classification datasets and 50% of regression ones, and the first
  pick and boosting swapped places on 10 of 20 and 4 of 20.

Counting the families one by one, which is how the first attempt at this
analysis did it, made the prior's regression regret 0.007 and made the
interactions order look better than the prior at p = 0.0008. Both effects
came from 54 sister datasets being ranked by weights learned on each other.

The run also found a rule that throws away winners. Models that need numeric
features (A31) are ruled out on all-categorical tables (A32), but the pipeline
one-hot encodes categories, and a ruled-out model won on 9 datasets: Logistic
Regression, Naive Bayes and Bayesian linear regression among them, by up to
0.13 R² on solar_flare (elsewhere by 0.002 to 0.023). The details are in `summary.json`
under `ruled_out_winners`.

## Is it more than luck?

Every comparison the site publishes goes through `dcn/significance.py`:

- **an interval** around each median, from resampling which datasets were in
  the benchmark (10,000 bootstrap draws)
- **a paired test**: the Wilcoxon signed-rank test on per-dataset differences,
  since every strategy was judged on the same datasets, with Holm's correction
  when the order in use is compared against several alternatives at once
- **a ranking of every model**: the Friedman test, and the Nemenyi critical
  difference from Demšar, *Statistical Comparisons of Classifiers over
  Multiple Data Sets* (JMLR, 2006). Two models whose average ranks differ by
  less than it cannot be told apart on this benchmark

The order in use is compared with the two things it claims to beat: counting
coordinates and always using boosting, Holm-corrected over those two. The
results are in "What the full run found" above.

Repeated cross-validation seeds are averaged into one score per dataset and
model before any of this. Seeds measure how much one split can move a score;
counting them as extra datasets would make every difference look more certain
than it is.

`python -m dcn.learn` prints the paired comparisons; `python -m dcn.evidence`
publishes them, and `tests/evidence.test.js` recomputes the medians, the win
counts and every model's average rank from the committed data.

### A family counts once

PMLB's regression collection is mostly generated, not collected: of the 101
regression datasets that fit, 54 come from Friedman's benchmark functions and
14 from Strogatz's equations. Sisters from one generator share a winner, so
treating them as separate datasets does two things wrong. Leave-one-dataset-out
lets a Friedman dataset be ranked by weights learned on its 53 siblings, and
every test counts 54 correlated results as 54 pieces of evidence.

`family_of()` in `dcn/significance.py` groups them (Friedman, Strogatz,
Feynman, BNG; every other dataset is its own family). `learn.py` holds a
family out whole when it ranks a member, and every published number counts a
family once: medians, intervals and paired tests over units, where a family's
unit is the mean over its datasets, and model ranks over family-averaged
scores. The coverage threshold and the neighbour bandwidth ignore a dataset's
own family, since a sister is always close.

## Which learned order ships

`learn.py` judges three orders leave-one-dataset-out, simplest first:

| order | what it is |
|---|---|
| `prior` | each model's average percentile rank, shrunk toward the middle |
| `prior_fit` | the prior plus ridge-fitted interactions between the dataset's features and the model's family |
| `prior_knn` | the prior blended with the model's percentile rank on the 10 nearest benchmark datasets |

The blend is `(4 * prior + sum w * rank) / (4 + sum w)` with
`w = exp(-(distance / h)^2)`, where distance is the same eight-feature distance
"datasets like yours" uses and `h` is the median distance from a benchmark
dataset to its nearest other one. Near neighbours pull a model's score toward
what it did on them; far ones barely move it. Those settings are fixed in
advance, not tuned on the results, and in the leave-one-dataset-out runs the
neighbours, the scale and `h` come from the other datasets only.

`choose()` applies a rule fixed before the full run: start from the prior, and
let a richer order replace it only if its median regret is no worse on either
task and it is better by more than luck on at least one (Wilcoxon,
Holm-corrected over the two tasks, more wins than losses). The decisions are
written into `ranking.json` and shown on the evidence tab. Neither richer order
qualified on the 40-dataset run or on the full one.

The page and the benchmark compute the neighbour order the same way:
`tests/test_agreement.py` fits it on the committed run, points the JavaScript
at it, and compares both rankings on every fixture.

## What is in results/

| file | one row per | what it holds |
|---|---|---|
| `pilot.csv` | dataset and model, before the fixes | signature, eligibility, rank, score, seconds, status |
| `after-fixes.csv` | dataset and model, after the fixes | the same columns |
| `full.csv` | dataset, model and seed, the full run | the same columns; what the site publishes |
| `meta.csv` | dataset | the eight meta-features "datasets like yours" and the neighbour order use |
| `per_dataset.csv` | dataset, full run | best model and score, the first pick, best of four, boosting, and the regret of each |
| `ranking-lodo.csv` | dataset, full run | the score and model each order picked, leave-one-dataset-out with families held out |
| `summary.json` | full run | the recorded headline, plus which models were ruled out and why |
| `delta.csv` | dataset | the two 40-dataset runs side by side and the change |
| `seeds.csv` | dataset, model and seed | an early three-seed check on seven datasets |

## Next

- let numeric-feature models run on all-categorical tables, since the pipeline
  encodes categories, and check the 9 lost winners come back
- weight a synthetic family once when fitting the prior, declared before the
  next run
- more collected regression data: 35 independent units is what limits every
  regression claim here, and more generated datasets would not help
