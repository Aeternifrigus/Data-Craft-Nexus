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
datasets collected from UCI, OpenML and elsewhere, all in one shape. 229 of
them are the right size to run here (128 classification, 101 regression,
between 200 and 20,000 rows and at most 100 columns). The simulated GAMETES
genetics datasets are excluded: the point is data somebody actually collected.

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

`evidence.py` writes `site/taxonomy/evidence.json`, which is what the site's
**The evidence** tab and the "Measured." line under each recommendation read.
A test in `tests/evidence.test.js` recomputes the published headline from
`results/per_dataset.csv`, so the page and the run cannot drift apart.

`run.py` fits every runnable model on every dataset, including the ones the
site rules out, and appends a row per result so it can be stopped and resumed.
`analyze.py` turns that into regret: how much worse than the best available
model each strategy was.

## What the runs found

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

## What is in results/

| file | one row per | what it holds |
|---|---|---|
| `pilot.csv` | dataset and model, before the fixes | signature, eligibility, rank, score, seconds, status |
| `after-fixes.csv` | dataset and model, after the fixes | the same columns |
| `per_dataset.csv` | dataset | best model and score, the first pick, best of four, boosting, and the regret of each |
| `delta.csv` | dataset | both runs side by side and the change |
| `summary.json` | run | the aggregate table above, plus which models were ruled out and why |

## Next

- rank by weights learned from these results instead of counting matched
  coordinates, evaluated leave-one-dataset-out so a dataset never scores its
  own recommendation
- run the full 229 datasets rather than 40
- publish the per-dataset table on the site next to each recommendation
