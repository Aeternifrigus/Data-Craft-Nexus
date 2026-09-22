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
python -m dcn.run --limit 20 --budget 60 --out results/pilot.csv
python -m dcn.analyze --results results/pilot.csv
```

`run.py` fits every runnable model on every dataset, including the ones the
site rules out, and appends a row per result so it can be stopped and resumed.
`analyze.py` turns that into regret: how much worse than the best available
model each strategy was.

## What the first run found

40 datasets, 760 model runs (20 classification, 20 regression). Committed in
`results/`.

| | classification | regression |
|---|---|---|
| median regret, the site's first pick | 0.024 | 0.312 |
| median regret, best of the four shown | 0.013 | 0.176 |
| median regret, always boosting | 0.023 | 0.007 |
| the first pick was the best model | 5% | 5% |
| always boosting was the best model | 20% | 15% |

Regret is measured in balanced accuracy for classification and R² for
regression, against the best of all 23 models run.

Three defects, in the order they cost the most:

1. **The ranking is wrong for regression.** Counting matched coordinates puts
   Linear Regression first on 17 of 20 regression datasets, because its codes
   (A11, A21, A31) match a numeric table exactly. On nonlinear data that costs
   0.3 to 0.7 R². Coordinate matching rewards a model for being *describable*,
   not for being *right*.
2. **Task tags are incomplete.** AdaBoost, both SVMs and the Perceptron are
   tagged classification-only, though each has a standard regressor. AdaBoost
   would have won 9 regression datasets, by up to 0.56 R².
3. **The modality code means different things on different models.**
   Logistic Regression is tagged A32 because its *target* is categorical;
   Random Forest is tagged A38 because its *features* can be mixed. The
   conflict rule reads both as features, so Logistic Regression is ruled out
   on numeric tables and Gradient Boosting on categorical ones. That cost a
   win on 5 datasets.

None of this was visible from the taxonomy alone. It took real data.

## Next

- fix the three defects above and re-run to see what the fixes are worth
- rank by learned weights instead of counting matches, evaluated
  leave-one-dataset-out
- run the full 229 datasets rather than 40
