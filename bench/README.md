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

## Next

- the runner: per-model time budget, checkpoints, results on disk
- regret against three baselines: always-boosting, a random eligible model,
  and the best model available
