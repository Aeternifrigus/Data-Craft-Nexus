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
dcn/corrupt.py   damage made on purpose: missing cells, junk text, wrong labels
dcn/messy.py     what that damage did, against the clean run
dcn/drift.py     the drift checker benchmark
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

## References

`BASE-HGB`, default histogram boosting, is what every recommendation has been
measured against. It is a low bar: a reviewer's first question is whether the
order beats boosting that somebody bothered to tune, or a model built for small
tables. Two references answer that. Neither is ever recommended; every code that
starts with `BASE-` is outside the taxonomy.

| code | what | where it runs |
|---|---|---|
| `BASE-HGB-TUNED` | histogram boosting, tuned: scikit-learn's defaults and 9 random configurations (learning rate, number of trees, tree and leaf size, regularisation, feature sampling), the best chosen by cross-validation inside each training fold | everywhere, as part of every run |
| `BASE-TABPFN` | [TabPFN](https://github.com/PriorLabs/TabPFN), a model pretrained for small tables | only where it is installed and its weights can be downloaded |

The tuning budget is small on purpose, and stated on the page: ten
configurations, a standard practitioner's search, not a competition. Two
lessons from building it:

- **The defaults are a candidate.** scikit-learn's defaults are good, and a
  random search that cannot pick them sometimes does worse than not tuning.
- **Small data needs cross-validation to choose on.** A first version chose on
  one 80/20 split of the training fold. On a 200-row dataset that is 40 rows to
  judge ten configurations by, mostly noise, and tuned boosting lost to its own
  defaults on 11 of the first 20 small datasets. Three-fold cross-validation
  judges every configuration on every row of the training fold; only above
  3,000 rows, where a fifth of them is a real validation set, is one split
  used (`inner_splits`). Early stopping stays off, as scikit-learn leaves it
  below 10,000 rows; the number of trees is tuned instead.

References get their own time budget (`--reference-budget`, 600 seconds by
default), since tuning fits a model thirty times. Run the tuning with one
thread per process (`OMP_NUM_THREADS=1`): histogram boosting spends most of
its time on thread coordination on small tables.

### Running TabPFN on your own machine

TabPFN's weights are downloaded from Hugging Face behind a Prior Labs login,
which the environment that produced the committed results could not reach. It
is added to an existing results file like this:

```bash
pip install tabpfn
export TABPFN_TOKEN=...        # from your Prior Labs account; or let it open the browser
cd bench
python -m dcn.run --models BASE-TABPFN --out results/full.csv
```

Then re-run `dcn.analyze`, `dcn.learn` and `dcn.evidence` as below, and the
evidence tab picks TabPFN up.

Which weights, and what the licence allows:

- `DCN_TABPFN_VERSION` picks the version, `v2` by default. TabPFN-2's weights
  are under the Prior Labs License, Apache 2.0 with an attribution requirement.
  The later versions (`v2.5`, `v2.6`, `v3`, `v3.5`) are stronger and under
  non-commercial licences; check them before publishing results.
- `DCN_TABPFN_MAX_ROWS` is the largest dataset TabPFN is run on, 1,000 by
  default, which is what TabPFN-2 supports on a CPU. On a GPU, or with a later
  version, raise it (the later versions allow 5,000 rows on a CPU). Larger
  datasets are recorded as skipped with the reason, so TabPFN's comparisons are
  made only on the datasets it ran on, and the page says how many.

`tools/run_tabpfn.sh` does all of the above in one command, from installing
TabPFN to rebuilding the page and running every test. On an Apple-silicon Mac
TabPFN uses the GPU (MPS) without being told to. Its arguments go to
`dcn.run`, so `tools/run_tabpfn.sh --limit 5` is a quick first try. The whole
pipeline was dry-run with a stand-in model in place of TabPFN's weights.

### What TabPFN has to show, written before it ran

Of the benchmark's datasets, 138 have 1,000 rows or fewer, the size TabPFN-2
is run on by default: 48 classification datasets in 44 independent
units, and 90 regression datasets in only 24, because most small regression
sets come from a few generator families. TabPFN-2's authors report it beating
tuned tree ensembles on datasets of up to 10,000 rows ([Hollmann et al.,
Nature, 2025](https://www.nature.com/articles/s41586-024-08328-6)).

**The prediction:** on those small tables, TabPFN-2 beats tuned boosting, so
the size of a table decides which model to try first. That would be the first
place the data itself changes the advice.

**The rule** (`choose_small_lead()` in `dcn/learn.py`, committed before any
TabPFN result existed): on the datasets TabPFN ran on, TabPFN goes before tuned
boosting only if its median regret is no worse on either task and it is better
by more than luck on at least one (paired Wilcoxon over independent units,
Holm-corrected over the two tasks, more wins than losses). It is the rule every
other change of order has had to pass. Its verdict is written to
`ranking.json` as `small_lead`, with the size of the largest table it covers.

**What follows either way:** if TabPFN passes, the page's next change shows it
first on tables up to that size and tuned boosting first above it. If it fails,
tuned boosting stays first everywhere and the evidence tab says TabPFN was
tested, on how many units, and lost or tied. With 24 regression units, a tie on
regression is the likely outcome there, and it would not be evidence that the
two are equal, only that this benchmark cannot tell them apart.

### Which TabPFN ran, declared before it ran

TabPFN-2's weights could not be reached from where the benchmark runs. The
first TabPFN's could ([Hollmann et al., ICLR
2023](https://arxiv.org/abs/2207.01848)): its checkpoint is public on the
project's `tabpfn_v1` branch, under Apache 2.0, and `DCN_TABPFN_VERSION=v1`
runs it (`dcn/models.py` pins the checkpoint by its SHA-256 before loading it).
It is the model that started the family, and it is narrower than its
successors:

- **Classification only.** TabPFN-1 has no regressor, so the rule above is
  applied to classification alone. Holm's correction over one task changes
  nothing. Regression stays untested, and the page says so.
- **Up to 1,000 training rows, 100 features and 10 classes.** Datasets over
  1,000 rows are skipped as before. A table with more than 100 columns after
  one-hot encoding, or more than 10 classes, is recorded as an error with that
  reason, so TabPFN-1 is judged only where it ran.
- **32 ensemble members**, as its authors recommend. The package defaults to 3.
- **Seed 0 only**, like tuned boosting.

The prediction and the rule are unchanged: on the small classification tables
it ran on, TabPFN-1 goes before tuned boosting only if its median regret is no
worse and it is better by more than luck (paired Wilcoxon over independent
units, more wins than losses).

If it passes, the page shows it first on classification tables within its
limits, and names the version: a result for TabPFN-1 is evidence that a
pretrained small-table model beats tuned boosting on small tables, and a lower
bound for the later versions, which their authors report stronger. It is not a
measurement of them. If it fails, tuned boosting stays first, and the evidence
tab says which TabPFN lost, on how many units.

### What TabPFN-1 showed

It ran on 42 classification datasets (38 independent units); 47 were over
1,000 rows, 5 had more than 10 classes or 100 features after encoding, and one
had a class too small to split. Against tuned boosting it was better on 21
units, worse on 16 and level on one: p = 0.26, not more than luck, so
`small_lead.led` is false and tuned boosting stays first on small tables. Its
median regret was no worse. Against the order's own first pick it was better
on 25 units and worse on 12 (p = 0.13, Holm over the four references). The
size of a table still does not change the first recommendation, as far as
TabPFN-1 can say; TabPFN-2 and later remain unmeasured here.

## Messy data

PMLB's datasets are clean, and the files people upload are not: the quality
axis (A62 missing, A64 noisy) had no evidence behind it. `dcn/corrupt.py`
damages clean datasets in known ways, and `--corrupt` runs the benchmark on the
damaged copies:

| condition | what is done |
|---|---|
| `missing_10`, `missing_30` | 10% or 30% of feature cells blanked, completely at random |
| `dirty_5` | 5% of the cells in numeric columns replaced by spreadsheet junk (`?`, `n/a`, `#VALUE!`, ...) |
| `labels_10` | 10% of the training labels replaced by another row's label, inside each training fold only, so every model is still scored on the true labels |

The damage is seeded, so a condition damages a dataset the same way for every
model. The profiler then measures the damaged file, so the recorded signature
says whether it noticed. Damaged data goes through what any real pipeline does
with a messy file: junk in a numeric column becomes missing, and categories
become text.

```bash
OMP_NUM_THREADS=1 python -m dcn.run --limit 20 --corrupt missing_30 --budget 120 --out results/messy.csv
python -m dcn.messy --messy results/messy.csv --clean results/full.csv   # compare with the clean run
```

Results carry a `condition` column; a results file written before it existed
is upgraded in place the next time the runner appends to it.

`dcn/messy.py` compares each damaged run with the clean run of the same
datasets, and replays the order's first pick on the damaged file exactly as
the runner makes it (`results/messy-picks.csv`). All four conditions ran on the
same 40 datasets, 20 per task; the 20 regression datasets are 8 independent
units, 13 of them from Friedman's functions. Medians are over units:

| | 10% missing | 30% missing | junk text | wrong labels |
|---|---|---|---|---|
| the profiler noticed (flag) | 100% (A62) | 100% (A62) | 100% of 36 (A64) | cannot |
| median loss, classification | 0.027 | 0.069 | 0.011 | 0.017 |
| median loss, regression | 0.101 | 0.268 | 0.039 | 0.043 |
| first pick's regret, classification (0.016 clean) | 0.007 | 0.011 | 0.009 | 0.019 |
| first pick's regret, regression (0.003 clean) | 0.009 | 0.049 | 0.008 | 0.033 |

- The profiler sees missing cells every time. It first missed the junk in 4
  of the 36 tables that got some, because A64 was judged on the average over
  all columns: junk in the few numeric columns of a mostly categorical table
  stayed under the threshold. Three of the four were read as having missing
  values instead, since `n/a` and `-` are missing-value words; one as clean.
  A64 is now judged per column as well (more than 1% of a column's values
  looking wrong), in the page and in the port: all 36 are flagged, and none of
  the 196 clean benchmark datasets is.
- Label noise cannot be seen in a file, and on classification it cost about
  what 10% of missing cells cost.
- The order does not look at data quality: it is a fixed ranking per model,
  so LightGBM comes first on classification and gradient boosting on
  regression, damaged or not. On classification that pick held up: its regret
  did not rise by more than luck under any damage. On regression it did,
  under 30% missing cells (0.003 to 0.049, 95% interval of the rise 0.002 to
  0.175) and under noisy targets (0.003 to 0.033, interval 0.001 to 0.057).
  Tree ensembles chase noisy targets and lose the most to heavy gaps; Lasso,
  Elastic Net and the linear SVM lost the least. Eight units is thin, but both
  intervals exclude no change.
- On classification, Naive Bayes lost least to missing cells, and a single
  decision tree lost most under every kind of damage.
- An order that reads data quality does not help. Declared first
  (`quality_order()`): for each kind of damage, rank the models by their
  average percentile rank on the other datasets' damaged runs, and compare its
  first pick with the fixed order's, leave-one-dataset-out, by the rule
  `choose()` applies. It lost more often than it won on classification under
  every kind of damage (at 30% missing, 5 better and 12 worse) and was level
  or worse on regression, so it replaces the fixed order nowhere.

Damage made on purpose is random, and real gaps are not. PMLB has datasets
that arrived with missing values of their own: 10 classification datasets in
the full run (8 independent, the three horse colic tables being one) and one
regression dataset. OpenML, which has more, could not be reached from where
this ran. On those 8 units the order's leave-one-dataset-out regret was 0.020,
against 0.016 on the 70 complete ones (Mann-Whitney, p = 0.50), and default
boosting's 0.014 against 0.015: no evidence of a difference either way, at a
size that could only have shown a large one.

## Before you trust a score

The page's first section runs four checks on an upload (`site/js/checks.js`):
one column that predicts the target almost perfectly on its own, a column
that looks like a row ID, more repeated rows than chance, and a date column
with the rows declared independent. `dcn/checks.py` measures the first three
with the page's own JavaScript, on the CSV a user would upload:

```bash
python -m dcn.checks --out results/checks.csv   # about a minute
```

| check | fired on the 195 clean datasets | planted in 40 datasets | caught |
|---|---|---|---|
| one column predicts the target | 5 | the target renamed or rescaled | 40 |
| | | the same, 1% of rows changed | 19 |
| | | the same, 5% of rows changed | 0 |
| looks like an ID | 3 | the row numbers, shuffled | 40 |
| | | a random code per row | 40 |
| more repeated rows than chance | 22 | 2% of the rows copied | 32 |

The 40 are the datasets the messy-data benchmark damaged, 20 per task. The
date check is a rule, not an estimate, so there is nothing to measure.

**On clean data, the flags are mostly real.** The leak check fired on
560_bodyfat (body fat is computed from density by a formula), cars (every
brand belongs to exactly one class), collins, irish and strogatz_vdp2: in each,
one column nearly determines the target, used alone on rows it was not fitted
on. The ID check fired on a phone number in churn, a row counter and a text
label in collins, and a year that counts up row by row in 695_chatfield_4. The
last is a time index rather than an ID: the flag's advice to drop it is wrong
there, and a time split is right. Repeated rows exceed chance in 22 datasets,
among them the six thyroid tables (six names for one table) and both wine
quality tables, which the benchmark itself cross-validated at random.

**What the leak check cannot do.** It looks at one column at a time, and its
threshold, 0.99, was set before this ran. It catches a column that is the target
under another name. With 5% of the rows changed, a leak looks like a strong
honest column, and it caught none. A lower threshold would catch more and fire
more on clean data; the numbers for 0.95 and 0.97 are in `evidence.json`
(`checks.leak_thresholds`) and on the evidence tab, reported and not used:

| threshold | clean datasets flagged | caught, 1% changed | caught, 5% changed |
|---|---|---|---|
| 0.95 | 14 | 38 | 16 |
| 0.97 | 9 | 38 | 1 |
| 0.99 (in use) | 5 | 19 | 0 |

**Two rules changed after the first run, and here is how.** The repeat check
first compared copies only with what chance would give if the columns were
independent. It fired on 33 clean datasets, most of them tables made only of
categories, where identical rows are natural (the same voting record, the same
answers) and are not copies at all. It now also needs a measurement column, a
numeric one with at least 50 distinct values. That brought the clean flags to
22 and planted catches from 37 to 32 of 40. The eight it misses have no
numeric column with 50 distinct values: tables of categories or of on/off
switches, and letter, whose 16 numeric columns take 16 values each. And
the leak check first missed the target rescaled in 4 regression datasets with
skewed targets, where cutting the column into 32 bins loses the tails; a rank
correlation is now checked beside it, and it caught all 40.

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
OMP_NUM_THREADS=1 python -m dcn.run --models BASE-HGB-TUNED --reference-budget 1800 --out results/full.csv
python -m dcn.evidence --results results/full.csv --label "195 PMLB datasets, 5446 model runs" \
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
has a class with 3 rows and cannot be split five ways. 5,446 model runs, with
40 of the datasets under three cross-validation seeds and tuned boosting on
every dataset. Ten fits ran out of
their 60-second budget, all of them scikit-learn's GradientBoosting on large
multiclass tables. (An earlier attempt ran two processes on two cores; the
boosting libraries starved each other, and the baseline timed out on datasets
it normally fits in a second. Every timeout from that period was deleted and
refitted alone.)

Leave-one-dataset-out, with families held out whole and counted once (see "A
family counts once" below):

| median regret (95% interval) | classification, 94 datasets, 78 independent | regression, 101 datasets, 35 independent |
|---|---|---|
| counting coordinates | 0.048 (0.034 to 0.068) | 0.086 (0.038 to 0.257) |
| learned prior, in use | 0.016 (0.012 to 0.021) | 0.012 (0.002 to 0.027) |
| prior + interactions | 0.016 (0.010 to 0.021) | 0.012 (0.003 to 0.025) |
| prior + neighbours | 0.018 (0.013 to 0.023) | 0.012 (0.001 to 0.025) |
| always boosting | 0.014 (0.011 to 0.022) | 0.021 (0.009 to 0.055) |
| boosting, tuned (reference) | 0.011 (0.008 to 0.018) | 0.006 (0.002 to 0.017) |

Holm-corrected over three comparisons (coordinates, default boosting, tuned
boosting):

- The learned order beats counting coordinates by more than luck on both
  tasks (p = 3e-6 and 5e-4).
- Against default boosting it is level on both: 39 better and 35 worse of 78
  classification units (p = 0.61), 21 and 14 of 35 regression units (p = 0.091).
  With only two comparisons in the family, before tuned boosting was added,
  the regression edge was p = 0.046.
- Tuned boosting beats it on classification by more than luck (49 units to
  24, p = 0.028) and is level on regression (19 units to 16, p = 0.55).
- Neither richer order beats the prior, so it stays.
- Tuned boosting beat default boosting on 62 of 94 classification datasets (25
  worse) and 88 of 101 regression datasets (12 worse), and has the best average
  rank of everything that ran: 6.2 of 19 on classification, 6.0 of 21 on
  regression. With 78 independent units two classifiers need average ranks 3.2
  apart to be told apart, and 8 of the 19 are within that of it; on regression the 35
  units give a critical difference of 5.3, with 12 of 21 within it.
- The gap to the best of everything that ran is 0.017 on classification and
  0.013 on regression for the order in use, against 0.011 and 0.006 for tuned
  boosting.

Tuned boosting took about 30 fits per dataset and fold, 7 hours of fitting
for all 195 datasets, a little under 4 hours of wall time on two cores. It ran
with one thread per process (`OMP_NUM_THREADS=1`), two processes at once, and
none of its fits ran out of the 1,800-second reference budget; the slowest,
`splice` (60 categorical columns, one-hot encoded), took 1,439 seconds.
- One split decides a lot. Across three seeds the best model stayed the same on
  30% of classification datasets and 50% of regression ones, and the first
  pick and boosting swapped places on 10 of 20 and 4 of 20.

Counting the families one by one, which is how the first attempt at this
analysis did it, made the prior's regression regret 0.007 and made the
interactions order look better than the prior at p = 0.0008. Both effects
came from 54 sister datasets being ranked by weights learned on each other.

The run also found a rule that threw away winners. Models that need numeric
features (A31) were ruled out on all-categorical tables (A32), but the pipeline
one-hot encodes categories, and a ruled-out model won on 9 datasets: Logistic
Regression, Naive Bayes and Bayesian linear regression among them, by up to
0.13 R² on solar_flare (elsewhere by 0.002 to 0.023). The details are in `summary.json`
under `ruled_out_winners`, which records the rule as it was during the run.

The rule is gone: a model for numbers and independent rows is now usable on a
table of categories, with a caution to encode them (`encodes()` in
`recommend.js`, `_encodes()` in the port). Recomputed leave-one-dataset-out, it
changed none of the four models shown on the 26 categorical tables, because the
learned order ranks boosting above those models there. It did change the order
the learned one is compared with: counting coordinates, now allowed the same
models, dropped from 0.205 to 0.086 median regret on regression, and the
table above is with the rule fixed. A forecaster is still ruled out on a table
of categories: encoding does not make a numeric series.

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

The order in use is compared with the two things it claims to beat, counting
coordinates and always using boosting, and with tuned boosting as a reference,
Holm-corrected over the three. The results are in "What the full run found"
above.

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

The classification collection has the same problem in another form: tables
that appear under several names. The Garvan thyroid records are there six
times with six targets (allbp, allhyper, allhypo, allrep, dis, hypothyroid),
horse colic three times, the handwritten digits of mfeat as four feature
sets, chess and kr_vs_kp and mushroom and agaricus_lepiota are exact
duplicates, and one generator each is behind led7 and led24, waveform_21 and
waveform_40, and monk1 to monk3. Counted once, 94 classification datasets are
78 independent units. Before they were, the order in use lost to tuned
boosting at p = 0.016 and drew with default boosting at 47 to 40; now it is
p = 0.028 and 39 to 35. Nothing changed direction.

`family_of()` in `dcn/significance.py` groups them (Friedman, Strogatz,
Feynman, BNG, and the shared tables in `SHARED_TABLES`; every other dataset is
its own family). `learn.py` holds a
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

`choose_lead()` applies the same rule to tuned boosting against the order in
use. It was no worse on either task and better by more than luck on
classification (49 of 78 units to 24, p = 0.029; regression 19 of 35 to 16,
p = 0.53), so `ranking.json` names it as the lead and the page shows it first,
above the order's picks, on labelled tables of numbers, categories or both
predicting a category or a number: what the benchmark covered. The rule was
written for the learned orders before the full run; applying it to the
reference came after tuned boosting's result was known, and the page says so.

The page and the benchmark compute the neighbour order the same way:
`tests/test_agreement.py` fits it on the committed run, points the JavaScript
at it, and compares both rankings on every fixture.

## Drift checkers

The site recommends drift checkers too, and until now it could only say which
ones match the data's coordinates. `dcn/drift.py` puts them through the same
kind of test the models get: drift of a known kind is injected into real data,
and every checker is asked whether it sees it.

A case is one PMLB classification dataset (2,000 rows or more, at least one
continuous feature, no missing values, one per family: 18 datasets), shuffled
and cut into a model's training rows, a reference window of 500 rows and a
current window of 250. Each dataset is cut five ways, and each cut gets seven
scenarios, 630 cases in all:

| scenario | what changes in the current window |
|---|---|
| `none` | nothing. Any alarm is a false alarm |
| `shift` | one continuous feature moves by 0.3 of its standard deviation |
| `scale` | its spread widens by half, around the same centre |
| `correlation` | the feature most correlated with another is shuffled between rows: each feature looks exactly as before on its own, but the relationship is gone |
| `selection` | rows are drawn more often the higher one feature is, the way a change in who shows up changes the data |
| `label_shift` | the most common class is drawn half as often |
| `concept` | the features stay exactly as they were, but where one feature is above its median, 30% of the labels change |

Every checker runs as its card says, at the threshold the card states. Tests
on single features run on every feature, Bonferroni-corrected, which is what
DR-M9's card says multi-feature use needs. Where a card leaves the threshold to
the user (KL, and Wasserstein, whose card says to set it in each feature's own
units), it is calibrated on the reference window: the 95th percentile of the
same number between random parts of the reference. The checkers that watch a
model's errors are given gradient boosting's error stream, reference rows then
current rows, and count only if they fire on the current rows. The streaming
detectors run at river's defaults, which is how most people meet them. Five
cards are not run, each with its reason in the evidence: DR-M9 is not a
checker on its own, DR-M11 has no implementation among the dependencies,
river does not ship DR-C8, and DR-DL1 and DR-DL2 are made for images and audio.

The score, declared before the run and what the site orders drift checkers by:
the share of the six kinds of drift caught, on average, minus the false alarm
rate. Every rate is a mean over datasets of each dataset's own rate, so a
dataset counts once.

```bash
OMP_NUM_THREADS=1 python -m dcn.drift --reps 5 --out results/drift.csv   # about half an hour
python -m dcn.drift --rerun DR-M12 --out results/drift.csv              # recompute one checker on the same cases
```

What it found, as a share of cases (the full table is in the evidence tab):

| checker | false alarms | shifted feature | wider spread | broken correlation | who is sampled | class mix | label meaning | net |
|---|---|---|---|---|---|---|---|---|
| DR-M13 Anderson-Darling | 7% | 96% | 89% | 13% | 99% | 33% | 9% | 0.50 |
| DR-M3 chi-square on binned values | 1% | 70% | 89% | 9% | 92% | 17% | 4% | 0.46 |
| DR-M12 Cramér-von Mises | 2% | 91% | 57% | 7% | 98% | 22% | 3% | 0.44 |
| DR-M14 energy distance | 2% | 56% | 26% | 33% | 99% | 38% | 7% | 0.41 |
| DR-CV1 adversarial validation | 0% | 73% | 78% | 76% | 17% | 0% | 0% | 0.41 |
| DR-M1 Kolmogorov-Smirnov | 7% | 84% | 62% | 3% | 96% | 17% | 1% | 0.37 |
| DR-M2 PSI above 0.25 | 0% | 50% | 61% | 0% | 63% | 0% | 0% | 0.29 |
| DR-MV1 Mahalanobis | 24% | 96% | 53% | 18% | 96% | 43% | 18% | 0.29 |
| DR-M8 error drift | 26% | 41% | 32% | 54% | 24% | 40% | 94% | 0.22 |
| DR-C1 DDM | 16% | 17% | 11% | 32% | 8% | 13% | 81% | 0.12 |
| DR-M5 Jensen-Shannon above 0.1 | 0% | 17% | 13% | 0% | 3% | 0% | 0% | 0.06 |
| DR-C3 Page-Hinkley | 0% | 0% | 0% | 6% | 0% | 0% | 0% | 0.01 |

- Tests on the whole distribution of each feature did best: Anderson-Darling,
  chi-square and Cramér-von Mises caught 46% to 57% of the injected drift, with
  1% to 7% false alarms.
- No test on single features can see a broken correlation or a change in what
  the labels mean, by construction. Adversarial validation caught 76% of the
  broken correlations with no false alarm, and only the checkers that watch
  the model's errors caught changed labels (error drift 94%, EDDM 98%, DDM
  81%), at a price: they fired on 26%, 41% and 16% of the windows where
  nothing had changed. Error drift's 20% rule is too tight for 250 rows.
- Conventional thresholds are cautious. PSI above 0.25 never fired falsely
  but caught half of the 0.3 standard deviation shifts, and Jensen-Shannon at
  the card's 0.1 on the divergence almost never fired at all.
- Mahalanobis distance fired on 24% of the quiet windows: the chi-square its
  card calls a principled p-value assumes Gaussian features.
- At river's defaults, 250 rows after a change are too few for most streaming
  detectors: Page-Hinkley and KSWIN caught almost nothing, ADWIN and HDDM_A
  about 40% of the changed labels.

The benchmark also found five things wrong, all fixed:

- Six drift cards linked to the wrong function: KL and Wasserstein had each
  other's, ANOVA pointed at Alibi Detect, error drift at `anderson_ksamp`, MMD
  at `cramervonmises_2samp`, and the p-value card at `mannwhitneyu`.
- Error drift (DR-M8) claimed to work without labels. It compares predictions
  with the truth, so it now needs them, like every checker that watches errors.
- A mixed table, numbers and categories together, which is most real files,
  was never offered a distribution test: only checkers for numeric (A31) or
  categorical (A32) columns exist, and A38 matched neither. They now fit it.
- `scipy.stats.cramervonmises_2samp` (1.17) gives tied values their average
  rank and then treats the ranks as distinct, so on a column that is mostly
  zeros it calls two samples of the same data different at p < 1e-9. The
  benchmark computes the statistic from the empirical CDFs, and on columns
  with few values takes the p-value from permutations, because there the
  limiting distribution fires too often: on a two-valued column, more than
  fifty times too often at p < 0.001.
- river's FHDDM is documented to take 1 for an error, but fires when the
  stream's mean falls, so given errors it fires when the model gets better.
  It is given correct predictions, which is the paper's reading.

What it cannot say: each kind of drift was injected at one strength, so the
rates are for that strength; the windows are 500 and 250 rows; the datasets are
classification only; detectors tuned for their stream would do better than
river's defaults; and the net score weights the six kinds equally, which is a
choice. `tests/drift.test.js` recomputes every published rate from
`results/drift.csv`.

## Forecasting

Until this run, "A future value" put its cards in coordinate order, because
nothing had been measured. `dcn/forecast.py` measures it: it runs the take-home
script's own forecasting functions, generated by the page's JavaScript and
executed here, on real series, and `dcn/forecast_learn.py` asks whether the
kind of series should change which forecaster comes first.

```bash
python -m dcn.forecast --out results/forecast.csv          # resumable
python -m dcn.forecast_learn --results results/forecast.csv
```

### What the data has to show, written before it ran

- **The methods** are the script's: carrying the last value forward, the
  average so far, the value one season earlier, ARIMA (its order chosen by AIC
  among the small ones), exponential smoothing (a trend, and a season of at most
  24 rows, where AIC prefers them), Croston's method with Syntetos and Boylan's
  correction, and TSB. Each predicts every scored value from the values before
  it, one step ahead, in the script's five time-ordered folds, scored by R²
  (the page's default) and by mean absolute error.
- **The series**: 20 drawn with a fixed seed from each of 12 public
  collections, the most recent 200 values of each. M4's yearly, quarterly,
  monthly, weekly, daily and hourly series; monthly car-part sales (Hyndman et
  al., complete series only); monthly Australian prescriptions (PBS), retail
  turnover and livestock slaughtered (tsibbledata); and M5's daily Walmart
  sales of one item in one store, and the same summed by store and department.
  The two M5 collections are cut from one table and count as one unit: 11
  independent units. All are fetched from GitHub copies.
- **What the page measures**: each series gets dates at its step, and the
  page's own rules (`site/js/series.js`, held to `dcn/series.py` by
  `tests/test_series.py`) read its period from them and sort it into one of six
  kinds. Intermittent or lumpy by Syntetos, Boylan and Croston's cut-offs
  (average demand interval at least 1.32, and for lumpy a squared coefficient
  of variation of the sizes at least 0.49); otherwise by seasonal and trend
  strength from a classical decomposition, each at least 0.5 or not: seasonal
  and trending, seasonal, trending, or neither.
- **The prediction**: the kind decides which forecaster is best. Croston-type
  methods on intermittent demand, seasonal smoothing on seasonal series.
- **The rule** (`choose()` in `dcn/forecast_learn.py`): two orders, each fitted
  on the other units and judged on the one held out. One fixed order for every
  series, and one kept per kind, shrunk toward the fixed order by 20 series'
  worth. The kind order replaces the fixed one only if its median regret over
  units is no worse under either score and it is better by more than luck under
  at least one: paired Wilcoxon over the units, Holm-corrected over the two
  scores, more wins than losses. Regret is in R², and in mean absolute error
  divided by what carrying the last value forward missed by on the same rows.
  The six kinds, their thresholds and the shrinkage were fixed here and are not
  tuned on the result.
- **What follows either way**: the forecasting cards are ordered by what was
  measured, not by coordinates. If the kind order passes, the order depends on
  the kind of the upload; if it fails, every series gets the fixed order and
  the evidence tab says the kind did not earn it. Either way each card says how
  often it was the best forecaster on series of the upload's kind, and how
  often doing nothing beat every model there.
- **What it cannot say**: 20 series a collection is enough to compare orders
  over units and thin inside a kind. Tuned boosting on recent changes, the
  script's reference, is not in this run: it fits 150 models a series, more
  than the machine this ran on could afford. Prophet is not run (it needs Stan).
  ARIMA is not seasonal, as in the script, and smoothing seasons stop at 24
  rows, as R's `forecast::ets` does, so a weekly series' yearly cycle is used
  only by doing nothing.

### What the run found

240 series, 1,656 forecasts scored; exponential smoothing could not be fitted
on 8. How often each forecaster was the best of the four, by R², and how often
doing nothing beat all four:

| kind | series | collections | ARIMA | smoothing | Croston (SBA) | TSB | nothing beat all four |
|---|---|---|---|---|---|---|---|
| seasonal and trending | 68 | 8 | 7% | 91% | 0% | 2% | 19% |
| seasonal | 17 | 6 | 35% | 65% | 0% | 0% | 47% |
| trending | 100 | 9 | 41% | 47% | 0% | 12% | 33% |
| neither | 13 | 4 | 15% | 46% | 8% | 31% | 8% |
| intermittent | 34 | 4 | 9% | 12% | 38% | 41% | 21% |
| lumpy | 8 | 2 | 25% | 0% | 50% | 25% | 38% |

The rule's verdict: the kind order does not replace the fixed one. By R² it was
better on 4 collections, worse on none and level on 7; by mean absolute error
better on 3 and worse on 1 (M5). Holm-corrected p = 0.25 under both. The median
over collections is the same for both orders (0.177 R², 0.061 in units of doing
nothing's error), because the two orders pick the same forecaster, exponential
smoothing, on every kind except intermittent demand, and intermittent series
came from only four collections (car parts, M5, PBS, livestock). With four
untied pairs the smallest two-sided p-value Wilcoxon can give is 0.125, so the
test could not have passed. Inside the intermittent kind the gain was large:
median regret 0.121 R² with the fixed order, 0.034 with the kind order; on car
parts, the collection that is all intermittent, 0.88 against 0.31.

What the page does with it: the forecasting cards are ordered by the fixed
prior (exponential smoothing, ARIMA, TSB, Croston) instead of by coordinates,
each card says how often it was the best on the upload's kind of series, and
when the kind order would have put a different forecaster first, the note
above the cards says which, how often it won, and why the order did not
change. The rule is not loosened after the fact: a follow-up run with enough
collections of intermittent demand for the orders to disagree in more of them,
declared before it runs, is what would change the order.

### Forecasting, second run: declared before it ran

The first run could not decide, for want of collections where the two orders
disagree. This run is built to decide, and nothing about the orders or the rule
is changed for it.

- **What is tested**: the two orders exactly as the first run fitted them,
  frozen (`frozen_orders()` refits them from `results/forecast.csv` alone, so
  nothing here can leak into them). On a series the page reads as intermittent
  the kind order puts TSB first, by R² and by mean absolute error, and the fixed
  order puts exponential smoothing first; on every other kind they agree.
- **The series**: up to 20 per collection, drawn with a fixed seed from the
  series the page reads as intermittent (the most recent 200 values; the kind is
  measured on those, with dates at the collection's step), from nine public
  sources the first run never saw: daily departures on each New York route
  (nycflights13), weekly syphilis cases per US state (ZIM), US births of each
  rarer name per year since 1950 (babynames), daily trips of ten Citi Bikes
  (tsibbledata), US police officers killed on duty per state and month
  (fivethirtyeight), yearly cases of seven diseases per US state (dslabs), monthly
  ratings of each film on MovieLens (dslabs), weekly purchases of each CDNOW
  customer (lifetimes), and Atlantic storms active each month by status (dplyr).
  Each is its own unit: nine units, 137 series. Two collections hold fewer than
  20 intermittent series (US diseases 2, storms 7) and give all they have.
- **The rule** is `choose()` unchanged, applied by `confirm()` to the new units
  only: the kind order replaces the fixed one only if its median regret over
  units is no worse under either score and it is better by more than luck under
  at least one (paired Wilcoxon, Holm over the two scores, more wins than
  losses). With Holm over two scores, seven units that all go one way are the
  fewest that can pass; nine leave room for a loss or two.
- **What follows**: this run's verdict replaces the first run's as the one the
  page follows. If it passes, the page orders forecasters by kind, using the
  frozen first-run orders, which are what was tested. If it fails, the fixed
  order stays, and the evidence tab says the kind was tested twice and did not
  earn it.

```bash
python -m dcn.forecast --run 2 --out results/forecast-2.csv
python -m dcn.forecast_learn --results results/forecast.csv --confirm results/forecast-2.csv
```

### What the second run found

137 intermittent series from nine new collections, 917 forecasts;
exponential smoothing could not be fitted on 49 of them (mostly baby names,
MovieLens and flights), where the fixed order falls back to its next choice.
Regret of each frozen order, averaged over a collection's series:

| collection | R², smoothing first | R², TSB first | absolute error, smoothing first | absolute error, TSB first |
|---|---|---|---|---|
| babynames | 0.380 | 0.171 | 0.207 | 0.145 |
| cdnow | 0.100 | 0.069 | 0.226 | 0.017 |
| movielens | 0.194 | 0.008 | 0.970 | 0.039 |
| nyc_bikes | 0.367 | 0.036 | 0.109 | 0.027 |
| nycflights | 0.096 | 0.101 | 0.294 | 0.261 |
| police_deaths | 0.106 | 0.021 | 0.153 | 0.052 |
| storms | 0.037 | 0.217 | 0.050 | 0.294 |
| syphilis | 0.107 | 0.010 | 0.162 | 0.018 |
| us_diseases | 0.508 | 0.369 | 0.173 | 0.000 |

The rule's verdict: the kind order does not replace the fixed one. It was
better on 7 collections by R² and on 8 by mean absolute error, and worse on 2
and 1; median regret fell from 0.107 to 0.069 R², and from 0.173 to 0.039 in
units of doing nothing's error. But Wilcoxon's p was 0.074 and 0.098 before
correction and 0.15 after Holm's, above 0.05, because one of the losses is
large: Atlantic storms, whose intermittent counts follow the hurricane season,
which smoothing with a twelve-month season can follow and TSB, which has no
season, cannot. The rule was set before the run and is not loosened after it:
the page keeps one order for every series, and on an intermittent upload it
says what both runs found.

Two observations, recorded as observations and not used by the page: apart
from the storms, the kind order's losses in both runs were small (M5 by
0.0015 in absolute error in the first, New York flights by 0.005 R² in the
second); and intermittent series with a strong season behave as a kind of
their own, which a future run could declare before it runs.

## Anomalies

"Unusual records" put its cards in coordinate order, because nothing had
measured them. `dcn/anomaly.py` fits every detector the page can recommend,
without labels, on tables whose anomalies are known, and scores each against
the labels afterwards. `dcn/anomaly_learn.py` asks whether the width of a table
should change which detector comes first.

```bash
python -m dcn.anomaly --out results/anomaly.csv
python -m dcn.anomaly_learn --results results/anomaly.csv
```

### What the data has to show, written before it ran

- **The detectors**: every one the page can recommend for "Unusual records"
  that runs on a table, Isolation Forest (TR4), One-Class SVM (SV3) and DBSCAN's
  noise points (CL2), and three that join the taxonomy with this run because
  they are the standard representatives of the other families of method: the
  Local Outlier Factor (IB3), the distance to the fifth nearest neighbour (IB4),
  and robust covariance, the Mahalanobis distance from a Minimum Covariance
  Determinant fit (PR6). scikit-learn's defaults, on standardised columns
  except the forest, which splits one column at a time and does not care about
  scale. Each is fitted without labels and scored afterwards by ROC AUC and by
  average precision, as ADBench scores them.
- **The tables**: ADBench's 47 classical tables (Han et al., NeurIPS 2022),
  fetched from its repository. A table over 5,000 rows is replaced by a seeded
  sample of 5,000, stratified by the label, keeping at least 20 anomalies (or
  all there are). Tables cut from one source count once: thyroid (2),
  cardiotocography (2), the Wisconsin breast-cancer tables (4), Statlog Landsat
  (3) and KDD Cup 1999 (2). That leaves 39 independent units.
- **What the page measures**: how many columns the table has. Narrow is at most
  10, middling 11 to 50, wide more than 50. The reason is mathematical: as
  columns are added, the distances from a point to its nearest and its farthest
  neighbour become relatively alike (Beyer et al., 1999), so detectors that rank
  points by the distances to their neighbours (LOF, k-NN) have less to work with
  on wide tables, and a covariance fit needs more rows than columns.
- **The prediction**: the width decides which detector comes first.
- **The rule** is `choose()` from `dcn/forecast_learn.py`, unchanged: an order
  kept per width replaces one order for every table only if its median regret
  over units is no worse by either score and it is better by more than luck by
  at least one (paired Wilcoxon over units, Holm over the two scores, more wins
  than losses), both orders fitted leave-one-unit-out. The width order is shrunk
  toward the fixed one by five tables' worth. The cut-offs and the shrinkage are
  fixed here and not tuned on the result.
- **What follows either way**: the cards for "Unusual records" are ordered by
  what was measured. If the width order passes, the order depends on the width
  of the upload; if not, every table gets the same order, and the page says so.
- **What it cannot say**: many of ADBench's anomalies are a rare class of a
  classification table relabelled as anomalous, not anomalies that occurred as
  such. Every detector runs at its defaults, the largest tables are sampled, and
  there is one seed.

### What the run found

47 tables, 282 fits. Robust covariance could not run on one table
(InternetAds: 1,966 rows for 1,555 columns); everything else ran. How often
each detector was the best of the six, and its median ROC AUC:

| width | tables | sources | Isolation Forest | One-Class SVM | LOF | k-NN distance | robust covariance | DBSCAN |
|---|---|---|---|---|---|---|---|---|
| narrow (≤ 10 columns) | 19 | 16 | 21% (0.86) | 11% (0.87) | 0% (0.70) | 21% (0.83) | 37% (0.86) | 11% (0.55) |
| middling (11 to 50) | 19 | 15 | 16% (0.74) | 16% (0.72) | 21% (0.61) | 21% (0.74) | 26% (0.80) | 0% (0.50) |
| wide (> 50) | 9 | 9 | 44% (0.69) | 0% (0.66) | 11% (0.61) | 0% (0.68) | 44% (0.73) | 0% (0.50) |

The rule's verdict: the width order does not replace the fixed one. By ROC AUC
it was better on 17 sources and worse on 4 (p = 0.016, Holm), with median
regret 0.053 against 0.086. By average precision it was better on 8 and
worse on 12 (p = 0.25), and its median regret was higher, 0.060 against 0.048.
The rule asks for no worse under both scores, so it fails on the second. The
two scores disagree because they reward different things: ROC AUC the ranking
of every anomaly against every normal row, average precision the top of the
list, where a user actually looks.

What it says about the mathematics: on wide tables the detectors that rank by
distances to neighbours were rarely best (LOF on one of nine, k-NN distance on
none), and Isolation Forest and robust covariance shared the rest, as
concentration of distances predicts (math AD6). On middling tables k-NN
distance was as good as anything, so width alone is not the whole story.

What the page does with it: the cards for "Unusual records" are ordered by the
fixed order by ROC AUC (robust covariance, Isolation Forest, k-NN distance,
One-Class SVM, LOF, DBSCAN; the first two are within 0.004 of each other), each
card says how the detector did on tables of the upload's width, and robust
covariance is ruled out when a table has no more than twice as many rows as
columns, the condition its fit needs (math AD5).

## What is in results/

| file | one row per | what it holds |
|---|---|---|
| `pilot.csv` | dataset and model, before the fixes | signature, eligibility, rank, score, seconds, status |
| `after-fixes.csv` | dataset and model, after the fixes | the same columns |
| `full.csv` | dataset, model and seed, the full run | the same columns, references included; what the site publishes |
| `meta.csv` | dataset | the eight meta-features "datasets like yours" and the neighbour order use |
| `per_dataset.csv` | dataset, full run | best model and score, the first pick, best of four, boosting, and the regret of each |
| `ranking-lodo.csv` | dataset, full run | the score and model each order picked, leave-one-dataset-out with families held out |
| `summary.json` | full run | the recorded headline, plus which models were ruled out and why |
| `delta.csv` | dataset | the two 40-dataset runs side by side and the change |
| `seeds.csv` | dataset, model and seed | an early three-seed check on seven datasets |
| `drift.csv` | dataset, cut, scenario and drift checker | whether it fired, its statistic, and the seconds it took |
| `messy.csv` | dataset, model and kind of damage | the same columns as `full.csv`, on damaged copies of 40 datasets |
| `messy-picks.csv` | dataset and kind of damage | the order's first pick and the signature the profiler measured, clean and damaged |
| `checks.csv` | dataset and planted problem | what the page's checks flagged, whether the planted problem was caught, and the best single column's leak score |
| `forecast.csv` | series and forecasting method | the series' collection, period, season, kind and the measurements behind it, and the method's R², mean absolute error and seconds |
| `forecast-lodo.csv` | score, order and series | which forecaster each order picked, leave one collection out, and its regret |
| `forecast-2.csv` | series and forecasting method | the confirmation run, the same columns as `forecast.csv` |
| `forecast-2-picks.csv` | score, order and series | which forecaster each frozen order picked on the confirmation run, and its regret |
| `anomaly.csv` | table and detector | the table's source, size, width and anomaly share, and the detector's ROC AUC, average precision and seconds |
| `anomaly-lodo.csv` | score, order and table | which detector each order picked, leave one source out, and its regret |

## Next

- run TabPFN-2 or later (`tools/run_tabpfn.sh`, with a Prior Labs login),
  judged by the same rule TabPFN-1 was
- tuned boosting on recent changes, the script's reference, in a forecasting
  run
- tune every family the way boosting was tuned, and rank them on that: the
  page puts tuned boosting first, but ranks the rest at their defaults
- weight a synthetic family once when fitting the prior, declared before the
  next run
- more collected regression data: 35 independent units is what limits every
  regression claim here, and more generated datasets would not help
- something that does help regression with heavy gaps or noisy targets: the
  fixed first pick lost ground there, and an order learned from damaged runs
  did not recover it
