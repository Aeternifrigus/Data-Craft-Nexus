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
| the profiler noticed (flag) | 100% (A62) | 100% (A62) | 89% of 36 (A64) | cannot |
| median loss, classification | 0.027 | 0.069 | 0.011 | 0.017 |
| median loss, regression | 0.101 | 0.268 | 0.039 | 0.043 |
| first pick's regret, classification (0.016 clean) | 0.007 | 0.011 | 0.009 | 0.019 |
| first pick's regret, regression (0.003 clean) | 0.009 | 0.049 | 0.008 | 0.033 |

- The profiler sees missing cells every time. It missed the junk in 4 of the
  36 tables that got some, because A64 is judged on the average over all
  columns: junk in the few numeric columns of a mostly categorical table stays
  under the threshold. Three of the four were read as having missing values
  instead, since `n/a` and `-` are missing-value words; one as clean.
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

Damage made on purpose is random, and real gaps are not. PMLB has datasets
that arrived with missing values of their own: 10 classification datasets in
the full run (8 independent, the three horse colic tables being one) and one
regression dataset. OpenML, which has more, could not be reached from where
this ran. On those 8 units the order's leave-one-dataset-out regret was 0.020,
against 0.016 on the 70 complete ones (Mann-Whitney, p = 0.50), and default
boosting's 0.014 against 0.015: no evidence of a difference either way, at a
size that could only have shown a large one.

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
| counting coordinates | 0.047 (0.035 to 0.056) | 0.205 (0.071 to 0.303) |
| learned prior, in use | 0.016 (0.012 to 0.021) | 0.012 (0.002 to 0.027) |
| prior + interactions | 0.016 (0.010 to 0.021) | 0.012 (0.003 to 0.025) |
| prior + neighbours | 0.018 (0.013 to 0.023) | 0.012 (0.001 to 0.025) |
| always boosting | 0.014 (0.011 to 0.022) | 0.021 (0.009 to 0.055) |
| boosting, tuned (reference) | 0.011 (0.008 to 0.018) | 0.006 (0.002 to 0.017) |

Holm-corrected over three comparisons (coordinates, default boosting, tuned
boosting):

- The learned order beats counting coordinates by more than luck on both
  tasks (p = 2e-6 and 3e-5).
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

## Next

- run TabPFN (see "Running TabPFN on your own machine") and let the evidence
  pick it up
- fold tuning into the advice: the site recommends families at their
  defaults, and ten configurations of boosting beat that on classification
- let numeric-feature models run on all-categorical tables, since the pipeline
  encodes categories, and check the 9 lost winners come back
- weight a synthetic family once when fitting the prior, declared before the
  next run
- more collected regression data: 35 independent units is what limits every
  regression claim here, and more generated datasets would not help
- an order that reads data quality: on regression with heavy gaps or noisy
  targets the fixed prior's first pick lost ground the linear models kept.
  Declare the rule before running it, as `choose()` does
- judge A64 per column: averaged over all columns, junk in the few numeric
  columns of a wide categorical table goes unflagged
