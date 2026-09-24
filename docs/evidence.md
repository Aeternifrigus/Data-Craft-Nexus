# The evidence

[README](../README.md) · [Docs](README.md) · [How it works](how-it-works.md) · **The evidence** · [Development](development.md)

Every recommendation on the page is measured, and this is the record: what each benchmark ran, what it found, the rule written down before it ran that decided what the page does with the result, and, where the mathematics can say why, the argument. The page shows the same record in its **The evidence** tab. Every number here is also in [`bench/results/`](../bench/results), with the places where the site loses, and how each run was set up is in [`bench/README.md`](../bench/README.md).

Regret means how far a choice landed below the best one available. Independent units are datasets counted once per source.

## Contents

- [Why not just tune boosting?](#why-not-just-tune-boosting)
- [1. Tables: a number or a category](#1-tables-a-number-or-a-category)
  - [The order in use](#the-order-in-use) · [Is it more than luck?](#is-it-more-than-luck) · [What that means for the advice](#what-that-means-for-the-advice) · [Richer orders](#richer-orders)
  - [What it was worth, as recorded](#what-it-was-worth-as-recorded) · [The luck of the split](#the-luck-of-the-split) · [When the data is messy](#when-the-data-is-messy) · [Small tables: a pretrained model](#small-tables-a-pretrained-model)
- [2. Forecasting: a future value](#2-forecasting-a-future-value)
- [3. Anomalies: unusual records](#3-anomalies-unusual-records)
- [4. Drift checkers](#4-drift-checkers)
- [5. What the checks catch](#5-what-the-checks-catch)
- [6. Why, mathematically](#6-why-mathematically)
- [7. The evidence tab](#7-the-evidence-tab)
- [Not done yet](#not-done-yet)

## Why not just tune boosting?

The page tells you to, first, because it measured the alternative. The site was first built to pick models by matching a dataset's six-axis signature. On 195 PMLB datasets and 5,541 model runs, with every ranking judged leave-one-dataset-out and every family of related datasets counted once, that lost. Balanced accuracy lost against the best model, median over 78 independent classification datasets:

| how the first model is chosen | points lost |
|---|---|
| matching models to the data's signature (how the site first ranked them) | 4.79 |
| a ranking learned from the benchmark | 1.60 |
| always using gradient boosting | 1.45 |
| gradient boosting, tuned over 10 settings | **1.12** |

So tuned boosting leads, and the signature does what it is good at: telling you which problem you have and what cannot work on it. The other two questions have no one-line answer:

- **What usually wins may not win on your file, and one split can mislead.** On 70% of the classification datasets re-run under three cross-validation seeds, the winner changed with the seed. The script runs the shortlist on your data and prints each score with its spread across folds, so you can see whether a gap is real.
- **Drift detectors see different things.** Across 23 detectors and 630 test cases, adversarial validation caught 76% of broken correlations, where tests that check one column at a time caught at most 13%. Only detectors that watch the model's errors noticed labels changing meaning, and they paid with 26% to 41% false alarms.

AutoML answers the second question with more search, once installed and given compute. It assumes rows are independent unless told otherwise, does not say what it ruled out or why, and leaves the third question to you.

## 1. Tables: a number or a category

The instrument publishes its own scoreboard: 195 datasets from [PMLB](https://github.com/EpistasisLab/pmlb), every runnable model fitted on every one of them, five-fold cross-validated, 5,541 model runs (40 of the datasets under three cross-validation seeds, to measure how much a split alone moves things, and tuned boosting on every dataset as a reference). Each recommendation also carries a line saying how that model did.

### The order in use

The weights are a prior per model, fitted in `bench/dcn/learn.py` on 195 real datasets and judged leave-one-dataset-out, so a dataset never contributes to the weights that rank it:

| median regret, leave-one-dataset-out (95% interval) | classification (94 datasets, 78 independent) | regression (101 datasets, 35 independent) |
|---|---|---|
| counting matched coordinates (before) | 0.048 (0.034 to 0.068) | 0.086 (0.038 to 0.257) |
| learned from the benchmark (now) | 0.016 (0.012 to 0.021) | 0.012 (0.002 to 0.027) |
| always use boosting | 0.014 (0.011 to 0.022) | 0.021 (0.009 to 0.055) |
| always use boosting, tuned (reference) | 0.011 (0.008 to 0.018) | 0.006 (0.002 to 0.017) |

Datasets generated from one function are not independent. Of the 101 regression datasets, 54 come from Friedman's benchmark functions and 14 from Strogatz's equations: sisters that share a winner. Such a family is held out whole when its members are ranked, and counted once in every number above, so 101 regression datasets are 35 independent units. Counting them one by one made the learned order look better on regression than it is (0.007 instead of 0.012). The same holds for datasets cut from one table: PMLB has the Garvan thyroid records under six names with six targets, three horse colic targets, four feature sets of the same handwritten digits, two exact duplicates, and one generator behind each of led7 and led24, waveform_21 and waveform_40, and the three MONK's problems. Counted once, 94 classification datasets are 78 independent units.

Before this order, the page counted matched coordinates, and it put plain Linear Regression first on 17 of 20 regression datasets at a cost of up to 0.7 R². Letting models that assume independent rows back onto ordered data, and models built for numbers back onto tables of categories (each with a caution), changed none of the four models shown on the 26 categorical tables, and made counting coordinates, the order both are compared with, better; the numbers above are with that rule fixed.

### Is it more than luck?

The intervals come from resampling those units. The order in use is also compared, unit by unit, with the two things it claims to beat, and with tuned boosting, a reference it makes no claim to beat but a reader will ask about (Wilcoxon signed-rank, Holm-corrected for the three comparisons):

- **Against counting coordinates it is better by more than luck:** on 56 of 78 classification units, worse on 21 (p = 3 × 10⁻⁶), and on 24 of 35 regression units (p = 5 × 10⁻⁴).
- **Against always using default boosting, it is level:** better on 39 classification units and worse on 35 (p = 0.61), better on 21 of 35 regression units and worse on 14 (p = 0.091). With two comparisons the regression edge was p = 0.046; adding a third comparison to the family, as honesty requires, puts it back inside luck.
- **Against tuned boosting, it loses on classification by more than luck:** better on 24 units, worse on 49 (p = 0.028). On regression they are level: better on 16 of 35 units, worse on 19 (p = 0.55).

### What that means for the advice

The site recommends model families, at their default settings. Choosing among the top families is worth about as much as reaching for boosting; spending ten configurations tuning boosting is worth more. Tuned boosting beat its own defaults on 62 of 94 classification datasets and 88 of 101 regression datasets, and has the best average rank of anything that ran in both tasks. So the page now shows tuned boosting first, above the order's picks, on every labelled table predicting a category or a number: it passes the same rule that decides which learned order ships (no worse on either task, better by more than luck on one), applied to it after its result was known, which the evidence tab says. Whichever family you pick after that, tune it before trusting its score. The tuning budget is small on purpose (scikit-learn's defaults and nine random configurations, chosen by cross-validation inside each training fold) and the details are in [`bench/README.md`](../bench/README.md#references).

### Richer orders

Two richer orders are built and judged the same way, and either can replace the per-model prior: one adds interactions between the dataset's measured features and each model's family, and one weights each model toward what it did on the benchmark datasets nearest to yours. Which one ships is decided by a rule fixed before the results were seen (`choose()` in `bench/dcn/learn.py`): a richer order replaces the prior only if it is no worse on either task and better by more than luck on at least one. Neither qualified on the full run (interactions: 9 better and 10 worse of 35 regression units; neighbours: 8 and 6), so the order in use is still the prior. The evidence tab reports each decision with its numbers.

### What it was worth, as recorded

| median regret, as recorded | classification | regression |
|---|---|---|
| the first model shown | 0.012 | 0.011 |
| best of the four shown | 0.005 | 0.002 |
| always use boosting | 0.015 | 0.021 |
| a model picked at random from the eligible ones | 0.051 | 0.162 |

Regret is how far below the best model that ran a choice landed, in balanced accuracy and in R². These are the recommendations as the run recorded them, ordered by the prior fitted on the earlier 40-dataset run, which had already seen 40 of these datasets; the leave-one-dataset-out table above is the fairer test. The four models shown contain the best available choice 31% of the time on classification and 43% on regression.

The numbers on the page are generated from `bench/results/`, and a test recomputes them from that data on every run, so the page cannot quietly disagree with the run behind it.

### The luck of the split

The full run also measured how much one split decides. Between cross-validation seeds a model's score moves by a median of 0.004 on classification and 0.005 on regression. The best model was the same under all three seeds on only 30% of classification datasets and 50% of regression ones, and the first pick and boosting swapped places, depending only on the split, on 10 of 20 classification datasets and 4 of 20 regression ones. A single dataset's winner is weak evidence, which is why every comparison here is made across datasets.

### When the data is messy

Benchmark datasets are clean and uploads are not, so 40 of the datasets were damaged on purpose and every model run again: 10% or 30% of cells blanked, junk text (`?`, `n/a`, `#VALUE!`) in 5% of numeric cells, or 10% of training labels swapped.

- **The profiler catches it.** It flagged every file with blanked cells (A62). It first flagged only 89% of those with junk (A64): the ones it missed had junk in the few numeric columns of a mostly categorical table, where an average over all columns stays under the threshold. Dirt is now judged per column as well, and all of them are flagged, while none of the 196 clean benchmark datasets is.
- **The first pick mostly holds.** On classification the order's first pick held up under every kind of damage. On regression it did not under 30% missing cells or noisy targets (its regret rose from 0.003 to 0.049 and 0.033), where the linear models and the linear SVM lost least and tree ensembles most; that rests on 8 independent units.
- **Reading data quality did not help the order.** An order that reads data quality, fitted on the other datasets' damaged runs, was tested leave-one-dataset-out and did not beat the fixed order under any kind of damage, so a messy file gets the same order. On the 8 classification units that arrived with missing values of their own, the order did as well as on the complete ones, as far as 8 units can tell.

A file flagged A62 or A64 now gets a line under the models saying what the same kind of damage did on the benchmark. Details in [`bench/README.md`](../bench/README.md#messy-data).

### Small tables: a pretrained model

TabPFN is a transformer pretrained on synthetic tables to predict a small table in one pass. TabPFN-2's weights sit behind a login this environment could not reach; TabPFN-1's are public (Apache 2.0), so it ran, within its limits: classification only, up to 1,000 rows, 100 features and 10 classes. That is 42 datasets in 38 independent units. What it had to show was written down before it ran (`bench/README.md`, "Which TabPFN ran").

Against tuned boosting it was better on 21 units and worse on 16 (p = 0.26). The rule asks for more than luck, so tuned boosting stays first on small tables too, and the size of a table still does not change the first recommendation. It did beat the order's own first pick more often than not (25 units better, 12 worse, p = 0.13 after correcting for four comparisons). None of this measures TabPFN-2 or later, which their authors report stronger.

## 2. Forecasting: a future value

The take-home script's forecasters have a benchmark of their own (`bench/dcn/forecast.py`): the functions the page writes, run on 240 real series from 12 public collections (M4 at six frequencies, car parts, Australian prescriptions, retail and livestock, and M5's Walmart sales), 11 independent units, each value predicted from the values before it. Every series was sorted into a kind the way the page sorts an upload (`site/js/series.js`): intermittent or lumpy demand by Syntetos, Boylan and Croston's cut-offs, otherwise by seasonal and trend strength.

| kind of series | series | ARIMA | smoothing | Croston (SBA) | TSB | nothing beat all four |
|---|---|---|---|---|---|---|
| seasonal and trending | 68 | 7% | 91% | 0% | 2% | 19% |
| seasonal | 17 | 35% | 65% | 0% | 0% | 47% |
| trending | 100 | 41% | 47% | 0% | 12% | 33% |
| neither | 13 | 15% | 46% | 8% | 31% | 8% |
| intermittent | 34 | 9% | 12% | 38% | 41% | 21% |
| lumpy | 8 | 25% | 0% | 50% | 25% | 38% |

- **The kind of series changes the winner.** Smoothing wins nine seasonal-and-trending series in ten; on intermittent demand the two methods built for it win four in five, and smoothing one in eight.
- **Doing nothing is a real contender.** It beat every forecaster on almost half the seasonal series without a trend and a third of the trending ones, which is why the script scores it beside them.
- **Whether the page should read the kind was decided by a rule fixed before the run.** An order kept by kind cut the median regret on intermittent series from 0.121 to 0.034 R², and by R² it was worse on no collection. But the two orders chose differently in only 4 of the 11 collections, and with 4 the smallest p-value the test can give is 0.125: it could not have passed. So the cards keep one order for every series (exponential smoothing, ARIMA, TSB, Croston), and on a series of a kind where the evidence points elsewhere the page says so, with the numbers.
- **A second run was declared to settle it, and came close.** Both orders were frozen and tried on 137 intermittent series from nine sources the first run never saw (New York flights, US syphilis cases, US baby names, Citi Bike trips, police deaths, disease counts, MovieLens ratings, CDNOW purchases, Atlantic storms). Putting TSB first was better on 7 of 9 collections by R² and 8 of 9 by absolute error, and cut median regret from 0.107 to 0.069 R² and from 0.173 to 0.039 in units of doing nothing's error. But one loss was large (the storms, whose intermittent counts follow the hurricane season, which TSB cannot model), and p was 0.15 after correcting for two scores. The rule was fixed before the run and is not loosened after it, so the order stays, and an intermittent upload is told what both runs found.

Both runs, with what they had to show written down first, are in [`bench/README.md`](../bench/README.md#forecasting).

## 3. Anomalies: unusual records

"Unusual records" has a benchmark too (`bench/dcn/anomaly.py`): every detector the page can recommend, fitted without labels on ADBench's 47 classical tables with known anomalies (39 independent sources) and scored against the labels afterwards. Three detectors joined the taxonomy for it, as the standard representatives of the other families of method: the Local Outlier Factor, the distance to the fifth nearest neighbour, and robust covariance (Minimum Covariance Determinant).

| width of table | tables | Isolation Forest | One-Class SVM | LOF | k-NN distance | robust covariance | DBSCAN |
|---|---|---|---|---|---|---|---|
| narrow (up to 10 columns) | 19 | 21% | 11% | 0% | 21% | 37% | 11% |
| middling (11 to 50) | 19 | 16% | 16% | 21% | 21% | 26% | 0% |
| wide (over 50) | 9 | 44% | 0% | 11% | 0% | 44% | 0% |

- **The mathematics predicted which detectors would fade on wide tables, and they did.** Distances concentrate as columns are added, so detectors that rank rows by the distances to their neighbours have less to work with: on wide tables LOF was best once in nine and k-NN distance never, while Isolation Forest and robust covariance shared the rest.
- **Reading the width did not pass the rule.** By ROC AUC an order kept per width was better on 17 sources and worse on 4 (p = 0.016), but by average precision, which rewards the top of the list, it was worse on 12 and better on 8. The rule asks for no worse under both, so the cards keep one order: robust covariance, Isolation Forest, k-NN distance, One-Class SVM, LOF, DBSCAN. Robust covariance is ruled out when a table has no more than twice as many rows as columns, the condition its fit needs.

The run's setup is in [`bench/README.md`](../bench/README.md#anomalies).

## 4. Drift checkers

The drift checkers get a benchmark of their own (`bench/dcn/drift.py`): 18 real datasets, each cut five ways into a 500-row reference window and a 250-row current window, and seven scenarios per cut: nothing changes, one feature shifts by 0.3 standard deviations, its spread widens by half, its link to the other features is broken while every feature looks the same on its own, the rows are drawn with a bias, the class mix changes, or the labels change meaning while the features stay put. Every checker ran as its card says, at its card's threshold.

- **The best all-rounders test each feature's whole distribution:** Anderson-Darling, chi-square on binned values and Cramér-von Mises caught 46% to 57% of the injected drift, with 1% to 7% false alarms.
- **Some drift is invisible feature by feature.** Adversarial validation caught 76% of broken correlations; the single-feature tests caught at most 13%. Only the checkers that watch the model's errors saw labels change meaning (error drift 94%, EDDM 98%), and they paid for it with false alarms on 26% and 41% of quiet windows.
- **Convention is cautious, and some cards overpromise.** PSI above 0.25 never fired on a quiet window but caught half the shifts; Jensen-Shannon at 0.1 almost never fired; Mahalanobis distance, whose card calls its chi-square p-value principled, fired on 24% of quiet windows. At river's defaults, Page-Hinkley and KSWIN caught almost nothing in 250 rows.

It also found five defects, now fixed: six drift cards linked to the wrong function, error drift claimed to work without labels, mixed tables (most real files) were never offered a distribution test, SciPy's Cramér-von Mises test calls two samples of one mostly-zero column different at p < 10⁻⁹, and river's FHDDM, fed errors as its documentation says, fires when the model gets better. The full table is in the evidence tab and in [`bench/README.md`](../bench/README.md#drift-checkers).

## 5. What the checks catch

The first section of the results, **Before you trust a score**, was measured the same way: on the 195 benchmark datasets as a user would upload them, and on 40 of them with a problem planted.

- **Planted problems:** it caught the target renamed or rescaled in 40 of 40, shuffled row numbers and random row codes in 40 of 40, and 2% of rows copied in 32 of 40.
- **On clean data its flags are mostly real:** body fat computed from density, a phone number in a churn table, and 22 datasets with more copied rows than chance, the thyroid and wine quality tables among them.
- **Its limits:** it cannot catch a leak with a few percent of mistakes in it (0 of 40 at 5%), and it looks at one column at a time.

Details in [`bench/README.md`](../bench/README.md#before-you-trust-a-score).

## 6. Why, mathematically

Where the mathematics can say why a model is chosen, the evidence tab says it, beside the measurement it explains, and each argument is a formula in the reference that opens from any card. Among them:

- **Why models are measured, not derived** (no free lunch): averaged over every possible problem, no learner beats another, so only the problems that actually occur can decide.
- **Why doing nothing wins so often** (random walk): if tomorrow is today plus unpredictable noise, today is the best forecast under squared error, and no model can do better on average.
- **Why exponential smoothing leads** (Muth, 1960): it is the optimal forecast for a level that wanders with noise on top.
- **Why Croston's method is corrected** (Syntetos and Boylan): dividing by an estimated interval inflates the forecast, by a factor that tends to 1 / (1 − α/2).
- **Why the score you choose matters** (optimal point forecasts): squared error rewards the conditional mean and absolute error the median, which is zero on demand that is empty more than half the time.
- **Why one-column drift tests miss broken correlations**, and why a classifier between the windows does not: if every column's distribution is unchanged, a one-column test fires only at its false-alarm rate, while the best classifier gains exactly the total variation between the windows.
- **Why PSI misbehaves on small windows**: with no drift its expected value is about (bins − 1)(1/n + 1/m), 0.05 on the drift benchmark's windows and 0.6 on two windows of 30 rows.
- **Why wide tables hurt neighbour-based detectors** (concentration of distances), and what each detector's score means.
- **Why a handful of units cannot pass** (Wilcoxon's floor): with n pairs the smallest possible p is 2/2^n, which is why the first forecasting run could not have passed and the second was built bigger.

## 7. The evidence tab

The evidence tab opens with a contents list and is split into numbered parts, one per kind of advice: tables (the order, the significance tests, the luck of the split, TabPFN, messy data), forecasting (winners by kind, the first run, the confirmation run), anomalies, drift checkers, the checks before a score, and the reference tables. Each part ends with its mathematics.

## Not done yet

- **Only boosting was tuned.** The page puts tuned boosting first because it beat the order's first pick, but the families below it are still ranked by what they were worth at their defaults. Tuning each of them the same way, and ranking them on that, would need a run the size of the tuned one for every family.
- **Run it here has been run in Chromium only**, with Pyodide's files served locally rather than from the CDN. Firefox and Safari support module workers and should work; they have not been tried. If it fails in yours, the downloaded script runs the same thing.
- **Only TabPFN-1 has been run.** TabPFN-2 and later need a Prior Labs login this benchmark's environment could not reach. `bench/tools/run_tabpfn.sh` adds them in one command, judged by the same rule TabPFN-1 was.
- **The order in use is a per-model prior, not yet a per-dataset one.** Both ways of making it depend on your data (interactions, and weighting toward the nearest benchmark datasets) are built, tested against the JavaScript, and judged leave-one-dataset-out, and neither beat the prior by more than luck on 195 datasets once families count once. With only 35 independent regression units, a per-dataset order needs more collected data to prove itself, not more of the same generators.
- **The prior itself still counts a family's datasets one by one when it is fitted**, so on regression it leans toward what wins on Friedman's functions. Weighting a family once in the fit is the obvious change, and it should be declared before the next run rather than tried after this one.
- **Heavy gaps and noisy targets still cost the first pick on regression.** An order that reads data quality was tested and did not help, so nothing here fixes that yet; it rests on 8 independent units either way.
- **Drift was injected at one strength per kind**, on classification datasets, with 500- and 250-row windows, so the rates hold for that setting. Detectors tuned to their stream would beat river's defaults.
- **Classification, regression, forecasting and anomaly detection are benchmarked.** Survival, grouping, compression and generation still fall back to counting coordinates.
- **The forecasting kinds are not settled.** A per-kind order was better on most collections in two runs and passed the rule in neither. Intermittent series with a strong season look like a kind of their own, and a run declared for that is the next step. Tuned boosting on recent changes, the script's reference, is not in the forecasting runs yet.
- **Anomaly detectors ran at their defaults only**, the largest tables were sampled to 5,000 rows, and many of ADBench's anomalies are a rare class relabelled, not anomalies that occurred as such.
- Image, audio, graph and spatial data cannot be detected from a CSV, so those models are reachable in the reference but never recommended from an upload.
- Separability (A55/A56) and weak or self-supervised labelling (A13 to A15) are not measured yet.
