Drift is not one phenomenon : it's a family of ways the world a model sees in production can diverge from the world it was trained on. It is implemented on [[ML Models]] and also on [[Model Stacking]] scenarios on basis of [[Math]]. 

==**Hence Drift has: 1 Drift Type, 1 Temporal Pattern, 1+ Detection Test (mapped to Math code), a Data Modality it applies to (mapped to Data code), a Monitoring Window strategy, and a Response Action : and only then does it inherit a final Math+Data signature.**==

## Domain: Full Drift-Checker Count [DR-COUNT]

|Domain|Codes|Count|
|---|---|---|
|[DR-M] (core + extended)|DR-M1–DR-M14|14|
|[DR-C] (core + extended)|DR-C1–DR-C9|9|
|[DR-MV]|DR-MV1–DR-MV2|2|
|[DR-CV]|DR-CV1|1|
|[DR-DL]|DR-DL1–DR-DL2|2|
|**Total**||**28**|

Not counted as checkers: [DR-T]/[DR-T6] (drift types, not tests), [DR-P] (temporal patterns), [DR-D] (routing table), [DR-W] (windowing strategy), [DR-R]/[DR-CR] (response/correction), [DR-MH]/[DR-TH] (post-processing on a checker's output), [DR-RS] (a workflow reusing existing checkers), [DR-X] (disambiguation).

---

## Domain: Drift Types : The Big Picture [DR-T]

|Code|Drift Type|Definition|Metaphor|
|---|---|---|---|
|DR-T1|Covariate Shift (Data / Feature Drift)|P(X) changes, but the relationship P(Y\|X) stays the same|The customers walking into the store changed, but what actually makes any given customer buy hasn't|
|DR-T2|Concept Drift|P(Y\|X) itself changes : the relationship between input and outcome shifts|Same customers, same store, but their reasons for buying flipped|
|DR-T3|Label / Prior Drift|P(Y) changes on its own, independent of X|The ratio of guilty-to-innocent verdicts shifted even though the evidence patterns coming in look the same|
|DR-T4|Prediction Drift|The distribution of the model's own outputs shifts : a proxy signal that something changed, without saying what|The judge starts handing out very different verdicts than before : an alarm bell whether or not the courtroom itself actually changed|
|DR-T5|Upstream (Operational) Data Drift|A pipeline, schema, or source change silently corrupts what a feature means : not a real-world change at all|Someone quietly swapped the measuring tape from inches to centimeters upstream, and never told the model|
|DR-T6|Sample Selection Bias|The training sample was drawn through a biased selection mechanism, so P(X) at training time never represented the true deployment population : no post-deployment change is required for this to be a problem|The store's original customer survey only ever interviewed people who walked in the front door, missing everyone who came in the back|

DR-T6 is the formal 4th category from the Moreno-Torres et al. dataset-shift taxonomy, sitting alongside covariate/label/concept shift. It differs from DR-T1 in timing: DR-T1 is a real-world shift _after_ deployment; DR-T6 is a mismatch that existed _before_ deployment ever started.

---

## Domain: Temporal Drift Patterns [DR-P]

|Code|Pattern|Definition|Metaphor|
|---|---|---|---|
|DR-P1|Sudden Drift|An abrupt, one-time regime change|A policy change that takes effect overnight|
|DR-P2|Gradual Drift|A slow fade where old and new patterns overlap for a while before the new one wins|Dusk : neither day nor night for a stretch, until it's clearly night|
|DR-P3|Incremental Drift|Many small, steady steps in one direction, each too small to notice day-to-day|Inflation : invisible on any single day, undeniable over a year|
|DR-P4|Recurring / Seasonal Drift|Drift that predictably comes back on a cycle|Holiday shopping patterns returning every December|
|DR-P5|Blip / Outlier Event (Not True Drift)|A temporary spike that reverts back to normal : must be distinguished from real drift to avoid false alarms|A single unusually busy Tuesday that says nothing about the rest of the year|

---

## Domain: Detection Tests : Mapped to Mathnote Codes [DR-M]

|Code|Test|Math Code|Metaphor|Best Suited For|
|---|---|---|---|---|
|DR-M1|Kolmogorov–Smirnov (KS) Test|PS7|Finds the single biggest gap between two eyewitness accounts of the same event|Continuous features|
|DR-M2|Population Stability Index (PSI)|PS8|A bank auditor's go-to : buckets the data and checks how much each bucket's share has shifted|Continuous or categorical, industry-standard for score/feature drift|
|DR-M3|Chi-Square Statistic|PS6|Checks whether the mix of categories in a lineup looks like it used to|Categorical features|
|DR-M4|Kullback–Leibler (KL) Divergence|IT4|Measures how surprised you'd be describing the new distribution using the old one's vocabulary|Any distribution, especially model probability outputs|
|DR-M5|Jensen–Shannon (JS) Divergence|PS19|KL Divergence's well-behaved sibling : symmetric and bounded, easier to put on a dashboard|Any distribution, especially when you need a stable 0-1 drift score|
|DR-M6|Wasserstein Distance (Earth Mover's)|PS18|The literal cost of physically reshaping one pile of dirt into another|Continuous features where the _magnitude_ of the shift matters, not just its presence|
|DR-M7|ANOVA F-statistic|PS21|Checks whether several subgroups (regions, cohorts, time windows) still share the same average|Drift across multiple segments at once|
|DR-M8|Mean Absolute Error Drift|PS5|Tracks whether the judge's rulings are simply getting worse over time, whatever the cause|Performance-based drift monitoring (requires ground-truth labels, always lagged)|
|DR-M9|p-value / Hypothesis Testing|PS15|The referee behind KS/Chi-Square : decides whether an observed gap is real or just noise|Significance layer under DR-M1/DR-M3|
|DR-M10|Maximum Mean Discrepancy (MMD)|No existing Mathnote code; closest analog is LA22 (RBF/Gaussian Kernel), which MMD is typically built on|Comparing two crowds not by their raw appearance but by where their "center of gravity" lands once everyone's mapped into a richer space|The most widely used multivariate two-sample test in production drift tooling (Alibi Detect, Evidently)|
|DR-M11|Least-Squares Density Difference (LSDD)|No existing Mathnote code; also builds on LA22|Estimating the gap between two crowds' density maps directly, instead of comparing their centers of gravity|A kernel-based alternative to MMD, estimating the density difference directly rather than via kernel-mean embeddings|
|DR-M12|Cramér-von Mises Test|No existing Mathnote code; closest analog is PS7 (KS Statistic), which it generalizes|A more patient version of DR-M1 : instead of the single worst moment of disagreement, it tallies the disagreement everywhere along the way|Continuous features, where the whole CDF gap matters, not just its single largest point|
|DR-M13|Anderson-Darling Test|No existing Mathnote code; closest analog is PS7|KS's cousin who cares far more about what's happening at the extremes than in the crowded middle|Continuous features where rare-event / tail drift matters most|
|DR-M14|Energy Distance|No existing Mathnote code; closest analog is PS18 (Wasserstein Distance)|A cousin of the "reshaping dirt" metaphor from DR-M6, but costed by pairwise distances between individual particles rather than an optimal transport plan|Continuous or multivariate features, distance-based alternative to Wasserstein|

---

## Domain: Streaming / Concept-Drift Detectors [DR-C]

Named, purpose-built online detectors : distinct from the generic distributional tests in [DR-M], which mostly assume batch access to two full samples. These operate point-by-point on a live stream and are the actual algorithms behind [DR-T2] (Concept Drift), replacing the crude PS5+PS15 proxy with real detectors. All pair naturally with the sequential windowing logic in [DR-W3].

|Code|Detector|Definition|Metaphor|Best Suited For|Math Code|
|---|---|---|---|---|---|
|DR-C1|DDM (Drift Detection Method)|Tracks the running error rate and its standard deviation; raises "warning" then "drift" as the error rate climbs past statistical thresholds|A quality-control chart that flags a machine once its defect rate creeps outside normal bounds|Sudden drift [DR-P1] in classifier error|PS4+PS12+PS14|
|DR-C2|EDDM (Early DDM)|Tracks the _distance between consecutive errors_ rather than the raw error rate, catching slow drifts DDM misses|Watching how far apart mistakes are spaced, not just how many happen|Gradual drift [DR-P2]|PS4+PS12|
|DR-C3|Page-Hinkley Test|Accumulates the running deviation of observed values from their mean; fires once the cumulative sum crosses a threshold|The CUSUM logic of [DR-W3] made concrete: a scale that only rings once total drift-weight crosses a limit|Incremental drift [DR-P3], performance-based monitoring|PS4 (extends [DR-W3])|
|DR-C4|ADWIN (Adaptive Windowing)|Maintains a variable-length window, splitting it in two whenever the means of the two sub-windows differ by more than a Hoeffding-bound threshold|A window that keeps stretching until it notices its older half no longer resembles its newer half|Sudden or gradual drift where the "right" window size isn't known in advance|No existing Mathnote code (Hoeffding bound); closest analog is PS15|
|DR-C5|KSWIN (KS-test in a sliding Window)|Applies the KS test [DR-M1] between a small recent window and a larger reference window, rather than two fixed batches|The same eyewitness comparison as DR-M1, just run continuously instead of once|Continuous features, online setting|PS7 (reuses DR-M1 directly)|
|DR-C6|HDDM_A (Hoeffding's Drift Detection Method : Average)|Uses a Hoeffding-bound test on the _moving average_ of the error rate to decide when a statistically significant change has occurred|A quality-control chart like DR-C1, but with the alarm threshold derived from a formal concentration bound instead of a fixed std-dev rule|Sudden drift [DR-P1] in classifier error, same family as ADWIN [DR-C4]|No existing Mathnote code (Hoeffding bound), same gap as DR-C4|
|DR-C7|HDDM_W (Hoeffding's Drift Detection Method : Weighted)|Same Hoeffding-bound logic as HDDM_A, but weights recent errors more heavily via an EWMA, making it more responsive to gradual drift|HDDM_A's cousin who pays more attention to what just happened than to the whole history|Gradual drift [DR-P2]|Same gap as DR-C6, conceptually close to TS5 (Exponential Smoothing)|
|DR-C8|STEPD|Compares the accuracy over a recent window against the accuracy over the whole stream using a statistical test, flagging drift when the two diverge significantly|Checking whether this week's report card looks like the semester average, not just yesterday's|Sudden drift [DR-P1] with a known "before" baseline|PS6 (Chi-Square, used internally for the significance test)|
|DR-C9|FHDDM (Fast Hoeffding Drift Detection Method)|A sliding-window variant of the Hoeffding-bound family (DR-C6/DR-C7) tuned to detect drift faster at the cost of more false positives|The twitchy sibling of HDDM : trades a few false alarms for catching real ones sooner|Sudden drift [DR-P1] where detection speed matters more than false-positive rate|Same gap as DR-C6/DR-C7|

---

## Domain: Multivariate Drift Detection [DR-MV]

Every test in [DR-M] is univariate : one feature at a time. Real drift often only shows up in the _joint_ distribution across features, which this domain covers.

|Code|Method|Definition|Metaphor|Math Code|Data Fit|
|---|---|---|---|---|---|
|DR-MV1|Mahalanobis Distance Drift|Measures how far a new batch's mean sits from the reference distribution's mean, scaled by the reference covariance structure|Checking whether a new suspect fits the _whole_ profile : not just height, or just weight, but both together, correlations included|LA21|A31…A38, A41/A42|
|DR-MV2|PCA Reconstruction-Error Drift|Projects new data through a PCA model [DR1] fit on the reference window; a rising reconstruction error signals the new data no longer lives on the old subspace|The photograph from [DR1]'s metaphor stops looking like the sculpture : the angle that used to hide the least suddenly hides very little|LA5+LA6+CA8|A42 (high-dimensional)|

---

## Domain: Classifier-Based / Adversarial Validation Drift [DR-CV]

Extremely common in practice, and absent from [DR-M] entirely: instead of comparing distributions statistically, train a model to tell "old" and "new" data apart.

|Code|Method|Definition|Metaphor|Model Code|Data Fit|
|---|---|---|---|---|---|
|DR-CV1|Adversarial Validation Drift|Label old-window rows 0 and new-window rows 1, train a classifier, and evaluate its discriminative power; if the classifier can separate them well above chance, that gap _is_ the drift signal|Hiring a detective whose only job is to tell two lineups apart : if they succeed easily, the lineups were never the same crowd|LM2 or EN3 (any [SL] classifier)|A11-A21-A31…A38 (inherits chosen classifier's Data Code)|

Note: the natural evaluation metric here is AUC-ROC, which has no existing Mathnote code : a real gap in [PS] alongside the ones already flagged.

---

## Domain: Deep-Learning-Native Drift Detection [DR-DL]

Detectors built specifically for high-dimensional, deep-learning-native data (images, audio, embeddings), where classical [DR-M] tests are impractical directly on raw pixels/waveforms. Evolves the ideas in [DR-CV] (classifier-based) and [DR-MV2] (reconstruction-error) into deep-learning form.

|Code|Detector|Definition|Metaphor|Model Code|Math Code|
|---|---|---|---|---|---|
|DR-DL1|Learned-Kernel MMD (Spot-the-Diff)|Trains a small neural net to _learn_ the kernel that maximizes MMD's [DR-M10] power to separate old vs. new data, rather than using a fixed RBF kernel|DR-CV1's adversarial-validation witness, but instead of just voting old-vs-new, it also gets to design its own lie-detector test as it goes|NN2 (small trained network) feeding DR-M10|CA3 (trains the kernel network) + DR-M10's own gap|
|DR-DL2|Autoencoder Reconstruction-Error Drift|Same logic as PCA reconstruction-error drift [DR-MV2], but using a trained autoencoder [DR5] instead of a linear PCA subspace : common for images/audio where the "normal" manifold is nonlinear|DR-MV2's photograph metaphor, except the camera is now a deep autoencoder that learned a curved, not flat, way of looking at the sculpture|DR5 (Autoencoder)|CA3+CA8+LA2 (inherited from DR5)|

---

## Domain: Multiple-Hypothesis-Testing Correction [DR-MH]

Monitoring hundreds of features at once with per-feature p-values [DR-M9] guarantees false positives unless corrected. This sits as a required layer under any [DR-M] test that produces a p-value.

|Code|Technique|Definition|Metaphor|Math Code|
|---|---|---|---|---|
|DR-MH1|Bonferroni Correction|Divides the significance threshold by the number of features tested, so the _family-wise_ false-positive rate stays controlled|Lowering the bar for "suspicious" proportionally to how many suspects you're interrogating at once|PS15|
|DR-MH2|Benjamini-Hochberg FDR Correction|Ranks all p-values and accepts the largest prefix that keeps the expected false-discovery _rate_ below a target, less conservative than Bonferroni|Accepting that a few false alarms are fine, so long as most of the alarms that do fire are real|PS15|

---

## Domain: What Gets Monitored : Data-Type-to-Test Routing [DR-D]

Maps each Data Segmentation modality code to the detection test(s) that actually fit it.

|Code|Data Modality (Data code)|Recommended Test(s)|Why|
|---|---|---|---|
|DR-D1|Numeric / Continuous [A31]|KS Test [DR-M1], PSI [DR-M2], Wasserstein [DR-M6]|These tests are built for comparing two continuous distributions directly|
|DR-D2|Categorical [A32]|Chi-Square [DR-M3], PSI [DR-M2] (bucketed)|Category counts need a frequency-comparison test, not a continuous one|
|DR-D3|Ordinal [A33]|PSI [DR-M2] with ordered buckets, or KS Test [DR-M1] treating ranks as continuous|Order matters, so binning must respect rank rather than treating categories as unordered|
|DR-D4|Text [A34]|Vocabulary-frequency Chi-Square [DR-M3], or embedding-distribution drift via KL/JS/Wasserstein [DR-M4/M5/M6]|Raw text needs to be reduced to either token frequencies or embedding vectors before any distributional test applies|
|DR-D5|Image / Visual [A35]|Embedding-distribution drift via KL/JS/Wasserstein [DR-M4/M5/M6] on extracted feature vectors|Pixels themselves aren't directly comparable; drift is measured on learned feature representations|
|DR-D6|Audio [A36]|Spectral-feature drift via the same embedding-distribution tests [DR-M4/M5/M6]|Same logic as images : drift is measured on extracted spectral/embedding features|
|DR-D7|Graph-native [A37]|Structural drift via KS/Chi-Square [DR-M1/DR-M3] on degree-distribution or centrality sequences|Graphs need to be reduced to a structural summary statistic before a distributional test applies|
|DR-D8|Sequential / Temporal structure [A22]|Any test above, but only inside a time-aware window (pairs with [DR-W2])|Naively pooling all history erases exactly the shift you're trying to detect|
|DR-D9|Non-stationary Distributional Property [A54]|All of the above, by definition|This is the literal Data Segmentation code for "this dataset drifts" : any model consuming A54 data should have drift monitoring by default|

---

## Domain: Monitoring Window Strategies [DR-W]

|Code|Strategy|Metaphor|Math Codes|
|---|---|---|---|
|DR-W1|Fixed Reference Window|All future data is compared forever against one frozen training-time snapshot|The constitution : every future law is checked against this one origin document, never updated|
|DR-W2|Sliding / Rolling Reference Window|The reference window itself moves forward, always comparing "now" to "recently" rather than to day one|Comparing this week's weather to last week's, not to the very first week ever recorded|
|DR-W3|Sequential / CUSUM-Style Monitoring|Small deviations accumulate silently until their running total crosses a threshold|A scale that doesn't ring an alarm on the first gram, only once total added weight crosses a limit|
|DR-W4|Fixed-Interval Batch Monitoring|Drift is only checked at scheduled points rather than continuously|A monthly checkup instead of a heart monitor|

---

## Domain: Threshold / Severity Calibration [DR-TH]

Bridges "a test produced a number" to "should this fire an alert" : missing from [DR-M]/[DR-R] as written. Pairs each test's raw statistic with the cut-offs the industry actually uses.

|Code|Test|Stable|Moderate|Significant|Typical Response|
|---|---|---|---|---|---|
|DR-TH1|PSI [DR-M2]|< 0.1|0.1 – 0.25|> 0.25|Significant → [DR-R1]/[DR-R2]; moderate → [DR-R1] only|
|DR-TH2|KS Statistic [DR-M1] + p-value [DR-M9]|p ≥ 0.05|p < 0.05, small effect size|p < 0.05, large effect size|Statistical significance alone should not fire [DR-R2] : pair with an effect-size floor, not p-value alone|
|DR-TH3|JS Divergence [DR-M5]|< 0.05|0.05 – 0.15|> 0.15|Bounded [0,1], so thresholds are dataset-relative and should be tuned against a historical baseline, not assumed universal|

---

## Domain: Drift Response Actions [DR-R]

|Code|Action|Metaphor|
|---|---|---|
|DR-R1|Alert Only|A smoke detector that just beeps : a human decides what to do next|
|DR-R2|Automatic Retrain Trigger|A fire sprinkler that activates itself the moment smoke crosses a threshold, no human in the loop|
|DR-R3|Model Rollback|Undoing a bad software deploy : revert to the last known-good model version|
|DR-R4|Feature Reweighting / Importance Sampling|Recalibrating an existing scale rather than buying a new one : adjust how much weight recent data gets without a full retrain|
|DR-R5|Shadow / Champion-Challenger Deployment|Running the new judge in parallel with the old one, watching how they'd differ, before ever letting the new judge actually rule on a real case|

---

## Domain: Drift Correction (beyond detection/response) [DR-CR]

[DR-R] covers retrain/rollback/reweight at a coarse level; this adds the specific technique of correcting for drift _without_ a full retrain.

|Code|Technique|Definition|Metaphor|Math Code|Pairs With|
|---|---|---|---|---|---|
|DR-CR1|Importance Weighting / Density-Ratio Estimation|Re-weights training examples by the estimated ratio of new-distribution density to old-distribution density, so the existing model is corrected for drift without retraining from scratch|Adjusting an old survey's answers by how much the population has shifted, instead of running the whole survey again|IT4 (density-ratio estimation typically minimizes a KL-divergence objective)|DR-R4 (Feature Reweighting), sits underneath it as the actual mechanism|

---

## Domain: Root-Cause Segmentation [DR-RS]

DR-M7 (ANOVA) touches this but "which cohort/segment is driving the drift" deserves its own drill-down workflow rather than a single row in [DR-M].

|Code|Step|Definition|Metaphor|Math Code|
|---|---|---|---|---|
|DR-RS1|Segment-Level Re-Testing|Re-run the chosen [DR-M] test independently within each cohort/segment (region, channel, time window) instead of only on the pooled population|Interviewing each neighborhood separately instead of assuming the whole city changed together|Whichever [DR-M] test was used at the pooled level|
|DR-RS2|ANOVA Across Segments|Formally test whether segment means differ from each other, pinpointing which segment(s) are statistically distinct|DR-M7, promoted from a single detection row to the first formal step of attribution|PS21|
|DR-RS3|Feature-Attribution Drill-Down|Once a segment is implicated, decompose _which features_ within it drifted most, using per-feature [DR-M] tests restricted to that segment|Having isolated which neighborhood changed, now asking which street in it changed|DR-M1…DR-M6, applied per-segment|

This is [DR-X4] (Drift Detection vs. Drift Attribution)'s "which room did the smoke start in" question, formalized as a three-step workflow rather than left as a disambiguation note.

---

## Domain: Full Data-Axis Signature for Drift [DR-Ax]

Drift's defining Data Segmentation code is A54 (Non-stationary) at Axis 5 : but detecting and explaining drift touches every other axis too.

|Axis Code|Axis Name|Codes Touched|Why|
|---|---|---|---|
|A1|Supervision / Label Density|A11 (for DR-T3/label drift, DR-M8/performance drift) or A12 (for DR-T1/DR-T2, which can be monitored without any new labels at all)|Covariate and concept drift can be caught unsupervised on features alone; label drift and performance drift require ground truth|
|A2|Relational Structure|A22 (Sequential/temporal) almost always : drift is fundamentally a comparison across time|Drift monitoring only makes sense with an ordered "before" and "after"|
|A3|Modality|Any : A31 through A38, routed per [DR-D]|Every modality can drift; only the test used to detect it changes|
|A4|Dimensionality & Scale|A41/A42|High-dimensional feature spaces often need dimensionality reduction (see ML-Model-Taxonomy [DR1]-[DR4]) before drift tests are even feasible|
|A5|Distributional Properties|A54 (Non-stationary) is the core signature; A52 (Imbalanced) matters because rare-class drift is easy to miss|This axis literally defines what drift is|
|A6|Missingness / Quality|A62/A63/A64|A spike in missingness or noise can masquerade as drift (see [DR-X2]) : must be ruled out before declaring real drift|

---

## Domain: Full Combined Signature : Per Drift Type [DR-F]

|Code|Drift Type|Typical Math Signature|Typical Data Signature|Requires Ground-Truth Labels?|
|---|---|---|---|---|
|DR-F1|DR-T1 Covariate Shift|PS7+PS8+PS18 (feature-level tests)|A12-A22-A31…A38|No|
|DR-F2|DR-T2 Concept Drift|PS5+PS15 (performance-based, since the X→Y mapping itself must be inferred from outcomes)|A11-A22-A31…A38|Yes|
|DR-F3|DR-T3 Label / Prior Drift|PS6+PS8 (frequency/bucket shift on the label itself)|A11-A22-A32/A33|Yes|
|DR-F4|DR-T4 Prediction Drift|IT4+PS19+PS8 (distribution tests applied to model outputs, not raw features)|A12-A22-A31/A32|No (uses predictions, not truth)|
|DR-F5|DR-T5 Upstream / Operational Drift|PS6+PS8 applied per-feature, cross-checked against [DR-X2] missingness/quality signals|A12-A22-A61…A64|No|

---

## Domain: Drift vs. Related Concepts : Disambiguation [DR-X]

Drift gets confused with a few neighboring phenomena. This table exists so the taxonomy doesn't blur them together.

|Code|Concept|How It Differs From Drift|Metaphor|
|---|---|---|---|
|DR-X1|Outlier / Anomaly Detection|A single unusual point, evaluated against a distribution assumed to be stable : not a shift in the distribution itself|One oddly-dressed guest at a party, vs. the whole guest list's fashion sense changing|
|DR-X2|Data Quality Issue (e.g. missingness spike, schema bug)|Looks identical to drift on a dashboard but is caused by a pipeline defect, not a real-world change : must be ruled out first (see [DR-T5])|The thermometer broke, the weather didn't actually change|
|DR-X3|Adversarial Attack|A deliberately engineered input shift designed to fool the model : organic drift is unintentional, an attack is targeted|Someone deliberately feeding the judge false evidence, vs. the world itself just changing|
|DR-X4|Drift Detection vs. Drift Attribution|Detection answers "did something change?"; attribution answers "which feature(s) caused it?" : the tests in [DR-M] mostly do the former, not the latter, though [DR-RS] now formalizes the attribution workflow|A smoke detector tells you there's smoke; it doesn't tell you which room it started in|

---

