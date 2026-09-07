Stacking is not one model : it is a meta-architecture that wraps other models. It is made up of different [[ML Models]] for better mechanism over types of [[Data]].


==**Hence Stacking has: N Level-0 base learners (referenced by Model code), 1 fold/validation mechanism, 1+ Level-1 meta-learner(s), and a structural variant : and only then does it inherit a final Math+Data signature.**==

---

## Domain: Stacking Variants : The Big Picture [ST-V]

|Code|Variant|Metaphor|Mechanism|
|---|---|---|---|
|ST-V1|Classic (K-Fold) Stacking|Witnesses are cross-examined only on cases they weren't personally involved in, so their notes to the judge aren't tainted by having already seen the answer|Base learners trained on K-1 folds, predict on the held-out fold; out-of-fold predictions become Level-1 training data|
|ST-V2|Blending (Holdout Stacking)|One dress rehearsal instead of a full jury rotation : faster, but the judge trains on less|Base learners trained on one training split, predict once on a single holdout set; those predictions train the meta-learner|
|ST-V3|Multi-Level (Deep) Stacking|A court of appeals : judges reviewing judges reviewing witnesses|3+ levels: Level-0 base learners → Level-1 meta-learners → Level-2 "judge of judges"|
|ST-V4|Feature-Augmented Stacking (Restacking)|The judge also gets to read the original case file, not just hear the witnesses summarize it|Meta-learner is trained on base predictions **plus** the original input features|
|ST-V5|Homogeneous Stacking|A jury of clones : same algorithm, different seeds or bootstrapped subsamples, each forming a slightly different opinion|All Level-0 base learners are the same Model code (e.g. five TR2 Random Forests with different seeds)|
|ST-V6|Heterogeneous Stacking|A jury deliberately hand-picked from different professions|Level-0 base learners are drawn from different Model families (e.g. LM2 + TR2 + EN3 + SV2 together)|
|ST-V7|Weighted Rank Averaging : Hand-Tuned (Non-Learned Combiner)|The judge doesn't retrain at all : just applies a fixed formula, weighting each witness by gut feel or past reputation|No trained meta-learner; base predictions combined via a hand-picked fixed weighted average instead of an ST-M layer|
|ST-V8|Multi-Layer Stacked Generalization (Wolpert's original)|The founding format all the above variants branch from : one clean K-fold Level-0 → Level-1 pipeline|The textbook definition; equivalent to ST-V1 with a single meta-learner|
|ST-V9|Optimized-Weight Blending|The judge still doesn't retrain a full model, but does sit down and _solve_ for the exact weights that would have made the panel most accurate in hindsight|Base predictions combined via weights found by constrained least-squares / Nelder-Mead optimization (weights sum to 1, non-negative) rather than hand-tuned|
|ST-V10|Time-Series-Safe Stacking|A witness is never allowed to testify about a case using information from the future|Same as ST-V1 but folds/holdouts respect chronological order (pairs with ST-CV5)|
|ST-V11|Mixture-of-Experts Stacking (Gated)|Instead of one judge weighing all witnesses equally on every case, a separate gatekeeper decides _which_ witness (or blend of witnesses) is even worth listening to for this specific case|A gating network/model outputs per-input weights over base learners, rather than one fixed meta-learner applied identically to all inputs|

---

## Domain: Level-0 Base Learners : Mapped to ML-Model-Taxonomy Codes [ST-B]

Any model from ML-Model-Taxonomy.md can serve as a Level-0 base learner. Each row below pulls its Math Codes and Data Codes directly from that model's own entry : nothing is re-derived.

|Code|Base Learner (Model Code)|Metaphor as a "Witness"|Math Codes (inherited)|Data Codes (inherited)|
|---|---|---|---|---|
|ST-B1|Logistic Regression [LM2]|The statistician witness, testifies in straight lines|LA1+CA10+CA22+PS3|A11-A21-A32|
|ST-B2|Random Forest [TR2]|The democracy witness, shows up as a hundred smaller witnesses who already voted|IT1+IT2+PS20|A11-A21-A38|
|ST-B3|XGBoost [EN3]|The CCP-govt witness, testifies only after correcting its own past errors under strict penalty|CA3+CA23+CA6+LA4|A11-A21-A42|
|ST-B4|Kernel SVM [SV2]|The boundary-drawing witness, testifies from the widest gap it could carve out|LA22+LA23+CA14+CA20|A11-A21-A56|
|ST-B5|MLP [NN2]|The assembly-line witness, testifies from layers of refined signal|CA4+CA12+CA3|A11-A21-A38|
|ST-B6|CNN [NN3]|The vision-specialist witness, only called when the case involves images|CA7+CA12+CA24|A11-A23-A35|
|ST-B7|Naive Bayes [PR1]|The odds-calculating witness, testifies purely in likelihoods|PS1+PS9+PS2|A11-A21-A32|
|ST-B8|k-Nearest Neighbors [IB1]|The neighbor-polling witness, testifies based on who's standing closest|LA9+LA19+LA20|A11-A21-A31|
|ST-B9|Gradient Boosting [EN2]|The relay-team witness, hands off a corrected version of the story|CA2+CA3+CA8|A11-A21-A31|
|ST-B10|LightGBM [EN4]|The startup witness, testifies fast and leaf-wise instead of level-by-level|CA3+IT2|A11-A21-A42|

_(Any other Model code from ML-Model-Taxonomy.md : LM1, LM3-7, TR1/TR3/TR4, EN1/EN5, SV1/SV3, PR2-5, NN1/4-14, GR1-3, etc. : can equally serve as a Level-0 base learner; the table above lists the most commonly stacked ones.)_

---

## Domain: Diversity Injection for Base Learners [ST-DIV]

Stacking only helps if the base learners actually disagree with each other; if every witness tells the same story, the judge learns nothing extra. These are the standard ways to force that disagreement.

|Code|Technique|Metaphor|Math Codes|
|---|---|---|---|
|ST-DIV1|Different Random Seeds (same Model code)|Clones of the same witness, each raised with slightly different early-life randomness|PS20|
|ST-DIV2|Bootstrapped Subsampling (Bagging-style, per base learner)|Each witness only ever saw a randomly resampled slice of the case files|PS20|
|ST-DIV3|Feature-Subspace Sampling|Each witness was only shown a random subset of the evidence, so no two witnesses build their opinion from identical clues|LA15 (rank/subspace selection)|
|ST-DIV4|Different Hyperparameters (same Model code)|Clones raised under different house rules : one strict, one lenient|Depends on chosen Model code|
|ST-DIV5|Different Model Families (Heterogeneous, = ST-V6)|Witnesses from entirely different professions, guaranteed to reason differently|Full union of chosen ST-B codes|

---

## Domain: Fold / Validation Mechanism [ST-CV]

This is the part that makes stacking actually work rather than leak and overfit : none of it lives in ML-Model-Taxonomy.md since it's a stacking-specific procedure, not a model.

|Code|Mechanism|Metaphor|Math Codes|
|---|---|---|---|
|ST-CV1|K-Fold Out-of-Fold Generation|Witnesses only testify about cases they personally weren't involved in deciding, so their notes to the judge are unbiased|PS20+PS4|
|ST-CV2|Holdout Split (used in Blending, ST-V2)|One dress rehearsal instead of jury duty on every case|PS4|
|ST-CV3|Leave-One-Out Stacking|The extreme version : each witness testifies on exactly one case at a time, informed by every other case|PS20|
|ST-CV4|Repeated K-Fold Stacking|Running the whole jury rotation multiple times with different splits and averaging, to calm down noisy folds|PS20+PS4+PS12|
|ST-CV5|Blocked / Purged Time-Series CV|A witness is never allowed to testify about a case using knowledge of what happened afterward|TS1+TS2+PS20 (folds walk forward chronologically; a "purge" gap removes cases too close to the boundary to prevent leakage)|

---

## Domain: Probability Calibration Before Stacking [ST-CAL]

Base learners' probability outputs are often systematically over- or under-confident. Feeding raw, miscalibrated probabilities into the meta-learner forces it to waste capacity correcting for that instead of learning genuine signal : so calibration is usually applied to each base learner's output before it ever reaches Level-1.

|Code|Technique|Metaphor|Math Codes|
|---|---|---|---|
|ST-CAL1|Platt Scaling|Teaching an overconfident witness to translate their "100% certain" into what that actually means historically|CA10+PS3|
|ST-CAL2|Isotonic Regression Calibration|A more flexible, step-wise version of the same translation : doesn't assume the witness's overconfidence is a clean straight-line bias|PS4+PS12|
|ST-CAL3|Temperature Scaling (common for NN base learners)|Turning down an overly confident witness's volume by one single dial, uniformly|CA11|

---

## Domain: Level-1 / Level-2 Meta-Learners : Mapped to ML-Model-Taxonomy Codes [ST-M]

The "judge." Classic stacking uses a simple linear judge, but any model can serve here : a more complex judge enables ST-V3 (Deep Stacking).

|Code|Meta-Learner (Model Code)|Metaphor|Math Codes (inherited)|
|---|---|---|---|
|ST-M1|Logistic Regression Meta [LM2]|The judge who weighs each witness's testimony and casts a verdict|LA1+CA10+CA22+PS3|
|ST-M2|Linear Regression Meta [LM1]|The judge who averages the testimonies into a single number|LA1+CA8+PS3|
|ST-M3|Gradient Boosting Meta [EN2]|A judge who is themselves a relay team, correcting their own mistakes about how much to trust each witness|CA2+CA3+CA8|
|ST-M4|Random Forest Meta [TR2]|A panel of judges voting on how much to trust the witnesses, rather than one judge deciding alone|IT1+IT2+PS20|
|ST-M5|Small Neural Net Meta [NN2]|A judge capable of learning genuinely non-obvious, non-linear rules for weighing testimony|CA4+CA12+CA3|
|ST-M6|Gating Network Meta (Mixture-of-Experts, = ST-V11)|Not one judge weighing all witnesses equally : a gatekeeper who decides, case by case, whose testimony is even worth hearing|CA11 (softmax gate)+CA3|

---

## Domain: Meta-Feature Engineering [ST-MF]

The meta-learner doesn't have to be fed only raw base-learner predictions : richer, derived features about the panel itself often help it learn faster and generalize better.

|Code|Meta-Feature|Metaphor|Math Codes|
|---|---|---|---|
|ST-MF1|Raw Base Predictions (default)|The judge hears each witness's final verdict only|: (passthrough)|
|ST-MF2|Base Prediction Variance / Disagreement|The judge also notices _how much the witnesses argued amongst themselves_, not just what they concluded|PS4 (variance)|
|ST-MF3|Prediction-Probability Entropy|The judge notices _how unsure_ each witness sounded, not just their final answer|IT1|
|ST-MF4|Pairwise Base-Model Correlation|The judge is told in advance which witnesses tend to agree with each other, so redundant testimony is weighted down|LA8|
|ST-MF5|Original Features Re-Included (= ST-V4)|The judge also gets to read the original case file directly|Whatever Math codes the meta-learner itself needs to process raw features|

---

## Domain: Full Data-Axis Coverage [ST-D]

Stacking is data-agnostic at the ensemble level : the meta-learner only requires labels (A11); every other axis is inherited from whichever ST-B base learners get chosen.

|Axis Code|Axis Name|Codes Covered|Why|
|---|---|---|---|
|A1|Supervision / Label Density|A11|The meta-learner is always trained on labels, even if individual base learners (e.g. an autoencoder used as a feature-generating base learner) are self-supervised internally|
|A2|Relational Structure|A21+A22+A23+A24+A25+A26|Any structure : different base learners can each specialize on a different relational type (e.g. ST-B6/CNN for spatial, ST-B9 for i.i.d.)|
|A3|Modality|A31+A32+A33+A34+A35+A36+A37+A38|Any modality : base learners can be mixed-modality specialists (ST-V6) feeding one meta-learner|
|A4|Dimensionality & Scale|A41+A42+A43+A44|Inherited per base learner : e.g. ST-B3/XGBoost handles high-dimensional, ST-B1/Logistic handles low-dimensional|
|A5|Distributional Properties|A51+A52+A55+A56|Base learners each absorb their own distributional quirks before the meta-learner ever sees the data|
|A6|Missingness / Quality|A61+A62+A63+A64|Individual base learners (e.g. tree-based ST-B2/ST-B3) can be deliberately chosen to tolerate missing/noisy data|

---

## Domain: Full Combined Signature : Per Variant [ST-F]

|Code|Variant|Full Math Signature (example: ST-B1+ST-B2+ST-B3 base, ST-M1 meta)|Full Data Signature|Paradigm|
|---|---|---|---|---|
|ST-F1|ST-V1 Classic K-Fold Stacking|LA1+CA10+CA22+IT1+IT2+PS20+CA3+CA23+CA6+LA4+PS3+PS4|A11-A21/A22/A23/A24/A25/A26-A31…A38-A41…A44-A51/A52/A55/A56-A61…A64|SL|
|ST-F2|ST-V2 Blending|Same as ST-F1, swap PS20→PS4 (holdout instead of k-fold)|Same as ST-F1|SL|
|ST-F3|ST-V3 Multi-Level (Deep) Stacking|ST-F1 signature + a second meta-layer's codes (e.g. ST-M3: CA2+CA3+CA8) stacked on top|Same as ST-F1 (meta-layers don't add new data types, only new math layers)|SL|
|ST-F4|ST-V4 Feature-Augmented Stacking|ST-F1 signature + whichever base-learner math the meta-learner itself now needs to process raw features directly|Same as ST-F1, plus the original feature's own A4/A5/A6 fully exposed to the meta-learner|SL|
|ST-F5|ST-V5 Homogeneous Stacking|Single base learner's Math Code repeated N times (no union needed : same family)|Single base learner's Data Code only|SL|
|ST-F6|ST-V6 Heterogeneous Stacking|Full union across every chosen ST-B code (worst case: all of ST-B1–ST-B10)|Full union, as in ST-D|SL|
|ST-F7|ST-V7 Weighted Rank Averaging (Hand-Tuned)|Whatever ST-B codes are used, **minus** any ST-M code (no trained meta-learner exists)|Same as base learners chosen|SL|
|ST-F8|ST-V9 Optimized-Weight Blending|Whatever ST-B codes are used + PS3 (weights fit via optimization, no full meta-model)|Same as base learners chosen|SL|
|ST-F9|ST-V10 Time-Series-Safe Stacking|Same as ST-F1, but ST-CV1 swapped for ST-CV5 (TS1+TS2+PS20 instead of PS20+PS4)|Same as ST-F1, restricted to A22 (Sequential/temporal) at Axis 2|SL|
|ST-F10|ST-V11 Mixture-of-Experts Stacking|ST-F1 signature + ST-M6's gating math (CA11+CA3)|Same as ST-F1|SL|
|ST-F11|With Calibration (any variant + ST-CAL)|Base ST-F signature + chosen ST-CAL code (e.g. +CA10+PS3 for Platt Scaling)|Same as base variant|SL|
|ST-F12|With Meta-Features (any variant + ST-MF)|Base ST-F signature + chosen ST-MF code (e.g. +PS4 for disagreement, +IT1 for entropy)|Same as base variant|SL|

---

## Domain: Stacking vs. Related Strategies : Disambiguation [ST-X]

Stacking gets confused with a few neighboring techniques. This table exists so the taxonomy doesn't blur them together.

|Code|Strategy|How It Differs From Stacking|Metaphor|
|---|---|---|---|
|ST-X1|Bagging (e.g. Random Forest, [TR2])|Base learners are trained on resampled data and combined by a _fixed_ rule (vote/average) : there is no trained meta-learner deciding how to weigh them|Democracy: every witness's vote counts equally, no judge involved at all|
|ST-X2|Boosting (e.g. XGBoost, [EN3])|Base learners are trained _sequentially_, each correcting the last one's errors : there's no separate Level-1 judge combining independent opinions after the fact|A relay race, not a courtroom : there's only ever one running story being corrected|
|ST-X3|Voting Ensemble ([EN7])|Same flat structure as Weighted Rank Averaging (ST-V7) but with _equal_, not weighted, votes : the simplest possible combiner|A jury that just counts raw hands, no weighting by reputation at all|
|ST-X4|Bayesian Model Averaging (BMA)|Combines models by their posterior probability of being the "correct" model, given the data : a principled probabilistic weighting, not a trained meta-learner|Instead of a judge learning who to trust, each witness is assigned a formal probability of being the one telling the truth|
|ST-X5|Mixture-of-Experts (as a native architecture, not stacking-flavored)|A single trained system where the gating and the "experts" are learned jointly end-to-end from scratch : stacking's Mixture-of-Experts flavor (ST-V11) mimics this after the fact, on top of already-independently-trained base learners|A courtroom built from day one to have a gatekeeper, vs. bolting a gatekeeper onto witnesses who already existed|

---