"""Taxonomy codes to real estimators.

Every model the site can recommend for a table has an entry here, so the
benchmark scores the taxonomy's own advice rather than a convenient subset.
Models that need something a CSV cannot hold (images, graphs, sequences) have
no entry, and the runner records them as not runnable rather than skipping
them silently.

Preprocessing follows the profiler's own column classification, so the
pipeline a model gets matches the axes the site reports: categorical columns
are one-hot encoded, numeric columns imputed, and scaled only for the models
that need it (linear, SVM, nearest neighbours, neural).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.stats import loguniform
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor, HistGradientBoostingClassifier,
    HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
    StackingClassifier, StackingRegressor, VotingClassifier, VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import (GridSearchCV, KFold, ParameterSampler, ShuffleSplit, StratifiedKFold,
                                     StratifiedShuffleSplit)
from sklearn.linear_model import (
    BayesianRidge, ElasticNet, Lasso, LinearRegression, LogisticRegression, Perceptron, Ridge,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

SEED = 0


@dataclass
class ModelSpec:
    code: str                      # the taxonomy code, e.g. "TR2"
    name: str
    build: Callable                # build(task) -> estimator
    scale: bool = False            # needs standardised features
    dense: bool = True             # cannot take a sparse matrix
    tasks: tuple[str, ...] = ("classification", "regression")
    notes: str = ""
    max_rows: int | None = None    # larger datasets are recorded as skipped, with the reason


def _hgb(task):
    return HistGradientBoostingClassifier(random_state=SEED) if task == "classification" \
        else HistGradientBoostingRegressor(random_state=SEED)


def _base_learners(task):
    """Small, fast, different in kind: what a stack or vote is usually built from."""
    if task == "classification":
        return [
            ("tree", DecisionTreeClassifier(max_depth=6, random_state=SEED)),
            ("knn", KNeighborsClassifier()),
            ("linear", LogisticRegression(max_iter=2000)),
        ]
    return [
        ("tree", DecisionTreeRegressor(max_depth=6, random_state=SEED)),
        ("knn", KNeighborsRegressor()),
        ("linear", Ridge()),
    ]


SPECS: list[ModelSpec] = [
    ModelSpec("LM1", "Linear Regression", lambda t: LinearRegression(), scale=True, tasks=("regression",)),
    ModelSpec("LM2", "Logistic Regression", lambda t: LogisticRegression(max_iter=3000), scale=True,
              tasks=("classification",)),
    ModelSpec("LM3", "Ridge Regression", lambda t: Ridge(random_state=SEED), scale=True, tasks=("regression",)),
    ModelSpec("LM4", "Lasso Regression", lambda t: Lasso(random_state=SEED), scale=True, tasks=("regression",)),
    ModelSpec("LM5", "Elastic Net", lambda t: ElasticNet(random_state=SEED), scale=True, tasks=("regression",)),
    ModelSpec("LM6", "Polynomial Regression",
              lambda t: Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)), ("ridge", Ridge())]),
              scale=True, tasks=("regression",), notes="degree 2 with a ridge penalty"),
    ModelSpec("LM7", "Bayesian Linear Regression", lambda t: BayesianRidge(), scale=True, tasks=("regression",)),

    ModelSpec("TR1", "Decision Tree",
              lambda t: DecisionTreeClassifier(random_state=SEED) if t == "classification"
              else DecisionTreeRegressor(random_state=SEED)),
    ModelSpec("TR2", "Random Forest",
              lambda t: RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
              if t == "classification" else RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)),
    ModelSpec("TR3", "Extra Trees",
              lambda t: ExtraTreesClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
              if t == "classification" else ExtraTreesRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)),

    ModelSpec("EN1", "AdaBoost",
              lambda t: AdaBoostClassifier(random_state=SEED) if t == "classification"
              else AdaBoostRegressor(random_state=SEED)),
    ModelSpec("EN2", "Gradient Boosting (GBM)",
              lambda t: GradientBoostingClassifier(random_state=SEED) if t == "classification"
              else GradientBoostingRegressor(random_state=SEED)),
    ModelSpec("EN6", "Stacking",
              lambda t: StackingClassifier(_base_learners(t), final_estimator=LogisticRegression(max_iter=2000), n_jobs=-1)
              if t == "classification" else StackingRegressor(_base_learners(t), final_estimator=Ridge(), n_jobs=-1),
              scale=True),
    ModelSpec("EN7", "Voting Ensemble",
              lambda t: VotingClassifier(_base_learners(t), voting="soft", n_jobs=-1) if t == "classification"
              else VotingRegressor(_base_learners(t), n_jobs=-1), scale=True),

    ModelSpec("SV1", "SVM (linear)",
              lambda t: LinearSVC(max_iter=5000) if t == "classification" else LinearSVR(max_iter=5000), scale=True),
    ModelSpec("SV2", "Kernel SVM (RBF/Poly)",
              lambda t: SVC(random_state=SEED) if t == "classification" else SVR(), scale=True),

    ModelSpec("PR1", "Naive Bayes", lambda t: GaussianNB(), tasks=("classification",)),
    ModelSpec("IB1", "k-Nearest Neighbors",
              lambda t: KNeighborsClassifier() if t == "classification" else KNeighborsRegressor(), scale=True),
    ModelSpec("DR4", "LDA (Linear Discriminant Analysis)", lambda t: LinearDiscriminantAnalysis(), scale=True,
              tasks=("classification",)),
    ModelSpec("NN1", "Perceptron", lambda t: Perceptron(random_state=SEED), scale=True, tasks=("classification",)),
    ModelSpec("NN2", "MLP (Feedforward)",
              lambda t: MLPClassifier(random_state=SEED, max_iter=600) if t == "classification"
              else MLPRegressor(random_state=SEED, max_iter=600), scale=True),
]

# Gradient boosting libraries, added when installed.
try:  # pragma: no cover - depends on the environment
    from xgboost import XGBClassifier, XGBRegressor

    SPECS.append(ModelSpec("EN3", "XGBoost",
                           lambda t: XGBClassifier(random_state=SEED, verbosity=0, n_jobs=-1)
                           if t == "classification" else XGBRegressor(random_state=SEED, verbosity=0, n_jobs=-1)))
except ImportError:
    pass

try:  # pragma: no cover
    from lightgbm import LGBMClassifier, LGBMRegressor

    SPECS.append(ModelSpec("EN4", "LightGBM",
                           lambda t: LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=-1)
                           if t == "classification" else LGBMRegressor(random_state=SEED, verbose=-1, n_jobs=-1)))
except ImportError:
    pass

BY_CODE = {spec.code: spec for spec in SPECS}

# Models the taxonomy can recommend for a table but this benchmark cannot run,
# each with the reason. Listed rather than dropped, so the coverage test fails
# if a new model appears with no decision made about it.
NOT_RUNNABLE = {
    "EN5": "CatBoost is not installed here; it would need its own build",
    "NN3": "a convolutional network needs images, which a CSV cannot hold",
    "NN8": "ResNet needs images",
    "NN9": "U-Net needs images",
    "NN10": "a Siamese network needs pairs, not rows",
    "NN4": "an RNN needs sequences",
    "NN5": "an LSTM needs sequences",
    "NN6": "a GRU needs sequences",
    "NN7": "a Transformer needs sequences",
    "PR3": "a hidden Markov model needs sequences",
    "PR4": "a Bayesian network needs a specified graph structure",
    "PR5": "a conditional random field needs sequences",
    "GR1": "a graph neural network needs a graph",
    "GR2": "a graph convolutional network needs a graph",
    "GR3": "a graph attention network needs a graph",
    "NLP1": "Word2Vec needs a text corpus",
    "NLP2": "BERT needs a text corpus",
    "NLP3": "GPT-style models need a text corpus",
    "DR5": "an autoencoder is trained on unlabelled data, not scored on a target",
}

# Runnable here, but not in the taxonomy's tabular scope: histogram boosting is
# the "always reach for this" baseline every recommendation is measured against.
BASELINE = ModelSpec("BASE-HGB", "Histogram Gradient Boosting (baseline)", _hgb)


# References: stronger than any default, and never recommended. They answer the
# question a reviewer asks first: the order beats default boosting, but does it
# beat boosting someone bothered to tune, or a model built for small tables?
# Every code that starts with "BASE-" is outside the taxonomy (is_reference).

TUNING_CONFIGS = 10
HGB_SPACE = {
    "learning_rate": loguniform(0.03, 0.3),
    "max_iter": [100, 200, 400],
    "max_leaf_nodes": [7, 15, 31, 63, 127],
    "min_samples_leaf": [5, 10, 20, 40, 80],
    "l2_regularization": loguniform(1e-4, 10),
    "max_features": [0.5, 0.75, 1.0],
}
# scikit-learn's own defaults, which are good: a search that could not pick
# them would sometimes do worse than not tuning at all.
HGB_DEFAULT = {"learning_rate": 0.1, "max_iter": 100, "max_leaf_nodes": 31, "min_samples_leaf": 20,
               "l2_regularization": 0.0, "max_features": 1.0}


def hgb_candidates(n: int = TUNING_CONFIGS) -> list[dict]:
    """The defaults, then n - 1 random configurations; fixed by SEED."""
    sampled = ParameterSampler(HGB_SPACE, n_iter=n - 1, random_state=SEED)
    tuned = [{k: float(v) if isinstance(v, np.floating) else v for k, v in c.items()} for c in sampled]
    return [HGB_DEFAULT, *tuned]


def inner_splits(n_rows: int, task: str):
    """How to choose between configurations, given the rows there are to choose on.

    One 80/20 split is cheap, but on 200 rows it leaves 40 to judge ten
    configurations by, which is mostly noise: a first version tuned that way
    lost to the untuned defaults on 11 of 20 small datasets. Three-fold
    cross-validation judges every configuration on every row of the training
    fold, which is what removes the noise, at three fits a configuration.
    Above 3,000 rows a fifth of them is a real validation set, and one split
    is enough.
    """
    classification = task == "classification"
    if n_rows <= 3000:
        return (StratifiedKFold if classification else KFold)(3, shuffle=True, random_state=SEED)
    return (StratifiedShuffleSplit if classification else ShuffleSplit)(n_splits=1, test_size=0.2,
                                                                      random_state=SEED)


class TunedHGB(BaseEstimator):
    """Histogram boosting with a small, standard tuning budget.

    Ten configurations: scikit-learn's defaults and nine random ones (learning
    rate, number of trees, tree size, leaf size, regularisation, feature
    sampling). Each is judged by cross-validation inside the training fold
    (inner_splits), and the best is refitted on the whole training fold. It
    runs inside the same five outer folds as everything else, so its score is
    never judged on data it was tuned on. Early stopping stays off, as
    scikit-learn leaves it below 10,000 rows: on small data its validation
    slice is too small to stop on. A bigger search would score a little higher;
    this is the budget a practitioner spends without thinking about it, and
    the page says so.
    """

    def __init__(self, task: str = "classification"):
        self.task = task

    def fit(self, X, y):
        if self.task == "classification":
            model, scoring = HistGradientBoostingClassifier(random_state=SEED), "balanced_accuracy"
        else:
            model, scoring = HistGradientBoostingRegressor(random_state=SEED), "r2"
        grid = [{k: [v] for k, v in config.items()} for config in hgb_candidates()]
        self.search_ = GridSearchCV(model, grid, cv=inner_splits(len(X), self.task), scoring=scoring,
                                    n_jobs=1, refit=True, error_score=np.nan).fit(X, y)
        self.best_params_ = self.search_.best_params_
        if self.task == "classification":
            self.classes_ = self.search_.classes_
        return self

    def predict(self, X):
        return self.search_.predict(X)


def _hgb_tuned(task):
    return TunedHGB(task)


TUNED = ModelSpec("BASE-HGB-TUNED", "Histogram Gradient Boosting, tuned", _hgb_tuned,
                  notes=f"the defaults and {TUNING_CONFIGS - 1} random configurations, chosen by cross-validation "
                        "inside each training fold")

# TabPFN, a model pretrained for small tables, when it is installed. Its weights
# come from Hugging Face behind a Prior Labs login (TABPFN_TOKEN), and their
# licence depends on the version: TabPFN-2 is Apache 2.0 with an attribution
# requirement, the later ones are non-commercial. TabPFN-2 is the default for
# that reason; DCN_TABPFN_VERSION picks another. On a CPU TabPFN-2 is built for
# up to 1,000 rows, so larger datasets are recorded as skipped unless
# DCN_TABPFN_MAX_ROWS says the machine can take more.
TABPFN_CODE = "BASE-TABPFN"
TABPFN_MAX_ROWS = int(os.environ.get("DCN_TABPFN_MAX_ROWS", "1000"))


def _installed_tabpfn() -> str | None:
    """The TabPFN release installed here, or None."""
    try:  # pragma: no cover - depends on the environment
        from importlib.metadata import version
        return version("tabpfn")
    except Exception:
        return None


# TabPFN-1 ships as tabpfn 0.1.x, a different API from the later releases.
_RELEASE = _installed_tabpfn()
TABPFN_VERSION = os.environ.get("DCN_TABPFN_VERSION") or ("v1" if (_RELEASE or "").startswith("0.1.") else "v2")


def _tabpfn(task):
    if TABPFN_VERSION == "v1":
        return TabPFNv1()
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.constants import ModelVersion

    cls = TabPFNClassifier if task == "classification" else TabPFNRegressor
    return cls.create_default_for_version(ModelVersion(TABPFN_VERSION), random_state=SEED,
                                          ignore_pretraining_limits=TABPFN_MAX_ROWS > 1000)


# TabPFN-1 (Hollmann et al., ICLR 2023). Its checkpoint lives on the project's
# tabpfn_v1 branch, Apache 2.0. The package's own loader looks for the highest
# epoch first and tries to download each missing one from a URL that no longer
# serves it, so the verified checkpoint is put where it looks first.
TABPFN_V1_URL = ("https://raw.githubusercontent.com/automl/TabPFN/tabpfn_v1/"
                 "tabpfn/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt")
TABPFN_V1_SHA256 = "3c9aadaeddbf51462af8c0ee4b3ca3c697890f77e92318abbb0821b75261c392"
TABPFN_V1_ENSEMBLE = 32          # its authors' recommendation; the package defaults to 3
TABPFN_V1_LIMITS = {"features": 100, "classes": 10}


def _tabpfn_v1_checkpoint() -> None:  # pragma: no cover - needs the network the first time
    """Put TabPFN-1's checkpoint where its loader looks, after checking its hash."""
    import hashlib
    import urllib.request
    from pathlib import Path

    import tabpfn

    target = Path(tabpfn.__file__).parent / "models_diff" / "prior_diff_real_checkpoint_n_0_epoch_100.cpkt"
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == TABPFN_V1_SHA256:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    data = urllib.request.urlopen(TABPFN_V1_URL, timeout=300).read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != TABPFN_V1_SHA256:
        raise RuntimeError(f"TabPFN-1 checkpoint has SHA-256 {digest}, expected {TABPFN_V1_SHA256}")
    target.write_bytes(data)


def _tabpfn_v1_sklearn_shim() -> None:  # pragma: no cover - depends on the environment
    """tabpfn 0.1.x passes force_all_finite, which scikit-learn 1.8 renamed ensure_all_finite."""
    import functools

    import tabpfn.scripts.transformer_prediction_interface as tpi
    from sklearn.utils import validation

    def renamed(fn):
        @functools.wraps(fn)
        def call(*args, force_all_finite=None, **kwargs):
            if force_all_finite is not None:
                kwargs.setdefault("ensure_all_finite", force_all_finite)
            return fn(*args, **kwargs)
        return call

    tpi.check_X_y = renamed(validation.check_X_y)
    tpi.check_array = renamed(validation.check_array)


class TabPFNv1(ClassifierMixin, BaseEstimator):
    """TabPFN-1 as a scikit-learn classifier, refusing what it was not built for.

    Too many features or classes raise, so the runner records an error with the
    reason and TabPFN-1 is judged only where it ran.
    """

    def fit(self, X, y):  # pragma: no cover - needs the checkpoint
        X = np.asarray(X, dtype=float)
        classes = np.unique(y)
        if X.shape[1] > TABPFN_V1_LIMITS["features"]:
            raise ValueError(f"TabPFN-1 takes at most {TABPFN_V1_LIMITS['features']} features, this has {X.shape[1]}")
        if len(classes) > TABPFN_V1_LIMITS["classes"]:
            raise ValueError(f"TabPFN-1 takes at most {TABPFN_V1_LIMITS['classes']} classes, this has {len(classes)}")
        _tabpfn_v1_checkpoint()
        _tabpfn_v1_sklearn_shim()
        from tabpfn import TabPFNClassifier

        # The checkpoint is a pickle from before PyTorch's weights-only loading;
        # its hash was checked above, so it is loaded the old way, once.
        previous = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        try:
            self.model_ = TabPFNClassifier(device="cpu", N_ensemble_configurations=TABPFN_V1_ENSEMBLE, seed=SEED)
        finally:
            if previous is None:
                os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
            else:
                os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):  # pragma: no cover - needs the checkpoint
        return self.model_.predict(np.asarray(X, dtype=float))

    def predict_proba(self, X):  # pragma: no cover - needs the checkpoint
        return self.model_.predict_proba(np.asarray(X, dtype=float))


REFERENCES: list[ModelSpec] = [TUNED]
if _RELEASE:  # pragma: no cover - depends on the environment
    REFERENCES.append(ModelSpec(
        TABPFN_CODE, f"TabPFN ({TABPFN_VERSION})", _tabpfn, max_rows=TABPFN_MAX_ROWS,
        tasks=("classification",) if TABPFN_VERSION == "v1" else ("classification", "regression"),
        notes=f"pretrained for small tables; run on up to {TABPFN_MAX_ROWS} rows"))

REFERENCE_BY_CODE = {spec.code: spec for spec in REFERENCES}
# Names for every reference a results file may hold, installed here or not.
REFERENCE_NAMES = {BASELINE.code: BASELINE.name, TUNED.code: TUNED.name, TABPFN_CODE: "TabPFN"}


def is_reference(code: str) -> bool:
    """The baseline and the references: run and reported, never recommended."""
    return str(code).startswith("BASE-")


def runnable_codes(task: str) -> list[str]:
    return [s.code for s in SPECS if task in s.tasks]


def build_pipeline(spec: ModelSpec, task: str, numeric_cols, categorical_cols):
    """The estimator wrapped in the preprocessing its kind needs."""
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if spec.scale:
        numeric_steps.append(("scale", StandardScaler()))

    transformers = [("num", Pipeline(numeric_steps), numeric_cols)]
    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=50)),
            ]),
            categorical_cols,
        ))

    pre = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)
    return Pipeline([("pre", pre), ("model", spec.build(task))])
