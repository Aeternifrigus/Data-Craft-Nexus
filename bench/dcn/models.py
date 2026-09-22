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

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor, HistGradientBoostingClassifier,
    HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
    StackingClassifier, StackingRegressor, VotingClassifier, VotingRegressor,
)
from sklearn.impute import SimpleImputer
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
