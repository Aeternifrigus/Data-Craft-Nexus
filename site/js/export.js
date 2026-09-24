// "Take it home": a Python script that runs the shortlist on your own file.
//
// The page recommends; this lets you check. The script uses the same
// preprocessing the benchmark uses (bench/dcn/models.py build_pipeline, with
// the profiler's own split of numeric and categorical columns), the same
// estimators, the same cross-validation, and puts tuned boosting beside the
// shortlist, because on the benchmark a small tuning budget was worth more
// than the choice among the top families.
//
// What it scores by is what the page was told a wrong answer costs
// (costs.js). Left alone, that is the benchmark's own score.
//
// Nothing runs here. The script is text; tests/export.test.js checks what goes
// in, and bench/tests/test_export.py runs generated scripts with Python and
// checks every estimator against the benchmark's registry. verify.js can run
// the same text in the browser, through the script's own load() and evaluate().

// The benchmark's estimators, as Python. `scale` mirrors ModelSpec.scale;
// `needs` names a library that is only there if installed.
export const MODEL_CODE = {
  LM1: { name: 'Linear Regression', scale: true, reg: 'LinearRegression()' },
  LM2: { name: 'Logistic Regression', scale: true, cls: 'LogisticRegression(max_iter=3000)' },
  LM3: { name: 'Ridge Regression', scale: true, reg: 'Ridge(random_state=SEED)' },
  LM4: { name: 'Lasso Regression', scale: true, reg: 'Lasso(random_state=SEED)' },
  LM5: { name: 'Elastic Net', scale: true, reg: 'ElasticNet(random_state=SEED)' },
  LM6: { name: 'Polynomial Regression', scale: true,
    reg: 'Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)), ("ridge", Ridge())])' },
  LM7: { name: 'Bayesian Linear Regression', scale: true, reg: 'BayesianRidge()' },
  TR1: { name: 'Decision Tree', cls: 'DecisionTreeClassifier(random_state=SEED)',
    reg: 'DecisionTreeRegressor(random_state=SEED)' },
  TR2: { name: 'Random Forest', cls: 'RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)',
    reg: 'RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)' },
  TR3: { name: 'Extra Trees', cls: 'ExtraTreesClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)',
    reg: 'ExtraTreesRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)' },
  EN1: { name: 'AdaBoost', cls: 'AdaBoostClassifier(random_state=SEED)', reg: 'AdaBoostRegressor(random_state=SEED)' },
  EN2: { name: 'Gradient Boosting (GBM)', cls: 'GradientBoostingClassifier(random_state=SEED)',
    reg: 'GradientBoostingRegressor(random_state=SEED)' },
  EN6: { name: 'Stacking', scale: true,
    cls: 'StackingClassifier(base_learners(), final_estimator=LogisticRegression(max_iter=2000), n_jobs=-1)',
    reg: 'StackingRegressor(base_learners(), final_estimator=Ridge(), n_jobs=-1)' },
  EN7: { name: 'Voting Ensemble', scale: true,
    cls: 'VotingClassifier(base_learners(), voting="soft", n_jobs=-1)', reg: 'VotingRegressor(base_learners(), n_jobs=-1)' },
  SV1: { name: 'SVM (linear)', scale: true, cls: 'LinearSVC(max_iter=5000)', reg: 'LinearSVR(max_iter=5000)' },
  SV2: { name: 'Kernel SVM (RBF/Poly)', scale: true, cls: 'SVC(random_state=SEED)', reg: 'SVR()' },
  PR1: { name: 'Naive Bayes', cls: 'GaussianNB()' },
  IB1: { name: 'k-Nearest Neighbors', scale: true, cls: 'KNeighborsClassifier()', reg: 'KNeighborsRegressor()' },
  DR4: { name: 'LDA (Linear Discriminant Analysis)', scale: true, cls: 'LinearDiscriminantAnalysis()' },
  NN1: { name: 'Perceptron', scale: true, cls: 'Perceptron(random_state=SEED)' },
  NN2: { name: 'MLP (Feedforward)', scale: true, cls: 'MLPClassifier(random_state=SEED, max_iter=600)',
    reg: 'MLPRegressor(random_state=SEED, max_iter=600)' },
  EN3: { name: 'XGBoost', needs: 'xgboost', cls: 'xgboost.XGBClassifier(random_state=SEED, verbosity=0, n_jobs=-1)',
    reg: 'xgboost.XGBRegressor(random_state=SEED, verbosity=0, n_jobs=-1)' },
  EN4: { name: 'LightGBM', needs: 'lightgbm', cls: 'lightgbm.LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=-1)',
    reg: 'lightgbm.LGBMRegressor(random_state=SEED, verbose=-1, n_jobs=-1)' },
  // Forecasting models: `fc` names the script's function that predicts each
  // row from the rows before it. bench/dcn/forecast.py runs these same
  // functions on real series.
  TSM1: { name: 'ARIMA', needs: 'statsmodels', fc: 'arima' },
  TSM2: { name: 'Exponential Smoothing (Holt-Winters)', needs: 'statsmodels', fc: 'smoothing' },
  TSM4: { name: "Croston's method (SBA)", fc: 'croston_sba' },
  TSM5: { name: 'TSB (Teunter-Syntetos-Babai)', fc: 'tsb' },
};

// Models the page can recommend that the script does not run, and why.
export const NOT_RUNNABLE = {
  TSM3: 'Prophet needs Stan, a compiled backend that the page cannot load and the script does not install',
};

const py = (value) => JSON.stringify(value);   // a JSON string or list is a valid Python literal
const pyBool = (b) => (b ? 'True' : 'False');
const oneLine = (text) => String(text).replace(/[\r\n]+/g, ' ');   // safe inside a Python comment
const fmtNumber = (x) => (Number.isInteger(x) ? String(x) : String(Math.round(x * 1e4) / 1e4));

// How the script scores, from a resolved cost (costs.js resolveCost), or the
// benchmark's own score when there is none.
function scoringCode(cost, bench) {
  const shown = (c) => c.metric + (c.higher ? '' : ' (shown negative: closer to zero is better)');
  if (cost?.id !== 'miss') {
    const c = cost ?? (bench === 'classification'
      ? { bench: true, scoring: 'balanced_accuracy', metric: 'balanced accuracy', higher: true }
      : { bench: true, scoring: 'r2', metric: 'R squared', higher: true });
    const why = c.bench ? 'the benchmark\'s own score'
      : `what you said a wrong answer costs: ${oneLine(c.label.charAt(0).toLowerCase() + c.label.slice(1))}`;
    return `# How every model is scored, and what tuned boosting is tuned for: ${why}.
SCORING = ${py(c.scoring)}
METRIC = ${py(shown(c))}`;
  }
  return `# How every model is scored, and what tuned boosting is tuned for: what its
# mistakes cost. You said a missed ${oneLine(py(cost.positive))} costs ${fmtNumber(cost.ratio)} false alarms. A row
# is flagged when its chance of being POSITIVE is above THRESHOLD, where a miss
# and a false alarm cost the same if the chances are right; a model that gives
# no chances is scored on its plain answer.
POSITIVE = ${py(cost.positive)}
MISS_COST = ${fmtNumber(cost.ratio)}
THRESHOLD = 1 / (1 + MISS_COST)


def cost_per_row(y, flagged):
    """A missed case costs MISS_COST, a false alarm costs 1."""
    y = np.asarray(y)
    return float(MISS_COST * np.sum((y == 1) & ~flagged) + np.sum((y == 0) & flagged)) / len(y)


def chances(model, X):
    """Each row's chance of being POSITIVE, or None from a model that gives none."""
    if not hasattr(model, "predict_proba"):
        return None
    classes = list(model.classes_)
    if 1 not in classes:   # a training fold with no case in it
        return np.zeros(len(X))
    return model.predict_proba(X)[:, classes.index(1)]


def cost_score(model, X, y):
    p = chances(model, X)
    flagged = np.asarray(model.predict(X)) == 1 if p is None else p > THRESHOLD
    return -cost_per_row(y, flagged)


SCORING = cost_score
METRIC = ${py(shown(cost))}`;
}

// A future value, forecast for real: the methods, which the SHORTLIST names,
// so they come before it. bench/tests/test_export.py checks that no method
// ever sees a row after the one it predicts.
const FORECAST_METHODS = `

# Forecasting. Every method gets the whole series and one fold (the rows it
# may learn from, and the rows it is scored on), and predicts each scored row
# using only the rows before it.
def last_value(y, train, test):
    return y[test - 1]


def running_average(y, train, test):
    return np.cumsum(y)[test - 1] / test


def one_step(fitted, y, test):
    """One step ahead through the scored rows, with the parameters fitted on the rows before them."""
    extended = fitted.append(y[test[0]:test[-1] + 1])
    return np.asarray(extended.predict(start=int(test[0]), end=int(test[-1])))


def arima(y, train, test):
    """ARIMA, its order chosen by AIC on the training rows, among the small ones."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # statsmodels turns its convergence warnings on when imported
        for d in (0, 1):
            for p in (0, 1, 2):
                for q in (0, 1):
                    try:
                        fitted = SARIMAX(y[train], order=(p, d, q), trend="c" if d == 0 else "n").fit(disp=False)
                    except Exception:
                        continue
                    if np.isfinite(fitted.aic) and (best is None or fitted.aic < best.aic):
                        best = fitted
        if best is None:
            raise RuntimeError("no ARIMA order could be fitted")
        return one_step(best, y, test)


def smoothing(y, train, test):
    """Exponential smoothing of the level, with a trend and with a SEASON-row season where AIC prefers them.

    A season is tried when the training rows hold two full cycles of it and it is at most 24 rows long, the
    limit R's forecast::ets keeps: a longer one has more starting values to fit than the method handles well."""
    from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
    best = None
    seasons = (None, SEASON) if SEASON and 1 < SEASON <= 24 and len(train) >= 2 * SEASON else (None,)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for trend in (False, True):
            for seasonal in seasons:
                try:
                    fitted = ExponentialSmoothing(y[train], trend=trend, seasonal=seasonal).fit(disp=False)
                except Exception:
                    continue
                if np.isfinite(fitted.aic) and (best is None or fitted.aic < best.aic):
                    best = fitted
        if best is None:
            raise RuntimeError("exponential smoothing could not be fitted")
        return one_step(best, y, test)


def seasonal_last(y, train, test):
    """Doing nothing, a season at a time: the value SEASON rows earlier (the last value before a season has passed)."""
    back = test - SEASON
    return np.where(back >= 0, y[np.maximum(back, 0)], y[test - 1])


# Intermittent demand: most periods sell nothing. Both methods smooth the size
# of a demand separately from how often one comes, and never see a row at or
# after the one they predict. The smoothing constants are chosen on the
# training rows by the squared error of their own one-step forecasts.
SMOOTHING_GRID = (0.05, 0.1, 0.2, 0.3)


def demand_only(y, train):
    if np.any(y[train] < 0):
        raise ValueError("needs demand that is never negative")


def croston_path(y, alpha, upto):
    """Row t's forecast from rows before t: (1 - alpha/2) times smoothed size over smoothed interval
    (Syntetos and Boylan's correction of Croston's method)."""
    out = np.zeros(upto)
    size = interval = None
    since = 1
    for t in range(upto):
        out[t] = 0.0 if size is None else (1 - alpha / 2) * size / interval
        if y[t] > 0:
            if size is None:
                size, interval = y[t], since
            else:
                size += alpha * (y[t] - size)
                interval += alpha * (since - interval)
            since = 1
        else:
            since += 1
    return out


def croston_sba(y, train, test):
    demand_only(y, train)
    end = int(train[-1]) + 1
    alpha = min(SMOOTHING_GRID, key=lambda a: np.mean((croston_path(y, a, end) - y[:end]) ** 2))
    return croston_path(y, alpha, int(test[-1]) + 1)[test]


def tsb_path(y, alpha, beta, upto, start):
    """Row t's forecast from rows before t: smoothed chance of a demand times smoothed size
    (Teunter, Syntetos and Babai), starting from the training rows' share and mean size."""
    chance, size = start
    out = np.zeros(upto)
    for t in range(upto):
        out[t] = chance * size
        if y[t] > 0:
            chance += beta * (1 - chance)
            size += alpha * (y[t] - size)
        else:
            chance -= beta * chance
    return out


def tsb(y, train, test):
    demand_only(y, train)
    seen = y[train]
    start = (float(np.mean(seen > 0)), float(seen[seen > 0].mean()) if np.any(seen > 0) else 0.0)
    end = int(train[-1]) + 1
    alpha, beta = min(((a, b) for a in SMOOTHING_GRID for b in SMOOTHING_GRID),
                      key=lambda ab: np.mean((tsb_path(y, *ab, end, start) - y[:end]) ** 2))
    return tsb_path(y, alpha, beta, int(test[-1]) + 1, start)[test]


def recent_changes(y, lags):
    """Row t's features: the last lags changes before it (row t-1 minus row t-2, and so on), missing
    where the series has not started. Changes rather than values, so trees can follow a trend."""
    change = np.r_[np.nan, np.diff(y)]
    return np.column_stack([np.r_[np.full(k, np.nan), change[:-k]] for k in range(1, lags + 1)])


def boosted_lags(y, train, test, lags):
    """Tuned boosting that predicts the next change from the recent ones, trained on the rows before the
    scored ones: its forecast is the last known value plus that change."""
    X = recent_changes(y, lags)
    change = np.r_[np.nan, np.diff(y)]
    rows = train[train >= 2]
    model = TunedHGB("regression").fit(X[rows], change[rows])
    return y[test - 1] + model.predict(X[test])

`;

const FORECAST_EVALUATE = `def series(frame):
    """The target as numbers, oldest row first, with the rows that have no target dropped."""
    NOTES.clear()
    if DATE_COLUMN and DATE_COLUMN in frame:
        text = frame[DATE_COLUMN].map(as_text)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                dates = pd.to_datetime(text, errors="coerce", format="mixed")
            except (TypeError, ValueError):   # pandas before 2.0
                dates = pd.to_datetime(text, errors="coerce")
        steps = dates.dropna().diff().dropna()
        steps = steps[steps != pd.Timedelta(0)]
        if len(steps) and (steps < pd.Timedelta(0)).mean() > 0.9:
            frame = frame.iloc[::-1]
            NOTES.append(f"The rows ran newest first by {DATE_COLUMN}, so they were turned around.")
        elif len(steps) and (steps > pd.Timedelta(0)).mean() < 0.9:
            NOTES.append(f"The rows are not in {DATE_COLUMN} order, so they were taken in file order, "
                         "which you said is the time order.")
        known = dates.dropna()
        if len(known) and known.duplicated().mean() > 0.2:
            NOTES.append(f"Many rows share a {DATE_COLUMN}. If the file holds several series (one per product "
                         "or place), these methods treat them as one; forecast each on its own.")
    y = frame[TARGET].map(as_number).astype(float).to_numpy()
    return y[~np.isnan(y)]


METRIC_FN = {"r2": (r2_score, 1), "neg_mean_absolute_error": (mean_absolute_error, -1),
             "neg_mean_absolute_percentage_error": (mean_absolute_percentage_error, -1)}


def forecast_folds(n):
    """Five time-ordered folds, keeping those with at least MIN_HISTORY earlier rows to learn from."""
    if n < MIN_HISTORY + 2:
        return []
    return [(train, test) for train, test in TimeSeriesSplit(5).split(np.zeros(n)) if len(train) >= MIN_HISTORY]


def forecast_runs(y):
    """What is scored: doing nothing, each forecasting model, and tuned boosting on the last few changes."""
    lags = max(1, min(7, len(y) // 20))
    runs = [("NAIVE-LAST", "Do nothing: the last known value", None, last_value),
            ("NAIVE-AVERAGE", "Do nothing: the average of every earlier value", None, running_average)]
    if SEASON and SEASON > 1:
        runs.append(("NAIVE-SEASON", f"Do nothing: the value {SEASON} rows earlier", None, seasonal_last))
    runs += [(code, name, needs, method) for code, (name, needs, method) in SHORTLIST.items()]
    recent = f"the last {lags} changes" if lags > 1 else "the last change"
    runs.append(("BASE-HGB-TUNED", f"Tuned boosting on {recent} (reference)", None,
                 lambda y, train, test: boosted_lags(y, train, test, lags)))
    return runs


def fold_forecasts(y, folds, method):
    """A method's forecasts for each fold's scored rows, each from the rows before it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [(test, np.asarray(method(y, train, test), dtype=float)) for train, test in folds]


def evaluate(frame, progress=print):
    """Doing nothing, each forecasting model, and tuned boosting on the last few values, each row predicted
    from the rows before it, in the same folds."""
    y = series(frame)
    folds = forecast_folds(len(y))
    measure, sign = METRIC_FN[SCORING]
    results = []
    for code, name, needs, method in forecast_runs(y):
        row = {"code": code, "name": name}
        if needs and not globals().get(needs):
            row.update(status="skipped", detail=f"{needs} is not installed")
        elif not folds:
            row.update(status="skipped", detail=f"too few rows: it needs at least {MIN_HISTORY + 2} with a {TARGET}")
        else:
            started = time.time()
            try:
                scores = [sign * measure(y[test], forecast) for test, forecast in fold_forecasts(y, folds, method)]
                row.update(status="ok", score=float(np.mean(scores)), spread=float(np.std(scores)),
                           seconds=round(time.time() - started, 1))
            except Exception as exc:  # a method that cannot handle this series is a result too
                row.update(status="error", detail=f"{type(exc).__name__}: {exc}"[:200])
        results.append(row)
        progress(row)
    return results`;

// The threshold that cost least on the user's file, when a miss is priced.
const THRESHOLD_CODE = `

def threshold_report(results, frame):
    """The threshold on the chances that cost least on your file, for the best model that gives chances.

    The chances come from rows each model did not train on, in the same folds as above.
    """
    X, y = prepare(frame)
    cv = splits(y)
    builds = {code: (scale, build) for code, (name, scale, needs, build) in SHORTLIST.items()}
    builds["BASE-HGB-TUNED"] = (False, lambda: TunedHGB(TASK))
    for row in sorted((r for r in results if r["status"] == "ok"), key=lambda r: -r["score"]):
        if row["code"] not in builds:   # doing nothing has no threshold to tune
            continue
        scale, build = builds[row["code"]]
        if not hasattr(pipeline(build(), scale), "predict_proba"):
            continue
        print(f"\\nThreshold, for {row['name']}, the best model that gives chances:")
        p, seen = np.zeros(len(y)), np.zeros(len(y), dtype=bool)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for train, test in cv.split(X, y):
                fitted = pipeline(build(), scale).fit(X.iloc[train], y[train])
                p[test], seen[test] = chances(fitted, X.iloc[test]), True
        p, truth = p[seen], y[seen]
        order = np.argsort(p)
        cases_below = np.concatenate([[0], np.cumsum(truth[order] == 1)])   # cases among the k lowest chances
        grid = np.linspace(0, 1, 1001)
        kept = np.searchsorted(p[order], grid, side="right")               # rows at or below each threshold
        missed = cases_below[kept]
        false_alarms = (len(p) - kept) - (cases_below[-1] - missed)
        costs = (MISS_COST * missed + false_alarms) / len(p)
        ties = grid[costs == costs.min()]   # of equally cheap thresholds, the one nearest THRESHOLD
        best = float(ties[np.argmin(np.abs(ties - THRESHOLD))])
        print(f"  flag a row as {POSITIVE!r} when its chance is above")
        print(f"    {best:.3f}  cost {costs.min():.4f} per row on your file, the least of any threshold")
        print(f"    {THRESHOLD:.3f}  cost {cost_per_row(truth, p > THRESHOLD):.4f}, where a miss and a false alarm cost the same")
        print(f"    0.500  cost {cost_per_row(truth, p > 0.5):.4f}, what a plain yes or no from the model does")
        print(f"  The first was picked on the rows it is scored on, so it flatters itself; {THRESHOLD:.3f} was set")
        print("  before looking. Prefer it unless the gap is large.")
        return {"code": row["code"], "best": best, "cost": float(costs.min())}
    print("\\nNo model that finished gives chances, so there is no threshold to tune.")
    return None
`;

// The shortlist codes the script can run, in the page's order, and the ones
// it cannot, with why.
export function scriptable(codes, task) {
  const key = task === 'category' ? 'cls' : task === 'forecast' ? 'fc' : 'reg';
  const run = codes.filter(c => MODEL_CODE[c]?.[key]);
  const skipped = codes.filter(c => !MODEL_CODE[c]?.[key]);
  return { run, skipped };
}

// What the take-home script can check for a reading. It cross-validates
// models that predict a target from the other columns, so it runs as it is
// for a number or a category. A future number on rows in time order is
// forecast for real: each row predicted from the rows before it, by the
// forecasting models the page showed and two ways of doing nothing
// (`forecast: true`). A future category is checked the nearest way it can
// be, predicted from the other columns and split by time, and the page says
// so. Anything else gets a reason instead of a silent gap.
export function takeHomeTask(task, sig, profile, taskLabel = task) {
  if (!sig.target) {
    return { task: null, why: 'The script checks models that predict a column, and none was picked above. Pick the column to predict, and it appears here.' };
  }
  if (task === 'category' || task === 'number') return { task, framed: false };
  if (task === 'forecast') {
    if (sig.codes[1] !== 'A22') {
      return { task: null, why: 'A future value needs rows in time order. Answer "Yes, it is a sequence" above, and the script can check it split by time.' };
    }
    const target = profile?.columns?.find(c => c.name === sig.target);
    return target && !target.numeric ? { task: 'category', framed: true } : { task: 'number', framed: true, forecast: true };
  }
  return {
    task: null,
    why: `The script checks models that predict a number or a category from the other columns, and "${taskLabel}" `
      + 'is a different kind of answer, so there is no script for it. Answer "A number" or "A category" above to get one.',
  };
}

// Which columns are which, exactly as the benchmark splits them
// (bench/dcn/run.py profile_dataset): numeric where the profiler read numbers,
// everything else categorical, the target in neither.
// `leftOut` are columns the checks found should not be features (ID-like ones).
export function columnRoles(profile, target, leftOut = []) {
  const features = profile.columns.filter(c => c.name !== target && !leftOut.includes(c.name));
  return {
    numeric: features.filter(c => c.numeric).map(c => c.name),
    categorical: features.filter(c => !c.numeric).map(c => c.name),
  };
}

export function pythonScript({ fileName, read, columns, target, task, ordered, numeric, categorical, shortlist,
  leftOut = [], cost = null, forecast = null, pageUrl = 'https://aeternifrigus.github.io/Data-Craft-Nexus/',
  date = new Date().toISOString().slice(0, 10) }) {
  // `forecast` ({ dateColumn }) forecasts a number from its own past instead
  // of predicting it from the other columns.
  const fc = !!forecast && task === 'number';
  const bench = task === 'category' ? 'classification' : 'regression';
  const { run, skipped } = scriptable(shortlist, fc ? 'forecast' : task);
  const key = task === 'category' ? 'cls' : 'reg';
  const needs = (c) => (MODEL_CODE[c].needs ? py(MODEL_CODE[c].needs) : 'None');
  const entries = fc
    ? run.map(c => `    ${py(c)}: (${py(MODEL_CODE[c].name)}, ${needs(c)}, ${MODEL_CODE[c].fc}),`).join('\n')
    : run.map(c => `    ${py(c)}: (${py(MODEL_CODE[c].name)}, ${pyBool(!!MODEL_CODE[c].scale)}, ${
      needs(c)}, lambda: ${MODEL_CODE[c][key]}),`).join('\n');
  const skippedNote = skipped.length
    ? `\n# Recommended but not runnable ${fc ? 'here' : 'on a table here'}: ${skipped.map(c =>
      (NOT_RUNNABLE[c] ? `${c} (${oneLine(NOT_RUNNABLE[c])})` : c)).join(', ')}.` : '';
  // A cost set for the other kind of answer (the answer changed after it was set) is not used.
  const priced = cost && cost.kind === task ? cost : null;
  const miss = priced?.id === 'miss';
  const leftOutNote = leftOut.length
    ? `\n# Left out of the features: ${leftOut.join(', ')}. The page found a different value in almost every row,\n# like an ID, which a model can memorise and which says nothing about new rows.` : '';

  const pricedNote = !priced || priced.bench ? ''
    : fc ? `\nIt scores by ${priced.metric}, from what you said a wrong answer costs.\n`
      : `\nIt scores by ${priced.metric}, from what you said a wrong answer costs. The
benchmark scored by ${bench === 'classification' ? 'balanced accuracy' : 'R squared'}, so the order here can differ from the
benchmark's; for your costs, this one counts.\n`;
  const doc = fc ? `Test the forecasting models Data Craft Nexus showed, on your own data.

Generated ${date} by ${pageUrl}
for ${fileName}, forecasting ${target} from its own past values.

Every method predicts each row's value from the rows before it and never
after: one step ahead, in time-ordered folds. Two ways of doing nothing are
scored the same way, carrying the last value forward and the average of
every earlier value, and tuned boosting on the last few values runs beside
them as the reference: on real business data it is often what wins. The
site's forecasting benchmark ran these same functions on real series, and
this run is the evidence on yours. The other columns are not used.
${pricedNote}
    python dcn_shortlist.py path/to/${fileName}

Needs pandas, scikit-learn and statsmodels (a missing one is reported and
skipped). Every row of the file is used; the page read at most the first 5,000.`
    : `Run the shortlist Data Craft Nexus recommended, on your own data.

Generated ${date} by ${pageUrl}
for ${fileName}, predicting ${target} (${bench}).

It uses the preprocessing, estimators and cross-validation the site's
benchmark uses, and runs tuned boosting beside the shortlist: on the
benchmark, ten configurations of boosting were worth more than the choice
among the top model families, so a recommendation should be checked against it.
Doing nothing is scored the same way (the most common answer, the average, or
the last known value), so a model that cannot beat it is plain to see.
${pricedNote}
    python dcn_shortlist.py path/to/${fileName}

Needs pandas and scikit-learn; XGBoost and LightGBM if the shortlist has them
(a missing one is reported and skipped). Every row of the file is used; the
page read at most the first 5,000.`;

  return `#!/usr/bin/env python3
"""${doc}
"""
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier, ExtraTreesRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor, HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
                              StackingClassifier, StackingRegressor, VotingClassifier, VotingRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (BayesianRidge, ElasticNet, Lasso, LinearRegression, LogisticRegression,
                                  Perceptron, Ridge)
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import (GridSearchCV, KFold, ParameterSampler, ShuffleSplit, StratifiedKFold,
                                     StratifiedShuffleSplit, TimeSeriesSplit, cross_val_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from scipy.stats import loguniform

# What the page read, so the script sees the same columns.
FILE = ${py(fileName)}
READ = {"sep": ${py(read?.sep ?? ',')}, "encoding": ${py(read?.encoding ?? 'utf-8')}}
DECIMAL_COMMA = ${pyBool(!!read?.decimalComma)}   # "1.234,56" means 1234.56, as the page read it
COLUMNS = ${py(columns)}
TARGET = ${py(target)}
TASK = ${py(bench)}
ORDERED = ${pyBool(ordered)}   # you said row order matters: split by time, never shuffle
NUMERIC = ${py(numeric)}
CATEGORICAL = ${py(categorical)}
SEED = 0${skippedNote}${leftOutNote}
FORECAST = ${pyBool(fc)}   # ${fc ? 'a future value: each row predicted from the rows before it, from its own past'
    : 'you asked for a number or a category: predicted from the other columns'}
NOTES = []   # what the run noticed about the file, printed under the scores${fc ? `
DATE_COLUMN = ${forecast.dateColumn ? py(forecast.dateColumn) : 'None'}   # checked to see the rows run oldest first
SEASON = ${forecast.season ? `${forecast.season}   # ${oneLine(forecast.seasonNote ?? 'rows in one cycle')}: smoothing may use it, and doing nothing repeats it`
    : 'None   # set it to 7, 12, ... if the values repeat every so many rows: smoothing may then use it'}
MIN_HISTORY = 10   # a fold is scored only when at least this many earlier rows are there to learn from` : ''}


def base_learners():
    if TASK == "classification":
        return [("tree", DecisionTreeClassifier(max_depth=6, random_state=SEED)), ("knn", KNeighborsClassifier()),
                ("linear", LogisticRegression(max_iter=2000))]
    return [("tree", DecisionTreeRegressor(max_depth=6, random_state=SEED)), ("knn", KNeighborsRegressor()),
            ("linear", Ridge())]


def optional(module):
    try:
        return __import__(module)
    except ImportError:
        return None


xgboost, lightgbm = optional("xgboost"), optional("lightgbm")${fc ? '\nstatsmodels = optional("statsmodels")' : ''}
${fc ? FORECAST_METHODS : ''}
${fc ? '# The forecasting models, in the order the page showed them: code -> (name, library, method).'
    : '# The shortlist, in the order the page showed it: code -> (name, scaled, library, estimator).'}
SHORTLIST = {
${entries}
}

# Tuned boosting: scikit-learn's defaults and nine random configurations, the
# best chosen by cross-validation inside each training fold (three folds, or
# one 80/20 split above 3,000 rows). The same as BASE-HGB-TUNED in the benchmark.
HGB_DEFAULT = {"learning_rate": 0.1, "max_iter": 100, "max_leaf_nodes": 31, "min_samples_leaf": 20,
               "l2_regularization": 0.0, "max_features": 1.0}
HGB_SPACE = {"learning_rate": loguniform(0.03, 0.3), "max_iter": [100, 200, 400],
             "max_leaf_nodes": [7, 15, 31, 63, 127], "min_samples_leaf": [5, 10, 20, 40, 80],
             "l2_regularization": loguniform(1e-4, 10), "max_features": [0.5, 0.75, 1.0]}


def hgb_candidates(n=10):
    sampled = ParameterSampler(HGB_SPACE, n_iter=n - 1, random_state=SEED)
    return [HGB_DEFAULT, *[{k: float(v) if isinstance(v, np.floating) else v for k, v in c.items()} for c in sampled]]


class TunedHGB(BaseEstimator):
    def __init__(self, task="classification"):
        self.task = task

    def fit(self, X, y):
        classification = self.task == "classification"
        model = (HistGradientBoostingClassifier if classification else HistGradientBoostingRegressor)(random_state=SEED)
        if len(X) <= 3000:
            inner = (StratifiedKFold if classification else KFold)(3, shuffle=True, random_state=SEED)
        else:
            inner = (StratifiedShuffleSplit if classification else ShuffleSplit)(n_splits=1, test_size=0.2,
                                                                                random_state=SEED)
        grid = [{k: [v] for k, v in c.items()} for c in hgb_candidates()]
        self.search_ = GridSearchCV(model, grid, cv=inner, scoring=SCORING, n_jobs=1, refit=True,
                                    error_score=np.nan).fit(X, y)
        if classification:
            self.classes_ = self.search_.classes_
        return self

    def predict(self, X):
        return self.search_.predict(X)

    def predict_proba(self, X):
        return self.search_.predict_proba(X)

    # A classifier to scikit-learn when it predicts a category, so a score
    # that needs chances can ask it for them.
    @property
    def _estimator_type(self):   # scikit-learn before 1.6
        return "classifier" if self.task == "classification" else "regressor"

    def __sklearn_tags__(self):   # scikit-learn 1.6 and later
        tags = super().__sklearn_tags__()
        tags.estimator_type = self._estimator_type
        return tags


${scoringCode(priced, bench)}


def pipeline(estimator, scale):
    """The benchmark's preprocessing: impute, scale where the model needs it, one-hot categories."""
    numeric_steps = [("impute", SimpleImputer(strategy="median"))] + ([("scale", StandardScaler())] if scale else [])
    transformers = [("num", Pipeline(numeric_steps), NUMERIC)]
    if CATEGORICAL:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=50)),
        ]), CATEGORICAL))
    return Pipeline([("pre", ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)),
                     ("model", estimator)])


# The words the page treats as a missing value (site/js/profile.js).
MISSING = {"", "na", "n/a", "null", "nan", "none", "-"}
DECIMAL = re.compile(r"^[+-]?(\\d+|\\d{1,3}(\\.\\d{3})+),\\d+$")   # site/js/csv.js DECIMAL_COMMA


def as_text(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value)
    return np.nan if text.strip().lower() in MISSING else text


def as_number(value):
    text = as_text(value)
    if isinstance(text, str) and DECIMAL_COMMA and DECIMAL.match(text.strip()):
        text = re.sub(r"\\.(?=\\d{3}(\\.|,))", "", text.strip()).replace(",", ".", 1)
    return pd.to_numeric(text, errors="coerce")


def prepare(frame):
    """Numeric columns as numbers (junk becomes missing), categories as text, rows with no target dropped."""
    frame = frame.copy()
    for col in NUMERIC:
        frame[col] = frame[col].map(as_number).astype(float)
    for col in CATEGORICAL:
        frame[col] = frame[col].map(as_text).astype(object)
    frame = frame[frame[TARGET].map(as_text).notna()]
    X = frame[NUMERIC + CATEGORICAL]
    if TASK == "classification":
        ${miss ? `y = (frame[TARGET].astype(str).str.strip() == POSITIVE).astype(int).to_numpy()   # 1 is a case
        if not y.any():
            raise SystemExit(f"No row of {TARGET} is {POSITIVE!r}, the case the page priced. Is this the same file?")`
    : 'y = LabelEncoder().fit_transform(frame[TARGET].astype(str))'}
    else:
        y = frame[TARGET].map(as_number).astype(float)
        X, y = X[y.notna()], y[y.notna()].to_numpy()
    return X, y


def splits(y):
    if ORDERED:
        return TimeSeriesSplit(5)
    if TASK == "classification":
        folds = int(min(5, np.bincount(y).min()))
        return StratifiedKFold(max(folds, 2), shuffle=True, random_state=SEED)
    return KFold(5, shuffle=True, random_state=SEED)


${fc ? FORECAST_EVALUATE : `class LastValue(BaseEstimator):
    """Doing nothing on rows in time order: every row gets the last value seen in training."""

    def __init__(self, task="classification"):
        self.task = task

    def fit(self, X, y):
        y = np.asarray(y)
        self.last_ = y[-1]
        if self.task == "classification":
            self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.last_)

    def predict_proba(self, X):
        return np.tile((self.classes_ == self.last_).astype(float), (len(X), 1))

    @property
    def _estimator_type(self):   # scikit-learn before 1.6
        return "classifier" if self.task == "classification" else "regressor"

    def __sklearn_tags__(self):   # scikit-learn 1.6 and later
        tags = super().__sklearn_tags__()
        tags.estimator_type = self._estimator_type
        return tags


def baselines():
    """Doing nothing, scored the same way as every model: the bar a model has to clear."""
    if TASK == "classification":
        runs = [("NAIVE-COMMON", "Do nothing: the most common answer", lambda: DummyClassifier(strategy="prior"))]
    elif SCORING == "r2":
        runs = [("NAIVE-AVERAGE", "Do nothing: the average", lambda: DummyRegressor(strategy="mean"))]
    else:   # the constant that misses by least, when every unit or share of a miss counts the same
        runs = [("NAIVE-AVERAGE", "Do nothing: the median", lambda: DummyRegressor(strategy="median"))]
    if ORDERED:
        runs.append(("NAIVE-LAST", "Do nothing: the last known value", lambda: LastValue(TASK)))
    return runs


def evaluate(frame, progress=print):
    """Doing nothing, every model on the shortlist, and tuned boosting, cross-validated the same way."""
    X, y = prepare(frame)
    cv = splits(y)
    runs = [(code, name, False, None, build) for code, name, build in baselines()]
    runs += [(code, name, scale, needs, build) for code, (name, scale, needs, build) in SHORTLIST.items()]
    runs.append(("BASE-HGB-TUNED", "Histogram Gradient Boosting, tuned (reference)", False, None,
                 lambda: TunedHGB(TASK)))
    results = []
    for code, name, scale, needs, build in runs:
        row = {"code": code, "name": name}
        if needs and not globals().get(needs):
            row.update(status="skipped", detail=f"{needs} is not installed")
        else:
            started = time.time()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scores = cross_val_score(pipeline(build(), scale), X, y, cv=cv, scoring=SCORING,
                                             error_score="raise")
                row.update(status="ok", score=float(np.mean(scores)), spread=float(np.std(scores)),
                           seconds=round(time.time() - started, 1))
            except Exception as exc:  # a model that cannot handle this data is a result too
                detail = ("gives no chances, which this score needs" if "predict_proba" in str(exc)
                          else f"{type(exc).__name__}: {exc}")
                row.update(status="error", detail=detail[:200])
        results.append(row)
        progress(row)
    return results`}


def verdict(results):
    """Whether any model beat doing nothing, compared at the four decimals shown."""
    ok = [r for r in results if r["status"] == "ok"]
    bars = [r for r in ok if r["code"].startswith("NAIVE-")]
    models = [r for r in ok if not r["code"].startswith("NAIVE-")]
    if not bars or not models:
        return None
    bar = max(bars, key=lambda r: r["score"])
    best = max(models, key=lambda r: r["score"])
    plain = bar["name"].replace("Do nothing: ", "")
    if round(best["score"], 4) <= round(bar["score"], 4):
        return (f"No model beat doing nothing ({plain}, {bar['score']:.4f}): on this file the models found "
                "nothing it does not already know.")
    return (f"The best model, {best['name']}, beat doing nothing ({plain}, {bar['score']:.4f}) by "
            f"{best['score'] - bar['score']:.4f}. Compare that with the spread beside each score.")


def report(results):
    how = ("one step ahead in time-ordered folds" if FORECAST
           else f"5-fold {'time-ordered ' if ORDERED else ''}cross-validation")
    print(f"\\n{len(results)} {'methods' if FORECAST else 'models'}, {how}, {METRIC}:\\n")
    for row in sorted(results, key=lambda r: -r.get("score", -np.inf)):
        if row["status"] == "ok":
            print(f"  {row['score']:.4f} ± {row['spread']:.4f}  {row['code']:15} {row['name']}  ({row['seconds']}s)")
        else:
            print(f"  {row['status']:>15}  {row['code']:15} {row['name']}: {row.get('detail', '')}")
    said = verdict(results)
    if said:
        print(f"\\n{said}")
    for note in NOTES:
        print(f"\\nNote: {note}")
${miss ? THRESHOLD_CODE : ''}

def load(source):
    """The file as the page read it: every cell as text, under the page's column names.

    \`source\` is a path, or an open text stream, which needs no encoding: the
    page hands Python running in the browser its own decoded copy that way.
    """
    options = dict(READ)
    if not isinstance(source, str):
        options.pop("encoding", None)
    frame = pd.read_csv(source, **options, dtype=str, keep_default_na=False)
    if len(frame.columns) == len(COLUMNS):
        frame.columns = COLUMNS   # the names the page gave them (blank or repeated headers renamed)
    return frame


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else FILE
    frame = load(path)
    results = evaluate(frame, progress=lambda row: print(".", end="", flush=True))
    report(results)${miss ? '\n    threshold_report(results, frame)' : ''}


if __name__ == "__main__":
    main()
`;
}

// The file the browser saves.
export function downloadScript(text, name = 'dcn_shortlist.py') {
  const url = URL.createObjectURL(new Blob([text], { type: 'text/x-python' }));
  const a = Object.assign(document.createElement('a'), { href: url, download: name });
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
