// What a reading finds, before anything is drawn: the checks, where the file
// sits among the benchmark datasets, the models, drift checkers and pipelines
// in order, and what the take-home script can run.
//
// The page draws this (results.js) and the MCP server (mcp/) hands it to a
// coding agent, so a file read either way gets the same answer.
//
// Pure functions: no DOM access.

import { runChecks } from './checks.js';
import { resolveCost } from './costs.js';
import { columnRoles, pythonScript, scriptable, takeHomeTask } from './export.js';
import { describeSeries, forecastMetric } from './forecasting.js';
import { metaFeatures, nearestDatasets } from './nearest.js';
import { leadRecommendation, rankDrifts, rankModels, rankPipelines } from './recommend.js';
import { forecastSetup } from './series.js';

// `sig` is the signature (profile.js), `task` the kind of answer asked for,
// `answer` what a wrong answer costs (costs.js) or null for the benchmark's
// own score. For a future number on rows in time order this sets sig.series,
// which the order of the forecasters depends on.
export function analyse(T, sig, task, profile, answer = null) {
  // What can make any score look better than it is, before any score.
  const checks = profile ? runChecks(profile, { target: sig.target, task, order: sig.codes[1] }) : null;
  const leftOut = checks ? checks.flags.filter(f => f.kind === 'id').map(f => f.column) : [];

  // Where this dataset sits among the benchmark datasets, measured the same
  // way. The order can depend on it, so it is measured first.
  const meta = profile ? metaFeatures(profile, sig.target ?? null) : null;
  if (meta && sig.drift) meta.drift_psi = Math.min(sig.drift.psi, 5);
  // A future number: measure the series the way the forecasting benchmark
  // did, since its kind (intermittent, seasonal, trending...) can decide the order.
  const targetColumn = profile?.columns.find(c => c.name === sig.target);
  if (task === 'forecast' && targetColumn?.numeric && sig.codes[1] === 'A22') {
    sig.series = describeSeries(forecastSetup(profile, sig.target),
      forecastMetric(resolveCost('number', answer, targetColumn)));
  }

  const models = rankModels(T, sig, task, 4, meta);
  const neighbours = meta ? nearestDatasets(T, meta, task) : [];
  const lead = leadRecommendation(T, sig, task);
  const drifts = rankDrifts(T, sig);
  const pipelines = rankPipelines(T, sig, task);

  // The script's task can differ from the answer: a future value is checked as a number, split by time.
  const home = takeHomeTask(task, sig, profile, T.TASKS.find(t => t.id === task)?.label ?? task);
  const cost = home.task && profile
    ? resolveCost(home.task, answer, profile.columns.find(c => c.name === sig.target)) : null;
  // A future number is forecast with the models on the cards; a future
  // category is checked with the category shortlist.
  const homeModels = home.task && home.task !== task && !home.forecast ? rankModels(T, sig, home.task, 4, meta) : models;
  const homeLead = home.task && !home.forecast ? leadRecommendation(T, sig, home.task) : null;

  return {
    checks, leftOut, meta, models, neighbours, lead, drifts, pipelines, home, cost,
    takeHome: { codes: homeModels.items.map(m => m.c), lead: homeLead },
  };
}

// Which of the take-home shortlist the script can run, which it cannot, and
// whether there is a script at all. A forecast still has something to run
// when no card can: doing nothing, and tuned boosting on recent changes.
export function scriptPlan(home, codes) {
  const { run, skipped } = home.task ? scriptable(codes, home.forecast ? 'forecast' : home.task) : { run: [], skipped: [] };
  return { run, skipped, runnable: Boolean(home.task) && (run.length > 0 || Boolean(home.forecast)) };
}

// The take-home script's text. `source` is how the file was read
// ({ fileName, read, columns }), so the script reads it the same way.
export function takeHomeScript({ sig, home, profile, source = null, codes, leftOut = [], cost = null }) {
  return pythonScript({
    fileName: source?.fileName ?? 'data.csv', read: source?.read, columns: source?.columns ?? profile.columns.map(c => c.name),
    target: sig.target, task: home.task, ordered: sig.codes[1] === 'A22', ...columnRoles(profile, sig.target, leftOut),
    shortlist: codes, leftOut, cost,
    forecast: home.forecast ? forecastSetup(profile, sig.target) : null,
  });
}
