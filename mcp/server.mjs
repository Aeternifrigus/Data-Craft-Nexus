// The Data Craft Nexus MCP server: the page's checks, signature, model
// shortlist and take-home script, as tools a coding agent can call on a CSV
// before it trains anything.
//
// Everything is read-only and local. The server reads the file it is pointed
// at and nothing else, sends nothing anywhere, and writes nothing.

import fs from 'node:fs';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';
import {
  checkDataset, InputError, LABELS_ARRIVE, loadTaxonomy, profileDataset, recommendModels, RUNS_AS, STAGES, takeHome, TASKS,
} from './tools.mjs';

const { name, version } = JSON.parse(fs.readFileSync(new URL('../package.json', import.meta.url), 'utf8'));

const file = {
  path: z.string().min(1).optional()
    .describe('Absolute path to a CSV, TSV or semicolon-separated file on this machine. Pass this or csv.'),
  csv: z.string().min(1).optional()
    .describe('The file\'s contents, when there is no path to give. Pass this or path.'),
};
const target = z.string().min(1)
  .describe('The column the model will predict, exactly as named in the header.');
const task = z.enum(TASKS).optional()
  .describe('The kind of answer: category (classification), number (regression), forecast (a future value), '
    + 'or survival, group, anomaly, compress, generate. Inferred from the target when left out.');
const ordered = z.boolean().optional()
  .describe('true when rows follow each other in time (a log, a series) and the model will be used on later rows, '
    + 'so the split must follow time. false (the default) when rows are independent.');

const readOnly = { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false };

// A tool's answer: the words for the agent, and the same content as data.
function reply(fn) {
  return async (args) => {
    try {
      const { text, data } = fn(args);
      return { content: [{ type: 'text', text }], structuredContent: data };
    } catch (err) {
      if (err instanceof InputError) return { content: [{ type: 'text', text: err.message }], isError: true };
      throw err;
    }
  };
}

export function createServer() {
  // The benchmark's sizes, from the committed evidence, so the descriptions
  // cannot drift from what the page says.
  const T = loadTaxonomy();
  const tables = T.EVIDENCE?.datasets?.length ?? 0;
  const series = T.FORECAST?.series ?? 0;
  const anomalyTables = T.ANOMALY?.datasets ?? 0;

  const server = new McpServer({ name, version }, {
    instructions: 'Data Craft Nexus reads a tabular dataset the way its web page does '
      + '(https://aeternifrigus.github.io/Data-Craft-Nexus/). Before training a model on a CSV, call check_dataset: '
      + 'it finds what makes a validation score look better than it will be on new data (a column that leaks the '
      + 'target, ID columns, repeated rows, dates split at random). Then recommend_models for a shortlist backed by '
      + `a ${tables}-dataset benchmark, and take_home_script for a script that scores that shortlist against a do-nothing `
      + 'baseline. Only the first 5,000 rows are read for the checks; the script reads the whole file.',
  });

  server.registerTool('check_dataset', {
    title: 'Check a dataset before trusting any score',
    description: 'Call this before training or evaluating a model on a CSV. Flags what makes any model\'s score look '
      + 'better than it will be on new data, which no choice of model fixes: a column that predicts the target almost '
      + 'perfectly on its own (usually recorded after the outcome), ID-like columns a model can memorise, more repeated '
      + 'rows than chance (copies on both sides of a random split), and date columns when rows are split at random. '
      + 'Each flag says what to do. How often each check fires and what it catches were measured on a benchmark, and '
      + 'that is included.',
    inputSchema: { ...file, target, task, rows_in_time_order: ordered },
    annotations: readOnly,
  }, reply(checkDataset));

  server.registerTool('profile_dataset', {
    title: 'Read a dataset\'s signature',
    description: 'Describes a CSV: its six-axis signature (labelled or not, row order, modality, scale, distribution '
      + 'and quality), class balance or drift between the first and second half of the file, and every column\'s '
      + 'kind, distinct values, missing share and share of odd values (text in a number column, labels differing only '
      + 'in case or spacing).',
    inputSchema: { ...file, target: target.optional(), rows_in_time_order: ordered },
    annotations: readOnly,
  }, reply(profileDataset));

  server.registerTool('recommend_models', {
    title: 'Shortlist models for a dataset',
    description: 'Ranks the models that can give the asked-for kind of answer on this data, by what each was worth '
      + `on a benchmark of ${tables} datasets (${series} series for forecasts, ${anomalyTables} tables for anomalies), with the measured `
      + 'record, cautions and failure modes of each. Says which models the data rules out and why, which drift '
      + 'checkers suit how the model will run, and which pipelines fit the part of the work being built. Also repeats '
      + 'any check_dataset flags, since they matter more than the choice of model.',
    inputSchema: {
      ...file, target: target.optional(), task, rows_in_time_order: ordered,
      runs_as: z.enum(RUNS_AS).optional().describe('How the model will run: batch (default) or streaming, row by row.'),
      labels_arrive: z.enum(LABELS_ARRIVE).optional()
        .describe('When the true answers arrive after a prediction: immediate, delayed (default) or none.'),
      stage: z.enum(STAGES).optional()
        .describe('Which part of the work is being built, for the pipelines: data, train, ship or llm. Default all.'),
    },
    annotations: readOnly,
  }, reply(recommendModels));

  server.registerTool('take_home_script', {
    title: 'Write a script that tests the shortlist',
    description: 'Returns a Python script (pandas and scikit-learn) that runs tuned gradient boosting and the '
      + 'recommended models on the whole file with the benchmark\'s preprocessing and cross-validation, next to a '
      + 'baseline that does nothing, so a model that cannot beat a guess is plain to see. ID-like columns are left out; '
      + 'rows in time order are split by time. Save it and run it with the file\'s path as the argument.',
    inputSchema: { ...file, target, task, rows_in_time_order: ordered },
    annotations: readOnly,
  }, reply(takeHome));

  return server;
}
