// The Data Craft Nexus MCP server: the page's checks and signature, as tools
// a coding agent can call on a CSV before it trains anything.
//
// Everything is read-only and local. The server reads the file it is pointed
// at and nothing else, sends nothing anywhere, and writes nothing.

import fs from 'node:fs';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';
import { checkDataset, InputError, profileDataset, TASKS } from './tools.mjs';

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
  const server = new McpServer({ name, version }, {
    instructions: 'Data Craft Nexus reads a tabular dataset the way its web page does '
      + '(https://aeternifrigus.github.io/Data-Craft-Nexus/). Before training a model on a CSV, call check_dataset: '
      + 'it finds what makes a validation score look better than it will be on new data (a column that leaks the '
      + 'target, ID columns, repeated rows, dates split at random). profile_dataset describes the file. '
      + 'Only the first 5,000 rows are read, as on the page.',
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

  return server;
}
