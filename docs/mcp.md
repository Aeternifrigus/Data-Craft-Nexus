# For coding agents

[README](../README.md) · [Docs](README.md) · [How it works](how-it-works.md) · [The evidence](evidence.md) · [Development](development.md) · **For coding agents**

Data Craft Nexus also runs as an [MCP](https://modelcontextprotocol.io) server, so a coding agent (Claude Code, Claude Desktop, Cursor, or anything else that speaks the Model Context Protocol) can read a CSV the way the page does before it trains anything. It is the page's own code reading the file from disk instead of from an upload: the same checks, the same order of models, the same take-home script.

## Contents

- [Why an agent needs it](#why-an-agent-needs-it)
- [Set it up](#set-it-up)
- [The tools](#the-tools)
- [What it reads, and what it does not](#what-it-reads-and-what-it-does-not)
- [Working on it](#working-on-it)
- [Publishing it](#publishing-it)

## Why an agent needs it

Ask an agent to "train the best model on this file" and it will usually split the rows at random, fit something, and report the score. If the file has one of the problems the page checks for, that score is wrong before any model is chosen, and no choice of model fixes it.

An example, on a generated table of 1,500 shipments made to show the effect: a shipment ID, a booking date, four honest features, whether the shipment was late, and `delay_compensation_eur`, which is only filled in after a shipment turns out late. The take-home script, split by time:

| | best balanced accuracy, 5 time-ordered folds |
|---|---|
| with `delay_compensation_eur` | 1.0000 (LightGBM) |
| without it | 0.6205 (tuned boosting) |
| doing nothing | 0.5000 |

`check_dataset` finds all three problems in one call, before anything is trained:

```
3 problems that would make any model's score look better than it will be on new data. No choice of
model fixes these; settle them before trusting a score.

1. [id] shipment_id looks like an ID
2. [leak] delay_compensation_eur predicts late almost perfectly on its own
   Used alone, on rows it was not fitted on, it reaches balanced accuracy 0.994. ...
   Fix: Check when delay_compensation_eur becomes known. If it is only known after late is, leave it out.
3. [time] booked_on holds dates, and the rows were declared independent
```

That table was built to have the problems, so it shows what the checks do, not how often agents make these mistakes. Measuring that, the same tasks run by an agent with and without this server, is the next step, and nothing here claims it yet.

<details>
<summary>The table</summary>

```python
import numpy as np, pandas as pd

rng = np.random.default_rng(3); n = 1500
df = pd.DataFrame({
    'shipment_id': [f'SHP-{i:05d}' for i in rng.permutation(n)],
    'booked_on': pd.date_range('2023-01-01', periods=n, freq='D').strftime('%Y-%m-%d'),
    'weight_kg': rng.gamma(2, 400, n).round(1),
    'distance_km': rng.integers(300, 12000, n),
    'carrier': rng.choice(['Maersk', 'MSC', 'CMA CGM', 'Hapag'], n),
    'priority': rng.choice(['low', 'high'], n, p=[.7, .3]),
})
logit = -1.2 + df.distance_km / 6000 + (df.carrier == 'MSC') * 0.8 - (df.priority == 'high') * 0.7
df['late'] = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), 'yes', 'no')
df['delay_compensation_eur'] = np.where(df.late == 'yes', rng.gamma(3, 120, n).round(2), 0.0)
df.to_csv('shipments.csv', index=False)
```

</details>

## Set it up

It needs Node 20 or later.

```bash
git clone https://github.com/Aeternifrigus/Data-Craft-Nexus.git
cd Data-Craft-Nexus
npm install
```

**Claude Code**

```bash
claude mcp add data-craft-nexus -- node /absolute/path/to/Data-Craft-Nexus/mcp/dcn-mcp.mjs
```

**Claude Desktop, Cursor and others** take the same entry in their MCP settings file (`claude_desktop_config.json` for Claude Desktop, `.cursor/mcp.json` for Cursor):

```json
{
  "mcpServers": {
    "data-craft-nexus": {
      "command": "node",
      "args": ["/absolute/path/to/Data-Craft-Nexus/mcp/dcn-mcp.mjs"]
    }
  }
}
```

Once it is on npm, `npx -y data-craft-nexus` replaces `node` and the path, and no clone is needed.

## The tools

| tool | what it answers |
|---|---|
| `check_dataset` | "Before you trust a score": a column that predicts the target almost perfectly on its own, ID-like columns, more repeated rows than chance, dates split at random. Each flag says what to do, and how often the check fired and what it caught on the benchmark. |
| `profile_dataset` | The six-axis signature, class balance or drift between the two halves of the file, and every column's kind, distinct values, missing share and odd values. |
| `recommend_models` | Tuned boosting first when it earned that place, then the shortlist in the page's order, each model with its benchmark record, cautions and failure modes; what the data rules out and why; the drift checkers that fit how the model will run. Any check flags come first. |
| `take_home_script` | The page's Python script: tuned boosting and the shortlist on the whole file with the benchmark's cross-validation, next to a baseline that does nothing. ID columns are left out, and rows in time order are split by time. |

Every tool takes the file as `path` (an absolute path) or `csv` (its contents), and the page's questions as arguments: `target`, `task` (`category`, `number`, `forecast`, and the page's other kinds of answer), and `rows_in_time_order`. `recommend_models` also takes `runs_as`, `labels_arrive` and `stage`, which decide the drift checkers and pipelines as they do on the page. When `task` is left out it is inferred from the target, and the reply says how, so it can be corrected.

Each reply is written for the agent to read, and carries the same content as structured data. A mistake in the request (no such file, no such column) comes back as a tool error that says what to fix, with the column names when a target is not found.

## What it reads, and what it does not

- **The same rows as the page.** The checks, the signature and the order use the first 5,000 rows, as the page does, and the reply says when a file was longer. The take-home script reads every row.
- **The same reading.** UTF-8, else Windows-1250 or -1252; comma, semicolon, tab or pipe; decimal commas read as numbers. The script is told how the file was read, so it reads it the same way.
- **Read-only and local.** Apart from its own files, the server reads only the file it is pointed at. It makes no network requests, sends no data anywhere, and writes nothing; every tool is marked read-only.
- **Not yet:** "What does a wrong answer cost?" is not a tool argument, so the script scores by the benchmark's own measure (balanced accuracy for a category, R² for a number).

## Working on it

```bash
npm run mcp                                            # the server on stdio
npx @modelcontextprotocol/inspector node mcp/dcn-mcp.mjs   # a browser UI to call the tools by hand
```

The server decides nothing itself. `site/js/analysis.js` works out a reading (the checks, the order, what the script can run) for both the page and the server, and `mcp/tools.mjs` only reads the file and says the result in words. A change to what the page recommends reaches the server with no change in `mcp/`.

`tests/mcp.test.js` plants each problem in a generated table and checks it is found and nothing else is; holds `profile_dataset`, `recommend_models` and `take_home_script` to the page's own analysis of the same file (the same signature, order, lead, drift checkers and script); checks that bad requests come back as tool errors; and runs the protocol both in memory and over stdio.

## Publishing it

The package is set up for npm (a `data-craft-nexus` bin, and a `files` list of the server, `site/js` and `site/taxonomy`, about 210 kB packed) and for the [MCP Registry](https://modelcontextprotocol.io/registry/about) (`mcpName` in `package.json`, and `server.json`). Nothing has been published yet. To publish a version:

```bash
npm publish                     # needs an npm account
mcp-publisher login github      # the CLI is on the registry's GitHub releases page
mcp-publisher publish
```

The version in `package.json` and both versions in `server.json` must match, and a test checks they do. The registry's name, `io.github.Aeternifrigus/data-craft-nexus`, has to match the GitHub account's name including its capital letter, since the registry compares them exactly.
