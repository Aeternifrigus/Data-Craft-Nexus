// "Take it home": the script's inputs. bench/tests/test_export.py runs the
// generated script with Python and holds it to the benchmark.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { parseCSV } from '../site/js/csv.js';
import { profileData } from '../site/js/profile.js';
import { MODEL_CODE, NOT_RUNNABLE, columnRoles, pythonScript, scriptable, takeHomeTask } from '../site/js/export.js';
import { loadTaxonomyFromDisk } from './helpers.js';

const T = loadTaxonomyFromDisk();
const ROOT = new URL('../', import.meta.url);

test('every runnable benchmark model has a Python estimator for each task it runs', () => {
  const ran = new Set(T.EVIDENCE.runnable);
  for (const code of ran) {
    assert.ok(MODEL_CODE[code], `${code} ran in the benchmark but the script cannot run it`);
    const tasks = Object.keys(T.EVIDENCE.models[code]?.tasks ?? {});
    for (const task of tasks) {
      assert.ok(MODEL_CODE[code][task === 'classification' ? 'cls' : 'reg'], `${code} has no ${task} estimator`);
    }
  }
});

test('columns are split the way the benchmark splits them, with the target in neither', () => {
  const { head, body } = parseCSV(fs.readFileSync(new URL('site/sample.csv', ROOT), 'utf8'));
  const roles = columnRoles(profileData(head, body), 'delayed');
  assert.ok(!roles.numeric.includes('delayed') && !roles.categorical.includes('delayed'));
  assert.equal(roles.numeric.length + roles.categorical.length, head.length - 1);
  assert.ok(roles.numeric.includes('transit_days'));
  assert.ok(roles.categorical.includes('origin_port'));
});

test('the script carries what the page read, and names what it cannot run', () => {
  const script = pythonScript({
    fileName: 'shipments;2026.csv', read: { sep: ';', encoding: 'cp1250', decimalComma: true },
    columns: ['a', 'b', 'target"quoted'], target: 'target"quoted', task: 'category', ordered: true,
    numeric: ['a'], categorical: ['b'], shortlist: ['TR2', 'NN9', 'EN4'], date: '2026-01-01',
  });
  assert.match(script, /READ = \{"sep": ";", "encoding": "cp1250"\}/);
  assert.match(script, /DECIMAL_COMMA = True/);
  assert.match(script, /ORDERED = True/);
  assert.match(script, /TARGET = "target\\"quoted"/, 'names are escaped as Python strings');
  assert.match(script, /Recommended but not runnable on a table here: NN9\./);
  assert.match(script, /"EN4": \("LightGBM", False, "lightgbm", lambda: lightgbm\.LGBMClassifier/);
  assert.deepEqual(scriptable(['TR2', 'NN9', 'LM1'], 'category'), { run: ['TR2'], skipped: ['NN9', 'LM1'] });
});

test('the script is offered for every answer it can check, and a reason is given for the rest', () => {
  const sig = (order, target = 'y') => ({ codes: ['A11', order, 'A38', 'A41', 'A51', 'A61'], flags: [], target });
  const profile = { columns: [{ name: 'y', numeric: true }, { name: 'kind', numeric: false }] };
  assert.deepEqual(takeHomeTask('number', sig('A21'), profile), { task: 'number', framed: false });
  assert.deepEqual(takeHomeTask('category', sig('A22'), profile), { task: 'category', framed: false });
  // A future number on rows in time order is forecast from its own past; a future category is
  // predicted from the other columns, split by time.
  assert.deepEqual(takeHomeTask('forecast', sig('A22'), profile), { task: 'number', framed: true, forecast: true });
  assert.deepEqual(takeHomeTask('forecast', sig('A22', 'kind'), profile), { task: 'category', framed: true });
  assert.match(takeHomeTask('forecast', sig('A21'), profile).why, /time order/);
  assert.match(takeHomeTask('group', sig('A21'), profile, 'Natural groupings').why, /"Natural groupings" is a different kind of answer/);
  assert.match(takeHomeTask('number', sig('A21', null), profile).why, /Pick the column to predict/);
});

test('a future number is forecast with the models on the cards, and the one that cannot run is named', () => {
  const forecasting = T.MODELS.filter(m => m.c.startsWith('TSM')).map(m => m.c);
  const { run, skipped } = scriptable(forecasting, 'forecast');
  assert.deepEqual(run, ['TSM1', 'TSM2', 'TSM4', 'TSM5']);
  assert.deepEqual(skipped, ['TSM3']);
  assert.match(NOT_RUNNABLE.TSM3, /Prophet needs Stan/);
  const base = {
    fileName: 'fx.csv', columns: ['day', 'rate'], target: 'rate', task: 'number', ordered: true,
    numeric: [], categorical: ['day'], shortlist: forecasting, date: '2026-01-01',
  };
  const text = pythonScript({ ...base, forecast: { dateColumn: 'day' } });
  assert.match(text, /^FORECAST = True/m);
  assert.match(text, /^DATE_COLUMN = "day"/m);
  assert.match(text, /^SEASON = None /m, 'no season unless the dates give one');
  assert.match(text, /"TSM4": \("Croston's method \(SBA\)", None, croston_sba\),/);
  const seasonal = pythonScript({ ...base, forecast: { dateColumn: 'day', season: 12, seasonNote: 'the dates step by a month, so a year is 12 rows' } });
  assert.match(seasonal, /^SEASON = 12   # the dates step by a month, so a year is 12 rows/m);
  assert.match(text, /"TSM1": \("ARIMA", "statsmodels", arima\),/);
  assert.match(text, /Recommended but not runnable here: TSM3 \(Prophet needs Stan/);
  assert.match(text, /^statsmodels = optional\("statsmodels"\)$/m);
  // Without the forecast flag the same answer is the old check: predicted from the other columns.
  const plain = pythonScript({ ...base, shortlist: ['TR2'] });
  assert.match(plain, /^FORECAST = False/m);
  assert.doesNotMatch(plain, /statsmodels|def arima/);
  assert.match(plain, /^def baselines\(\):/m);
});
