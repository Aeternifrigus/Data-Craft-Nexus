// "Take it home": the script's inputs. bench/tests/test_export.py runs the
// generated script with Python and holds it to the benchmark.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { parseCSV } from '../site/js/csv.js';
import { profileData } from '../site/js/profile.js';
import { MODEL_CODE, columnRoles, pythonScript, scriptable } from '../site/js/export.js';
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
