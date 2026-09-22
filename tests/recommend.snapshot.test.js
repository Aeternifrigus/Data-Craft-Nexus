// Records what the instrument recommends for every fixture, target, task and
// order answer. Any change to the profiler or the rankings shows up as a diff
// in tests/snapshots/recommendations.json.
//
// After an intended change, regenerate with:  npm run test:update
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { parseCSV } from '../site/js/csv.js';
import { profileData, signature } from '../site/js/profile.js';
import { rankModels, rankDrifts, rankPipelines } from '../site/js/recommend.js';
import { loadTaxonomyFromDisk, readFixture, fixtureNames } from './helpers.js';

const T = loadTaxonomyFromDisk();
const SNAPSHOT = new URL('./snapshots/recommendations.json', import.meta.url);

function buildSnapshot() {
  const out = {};
  for (const file of fixtureNames()) {
    const { head, body } = parseCSV(readFixture(file));
    const profile = profileData(head, body);
    const entry = { rows: profile.n, columns: profile.feat, runs: {} };
    for (const target of [...head, '__none__']) {
      for (const task of T.TASKS.map(t => t.id)) {
        for (const order of ['A21', 'A22']) {
          const sig = signature(profile, { target, task, order });
          const models = rankModels(T, sig, task);
          entry.runs[`${target} | ${task} | ${order}`] = {
            signature: sig.codes.join(' ') + (sig.flags.length ? ` (+${sig.flags.join(' ')})` : ''),
            models: models.items.map(m => `${m.c} (${m.score}/${m.of})`).join(', ') || 'none',
            tied: `${models.tied} of ${models.candidates}`,
            ruledOut: models.ruledOut.length,
            drift: rankDrifts(T, sig).items.map(d => d.c).join(', '),
            pipelines: rankPipelines(T, sig, task).items.map(p => p.c).join(', '),
          };
        }
      }
    }
    out[file] = entry;
  }
  return out;
}

test('recommendations match the recorded snapshot', () => {
  const current = buildSnapshot();
  if (process.env.UPDATE_SNAPSHOTS || !fs.existsSync(SNAPSHOT)) {
    fs.writeFileSync(SNAPSHOT, JSON.stringify(current, null, 1) + '\n');
    return;
  }
  const recorded = JSON.parse(fs.readFileSync(SNAPSHOT, 'utf8'));
  assert.deepEqual(current, recorded);
});
