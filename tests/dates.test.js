import { test } from 'node:test';
import assert from 'node:assert/strict';
import { isDateLike } from '../site/js/dates.js';

const DATES = [
  '2025', '2025-01', '2025-01-04', '2025-01-04T10:30:00Z', '2025-01-04 10:30',
  '2025-01-04T10:30:00.500+02:00', '2025/01/04', '01/04/2025', '04.01.2025',
  'Jan 4, 2025', 'January 4 2025', 'Jan 4', 'Jan 2025', '4 Jan 2025', '04-Jan-25',
  'Sat, 04 Jan 2025 10:00:00 GMT', 'JAN 2025',
];

const NOT_DATES = [
  '', '   ', '40ft', 'Gdynia', '12abc', 'basmati_rice', 'yes', 'n/a',
  '12,50', '1e5', 'Janitor 4', '2025-13-45-99-11', 'x2025-01-04',
];

test('the date shapes a CSV actually contains are recognised', () => {
  for (const v of DATES) assert.equal(isDateLike(v), true, v);
});

test('ordinary values are not mistaken for dates', () => {
  for (const v of NOT_DATES) assert.equal(isDateLike(v), false, v);
});

test('detection does not depend on the browser', () => {
  // Date.parse is engine-specific outside ISO strings: Chrome accepts
  // "04.01.2025" and Safari does not. These patterns are fixed, so a file is
  // measured the same everywhere.
  assert.equal(isDateLike('04.01.2025'), true);
  assert.equal(isDateLike('4-Jan-25'), true);
});
