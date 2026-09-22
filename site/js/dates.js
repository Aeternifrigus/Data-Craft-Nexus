// Date detection.
//
// This used to call Date.parse, which is only specified for ISO strings:
// everything else is up to the engine, so Chrome and Safari disagree and the
// same file could be measured differently in different browsers. These
// patterns are explicit instead, and bench/dcn/dates.py holds the same list so
// the benchmark measures what the page measures.

const MONTHS = 'january|february|march|april|august|september|october|november|december|june|july|' +
  'jan|feb|mar|apr|may|jun|jul|aug|sept|sep|oct|nov|dec';

export const DATE_PATTERNS = [
  // 2025, 2025-01, 2025-01-04, 2025-01-04T10:30:00.500Z, 2025-01-04 10:30
  /^\d{4}(-\d{1,2}(-\d{1,2})?)?([T ]\d{1,2}:\d{2}(:\d{2})?(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$/,
  // 2025/01/04, 01/04/2025
  /^\d{1,4}\/\d{1,2}\/\d{1,4}([T ]\d{1,2}:\d{2}(:\d{2})?)?$/,
  // 04.01.2025
  /^\d{1,2}\.\d{1,2}\.\d{2,4}$/,
  // Jan 4, 2025 / January 4 2025 / Jan 4
  new RegExp(`^(${MONTHS})\\.?\\s+\\d{1,2}(,)?(\\s*\\d{4})?$`, 'i'),
  // Jan 2025
  new RegExp(`^(${MONTHS})\\.?\\s+\\d{4}$`, 'i'),
  // 4 Jan 2025 / 04-Jan-25
  new RegExp(`^\\d{1,2}[\\s-](${MONTHS})\\.?[\\s-]\\d{2,4}$`, 'i'),
  // Sat, 04 Jan 2025 10:00:00 GMT
  new RegExp(`^[a-z]{3},?\\s+\\d{1,2}\\s+(${MONTHS})\\s+\\d{4}\\b.*$`, 'i'),
];

export function isDateLike(value) {
  const s = String(value).trim();
  if (!s) return false;
  return DATE_PATTERNS.some(p => p.test(s));
}

export function countDateLike(values) {
  return values.reduce((n, v) => n + (isDateLike(v) ? 1 : 0), 0);
}
