// Reads cases as JSON on stdin and prints what the site's series.js measures
// for each, as JSON. bench/tests/test_series.py holds the Python port to it.
//   [{"values": [...], "period": 12}, {"dates": ["2025-01-01", ...]}]
import { dayNumber, periodFromDays, seriesFeatures, seriesCell } from '../../site/js/series.js';

let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => { input += chunk; });
process.stdin.on('end', () => {
  const out = JSON.parse(input).map((c) => {
    if (c.dates) {
      const days = c.dates.map(dayNumber);
      return { days: days.map(d => (Number.isFinite(d) ? d : null)), period: periodFromDays(days) };
    }
    const f = seriesFeatures(c.values, c.period ?? null);
    return { ...f, adi: Number.isFinite(f.adi) ? f.adi : null, cell: seriesCell(f) };
  });
  process.stdout.write(JSON.stringify(out));
});
