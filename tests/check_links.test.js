// The link checker must fail on a dead link and not on a network hiccup.
// CI once reported a live SciPy page as broken because three quick retries
// all landed in the same burst of connections to one host. These run the
// checker against a fake network, so they need no internet.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { checkLinks, describe } from '../tools/links.mjs';

const instant = { sleep: async () => {} };

// A fake fetch: `plan` maps a URL to what each successive attempt does.
function network(plan) {
  const calls = new Map();
  let inFlight = new Map(), peak = new Map();
  const fetch = async (url) => {
    const host = new URL(url).host;
    inFlight.set(host, (inFlight.get(host) ?? 0) + 1);
    peak.set(host, Math.max(peak.get(host) ?? 0, inFlight.get(host)));
    const n = (calls.get(url) ?? 0) + 1;
    calls.set(url, n);
    await new Promise(r => setTimeout(r, 1));
    inFlight.set(host, inFlight.get(host) - 1);
    const steps = plan[url] ?? [200];
    const step = steps[Math.min(n - 1, steps.length - 1)];
    if (step === 'reset') {
      throw Object.assign(new TypeError('fetch failed'), { cause: Object.assign(new Error('read ECONNRESET'), { code: 'ECONNRESET' }) });
    }
    return { status: step, ok: step >= 200 && step < 300, body: null };
  };
  return { fetch, calls, peak };
}

test('a connection that resets and then answers is not a broken link', async () => {
  const url = 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html';
  const net = network({ [url]: ['reset', 'reset', 'reset', 200] });
  const failures = await checkLinks([url], { ...instant, fetch: net.fetch });
  assert.deepEqual(failures, []);
  assert.equal(net.calls.get(url), 4);
});

test('a link that never connects is reported, with the real reason', async () => {
  const url = 'https://docs.scipy.org/gone';
  const net = network({ [url]: ['reset'] });
  const failures = await checkLinks([url], { ...instant, fetch: net.fetch });
  assert.equal(failures.length, 1);
  assert.equal(failures[0].status, undefined);
  assert.match(failures[0].error, /ECONNRESET/, 'not just "fetch failed"');
  // Four tries in the main pass, then two more in the unhurried final pass.
  assert.equal(net.calls.get(url), 6);
});

test('a 404 is final on the first answer', async () => {
  const url = 'https://numpy.org/doc/stable/reference/nope.html';
  const net = network({ [url]: [404, 200] });
  const failures = await checkLinks([url], { ...instant, fetch: net.fetch });
  assert.deepEqual(failures.map(f => [f.url, f.status]), [[url, 404]]);
  assert.equal(net.calls.get(url), 1, 'a server that says gone is not asked again');
});

test('rate limits and server errors are retried', async () => {
  const limited = 'https://github.com/a/b';
  const flaky = 'https://scikit-learn.org/x';
  const net = network({ [limited]: [429, 429, 200], [flaky]: [503, 200] });
  assert.deepEqual(await checkLinks([limited, flaky], { ...instant, fetch: net.fetch }), []);
  assert.equal(net.calls.get(limited), 3);
  assert.equal(net.calls.get(flaky), 2);
});

test('no host gets more than two requests at once, however many of its links are queued', async () => {
  const urls = [
    ...Array.from({ length: 12 }, (_, i) => `https://docs.scipy.org/page${i}.html`),
    ...Array.from({ length: 4 }, (_, i) => `https://numpy.org/page${i}.html`),
  ];
  const net = network({});
  assert.deepEqual(await checkLinks(urls, { ...instant, fetch: net.fetch }), []);
  assert.equal(net.peak.get('docs.scipy.org'), 2);
  assert.ok(net.peak.get('numpy.org') <= 2);
  assert.equal([...net.calls.values()].reduce((a, b) => a + b, 0), urls.length, 'every link checked once');
});

test('the reason for a network failure is spelled out', () => {
  const err = Object.assign(new TypeError('fetch failed'),
    { cause: Object.assign(new Error('getaddrinfo ENOTFOUND example.invalid'), { code: 'ENOTFOUND' }) });
  assert.equal(describe(err), 'fetch failed: ENOTFOUND: getaddrinfo ENOTFOUND example.invalid');
  const timeout = Object.assign(new Error('The operation was aborted due to timeout'), { name: 'TimeoutError' });
  assert.equal(describe(timeout), 'The operation was aborted due to timeout: TimeoutError');
});
