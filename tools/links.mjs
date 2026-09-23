// Checking that links resolve, without crying wolf.
//
// A broken link is one the server says is gone: a 404, a 410, any status that
// is not a success after redirects. A connection that resets or times out
// says nothing about the page, only about the network between the checker and
// the host, and CI runners see plenty of those. So the two are treated
// differently:
//
//   a status   final on the first answer, except 429 (rate limited) and the
//              5xx family, which are the server asking to be tried again
//   a network  error retried with growing pauses, and anything still failing
//              is checked once more at the end, one at a time, after the
//              burst of the main pass is over
//
// Requests are also spread out: at most PER_HOST at once to any one host, so
// the dozens of links into docs.scipy.org or scikit-learn.org do not arrive
// as a burst that the host's edge may reset.
//
// Pure apart from `fetch` and `sleep`, which are passed in so the tests can
// run without a network.

export const DEFAULTS = {
  concurrency: 6,      // requests in flight overall
  perHost: 2,          // requests in flight to one host
  attempts: 4,         // tries per link in the main pass
  backoff: [2000, 6000, 15000],   // pause before each retry, in ms
  settle: 10000,       // pause before the final one-at-a-time pass
  timeout: 20000,      // per request
};

const RETRY_STATUS = (status) => status === 429 || status >= 500;

// What actually went wrong. Node's fetch reports every network failure as
// "fetch failed" and keeps the reason in err.cause.
export function describe(err) {
  const cause = err?.cause;
  const code = cause?.code || cause?.name || err?.name;
  const detail = cause?.message && cause.message !== err?.message ? cause.message : null;
  return [err?.message || String(err), code && code !== 'Error' ? code : null, detail]
    .filter(Boolean).join(': ');
}

async function attempt(url, { fetch, timeout, userAgent }) {
  try {
    const res = await fetch(url, {
      redirect: 'follow',
      headers: { 'user-agent': userAgent },
      signal: AbortSignal.timeout(timeout),
    });
    // Nothing is read from the body; release the connection.
    try { await res.body?.cancel?.(); } catch { /* already closed */ }
    return { status: res.status, ok: res.ok };
  } catch (err) {
    return { error: describe(err) };
  }
}

async function checkOne(url, options) {
  const { attempts, backoff, sleep } = options;
  let last;
  for (let i = 0; i < attempts; i++) {
    if (i > 0) await sleep(backoff[Math.min(i - 1, backoff.length - 1)]);
    last = await attempt(url, options);
    if (last.ok) return { url, ok: true, tries: i + 1 };
    if (last.status != null && !RETRY_STATUS(last.status)) break;   // the server has answered
  }
  return { url, ok: false, ...last };
}

// Check every URL; returns the ones that failed, each with its status or error.
export async function checkLinks(urls, options = {}) {
  const opts = {
    ...DEFAULTS,
    userAgent: 'data-craft-nexus link check (https://github.com/Aeternifrigus/Data-Craft-Nexus)',
    fetch: globalThis.fetch,
    sleep: (ms) => new Promise(r => setTimeout(r, ms)),
    onProgress: () => {},
    ...options,
  };

  const queue = [...urls];
  const inFlight = new Map();   // host -> count
  const results = [];
  let active = 0;

  await new Promise((resolve) => {
    const next = () => {
      if (!queue.length && active === 0) return resolve();
      while (active < opts.concurrency) {
        // The first queued URL whose host has room.
        const index = queue.findIndex(u => (inFlight.get(new URL(u).host) ?? 0) < opts.perHost);
        if (index === -1) return;
        const url = queue.splice(index, 1)[0];
        const host = new URL(url).host;
        inFlight.set(host, (inFlight.get(host) ?? 0) + 1);
        active++;
        checkOne(url, opts).then((result) => {
          results.push(result);
          inFlight.set(host, inFlight.get(host) - 1);
          active--;
          opts.onProgress(result);
          next();
        });
      }
    };
    next();
  });

  // A network error that survived the main pass gets one more, unhurried look.
  const network = results.filter(r => !r.ok && r.error);
  if (network.length) {
    await opts.sleep(opts.settle);
    for (const r of network) {
      const again = await checkOne(r.url, { ...opts, attempts: 2 });
      Object.assign(r, again);
    }
  }
  return results.filter(r => !r.ok);
}
