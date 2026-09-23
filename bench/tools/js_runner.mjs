// Prints the Python the page's "Run it here" worker runs (site/js/verify.js),
// for the test that runs it with CPython.
import { PY_RUNNER } from '../../site/js/verify.js';

process.stdout.write(PY_RUNNER);
