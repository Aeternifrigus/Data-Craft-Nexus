#!/usr/bin/env bash
# Adds TabPFN to the committed benchmark run and republishes everything the
# page reads from it, in one go. On an Apple-silicon Mac TabPFN uses the GPU
# (MPS) by itself; nothing needs configuring.
#
#   export TABPFN_TOKEN=...      # your Prior Labs API key; without it TabPFN opens a browser to log in
#   bench/tools/run_tabpfn.sh    # any arguments go to dcn.run: --limit 5 is a quick first try
#
# The run is resumable: stop it, start it again, and it carries on where it was.
# What the result decides was written down before it existed: "What TabPFN has
# to show" in bench/README.md, and choose_small_lead() in dcn/learn.py.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"

# 9.0.0 is the release whose API (ModelVersion, create_default_for_version)
# dcn/models.py was checked against.
"$PY" -m pip install --quiet -r requirements.txt "tabpfn==9.0.0"
[ -n "${TABPFN_TOKEN:-}" ] || echo "TABPFN_TOKEN is not set, so TabPFN will open a browser window to log in to Prior Labs."

"$PY" -m dcn.run --models BASE-TABPFN --out results/full.csv "$@"
"$PY" -m dcn.analyze --results results/full.csv
"$PY" -m dcn.learn --results results/full.csv
label=$("$PY" -c "import pandas as pd; print(f\"{len(pd.read_csv('results/per_dataset.csv'))} PMLB datasets, {len(pd.read_csv('results/full.csv'))} model runs\")")
"$PY" -m dcn.evidence --results results/full.csv --label "$label" \
    --ranked-by "the per-model prior fitted on the earlier 40-dataset run, which had already seen 40 of these datasets"
"$PY" -m pytest -q
cd ..
npm run build   # the page embeds evidence.json, so rebuild it before the tests compare it
npm test

echo
echo "Done: $label."
echo "The rule's verdict is under \"small_lead\" in site/taxonomy/ranking.json, and the lines above it."
echo "README.md writes its numbers by hand; if the run count changed, update it there too."
echo "Review with git diff, then commit."
