#!/usr/bin/env bash
# Upload the staged files (private). Run build.sh first.
set -euo pipefail
cd "$(dirname "$0")/../.."
K=.venv/bin/kaggle; OUT=build/kaggle_prism
if $K datasets status "$KAGGLE_USERNAME/openjev-prism-data" >/dev/null 2>&1; then
  $K datasets version -p "$OUT/dataset" -m "update" --dir-mode zip
else
  $K datasets create -p "$OUT/dataset" --dir-mode zip
fi
echo "Wait until the dataset is ready, then:"
echo "  $K kernels push -p $OUT/eda && $K kernels push -p $OUT/train"
