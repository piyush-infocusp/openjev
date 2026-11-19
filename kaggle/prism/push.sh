#!/usr/bin/env bash
# Upload the staged files (private). Run build.sh first.
# SKIP_DATASET=1 skips the dataset version: use it for an Optuna round where only
# trials.json changed (build.sh bakes the trials into the kernel's entry.py). Code
# changes (bench_trials.py, train_prism.py, openjev/) still need a dataset version.
set -euo pipefail
cd "$(dirname "$0")/../.."
K=.venv/bin/kaggle; OUT=build/kaggle_prism
if [ "${SKIP_DATASET:-0}" != 1 ]; then
  if $K datasets status "$KAGGLE_USERNAME/openjev-prism-data" >/dev/null 2>&1; then
    $K datasets version -p "$OUT/dataset" -m "update" --dir-mode zip
  else
    $K datasets create -p "$OUT/dataset" --dir-mode zip
  fi
  echo "Wait until the dataset is ready, then:"
fi
echo "  $K kernels push -p $OUT/eda && $K kernels push -p $OUT/train"
echo "  (staged kernel: $(head -c 120 "$OUT/train/kernel-metadata.json" | sed -n 's/.*"id": "\([^"]*\)".*/\1/p'))"
