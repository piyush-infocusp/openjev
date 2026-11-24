#!/usr/bin/env bash
# Stage the Unsloth spike kernel (piyushsar/openjev-unsloth-spike) in build/unsloth_spike:
# entry.py with the code (openjev/, train_prism.py, bench_trials.py,
# bench_lengths.json, bench_unsloth.py) baked in as a base64 zip.
# Push: .venv/bin/kaggle kernels push -p build/unsloth_spike
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=build/unsloth_spike; rm -rf "$OUT"; mkdir -p "$OUT"
Z=$(mktemp -d)/src.zip
zip -qr "$Z" openjev train_prism.py bench_trials.py -x '*__pycache__*'
zip -qj "$Z" runs/optuna/bench_lengths.json kaggle/unsloth_spike/bench_unsloth.py
{ printf 'SRC_B64 = "%s"\n' "$(base64 -w0 "$Z")"; cat kaggle/unsloth_spike/entry.py; } > "$OUT/entry.py"
cat > "$OUT/kernel-metadata.json" <<JSON
{"id": "piyushsar/openjev-unsloth-spike", "title": "openjev-unsloth-spike",
 "code_file": "entry.py", "language": "python", "kernel_type": "script",
 "is_private": true, "enable_gpu": true, "enable_internet": true, "machine_shape": "NvidiaTeslaT4",
 "dataset_sources": ["piyushsar/openjev-prism-data"], "competition_sources": [], "kernel_sources": [], "model_sources": []}
JSON
ls -la "$OUT"
