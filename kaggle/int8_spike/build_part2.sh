#!/usr/bin/env bash
# Stage the Part 2 kernel (same id piyushsar/openjev-int8-spike) in build/int8_spike:
# part2_entry.py with the code (openjev/, train_prism.py, bench_trials.py, bench_lengths.json,
# kaggle/int8_spike/*.py) baked in as a base64 zip. Push: .venv/bin/kaggle kernels push -p build/int8_spike
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=build/int8_spike; rm -rf "$OUT"; mkdir -p "$OUT"
Z=$(mktemp -d)/src.zip
zip -qr "$Z" openjev train_prism.py bench_trials.py -x '*__pycache__*'
zip -qj "$Z" runs/optuna/bench_lengths.json kaggle/int8_spike/int8_model.py kaggle/int8_spike/bench_int8.py
{ printf 'SRC_B64 = "%s"\n' "$(base64 -w0 "$Z")"; cat kaggle/int8_spike/part2_entry.py; } > "$OUT/part2_entry.py"
cat > "$OUT/kernel-metadata.json" <<JSON
{"id": "piyushsar/openjev-int8-spike", "title": "openjev-int8-spike",
 "code_file": "part2_entry.py", "language": "python", "kernel_type": "script",
 "is_private": true, "enable_gpu": true, "enable_internet": true, "machine_shape": "NvidiaTeslaT4",
 "dataset_sources": ["piyushsar/openjev-prism-data"], "competition_sources": [], "kernel_sources": [], "model_sources": []}
JSON
ls -la "$OUT"
