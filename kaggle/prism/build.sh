#!/usr/bin/env bash
# Stage the Kaggle dataset and the two kernels (EDA notebook + Prism training).
# Nothing is uploaded here. Upload with:  ./build.sh && ./push.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${KAGGLE_USERNAME:?set KAGGLE_USERNAME}"
DS=openjev-prism-data
OUT=build/kaggle_prism
rm -rf "$OUT"; mkdir -p "$OUT/dataset" "$OUT/eda" "$OUT/train"

# 1. Dataset: every split as its own jsonl file + the code as one zip.
cp data/*.{train,val,test}.jsonl data/STATS.md "$OUT/dataset/"
(zip -qr "$OUT/dataset/openjev_src.zip" openjev train_prism.py -x '*__pycache__*')
cat > "$OUT/dataset/dataset-metadata.json" <<JSON
{"title": "$DS", "id": "$KAGGLE_USERNAME/$DS", "licenses": [{"name": "other"}],
 "subtitle": "OpenJev decision tasks in the Jev request/answer format",
 "description": "Converted from public HF datasets (banking77, clinc150, diplomacy, helpsteer3, lichess puzzles, pokerbench, prm800k, support tickets). Each source keeps its own license. See STATS.md."}
JSON

# 2. EDA notebook kernel.
cp notebooks/explore_data.ipynb "$OUT/eda/"
cat > "$OUT/eda/kernel-metadata.json" <<JSON
{"id": "$KAGGLE_USERNAME/openjev-prism-eda", "title": "openjev-prism-eda",
 "code_file": "explore_data.ipynb", "language": "python", "kernel_type": "notebook",
 "is_private": true, "enable_gpu": false, "enable_internet": false,
 "dataset_sources": ["$KAGGLE_USERNAME/$DS"], "competition_sources": [], "kernel_sources": [], "model_sources": []}
JSON

# 3. Training kernel (GPU).
cp kaggle/prism/entry.py "$OUT/train/"
cat > "$OUT/train/kernel-metadata.json" <<JSON
{"id": "$KAGGLE_USERNAME/openjev-prism-train", "title": "openjev-prism-train",
 "code_file": "entry.py", "language": "python", "kernel_type": "script",
 "is_private": true, "enable_gpu": true, "enable_internet": true,
 "dataset_sources": ["$KAGGLE_USERNAME/$DS"], "competition_sources": [], "kernel_sources": [], "model_sources": []}
JSON
du -sh "$OUT"/*
