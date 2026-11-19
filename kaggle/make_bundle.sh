#!/usr/bin/env bash
# Build build/kaggle_dataset/ containing the source bundle (openjev_src.zip)
# and dataset-metadata.json, ready for `kaggle datasets create/version`.
#
# Usage:
#   kaggle/make_bundle.sh [run_args_file]
#
# If run_args_file is omitted, a default run_args.txt is generated.
# Requires KAGGLE_USERNAME to be set (used only to fill dataset-metadata.json's
# "id" field, never written as credentials).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build/kaggle_dataset"
DATASET_SLUG="openjev-data"

if [[ -z "${KAGGLE_USERNAME:-}" ]]; then
  echo "error: KAGGLE_USERNAME is not set." >&2
  exit 1
fi

RUN_ARGS_FILE="${1:-}"

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}/bundle/data"

echo "Copying openjev/ ..."
cp -r "${REPO_ROOT}/openjev" "${BUILD_DIR}/bundle/openjev"

echo "Copying converters/_common.py ..."
mkdir -p "${BUILD_DIR}/bundle/converters"
cp "${REPO_ROOT}/converters/_common.py" "${BUILD_DIR}/bundle/converters/_common.py"

echo "Copying train*.py ..."
for f in "${REPO_ROOT}"/train*.py; do
  [[ -e "$f" ]] && cp "$f" "${BUILD_DIR}/bundle/"
done

echo "Copying data/*.{train,val,test}.jsonl (falling back to data/*.jsonl) ..."
shopt -s nullglob
split_files=("${REPO_ROOT}"/data/*.train.jsonl "${REPO_ROOT}"/data/*.val.jsonl "${REPO_ROOT}"/data/*.test.jsonl)
if [[ ${#split_files[@]} -gt 0 ]]; then
  cp "${split_files[@]}" "${BUILD_DIR}/bundle/data/"
else
  echo "No split files found; falling back to data/*.jsonl"
  all_files=("${REPO_ROOT}"/data/*.jsonl)
  if [[ ${#all_files[@]} -gt 0 ]]; then
    cp "${all_files[@]}" "${BUILD_DIR}/bundle/data/"
  else
    echo "warning: no data/*.jsonl files found either; bundle will have no data." >&2
  fi
fi
shopt -u nullglob

echo "Writing run_args.txt ..."
if [[ -n "${RUN_ARGS_FILE}" ]]; then
  if [[ ! -f "${RUN_ARGS_FILE}" ]]; then
    echo "error: run args file '${RUN_ARGS_FILE}' not found." >&2
    exit 1
  fi
  cp "${RUN_ARGS_FILE}" "${BUILD_DIR}/bundle/run_args.txt"
else
  cat > "${BUILD_DIR}/bundle/run_args.txt" <<'EOF'
--model Qwen/Qwen3-0.6B --steps 3000 --bs 8 --grad-accum 4 --lr 2e-5 --head-lr 1e-3 --max-state-tokens 1024 --eval-every 500 --eval-max 2000 --grad-ckpt --fp16 --save-dir checkpoints/sft
EOF
fi

echo "Zipping bundle into openjev_src.zip ..."
( cd "${BUILD_DIR}/bundle" && zip -qr "${BUILD_DIR}/openjev_src.zip" . )

echo "Writing dataset-metadata.json ..."
cat > "${BUILD_DIR}/dataset-metadata.json" <<EOF
{
  "title": "${DATASET_SLUG}",
  "id": "${KAGGLE_USERNAME}/${DATASET_SLUG}",
  "licenses": [{"name": "other"}]
}
EOF

echo "Bundle ready at ${BUILD_DIR} (openjev_src.zip + dataset-metadata.json)."
