#!/usr/bin/env bash
# Orchestrates the Kaggle GPU workflow for OpenJev training.
#
# Subcommands:
#   bundle    build build/kaggle_dataset/ (source dataset bundle)
#   upload    kaggle datasets create (first time) or version (later)
#   push      render kernel-metadata.json into build/kaggle_kernel/, copy
#             entry.py, then `kaggle kernels push`
#   status    print `kaggle kernels status <user>/<slug>`
#   wait      poll status every 60s until complete/error/cancel; print each
#             status change; exit non-zero on error/cancel
#   pull      pull kernel output into runs/<timestamp>/
#
# Global flag --dry-run (must appear before the subcommand): only print the
# commands that would run, do not execute them.
#
# Requires KAGGLE_USERNAME env var and the `kaggle` CLI on PATH (both checks
# are skipped in --dry-run mode).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"
KERNEL_BUILD_DIR="${BUILD_DIR}/kaggle_kernel"
DATASET_SLUG="openjev-data"
KERNEL_SLUG="openjev-train"
POLL_INTERVAL=60

DRY_RUN=0

usage() {
  cat >&2 <<'EOF'
Usage: kaggle/run.sh [--dry-run] <subcommand>

Subcommands:
  bundle    build the source dataset bundle
  upload    create/version the Kaggle dataset
  push      render kernel metadata and push the kernel
  status    print the kernel's current status
  wait      poll status until complete/error/cancel
  pull      pull kernel output into runs/<timestamp>/
EOF
  exit 1
}

# --- parse global flags ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      break
      ;;
  esac
done

SUBCOMMAND="${1:-}"
[[ -n "${SUBCOMMAND}" ]] || usage
shift || true

maybe_run() {
  # Prints the command always; executes it unless DRY_RUN=1.
  echo "+ $*"
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    "$@"
  fi
}

check_env() {
  if [[ -z "${KAGGLE_USERNAME:-}" ]]; then
    echo "error: KAGGLE_USERNAME is not set. Export it before running kaggle/run.sh." >&2
    exit 1
  fi
  if ! command -v kaggle >/dev/null 2>&1; then
    echo "error: 'kaggle' CLI not found on PATH. Install it with: pip install kaggle" >&2
    exit 1
  fi
}

if [[ "${DRY_RUN}" -eq 0 ]]; then
  check_env
fi

USER="${KAGGLE_USERNAME:-}"
KERNEL_ID="${USER}/${KERNEL_SLUG}"

cmd_bundle() {
  maybe_run bash "${SCRIPT_DIR}/make_bundle.sh"
}

cmd_upload() {
  local dataset_dir="${BUILD_DIR}/kaggle_dataset"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "+ (check) test -f ${dataset_dir}/dataset-metadata.json"
    echo "+ kaggle datasets create -p ${dataset_dir}  # if dataset does not exist yet"
    echo "+ kaggle datasets version -p ${dataset_dir} -m <message>  # if it already exists"
    return
  fi
  if [[ ! -f "${dataset_dir}/dataset-metadata.json" ]]; then
    echo "error: ${dataset_dir}/dataset-metadata.json not found. Run 'bundle' first." >&2
    exit 1
  fi
  if kaggle datasets status "${USER}/${DATASET_SLUG}" >/dev/null 2>&1; then
    maybe_run kaggle datasets version -p "${dataset_dir}" -m "update $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  else
    maybe_run kaggle datasets create -p "${dataset_dir}"
  fi
}

cmd_push() {
  local dataset_source="${USER}/${DATASET_SLUG}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    dataset_source="{{USER}}/${DATASET_SLUG}"
  fi
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    mkdir -p "${KERNEL_BUILD_DIR}"
  fi
  local user_for_render="${USER}"
  if [[ "${DRY_RUN}" -eq 1 && -z "${user_for_render}" ]]; then
    user_for_render="{{USER}}"
  fi
  echo "+ render kernel-metadata.template.json -> ${KERNEL_BUILD_DIR}/kernel-metadata.json"
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    sed -e "s/{{USER}}/${user_for_render}/g" \
        -e "s/{{SLUG}}/${KERNEL_SLUG}/g" \
        -e "s#{{DATASET}}#${dataset_source}#g" \
        "${SCRIPT_DIR}/kernel-metadata.template.json" > "${KERNEL_BUILD_DIR}/kernel-metadata.json"
  fi
  echo "+ cp ${SCRIPT_DIR}/entry.py ${KERNEL_BUILD_DIR}/entry.py"
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    cp "${SCRIPT_DIR}/entry.py" "${KERNEL_BUILD_DIR}/entry.py"
  fi
  maybe_run kaggle kernels push -p "${KERNEL_BUILD_DIR}"
}

cmd_status() {
  maybe_run kaggle kernels status "${KERNEL_ID}"
}

cmd_wait() {
  echo "+ polling 'kaggle kernels status ${KERNEL_ID}' every ${POLL_INTERVAL}s until complete/error/cancel"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "+ kaggle kernels status ${KERNEL_ID}"
    return
  fi
  local last_status=""
  while true; do
    local output
    output="$(kaggle kernels status "${KERNEL_ID}")"
    if [[ "${output}" != "${last_status}" ]]; then
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) ${output}"
      last_status="${output}"
    fi
    if echo "${output}" | grep -qi "complete"; then
      exit 0
    fi
    if echo "${output}" | grep -qi -e "error" -e "cancel"; then
      echo "kernel run ended with error/cancel." >&2
      exit 1
    fi
    sleep "${POLL_INTERVAL}"
  done
}

cmd_pull() {
  local ts
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  local out_dir="${REPO_ROOT}/runs/${ts}"
  echo "+ mkdir -p ${out_dir}"
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    mkdir -p "${out_dir}"
  fi
  maybe_run kaggle kernels output "${KERNEL_ID}" -p "${out_dir}"
}

case "${SUBCOMMAND}" in
  bundle) cmd_bundle ;;
  upload) cmd_upload ;;
  push) cmd_push ;;
  status) cmd_status ;;
  wait) cmd_wait ;;
  pull) cmd_pull ;;
  *) usage ;;
esac
