#!/usr/bin/env bash
# Stage the Kaggle dataset and the two kernels (EDA notebook + Prism training).
# Nothing is uploaded here. Upload with:  ./build.sh && ./push.sh
#
# PROFILE=optuna ./build.sh  stages the speed-trial kernel instead (id openjev-prism-optuna):
# PROFILE (and TRIAL_SECONDS, default 50) and kaggle/prism/trials.json (from tools/optuna_loop.py propose) are baked into the
# staged entry.py, so a new round only needs a kernel push, not a new dataset version.
# trials.json is also put into openjev_src.zip as a fallback.
# Final run (docs/FINAL_RUN_PLAN.md), each with DDP=1 (2xT4 torchrun, OJ-062):
#   PROFILE=smoke    DDP=1 ./build.sh   pre-flight, kernel openjev-prism-smoke
#   PROFILE=full_sft DDP=1 ./build.sh   session 1 (SFT 2,200 steps), kernel openjev-prism-sft
#   PROFILE=full_rl  DDP=1 ./build.sh   session 2 (RL 1,000 steps), kernel openjev-prism-rl; it
#                                       attaches the openjev-prism-sft output (SFT best/)
#   SESSION_MINUTES=620 (default) is baked: each stage gets --max-minutes = this minus the time used.
#   After a killed session: EXTRA_KERNEL_SOURCES=<user>/<killed kernel> (comma list) attaches its
#   output; KERNEL_ID=<id> pushes under another id if Kaggle does not let a kernel attach its own output.
# PROFILE=optuna_length ./build.sh  stages the length-curriculum bench (id openjev-prism-length) from
# kaggle/prism/trials.json (propose on runs/length/space.yaml) + runs/length/stage_streams.json.
# PROFILE=optuna_validate ./build.sh  stages a quality checkpoint (id openjev-prism-validate) from
# kaggle/prism/validate.json (tools/optuna_loop.py validate-plan).
# Round 3+: runs/optuna/trial_stream.json and val_set.json (optuna_loop.py freeze) go into the zip,
# so the dataset must be pushed again after a freeze (and after any code change).
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${KAGGLE_USERNAME:?set KAGGLE_USERNAME}"
DS=openjev-prism-data
OUT=build/kaggle_prism
rm -rf "$OUT"; mkdir -p "$OUT/dataset" "$OUT/eda" "$OUT/train"

# 1. Dataset: every split as its own jsonl file + the code as one zip.
cp data/*.{train,val,test}.jsonl data/STATS.md "$OUT/dataset/"
(zip -qr "$OUT/dataset/openjev_src.zip" openjev train_prism.py bench_trials.py -x '*__pycache__*')
TRIALS=${TRIALS:-kaggle/prism/trials.json}   # TRIALS=... overrides (e.g. a length round)
VALIDATE=kaggle/prism/validate.json
if [ -f "$TRIALS" ]; then zip -qj "$OUT/dataset/openjev_src.zip" "$TRIALS"; fi
# Round 3+: the frozen trial stream and val set (tools/optuna_loop.py freeze; record ids only).
for f in runs/optuna/trial_stream.json runs/optuna/val_set.json runs/optuna/packed_layout.json \
         runs/length/stage_streams.json runs/optuna/bench_lengths.json; do
  if [ -f "$f" ]; then zip -qj "$OUT/dataset/openjev_src.zip" "$f"; fi
done
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

# 3. Training kernel (GPU). With PROFILE set, bake it (and the trials) into entry.py.
KERNEL=openjev-prism-train
if [ -n "${PROFILE:-}" ]; then
  [ "$PROFILE" = optuna ] && KERNEL=openjev-prism-optuna
  [ "$PROFILE" = optuna_length ] && KERNEL=openjev-prism-length
  [ "$PROFILE" = optuna_validate ] && KERNEL=openjev-prism-validate
  [ "$PROFILE" = smoke ] && KERNEL=openjev-prism-smoke
  [ "$PROFILE" = full_sft ] && KERNEL=openjev-prism-sft
  [ "$PROFILE" = full_rl ] && KERNEL=openjev-prism-rl
  case "$PROFILE" in short|full) echo "PROFILE=$PROFILE was replaced by smoke|full_sft|full_rl" >&2; exit 1;; esac
  case "$PROFILE" in optuna|optuna_length) [ -f "$TRIALS" ] || { echo "no $TRIALS: run tools/optuna_loop.py propose" >&2; exit 1; };; esac
  [ "$PROFILE" = optuna_validate ] && [ ! -f "$VALIDATE" ] && { echo "no $VALIDATE: run tools/optuna_loop.py validate-plan" >&2; exit 1; }
  python3 - "$PROFILE" "$TRIALS" kaggle/prism/entry.py "$VALIDATE" > "$OUT/train/entry.py" <<'PY'
import os, sys
profile, trials, entry, validate = sys.argv[1:]
print(f"import os; os.environ.setdefault('PROFILE', {profile!r})")
for v in ("TRIAL_SECONDS", "DDP", "SYNC_MODE", "SYNC_EVERY", "SESSION_MINUTES",
          "ALLOW_UNFINISHED_SFT"):   # DDP*: OJ-062
    if os.environ.get(v):
        print(f"os.environ.setdefault({v!r}, {os.environ[v]!r})")
if profile in ("optuna", "optuna_length"):
    print(f"OPTUNA_TRIALS = {open(trials).read()!r}")
if profile == "optuna_validate":
    print(f"OPTUNA_VALIDATE = {open(validate).read()!r}")
print(open(entry).read())
PY
else
  for v in DDP SYNC_MODE SYNC_EVERY SESSION_MINUTES; do
    [ -n "${!v:-}" ] && { echo "$v is baked only with PROFILE set (PROFILE=smoke|full_sft|full_rl $v=... ./build.sh)" >&2; exit 1; }
  done
  cp kaggle/prism/entry.py "$OUT/train/"
fi
KERNEL=${KERNEL_ID:-$KERNEL}   # KERNEL_ID=... overrides (a resumed session under a new id)
# Earlier kernel outputs to attach: full_rl reads the SFT best/; EXTRA_KERNEL_SOURCES resumes a kill.
KSRC=()
[ "${PROFILE:-}" = full_rl ] && KSRC+=("$KAGGLE_USERNAME/openjev-prism-sft")
IFS=',' read -ra EXTRA <<< "${EXTRA_KERNEL_SOURCES:-}"
for k in "${EXTRA[@]}"; do [ -n "$k" ] && KSRC+=("$k"); done
KSRC_JSON=$(printf '%s\n' "${KSRC[@]}" | python3 -c 'import json,sys; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))')
cat > "$OUT/train/kernel-metadata.json" <<JSON
{"id": "$KAGGLE_USERNAME/$KERNEL", "title": "$KERNEL",
 "code_file": "entry.py", "language": "python", "kernel_type": "script",
 "is_private": true, "enable_gpu": true, "enable_internet": true,
 "dataset_sources": ["$KAGGLE_USERNAME/$DS"], "competition_sources": [], "kernel_sources": $KSRC_JSON, "model_sources": []}
JSON
du -sh "$OUT"/*
