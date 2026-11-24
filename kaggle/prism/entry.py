"""Kaggle GPU kernel: SFT and/or RL (exact policy gradient, Brier reward) for
OpenJev-Prism on Gemma 4 E2B (PROFILE=smoke|full_sft|full_rl), or (PROFILE=optuna*) a
batch of speed trials with bench_trials.py. Outputs go to /kaggle/working/out."""
import glob, os, shutil, subprocess, sys, time, zipfile

T0 = time.time()                       # kernel start: the session clock for --max-minutes
work = "/kaggle/working/src"


def dataset_glob(pattern):
    """Matches in the dataset only. An attached earlier kernel output (a resumed session)
    also holds src/ and out/; its old train_prism.py must never shadow the new code."""
    return [p for p in glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
            if "/out/" not in p and "/src/" not in p]


# Kaggle may unpack a .zip inside a dataset, so accept the zip or its unpacked folder.
zips = dataset_glob("openjev_src.zip")
if zips and os.path.isfile(zips[0]):
    zipfile.ZipFile(zips[0]).extractall(work)
else:
    script = dataset_glob("train_prism.py")
    if not script:
        raise SystemExit("openjev_src not found; inputs: "
                         + str(glob.glob("/kaggle/input/**", recursive=True)[:50]))
    shutil.copytree(os.path.dirname(script[0]), work, dirs_exist_ok=True)
data = os.path.dirname(dataset_glob("banking77.train.jsonl")[0])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U",
                       "transformers>=5.17", "peft", "safetensors", "tensorboard", "optuna", "pyyaml",
                       "bitsandbytes==0.50.2"])   # OJ-075 quant=int8|nf4 (openjev.prism.BNB_PIN); sm75 OK
# The Kaggle image ships torchao 0.10; new peft rejects torchao < 0.16 at LoRA
# injection. We do not use torchao, so remove it.
subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
subprocess.check_call([sys.executable, "-c",
                       "import bitsandbytes, peft, torch, transformers; from peft.import_utils import "
                       "is_torchao_available as f; assert not f(); "
                       "print('env ok', torch.__version__, transformers.__version__, peft.__version__)"])
os.chdir(work)
OUT = "/kaggle/working/out"
os.makedirs(OUT, exist_ok=True)

# PROFILE "smoke" | "full_sft" | "full_rl": the final run's pre-flight and its two sessions
# (see STAGES below).
# PROFILE "optuna": speed trials from trials.json (tools/optuna_loop.py propose), no training run.
# PROFILE "optuna_length": speed trials like "optuna", on the frozen per-stage length streams
# (stage_streams.json, tools/length_curriculum.py freeze) instead of trial_stream.json.
# PROFILE "optuna_validate": a quality checkpoint from validate.json (optuna_loop.py validate-plan).
# A pushed script kernel gets no env vars, so `PROFILE=optuna build.sh` bakes PROFILE (and
# the trials, as OPTUNA_TRIALS) into the staged copy of this file.
PROFILE = os.environ.get("PROFILE", "smoke")
# OJ-062: DDP=1 (baked by `DDP=1 build.sh`) runs the smoke/full_* training under torchrun, one
# process per GPU (2xT4); SYNC_MODE=local_sgd SYNC_EVERY=K for weight averaging instead of the
# per-step gradient all-reduce. Default DDP=0: one process, unchanged. Optuna ddp=on trials
# get their own torchrun launch from bench_trials.py (no env needed).
os.environ.setdefault("NCCL_IB_DISABLE", "1")          # PCIe T4 pair, no InfiniBand
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")   # a dead rank ends the job

if PROFILE in ("optuna", "optuna_length"):
    if globals().get("OPTUNA_TRIALS"):            # baked by build.sh: newer than the bundle copy
        with open("trials.json", "w") as f:
            f.write(OPTUNA_TRIALS)
    print("=== OPTUNA trials ===\n" + open("trials.json").read(), flush=True)
    # Round 3+: the frozen trial stream (record ids, bundled by build.sh); bench checks its sha.
    stream = ["--trial-stream", "trial_stream.json"] if os.path.exists("trial_stream.json") else []
    import json, torch
    # Study v3+: trials proposed with a bench_set_sha run on the fixed multi-length bench set.
    # bench_trials refuses them without the file (or with another set), so a missing file
    # fails the round instead of timing other data.
    if any((t.get("bench") or {}).get("bench_set_sha") for t in json.load(open("trials.json"))):
        stream = ["--bench-lengths", "bench_lengths.json"]
    if PROFILE == "optuna_length":
        stream = ["--stage-streams", "stage_streams.json"]
    # Dual-GPU (2xT4) is the default: shards run one per GPU, GPU1 staggered after GPU0's load.
    # DUAL_GPU=0 (or a single visible GPU) falls back to sequential trials on GPU 0.
    dual = (os.environ.get("DUAL_GPU", "1") != "0" and torch.cuda.device_count() >= 2
            and len(json.load(open("trials.json"))) > 1)
    print(f"=== OPTUNA dual_gpu={dual} (gpus={torch.cuda.device_count()}) ===", flush=True)
    subprocess.check_call([sys.executable, "bench_trials.py", *(["--dual-gpu"] if dual else []),
                           "--trials", "trials.json",
                           "--trial-seconds", os.environ.get("TRIAL_SECONDS", "50"),
                           "--out", f"{OUT}/optuna_results.json",
                           "--model", "google/gemma-4-E2B-it", "--train", f"{data}/*.train.jsonl",
                           "--bs", "4", "--max-state-tokens", "1024", "--grad-ckpt", *stream])
    print("=== OPTUNA DONE ===", flush=True)
    sys.exit(0)

if PROFILE == "optuna_validate":
    # Quality checkpoint (tools/optuna_loop.py validate-plan): best vs baseline config, real
    # training steps, eval on the frozen val set. Not a speed trial.
    if globals().get("OPTUNA_VALIDATE"):
        with open("validate.json", "w") as f:
            f.write(OPTUNA_VALIDATE)
    print("=== OPTUNA VALIDATE ===\n" + open("validate.json").read(), flush=True)
    subprocess.check_call([sys.executable, "bench_trials.py", "--validate", "validate.json",
                           "--val-set", "val_set.json", "--out", f"{OUT}/validate_results.json",
                           "--model", "google/gemma-4-E2B-it", "--train", f"{data}/*.train.jsonl",
                           "--val", f"{data}/*.val.jsonl", "--max-state-tokens", "1024", "--grad-ckpt"])
    print("=== OPTUNA VALIDATE DONE ===", flush=True)
    sys.exit(0)

# Training profiles (docs/FINAL_RUN_PLAN.md; user decisions 2026-11-24: two sessions, packing
# off, length grouping on, KL reference = the frozen SFT adapter):
#   smoke     pre-flight (~40 min est., 3 staggered loads): SFT 30 steps, then RL exact 10 steps
#             run as 5 + --resume auto + 5 (resume, frozen SFT reference and GradScaler state on
#             the real model and 2xT4).
#   full_sft  session 1: SFT 2,200 steps.
#   full_rl   session 2: RL exact 1,000 steps from the SFT best/ (build.sh attaches the
#             openjev-prism-sft kernel output).
# Every stage: mb 8 x accum 4 (global 32), max_state 2048, packing off, length grouping on, fp16
# (--dtype auto on the T4), ple_cpu 1, grad ckpt on, torch_compile off, eval every 200 steps on
# 800 val records, a resume checkpoint every 100 steps, --resume auto, and --max-minutes =
# SESSION_MINUTES (620) minus the time already used by this kernel. Checkpoints are written
# straight into /kaggle/working/out/<stage> (the kernel output), so every save is in
# /kaggle/working at once. After a kill: push a new version with the killed output attached
# (EXTRA_KERNEL_SOURCES, build.sh); restore() copies it back and --resume auto continues.
STAGES = {
    "smoke": [("sft", 30, 15, 10), ("rl", 10, 5, 5)],     # (stage, steps, eval_every, save_every)
    "full_sft": [("sft", 2200, 200, 100)],
    "full_rl": [("rl", 1000, 200, 100)],
}
if PROFILE not in STAGES:
    raise SystemExit(f"PROFILE {PROFILE!r}: use one of {sorted(STAGES)} or optuna*")
SESSION_MINUTES = float(os.environ.get("SESSION_MINUTES", "620"))
EVAL_MAX = 160 if PROFILE == "smoke" else 800
import torch
sys.path.insert(0, work)
from openjev import resume as rs
N_GPU = torch.cuda.device_count()
NPROC = N_GPU if os.environ.get("DDP", "0") == "1" and N_GPU >= 2 else 1
LAUNCH, ENV = [sys.executable], dict(os.environ)
SYNC = []
if NPROC > 1:
    # --grad-accum stays global (must divide by NPROC): same global batch and lr schedule.
    LAUNCH = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(NPROC)]
    ENV.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 4) // NPROC)))
    SYNC = ["--sync-mode", os.environ.get("SYNC_MODE", "ddp"), "--sync-every", os.environ.get("SYNC_EVERY", "50")]
print(f"=== GPUs {N_GPU}, processes {NPROC} {SYNC} ===", flush=True)
common = ["--model", "google/gemma-4-E2B-it", "--train", f"{data}/*.train.jsonl",
          "--val", f"{data}/*.val.jsonl", "--bs", "8", "--grad-accum", "4",
          "--max-state-tokens", "2048", "--packing", "off", "--length-grouping", "on",
          "--dtype", "auto", "--ple-cpu", "1", "--grad-ckpt", "--torch-compile", "off",
          "--eval-max", str(EVAL_MAX), "--keep-resume", "2", *SYNC]
ROOT = f"{OUT}/smoke" if PROFILE == "smoke" else OUT
ALGO = {"sft": ["--algo", "sft", "--lr", "1e-4", "--head-lr", "1e-3"],
        "rl": ["--algo", "exact", "--reward", "brier", "--init", f"{ROOT}/sft/best",
               "--lr", "2e-5", "--head-lr", "1e-4", "--lam-kl", "0.02", "--lam-cal", "0.5"]}


def earlier_outputs(stage):
    """<stage> dirs of attached earlier kernel outputs (read-only under /kaggle/input)."""
    return [d for d in glob.glob(f"/kaggle/input/**/out/{stage}", recursive=True) if os.path.isdir(d)]


def restore(stage):
    """A new session of a killed stage: copy the newest attached output of `stage` (resume/,
    best/, last/, logs, DONE.json) into /kaggle/working/out/<stage>, unless one is there."""
    dst = f"{ROOT}/{stage}"
    if os.path.exists(f"{dst}/DONE.json") or rs.find_latest(f"{dst}/resume"):
        return
    cands = [(rs.read_meta(rs.find_latest(f"{d}/resume")) or {}, os.path.exists(f"{d}/DONE.json"), d)
             for d in earlier_outputs(stage)]
    if not cands:
        print(f"=== {stage}: no earlier output attached, fresh start ===", flush=True)
        return
    meta, done, src = max(cands, key=lambda c: (c[1], c[0].get("step", -1)))
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"=== {stage}: restored {src} (done={done}, resume step {meta.get('step')}) ===", flush=True)


def need_sft_best():
    best = f"{ROOT}/sft/best"
    if not os.path.exists(f"{best}/heads.pt"):
        srcs = [d for d in earlier_outputs("sft") if os.path.exists(f"{d}/best/heads.pt")]
        if not srcs:
            raise SystemExit("full_rl needs the SFT best/: attach the openjev-prism-sft output "
                             "(build.sh does) or run full_sft first")
        src = max(srcs, key=lambda d: os.path.exists(f"{d}/DONE.json"))
        for f in ("best", "DONE.json", "log.jsonl", "train_log.jsonl"):
            p = f"{src}/{f}"
            if os.path.isdir(p):
                shutil.copytree(p, f"{ROOT}/sft/{f}", dirs_exist_ok=True)
            elif os.path.exists(p):
                shutil.copy(p, f"{ROOT}/sft/{f}")
        print(f"=== SFT best from {src} ===", flush=True)
    if not os.path.exists(f"{ROOT}/sft/DONE.json"):
        if os.environ.get("ALLOW_UNFINISHED_SFT") != "1":
            raise SystemExit("the SFT stage has no DONE.json (stopped before its last step): resume "
                             "full_sft first, or bake ALLOW_UNFINISHED_SFT=1 to start RL from its best/")
        print("=== WARNING: SFT unfinished (ALLOW_UNFINISHED_SFT=1); RL starts from its best/ ===", flush=True)


def run(stage, steps, eval_every, save_every, *extra):
    left = SESSION_MINUTES - (time.time() - T0) / 60
    if left < 20:
        raise SystemExit(f"{stage}: only {left:.0f} min left of SESSION_MINUTES={SESSION_MINUTES:.0f}")
    cmd = [*LAUNCH, "train_prism.py", *common, *ALGO[stage], "--steps", str(steps),
           "--eval-every", str(eval_every), "--save-every", str(save_every),
           "--save-dir", f"{ROOT}/{stage}", "--resume", "auto", "--max-minutes", f"{left:.1f}", *extra]
    print(f"=== {stage}: {' '.join(cmd[len(LAUNCH):])} ===", flush=True)
    subprocess.check_call(cmd, env=ENV)


def report(stage):
    """s/step and GradScaler skips (the scale went down) from train_log.jsonl."""
    import json, statistics
    p = f"{ROOT}/{stage}/train_log.jsonl"
    if not os.path.exists(p):
        return
    rows = {r["step"]: r for r in map(json.loads, open(p))}
    rows = [rows[k] for k in sorted(rows)]
    dt = [b["t"] - a["t"] for a, b in zip(rows, rows[1:]) if b["t"] > a["t"]]
    skips = sum(1 for a, b in zip(rows, rows[1:]) if a["scale"] and b["scale"] and b["scale"] < a["scale"])
    print(json.dumps({"report": stage, "steps": len(rows), "s_per_step_median":
                      round(statistics.median(dt), 2) if dt else None, "scaler_skips": skips,
                      "done": os.path.exists(f"{ROOT}/{stage}/DONE.json")}), flush=True)


print(f"=== PROFILE {PROFILE}: {STAGES[PROFILE]}, session {SESSION_MINUTES:.0f} min ===", flush=True)
for stage, steps, eval_every, save_every in STAGES[PROFILE]:
    if PROFILE != "smoke":
        restore(stage)
    if stage == "rl":
        need_sft_best()
    if PROFILE == "smoke" and stage == "rl":
        run(stage, steps, eval_every, save_every, "--session-steps", str(steps // 2))
        assert not os.path.exists(f"{ROOT}/rl/DONE.json"), "smoke: RL should have stopped half way"
    run(stage, steps, eval_every, save_every)
    report(stage)
    if not os.path.exists(f"{ROOT}/{stage}/DONE.json"):
        print(f"=== {stage} STOPPED before the last step (time budget): push a new version with this "
              f"output attached to continue ===", flush=True)
        sys.exit(0)
if PROFILE == "smoke":
    import json
    log = [json.loads(x) for x in open(f"{ROOT}/rl/train_log.jsonl")]
    assert sorted({r["step"] for r in log}) == list(range(10)), "smoke: RL steps missing after resume"
    print(json.dumps({"smoke_rl_kl_step0": log[0].get("kl")}), flush=True)
print("=== DONE ===", flush=True)
