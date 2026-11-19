"""Kaggle GPU kernel: SFT then RL (exact policy gradient, Brier reward) for
OpenJev-Prism on Gemma 4 E2B, or (PROFILE=optuna) a batch of speed trials with
bench_trials.py. Outputs go to /kaggle/working/out."""
import glob, os, shutil, subprocess, sys, zipfile

work = "/kaggle/working/src"
# Kaggle may unpack a .zip inside a dataset, so accept the zip or its unpacked folder.
zips = glob.glob("/kaggle/input/**/openjev_src.zip", recursive=True)
if zips and os.path.isfile(zips[0]):
    zipfile.ZipFile(zips[0]).extractall(work)
else:
    script = glob.glob("/kaggle/input/**/train_prism.py", recursive=True)
    if not script:
        raise SystemExit("openjev_src not found; inputs: "
                         + str(glob.glob("/kaggle/input/**", recursive=True)[:50]))
    shutil.copytree(os.path.dirname(script[0]), work, dirs_exist_ok=True)
data = os.path.dirname(glob.glob("/kaggle/input/**/banking77.train.jsonl", recursive=True)[0])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U",
                       "transformers>=5.17", "peft", "safetensors", "tensorboard", "optuna", "pyyaml"])
# The Kaggle image ships torchao 0.10; new peft rejects torchao < 0.16 at LoRA
# injection. We do not use torchao, so remove it.
subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
subprocess.check_call([sys.executable, "-c",
                       "import peft, torch, transformers; from peft.import_utils import "
                       "is_torchao_available as f; assert not f(); "
                       "print('env ok', torch.__version__, transformers.__version__, peft.__version__)"])
os.chdir(work)
OUT = "/kaggle/working/out"
os.makedirs(OUT, exist_ok=True)

# PROFILE "short": a ~1.5 h end-to-end check (SFT then RL) with fast evals.
# PROFILE "full": the long run with a preflight first.
# PROFILE "optuna": speed trials from trials.json (tools/optuna_loop.py propose), no training run.
# A pushed script kernel gets no env vars, so `PROFILE=optuna build.sh` bakes PROFILE (and
# the trials, as OPTUNA_TRIALS) into the staged copy of this file.
PROFILE = os.environ.get("PROFILE", "short")

if PROFILE == "optuna":
    if globals().get("OPTUNA_TRIALS"):            # baked by build.sh: newer than the bundle copy
        with open("trials.json", "w") as f:
            f.write(OPTUNA_TRIALS)
    print("=== OPTUNA trials ===\n" + open("trials.json").read(), flush=True)
    subprocess.check_call([sys.executable, "bench_trials.py", "--trials", "trials.json",
                           "--trial-seconds", os.environ.get("TRIAL_SECONDS", "50"),
                           "--out", f"{OUT}/optuna_results.json",
                           "--model", "google/gemma-4-E2B-it", "--train", f"{data}/*.train.jsonl",
                           "--bs", "4", "--max-state-tokens", "1024", "--grad-ckpt"])
    print("=== OPTUNA DONE ===", flush=True)
    sys.exit(0)

if PROFILE == "short":
    STEPS_SFT, STEPS_RL, ACCUM, EVAL_EVERY, EVAL_MAX = 60, 20, 2, 30, 200
else:
    STEPS_SFT, STEPS_RL, ACCUM, EVAL_EVERY, EVAL_MAX = 2000, 500, 8, 250, 800
common = ["--model", "google/gemma-4-E2B-it", "--train", f"{data}/*.train.jsonl",
          "--val", f"{data}/*.val.jsonl", "--bs", "4", "--grad-accum", str(ACCUM),
          "--max-state-tokens", "1024", "--grad-ckpt", "--eval-every", str(EVAL_EVERY),
          "--eval-max", str(EVAL_MAX)]
run = lambda extra: subprocess.check_call([sys.executable, "train_prism.py", *common, *extra])

if PROFILE != "short":
    # Preflight: every code path on the real model before the long run.
    print("=== PREFLIGHT ===", flush=True)
    pf = ["--steps", "2", "--grad-accum", "1", "--eval-every", "2", "--eval-max", "16"]
    run(["--algo", "sft", *pf, "--save-dir", "pf/sft"])
    run(["--algo", "exact", "--reward", "brier", "--init", "pf/sft/best", *pf, "--save-dir", "pf/rl"])
    run(["--algo", "grpo", "--init", "pf/sft/best", *pf, "--save-dir", "pf/grpo"])
    shutil.rmtree("pf")
    print("=== PREFLIGHT OK ===", flush=True)

print(f"=== PROFILE {PROFILE}: SFT {STEPS_SFT} steps, RL {STEPS_RL} steps ===", flush=True)
run(["--algo", "sft", "--steps", str(STEPS_SFT), "--lr", "1e-4", "--save-dir", "ckpt/sft",
     "--tb-dir", f"{OUT}/tb/sft"])
shutil.copytree("ckpt/sft", f"{OUT}/sft", dirs_exist_ok=True)   # keep SFT even if RL fails
run(["--algo", "exact", "--reward", "brier", "--init", "ckpt/sft/best",
     "--steps", str(STEPS_RL), "--lr", "2e-5", "--head-lr", "1e-4", "--save-dir", "ckpt/rl",
     "--tb-dir", f"{OUT}/tb/rl"])
shutil.copytree("ckpt/rl", f"{OUT}/rl", dirs_exist_ok=True)
print("=== DONE ===", flush=True)
