"""Kaggle GPU kernel: SFT then RL (exact policy gradient, Brier reward) for
OpenJev-Prism on Gemma 4 E2B. Outputs go to /kaggle/working/out."""
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
                       "transformers>=5.17", "peft", "safetensors"])
# The Kaggle image ships torchao 0.10; new peft rejects torchao < 0.16 at LoRA
# injection. We do not use torchao, so remove it.
subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
subprocess.check_call([sys.executable, "-c",
                       "import peft, torch, transformers; from peft.import_utils import "
                       "is_torchao_available as f; assert not f(); "
                       "print('env ok', torch.__version__, transformers.__version__, peft.__version__)"])
os.chdir(work)

STEPS_SFT = int(os.environ.get("STEPS_SFT", 2000))
STEPS_RL = int(os.environ.get("STEPS_RL", 500))
common = ["--model", "google/gemma-4-E2B-it", "--train", f"{data}/*.train.jsonl",
          "--val", f"{data}/*.val.jsonl", "--bs", "4", "--grad-accum", "8",
          "--max-state-tokens", "1024", "--grad-ckpt", "--eval-every", "250",
          "--eval-max", "800"]
run = lambda extra: subprocess.check_call([sys.executable, "train_prism.py", *common, *extra])

# Preflight: every code path (load, SFT step, eval, save, RL from checkpoint,
# GRPO) on the real model, in a few minutes, before the long run.
print("=== PREFLIGHT ===", flush=True)
pf = ["--steps", "2", "--grad-accum", "1", "--eval-every", "2", "--eval-max", "16"]
run(["--algo", "sft", *pf, "--save-dir", "pf/sft"])
run(["--algo", "exact", "--reward", "brier", "--init", "pf/sft/best", *pf, "--save-dir", "pf/rl"])
run(["--algo", "grpo", "--init", "pf/sft/best", *pf, "--save-dir", "pf/grpo"])
shutil.rmtree("pf")
print("=== PREFLIGHT OK ===", flush=True)

run(["--algo", "sft", "--steps", str(STEPS_SFT), "--lr", "1e-4", "--save-dir", "ckpt/sft"])
run(["--algo", "exact", "--reward", "brier", "--init", "ckpt/sft/best",
     "--steps", str(STEPS_RL), "--lr", "2e-5", "--head-lr", "1e-4", "--save-dir", "ckpt/rl"])
shutil.copytree("ckpt", "/kaggle/working/out", dirs_exist_ok=True)
