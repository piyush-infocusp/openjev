"""OpenJev Kaggle kernel entry point.

Runs on a Kaggle script kernel (T4/P100, no bf16 support -> training must use
fp16). This file is pushed as the kernel's code_file; it does not train the
model itself. Instead it:

  1. Prints diagnostics (nvidia-smi, torch/cuda info).
  2. Installs extra pip deps quietly.
  3. Locates the source bundle dataset (openjev_src.zip) under
     /kaggle/input/{{DATASET}}/ and unzips it to /kaggle/working/src.
  4. Reads run_args.txt from the bundle for the training command's arguments.
  5. Runs train.py with those args if present, else falls back to
     train_toy.py --model Qwen/Qwen3-0.6B --steps 200 so the whole
     pipeline can be smoke-tested end to end before train.py exists.
  6. Copies checkpoints/metrics produced under the src dir to
     /kaggle/working/out.
"""

import glob
import os
import shutil
import subprocess
import sys
import zipfile

DATASET_GLOB = "/kaggle/input/*/"
SRC_DIR = "/kaggle/working/src"
OUT_DIR = "/kaggle/working/out"
BUNDLE_NAME = "openjev_src.zip"


def run(cmd, **kwargs):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kwargs)


def print_diagnostics():
    print("=== nvidia-smi ===", flush=True)
    try:
        subprocess.run(["nvidia-smi"], check=False)
    except FileNotFoundError:
        print("nvidia-smi not found (no GPU?)", flush=True)

    print("=== torch / cuda info ===", flush=True)
    try:
        import torch
        print("torch:", torch.__version__)
        print("cuda available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("device:", torch.cuda.get_device_name(0))
            print("bf16 supported:", torch.cuda.is_bf16_supported())
    except ImportError:
        print("torch not importable yet (installed later)", flush=True)


def pip_install_quiet():
    run([sys.executable, "-m", "pip", "install", "-q", "transformers", "safetensors"])


def find_dataset_dir():
    for d in sorted(glob.glob(DATASET_GLOB)):
        if os.path.exists(os.path.join(d, BUNDLE_NAME)):
            return d
    raise FileNotFoundError(
        f"Could not find {BUNDLE_NAME} under /kaggle/input/*/ "
        "(expected the openjev source bundle dataset to be attached)."
    )


def unzip_bundle(dataset_dir):
    bundle_path = os.path.join(dataset_dir, BUNDLE_NAME)
    os.makedirs(SRC_DIR, exist_ok=True)
    with zipfile.ZipFile(bundle_path) as zf:
        zf.extractall(SRC_DIR)
    print(f"Unzipped {bundle_path} -> {SRC_DIR}", flush=True)


def read_run_args():
    args_path = os.path.join(SRC_DIR, "run_args.txt")
    if not os.path.exists(args_path):
        print(f"No run_args.txt found at {args_path}, using empty args.", flush=True)
        return []
    with open(args_path) as f:
        text = f.read().strip()
    return text.split() if text else []


def run_training():
    train_py = os.path.join(SRC_DIR, "train.py")
    train_toy_py = os.path.join(SRC_DIR, "train_toy.py")

    args = read_run_args()

    if os.path.exists(train_py):
        cmd = [sys.executable, train_py] + args
    elif os.path.exists(train_toy_py):
        print("train.py not found in bundle; falling back to train_toy.py "
              "smoke test.", flush=True)
        cmd = [sys.executable, train_toy_py,
               "--model", "Qwen/Qwen3-0.6B", "--steps", "200"]
    else:
        raise FileNotFoundError(
            f"Neither train.py nor train_toy.py found under {SRC_DIR}."
        )

    run(cmd, cwd=SRC_DIR)


def collect_outputs():
    os.makedirs(OUT_DIR, exist_ok=True)
    patterns = ["**/*.safetensors", "**/*checkpoint*", "**/*metrics*",
                "**/*.json", "**/*.log"]
    seen = set()
    for pattern in patterns:
        for path in glob.glob(os.path.join(SRC_DIR, pattern), recursive=True):
            if os.path.isfile(path) and path not in seen:
                seen.add(path)
                rel = os.path.relpath(path, SRC_DIR)
                dest = os.path.join(OUT_DIR, rel)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(path, dest)
    print(f"Copied {len(seen)} output file(s) to {OUT_DIR}", flush=True)


def main():
    print_diagnostics()
    pip_install_quiet()
    print_diagnostics()  # torch info again in case it was just installed

    dataset_dir = find_dataset_dir()
    unzip_bundle(dataset_dir)
    run_training()
    collect_outputs()


if __name__ == "__main__":
    main()
