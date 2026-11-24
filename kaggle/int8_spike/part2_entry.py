"""Kaggle kernel for int8 spike Part 2 (staged by build_part2.sh; SRC_B64 is baked in there).
1) int8_model.py --real: E2B on 1xT4, int8 (torch._int_mm) on the 105 frozen MLP base linears,
   toggled in place vs fp16 on the <=512 bench bucket (mb 8, grad ckpt, LoRA unchanged).
2) bench_int8.py --sub torchao at M 4096/8192 with torchao 0.18 (--no-deps), incl. module_swap.
Data: the read-only dataset piyushsar/openjev-prism-data (record ids of bench_lengths.json)."""
import base64, glob, io, os, subprocess, sys, zipfile

SRC_B64 = globals().get("SRC_B64")
work = "/kaggle/working/src"
zipfile.ZipFile(io.BytesIO(base64.b64decode(SRC_B64))).extractall(work)
data = os.path.dirname(glob.glob("/kaggle/input/**/banking77.train.jsonl", recursive=True)[0])
py = sys.executable
subprocess.check_call([py, "-m", "pip", "install", "-q", "-U", "transformers>=5.17", "peft", "safetensors",
                       "tensorboard", "optuna", "pyyaml", "bitsandbytes==0.50.2"])
subprocess.call([py, "-m", "pip", "uninstall", "-y", "-q", "torchao"])   # as kaggle/prism/entry.py
subprocess.check_call([py, "-c", "import peft, torch, transformers; print('env ok', torch.__version__, "
                       "transformers.__version__, peft.__version__)"])
os.chdir(work)
subprocess.run(["nvidia-smi"])
r = subprocess.run([py, "int8_model.py", "--real", "--data", data, "--out", "/kaggle/working/int8_model.json"])
print("=== PART2 rc", r.returncode, "===", flush=True)
subprocess.run([py, "-m", "pip", "install", "-q", "--no-deps", "torchao==0.18.0"])
r = subprocess.run([py, "bench_int8.py", "--sub", "torchao", "--ms", "4096,8192",
                    "--json-out", "/kaggle/working/torchao_swap.json"])
print("=== TORCHAO rc", r.returncode, "===", flush=True)
print("=== INT8 PART2 DONE ===", flush=True)
