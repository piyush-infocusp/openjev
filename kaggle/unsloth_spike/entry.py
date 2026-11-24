"""Kaggle kernel for the Unsloth spike (staged by build.sh; SRC_B64 is baked in there).
1) our env (transformers pinned to 5.17.0, the newest unsloth 2026.11.9 accepts) -> bench --stage base
2) install unsloth / unsloth_zoo 2026.11.9 with --no-deps + their deps under a torch/triton
   constraint (Kaggle's torch must stay), torchao 0.18 --no-deps (zoo imports it; peft needs >= 0.16)
3) bench --stage unsloth16, then --stage unsloth4 (each its own process: unsloth patches globally)
Data: the read-only dataset piyushsar/openjev-prism-data (record ids of bench_lengths.json)."""
import base64, glob, io, json, os, subprocess, sys, zipfile

SRC_B64 = globals().get("SRC_B64")
UNSLOTH = "2026.11.9"
work = "/kaggle/working/src"
zipfile.ZipFile(io.BytesIO(base64.b64decode(SRC_B64))).extractall(work)
data = os.path.dirname(glob.glob("/kaggle/input/**/banking77.train.jsonl", recursive=True)[0])
py = sys.executable
log = {}


def pip(*args, check=False):
    r = subprocess.run([py, "-m", "pip", "install", "-q", *args], capture_output=True, text=True)
    print("PIP", args, "rc", r.returncode, r.stdout[-1500:], r.stderr[-2500:], flush=True)
    if check and r.returncode:
        raise SystemExit(f"pip failed: {args}")
    return r.returncode


def versions():
    code = ("import importlib, json\nout = {}\nfor m in ['torch','torchvision','triton','transformers','peft',"
            "'trl','torchao','bitsandbytes','xformers','unsloth','unsloth_zoo','cut_cross_entropy']:\n"
            "    try:\n        out[m] = importlib.metadata.version(m.replace('_','-') if m!='unsloth_zoo' else m)\n"
            "    except Exception as e:\n        out[m] = None\nprint(json.dumps(out))")
    r = subprocess.run([py, "-c", "import importlib.metadata\n" + code], capture_output=True, text=True)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return {"error": r.stderr[-1000:]}


def stage(name, timeout=3000):
    try:
        r = subprocess.run([py, "bench_unsloth.py", "--stage", name, "--data", data], timeout=timeout)
        rc = r.returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    print(f"=== {name} rc {rc} ===", flush=True)
    log[f"rc_{name}"] = rc
    json.dump(log, open("/kaggle/working/unsloth_env.json", "w"), indent=1)


log["kaggle_versions"] = versions()
pip("transformers==5.17.0", "peft", "safetensors", "pyyaml", check=True)
subprocess.call([py, "-m", "pip", "uninstall", "-y", "-q", "torchao"])   # as kaggle/prism/entry.py
log["base_versions"] = versions()
print("BASE_VERSIONS", log["base_versions"], flush=True)
os.chdir(work)
subprocess.run(["nvidia-smi"])
stage("base")

# constraint file: keep Kaggle's torch / torchvision / torchaudio / triton and our transformers
fr = subprocess.run([py, "-m", "pip", "freeze"], capture_output=True, text=True).stdout.splitlines()
keep = [l for l in fr if l.split("==")[0].lower() in ("torch", "torchvision", "torchaudio", "triton",
                                                      "transformers", "numpy")]
open("/kaggle/working/cons.txt", "w").write("\n".join(keep) + "\n")
log["constraints"] = keep
log["deps_rc"] = pip("-c", "/kaggle/working/cons.txt", "tyro", "hf_transfer", "cut_cross_entropy", "msgspec",
                     "einops", "trl<=1.15.0", "datasets<5", "sentencepiece", "structlog", "json5",
                     "nest-asyncio", "typer<0.29", "diffusers", "accelerate", "bitsandbytes==0.50.2",
                     "certifi>=2026.10.19")
log["unsloth_rc"] = pip("--no-deps", f"unsloth=={UNSLOTH}", f"unsloth_zoo=={UNSLOTH}", "torchao==0.18.0")
log["unsloth_versions"] = versions()
log["torch_unchanged"] = log["unsloth_versions"].get("torch") == log["base_versions"].get("torch")
print("UNSLOTH_VERSIONS", log["unsloth_versions"], "torch_unchanged", log["torch_unchanged"], flush=True)
r = subprocess.run([py, "-c", "import unsloth, unsloth_zoo; print('unsloth ok', unsloth.__version__)"],
                   capture_output=True, text=True, env={**os.environ, "UNSLOTH_ENABLE_LOGGING": "1"})
log["import_check"] = {"rc": r.returncode, "out": r.stdout[-3000:], "err": r.stderr[-5000:]}
print("IMPORT_CHECK", log["import_check"], flush=True)
json.dump(log, open("/kaggle/working/unsloth_env.json", "w"), indent=1)
stage("unsloth16")
stage("unsloth4", timeout=2400)
print("=== UNSLOTH SPIKE DONE ===", flush=True)
