"""Unsloth spike (Kaggle T4, sm75, fp16): does Unsloth make OUR training step faster, and can
it keep the Prism model (Gemma 4 E2B text decoder + PLE + Prism mask dict + flag head +
template tokens + LoRA)?

One stage per process, because `import unsloth` patches transformers globally:
  --stage base       our path (train_prism.build: fp16, ple_cpu, LoRA r16/a32, HF grad ckpt).
                     Saves the flag head, template rows and the batch-0 flag logits for parity.
  --stage unsloth16  FastModel.from_pretrained(text_only, 16-bit) + FastModel.get_peft_model
                     (same LoRA targets / rank, use_gradient_checkpointing="unsloth"), wrapped
                     in PrismOpenJev. Same micro-batches, same flag head, same template rows.
  --stage unsloth4   the same with load_in_4bit=True (memory only; last priority).

Work per stage (micro-batches: the first 32 records of the <=512 bucket of bench_lengths.json,
4 micro-batches of 8, prism_collate pad_to_multiple_of=8, max_state_tokens 2048):
  causal_nopad  raw decoder cost: every row cut to the shortest row of its micro-batch, no
                attention mask (plain causal + sliding band, so Unsloth's banded sliding SDPA
                can engage), loss = mean(flag_head(h)^2). No LM head: our model has none, so
                Unsloth's fused CE is not part of our step.
  prism         the real step: flag_logits with the {full_attention, sliding_attention} dict,
                Prism position ids, sft_loss.
Each: 1 warm-up pass (compiles), then 3 timed passes of the 4 micro-batches (fwd + bwd,
GradScaler-scaled loss / 4, no optimizer step), per-step wall ms (synchronised) and CUDA-event
ms per micro-batch, peak GB above nothing (max_memory_allocated), grads finite.

  python bench_unsloth.py --stage base --data DIR --src SRC
  python bench_unsloth.py --cpu-check          # base-stage helpers on the tiny Gemma 4 (CPU)
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import traceback

import torch

OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
MODEL = "google/gemma-4-E2B-it"
BUCKET, GB, MB = 512, 32, 8
REPS = 3
HEAD_FILE = os.path.join(OUT, "unsloth_head.pt")


def tb():
    return traceback.format_exc()[-4000:]


class Res(dict):
    """Result dict written to disk after every key."""

    def __init__(self, path):
        super().__init__(t0=time.time())
        self.path = path

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        print(f"RES {k} = {json.dumps(v, default=str)[:3000]}", flush=True)
        with open(self.path, "w") as f:
            json.dump(self, f, indent=1, default=str)


def step(res, key, fn):
    """Run fn() -> value, store it (or the error) under res[key]. Returns value or None."""
    t = time.perf_counter()
    try:
        v = fn()
        res[key] = v
        return v
    except Exception as e:
        res[key] = {"error": f"{type(e).__name__}: {str(e)[:1500]}", "tb": tb(),
                    "s": time.perf_counter() - t}
        print(res[key]["tb"], flush=True)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return None


def env_info():
    import peft
    import transformers
    out = {"torch": torch.__version__, "cuda": torch.version.cuda, "python": sys.version.split()[0],
           "transformers": transformers.__version__, "peft": peft.__version__}
    if torch.cuda.is_available():
        out |= {"gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability()}
    for m in ("triton", "unsloth", "unsloth_zoo", "bitsandbytes", "torchao", "xformers", "trl"):
        try:
            mod = sys.modules.get(m) or __import__(m)
            out[m] = getattr(mod, "__version__", "?")
        except Exception as e:
            out[m] = f"n/a ({type(e).__name__})"
    return out


def mem_gb():
    return round(torch.cuda.memory_allocated() / 2**30, 3) if torch.cuda.is_available() else 0.0


# ------------------------------------------------- data, losses

def load_bench(data_dir):
    import bench_trials as bt
    own, targs = bt.parse(["--trials", "none.json", "--bench-lengths", "bench_lengths.json",
                           "--model", MODEL, "--train", f"{data_dir}/*.train.jsonl",
                           "--bs", "4", "--max-state-tokens", "2048", "--grad-ckpt"])
    data, _probe, meta = bt.load_data(own, targs)
    return targs, data, {k: v for k, v in meta.items() if k != "stems"}


def make_batches(tok, data, device):
    from openjev.prism import prism_collate
    recs = data.buckets[BUCKET][:GB]
    bs = [prism_collate(tok, recs[i:i + MB], data.max_state_tokens, pad_to_multiple_of=8)
          for i in range(0, GB, MB)]
    h = hashlib.sha256()
    for b in bs:
        h.update(b.input_ids.numpy().tobytes())
    info = {"shapes": [list(b.input_ids.shape) for b in bs],
            "real_tokens": [int(b.attention_mask.sum()) for b in bs],
            "min_len": [int(b.attention_mask.sum(1).min()) for b in bs],
            "flags": [int(b.flag_index.shape[0]) for b in bs],
            "ids_sha": h.hexdigest()[:16]}
    return [b.to(device) for b in bs], info


def _ple(model, ids, kw):
    text = model._text_backbone()
    v_ple = getattr(text.config, "vocab_size_per_layer_input", None)
    if v_ple and hasattr(text, "get_per_layer_inputs"):
        from openjev.prism import host_to_device
        pad = text.config.pad_token_id or 0
        ple_ids = torch.where(ids >= v_ple, torch.full_like(ids, pad), ids)
        dev = text.embed_tokens_per_layer.weight.device
        kw["per_layer_inputs"] = host_to_device(text.get_per_layer_inputs(ple_ids.to(dev), None), ids.device)
    return kw


def loss_causal_nopad(model, b):
    L = int(b.attention_mask.sum(1).min())
    ids = b.input_ids[:, :L]
    kw = _ple(model, ids, {"inputs_embeds": model.embed(ids)})
    hs = model.backbone(**kw, use_cache=False).last_hidden_state
    return model.flag_head(hs.float()).pow(2).mean()


def loss_prism(model, b):
    import train_prism as tp
    return tp.sft_loss(model.flag_logits(b), b)


LOSSES = {"causal_nopad": loss_causal_nopad, "prism": loss_prism}


def time_fn(model, batches, loss_fn, use_autocast=True, reps=REPS):
    import train_prism as tp
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = torch.amp.GradScaler(dev, init_scale=2.0 ** 16, enabled=dev == "cuda")
    ac = (lambda: tp.autocast(dev, model)) if use_autocast else contextlib.nullcontext
    losses = []

    def one(b):
        with ac():
            loss = loss_fn(model, b)
        scaler.scale(loss / len(batches)).backward()
        losses.append(float(loss.detach()))

    def sync():
        if dev == "cuda":
            torch.cuda.synchronize()

    model.train()
    model.zero_grad(set_to_none=True)
    sync()
    t = time.perf_counter()
    for b in batches:
        one(b)
    sync()
    r = {"warm_s": round(time.perf_counter() - t, 2)}
    model.zero_grad(set_to_none=True)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    steps, mbs = [], []
    for rep in range(reps):
        sync()
        t = time.perf_counter()
        for b in batches:
            if dev == "cuda":
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                one(b)
                e1.record()
                e1.synchronize()
                mbs.append(e0.elapsed_time(e1))
            else:
                one(b)
        sync()
        steps.append((time.perf_counter() - t) * 1e3)
        if rep < reps - 1:
            model.zero_grad(set_to_none=True)
    g = [(n, p.grad) for n, p in model.named_parameters() if p.requires_grad]
    with_grad = [(n, x) for n, x in g if x is not None]
    r.update(step_ms_median=round(statistics.median(steps), 1), step_ms=[round(x, 1) for x in steps],
             mb_ms_median=round(statistics.median(mbs), 1) if mbs else None,
             mb_ms=[round(x, 1) for x in mbs],
             peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 3) if dev == "cuda" else None,
             losses_first4=[round(x, 5) for x in losses[:4]], loss_last=round(losses[-1], 5),
             trainable=len(g), with_grad=len(with_grad),
             nonzero_grad=sum(int(x.abs().sum() > 0) for _, x in with_grad),
             grads_finite=all(bool(torch.isfinite(x).all()) for _, x in with_grad),
             scaler_scale=scaler.get_scale() if scaler.is_enabled() else None)
    model.zero_grad(set_to_none=True)
    return r


@torch.no_grad()
def flag_logits_eval(model, b, use_autocast=True):
    import train_prism as tp
    dev = b.input_ids.device.type
    model.eval()
    with (tp.autocast(dev, model) if use_autocast else contextlib.nullcontext()):
        out = model.flag_logits(b).float().cpu()
    model.train()
    return out


def compare(l, ref, b):
    import train_prism as tp
    d = (l - ref).abs()
    agree, n = 0, 0
    for (a, _t, _), (r, _t2, _) in zip(tp.question_views(l, b), tp.question_views(ref, b)):
        agree += int(a.argmax() == r.argmax())
        n += 1
    return {"n_flags": int(ref.numel()), "max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "rel_l2": float((l - ref).norm() / ref.norm().clamp_min(1e-12)),
            "ref_std": float(ref.std()), "argmax_agree": f"{agree}/{n}",
            "loss": float(tp.sft_loss(l, b)), "loss_ref": float(tp.sft_loss(ref, b)),
            "all_finite": bool(torch.isfinite(l).all())}


def lora_count(model):
    return sum(1 for n, _ in model.named_modules() if n.endswith("lora_A"))


def gc_info(text):
    f = getattr(text, "_gradient_checkpointing_func", None)
    return {"is_gradient_checkpointing": bool(getattr(text, "is_gradient_checkpointing", False)),
            "func": getattr(f, "__qualname__", None) or repr(f)[:200],
            "func_module": getattr(f, "__module__", None)}


# ------------------------------------------------- stage: base (our path)

def stage_base(a):
    import bench_trials as bt
    import train_prism as tp
    res = Res(os.path.join(OUT, "unsloth_base.json"))
    res["env"] = env_info()
    targs, data, meta = load_bench(a.data)
    res["bench"] = meta
    random.seed(targs.seed)
    torch.manual_seed(targs.seed)
    t = time.perf_counter()
    model, tok, dtype = tp.build(bt.build_args(targs, {"ple_cpu": 1, "dtype": "fp16", "quant": "none"}), "cuda")
    model.train()
    bt.apply_params(model, {"grad_ckpt": "on", "torch_threads": 4}, targs, torch.get_num_threads())
    text = model._text_backbone()
    res["load"] = {"s": round(time.perf_counter() - t, 1), "mem_gb": mem_gb(), "dtype": str(dtype),
                   "lora_r": targs.lora_r, "lora_alpha": targs.lora_alpha, "lora_modules": lora_count(model),
                   "attn_impl": text.config._attn_implementation, "gc": gc_info(text),
                   "ple_device": str(text.embed_tokens_per_layer.weight.device)}
    batches, binfo = make_batches(tok, data, "cuda")
    res["batches"] = binfo
    ids = model.template_ids.tolist()
    emb = model.get_input_embeddings().weight
    ref = flag_logits_eval(model, batches[0])
    torch.save({"flag_head": {k: v.cpu() for k, v in model.flag_head.state_dict().items()},
                "template_ids": ids, "template_rows": emb[ids].detach().cpu(),
                "logits0": ref, "ids_sha": binfo["ids_sha"], "vocab": len(tok)}, HEAD_FILE)
    res["parity_ref"] = {"n": ref.numel(), "mean": float(ref.mean()), "std": float(ref.std()),
                         "loss": float(tp.sft_loss(ref, batches[0]))}
    for name in ("causal_nopad", "prism"):
        bt.free("cuda")
        step(res, name, lambda: time_fn(model, batches, LOSSES[name]))
    return res


# ------------------------------------------------- stage: unsloth

def stage_unsloth(a, four_bit):
    os.environ.setdefault("UNSLOTH_ENABLE_LOGGING", "1")
    res = Res(os.path.join(OUT, f"unsloth_{'4bit' if four_bit else '16bit'}.json"))
    t = time.perf_counter()
    try:
        import unsloth  # noqa: F401  (must be first: patches transformers / peft)
        from unsloth import FastModel
    except Exception as e:
        res["import"] = {"error": f"{type(e).__name__}: {str(e)[:1500]}", "tb": tb()}
        return res
    res["import"] = {"s": round(time.perf_counter() - t, 1)}
    res["env"] = env_info()
    import bench_trials as bt
    import train_prism as tp
    from transformers import AutoTokenizer
    from openjev import add_special_tokens
    from openjev.prism import PrismOpenJev
    from openjev.schema import SPECIAL_TOKENS

    targs, data, meta = load_bench(a.data)
    res["bench"] = meta
    head = torch.load(HEAD_FILE, map_location="cpu") if os.path.exists(HEAD_FILE) else None
    random.seed(targs.seed)
    torch.manual_seed(targs.seed)
    st = {}

    def load():
        t = time.perf_counter()
        m, _proc = FastModel.from_pretrained(MODEL, max_seq_length=2048, dtype=torch.float16,
                                             load_in_4bit=four_bit, full_finetuning=False, text_only=True)
        st["hf"] = m
        text = next(x for x in m.modules() if type(x).__name__ == "Gemma4TextModel")
        st["text"] = text
        q = text.layers[0].self_attn.q_proj
        return {"s": round(time.perf_counter() - t, 1), "class": type(m).__name__, "mem_gb": mem_gb(),
                "embed_dtype": str(text.embed_tokens.weight.dtype), "q_proj": type(q).__name__,
                "q_dtype": str(getattr(q, "weight", torch.empty(0)).dtype),
                "norm_dtype": str(text.norm.weight.dtype) if hasattr(text.norm, "weight") else None,
                "attn_impl": text.config._attn_implementation,
                "forced_float32": os.environ.get("UNSLOTH_FORCE_FLOAT32"),
                "fp16_sdpa_env": os.environ.get("UNSLOTH_GEMMA4_FP16_SDPA", "default(1)"),
                "ple_device": str(text.embed_tokens_per_layer.weight.device),
                "towers": [n for n, _ in m.named_children()]}
    if step(res, "load", load) is None:
        return res

    def wrap():
        model = PrismOpenJev(st["text"])
        tok = AutoTokenizer.from_pretrained(MODEL)
        add_special_tokens(tok, model)
        rows = model.get_input_embeddings().weight.shape[0]
        if len(tok) > rows:
            model.resize_token_embeddings(-(-len(tok) // 128) * 128)
        st["model"], st["tok"] = model, tok
        return {"vocab": len(tok), "rows": model.get_input_embeddings().weight.shape[0]}
    if step(res, "wrap", wrap) is None:
        return res

    def peft():
        t = time.perf_counter()
        hf = FastModel.get_peft_model(st["hf"], r=targs.lora_r, lora_alpha=targs.lora_alpha, lora_dropout=0,
                                      target_modules=list(tp.LORA_TARGETS), bias="none",
                                      use_gradient_checkpointing="unsloth", random_state=targs.seed)
        st["hf"] = hf
        model, text = st["model"], st["text"]
        same = any(x is text for x in hf.modules())
        model.set_template_ids(st["tok"].convert_tokens_to_ids(SPECIAL_TOKENS))
        dtypes = sorted({str(p.dtype) for n, p in model.named_parameters() if "lora_" in n})
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        if head is not None:
            model.flag_head.load_state_dict(head["flag_head"])
            with torch.no_grad():
                w = model.get_input_embeddings().weight
                w[head["template_ids"]] = head["template_rows"].to(w.device, w.dtype)
        tp.place(model, "cuda", True)
        return {"s": round(time.perf_counter() - t, 1), "text_module_kept": same,
                "lora_modules": lora_count(model), "lora_dtype_before_upcast": dtypes,
                "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "gc": gc_info(text), "mem_gb": mem_gb(),
                "ple_device": str(text.embed_tokens_per_layer.weight.device),
                "head_restored": head is not None}
    if step(res, "peft", peft) is None:
        return res
    model, tok = st["model"], st["tok"]
    batches, binfo = make_batches(tok, data, "cuda")
    binfo["same_ids_as_base"] = head is not None and head["ids_sha"] == binfo["ids_sha"]
    res["batches"] = binfo

    def engaged():
        try:
            from unsloth_zoo.temporary_patches import gemma4_flash_sliding as fs
            return {"causal_engaged": fs._CAUSAL_ENGAGED[0], "enabled": fs._enabled()}
        except Exception as e:
            return f"n/a ({type(e).__name__}: {e})"

    # 1) raw decoder speed (plain causal, no mask, no padding)
    res["engaged_before"] = engaged()
    step(res, "causal_nopad", lambda: time_fn(model, batches, loss_causal_nopad))
    res["engaged_after_causal"] = engaged()
    # 2) our masks: parity first (LoRA B = 0, same head + template rows), then timing
    if head is not None:
        step(res, "parity", lambda: compare(flag_logits_eval(model, batches[0]), head["logits0"], batches[0]))
    step(res, "prism", lambda: time_fn(model, batches, loss_prism))
    res["engaged_after_prism"] = engaged()
    # 3) attribution (not the main result)
    step(res, "prism_no_autocast", lambda: time_fn(model, batches, loss_prism, use_autocast=False, reps=2))

    def hf_gc():
        tp.set_grad_ckpt(model, False)
        tp.set_grad_ckpt(model, True)
        info = gc_info(st["text"])
        return {"gc": info, **time_fn(model, batches, loss_prism, reps=2)}
    if not four_bit:
        step(res, "prism_gc_true", hf_gc)
    return res


# ------------------------------------------------- CPU check (tiny Gemma 4, base helpers)

def cpu_check():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Gemma4TextConfig, Gemma4TextModel
    import train_prism as tp
    from openjev import Schema, add_special_tokens
    from openjev.data import Record
    from openjev.prism import PrismOpenJev, prism_collate
    from openjev.schema import SPECIAL_TOKENS

    tok = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen3ForCausalLM")
    torch.manual_seed(0)
    cfg = Gemma4TextConfig(vocab_size=len(tok) + 64, hidden_size=64, intermediate_size=128,
                           num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                           head_dim=16, sliding_window=64, vocab_size_per_layer_input=100,
                           hidden_size_per_layer_input=8, num_kv_shared_layers=0,
                           layer_types=["sliding_attention", "sliding_attention", "full_attention"],
                           initializer_range=0.1)
    cfg._attn_implementation = "sdpa"
    model = PrismOpenJev(Gemma4TextModel(cfg))
    add_special_tokens(tok, model)
    model.set_template_ids(tok.convert_tokens_to_ids(SPECIAL_TOKENS))
    model.backbone = get_peft_model(model.backbone, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                                                               target_modules=list(tp.LORA_TARGETS)))
    tp.set_grad_ckpt(model, True)
    rng = random.Random(0)
    words = ["red", "blue", "green", "pink", "black", "white", "gold", "gray"]
    recs = []
    for i in range(16):
        opts = rng.sample(words, 4)
        y = rng.choice(opts)
        recs.append(Record(f"r{i}", "toy", "train", f"the answer is {y} " * rng.randint(2, 5),
                           Schema().choice("c", {o: "" for o in opts}, "pick"), [y]))
    bs = [prism_collate(tok, recs[i:i + 8]) for i in (0, 8)]
    out = {"lora_modules": lora_count(model), "gc": gc_info(model._text_backbone())}
    for name, fn in LOSSES.items():
        out[name] = time_fn(model, bs, fn, reps=1)
        assert out[name]["grads_finite"] and out[name]["nonzero_grad"] > 0, out[name]
    ref = flag_logits_eval(model, bs[0])
    out["self_parity"] = compare(flag_logits_eval(model, bs[0]), ref, bs[0])
    assert out["self_parity"]["max_abs"] == 0.0
    print("CPU_CHECK", json.dumps(out, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["base", "unsloth16", "unsloth4"])
    ap.add_argument("--data", default="")
    ap.add_argument("--cpu-check", action="store_true")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:0] = [here, os.path.abspath(os.path.join(here, "..", ".."))]   # Kaggle: flat; repo: root
    if a.cpu_check:
        return cpu_check()
    if torch.cuda.is_available():
        subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.max.sm,temperature.gpu,memory.used",
                        "--format=csv,noheader"])
    if a.stage == "base":
        stage_base(a)
    else:
        stage_unsloth(a, four_bit=a.stage == "unsloth4")
    print(f"=== STAGE {a.stage} DONE ===", flush=True)


if __name__ == "__main__":
    main()
