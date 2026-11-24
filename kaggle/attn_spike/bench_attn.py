"""Attention-kernel spike (Kaggle T4, sm75, fp16): which fused / memory-efficient attention
kernels accept the Prism mask at Gemma 4 E2B shapes, and how fast are they compared with
math attention and with the current path?

No model load. Random q/k/v, batch 8, 8 query heads / 1 KV head (repeated to 8 heads, as
transformers' sdpa path does when a mask is given), head_dim 256 (28 sliding layers, sliding
mask, window 512) and 512 (7 global layers, full mask), seq 128..2048, fp16, requires_grad.
The mask is built with a verbatim copy of openjev.prism.layout + prism_mask (commit b71f4f0)
on synthetic requests (shared block, questions with 2-10 options sharing a start position, one
flag per option, right padding), so this file runs without the OpenJev bundle.

Per candidate: forward + backward time (CUDA events, 3 warm-up, median of 10), peak memory
above the inputs, max abs diff of out / dq / dk / dv vs an fp32 math reference.

  python bench_attn.py                      # full run (Kaggle kernel)
  python bench_attn.py --smoke              # tiny CPU run (syntax / mask check)
  python bench_attn.py --report res.json    # markdown table + end-to-end estimate
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import warnings

import torch
import torch.nn.functional as F

HEADS, KV_HEADS, WINDOW = 8, 1, 512
OPT, FLAG, TOK = 1, 2, 3
SHARED = 0
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."


# ------------------------------------------------- verbatim copies from openjev/prism.py (b71f4f0)

def layout(ids, readouts, opt_id, flag_id):
    T = len(ids)
    seg = [SHARED] * T
    qix = [-1] * T
    flag = [False] * T
    spans = []
    for qi, q in enumerate(readouts):
        for f in q["flags"]:
            s = f
            while ids[s] != opt_id:
                s -= 1
            spans.append((qi, s, f))
    for n, (qi, s, f) in enumerate(spans, start=1):
        for t in range(s, f + 1):
            seg[t], qix[t] = n, qi
        flag[f] = True
    pos = [0] * T
    starts = {}
    p, t = 0, 0
    while t < T:
        if seg[t] == SHARED:
            pos[t] = p
            p += 1
            t += 1
            continue
        qi = qix[t]
        base = starts.setdefault(qi, p)
        longest = 0
        while t < T and seg[t] != SHARED and qix[t] == qi:
            s0, k = t, seg[t]
            while t < T and seg[t] == k:
                pos[t] = base + (t - s0)
                t += 1
            longest = max(longest, t - s0)
        p = base + longest
    return seg, qix, flag, pos


def prism_mask(segment, question, is_flag, position_ids, window=None, doc=None):
    sq, sk = segment[:, :, None], segment[:, None, :]
    qq, qk = question[:, :, None], question[:, None, :]
    real_k = sk >= 0
    m = (sk == SHARED)
    m = m & (sq != SHARED) | (sq == SHARED) & (sk == SHARED)
    m = m | ((sq == sk) & (sq > 0))
    m = m | (is_flag[:, :, None] & is_flag[:, None, :] & (qq == qk) & (qk >= 0))
    m = m & real_k
    if doc is not None:
        m = m & (doc[:, :, None] == doc[:, None, :])
    if window is not None:
        d = (position_ids[:, :, None] - position_ids[:, None, :]).abs()
        m = m & (d < window)
    eye = torch.eye(segment.shape[1], dtype=torch.bool, device=segment.device)
    m = m | eye[None]
    return m[:, None]


# ------------------------------------------------- verbatim copy of openjev/attn_backends.py chunked_math

def _chunk_attn(q, k, v, mask, scaling):
    B, Hkv, G, c, D = q.shape
    s = torch.matmul(q.reshape(B, Hkv, G * c, D), k.transpose(-1, -2)) * scaling
    s = s.view(B, Hkv, G, c, -1)
    s = s.masked_fill(~mask, torch.finfo(s.dtype).min) if mask.dtype == torch.bool else s + mask
    p = F.softmax(s, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(p.view(B, Hkv, G * c, -1), v).view(B, Hkv, G, c, -1)


def chunked_math(q, k, v, mask, scaling, chunk=256):
    from torch.utils.checkpoint import checkpoint
    B, H, T, D = q.shape
    Hkv = k.shape[1]
    G = H // Hkv
    qg = q.reshape(B, Hkv, G, T, D)
    m = mask[:, :, None]                                     # (B, 1, 1, T, S)
    outs = []
    for s0 in range(0, T, chunk):
        outs.append(checkpoint(_chunk_attn, qg[:, :, :, s0:s0 + chunk], k, v,
                               m[..., s0:s0 + chunk, :], scaling, use_reentrant=False))
    return torch.cat(outs, dim=3).reshape(B, H, T, -1)


# ------------------------------------------------- synthetic Prism requests

def synth_request(L, rng):
    """ids + readouts of one request of exactly L tokens: shared state, then 1-3 questions,
    each a shared header followed by 2-10 options (<opt> label... <flag>)."""
    budget = max(4, int(0.3 * L))                       # option tokens <= ~30% of the request
    nq = rng.randint(1, 3 if L >= 256 else 1)
    qs = []
    for _ in range(nq):
        hdr = rng.randint(6, 20)
        opts = [rng.randint(1, 8) + 2 for _ in range(rng.randint(2, 10))]   # +2: <opt>, <flag>
        while len(opts) > 2 and sum(opts) > budget // nq:
            opts.pop()
        qs.append((hdr, opts))
    used = sum(h + sum(o) for h, o in qs)
    state = L - used
    if state < 4:                                         # tiny L: 1 question, 2 short options
        qs = [(2, [3, 3])]
        state = L - 8
    ids = [TOK] * state
    readouts = []
    for hdr, opts in qs:
        ids += [TOK] * hdr
        flags = []
        for n in opts:
            ids += [OPT] + [TOK] * (n - 2) + [FLAG]
            flags.append(len(ids) - 1)
        readouts.append({"flags": flags})
    assert len(ids) == L, (len(ids), L)
    return ids, readouts


def synth_masks(B, T, device, seed=0):
    """(full, sliding) bool masks (B, 1, T, T), length-grouped rows (0.85T..T, one row = T)."""
    rng = random.Random(seed)
    seg = torch.full((B, T), -1, dtype=torch.long)
    qix = torch.full((B, T), -1, dtype=torch.long)
    isf = torch.zeros((B, T), dtype=torch.bool)
    pos = torch.zeros((B, T), dtype=torch.long)
    doc = torch.full((B, T), -1, dtype=torch.long)
    for b in range(B):
        L = T if b == 0 else rng.randint(int(0.85 * T), T)
        ids, ro = synth_request(L, rng)
        s, q, f, p = layout(ids, ro, OPT, FLAG)
        seg[b, :L], qix[b, :L] = torch.tensor(s), torch.tensor(q)
        isf[b, :L], pos[b, :L] = torch.tensor(f), torch.tensor(p)
        doc[b, :L] = 0
        pos[b, L:] = torch.arange(max(p) + 1, max(p) + 1 + T - L)
    args = [t.to(device) for t in (seg, qix, isf, pos)]
    full = prism_mask(*args, doc=doc.to(device))
    slide = prism_mask(*args, window=WINDOW, doc=doc.to(device))
    return full, slide


# ------------------------------------------------- candidates: f(q, k1, v1, mask_bool) -> (B, H, T, D)

def repeat_kv(x, n):                                       # transformers' repeat_kv (a copy)
    B, Hkv, T, D = x.shape
    return x[:, :, None].expand(B, Hkv, n, T, D).reshape(B, Hkv * n, T, D)


def additive(mask, dtype, pad8=True):
    """Float additive mask; with pad8 the last dim is allocated padded to a multiple of 8
    and sliced back (mem-efficient / xformers want an aligned row stride)."""
    B, _, Tq, Tk = mask.shape
    Tp = -(-Tk // 8) * 8 if pad8 else Tk
    out = torch.zeros(B, 1, Tq, Tp, dtype=dtype, device=mask.device)
    out[..., :Tk].masked_fill_(~mask, torch.finfo(dtype).min)
    return out[..., :Tk]


def sdpa(backend, float_mask):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    be = None if backend is None else getattr(SDPBackend, backend)

    def prep(mask, q):
        return additive(mask, q.dtype) if float_mask else mask

    def f(q, k, v, mask, scale):
        m = prep(mask, q)
        kk, vv = repeat_kv(k, q.shape[1] // k.shape[1]), repeat_kv(v, q.shape[1] // k.shape[1])
        if be is None:
            return F.scaled_dot_product_attention(q, kk, vv, attn_mask=m, scale=scale)
        with sdpa_kernel(be):
            return F.scaled_dot_product_attention(q, kk, vv, attn_mask=m, scale=scale)
    return f


def xformers_fn(op=None):
    import xformers.ops as xo

    def f(q, k, v, mask, scale):
        H = q.shape[1]
        bias = additive(mask, q.dtype).expand(-1, H, -1, -1)
        kk, vv = repeat_kv(k, H // k.shape[1]), repeat_kv(v, H // k.shape[1])
        o = xo.memory_efficient_attention(q.transpose(1, 2), kk.transpose(1, 2), vv.transpose(1, 2),
                                          attn_bias=bias, scale=scale, op=op)
        return o.transpose(1, 2)
    return f


def candidates(which):
    c = {}
    if "sdpa" in which:
        c["a_math_bool"] = sdpa("MATH", False)             # math kernel (forced)
        c["a0_default_bool"] = sdpa(None, False)           # = attn_backend sdpa_bool (r7 path)
        c["a0_default_float"] = sdpa(None, True)           # = attn_backend sdpa_default
        c["b_efficient_float"] = sdpa("EFFICIENT_ATTENTION", True)
        c["b_efficient_bool"] = sdpa("EFFICIENT_ATTENTION", False)
        c["e_chunked_math"] = lambda q, k, v, m, s: chunked_math(q, k, v, m, s)
    if "xformers" in which:
        c["c_xformers"] = xformers_fn()
    return c


# ------------------------------------------------- measurement

def err(e):
    return f"{type(e).__name__}: {str(e).strip()}"[:400]


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def kernel_used(f, q, k, v, mask, scale):
    """aten SDPA op names seen in one forward (which kernel the dispatcher picked)."""
    from torch.profiler import ProfilerActivity, profile
    with torch.no_grad(), profile(activities=[ProfilerActivity.CPU]) as prof:
        f(q, k, v, mask, scale)
    names = {e.key for e in prof.key_averages()}
    hits = sorted(n for n in names if "scaled_dot_product" in n or "efficient_attention" in n
                  or "flash_attention" in n or "memory_efficient" in n.lower() or "fmha" in n.lower())
    return hits


def reference(q, k, v, mask, scale, g):
    """fp32 math attention, one batch row at a time (memory), with grads."""
    outs, dq, dk, dv = [], [], [], []
    for b in range(q.shape[0]):
        qb = q[b:b + 1].detach().float().requires_grad_()
        kb = k[b:b + 1].detach().float().requires_grad_()
        vb = v[b:b + 1].detach().float().requires_grad_()
        o = sdpa("MATH", False)(qb, kb, vb, mask[b:b + 1], scale)
        o.backward(g[b:b + 1].float())
        outs.append(o.detach()), dq.append(qb.grad), dk.append(kb.grad), dv.append(vb.grad)
    return [torch.cat(x) for x in (outs, dq, dk, dv)]


def bench_one(f, q, k, v, mask, scale, g, ref, dev, iters, warm):
    r = {}
    def step():
        q.grad = k.grad = v.grad = None
        o = f(q, k, v, mask, scale)
        o.backward(g)
        return o
    try:
        sync(dev)
        if dev == "cuda":
            torch.cuda.empty_cache()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        o = step()
        sync(dev)
        if dev == "cuda":
            r["peak_mb"] = round((torch.cuda.max_memory_allocated() - base) / 2**20, 1)
        got = [o.detach().float(), q.grad.float(), k.grad.float(), v.grad.float()]
        for name, a, b in zip(("out", "dq", "dk", "dv"), got, ref):
            r[f"diff_{name}"] = float((a - b).abs().max())
        r["finite"] = all(bool(torch.isfinite(a).all()) for a in got)
        del o, got
        for _ in range(warm):
            step()
        ts = []
        for _ in range(iters):
            if dev == "cuda":
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                step()
                e1.record()
                torch.cuda.synchronize()
                ts.append(e0.elapsed_time(e1))
            else:
                t0 = time.perf_counter()
                step()
                ts.append((time.perf_counter() - t0) * 1e3)
        r["ms"] = round(statistics.median(ts), 3)
        r["ms_all"] = [round(t, 3) for t in ts]
        r["status"] = "ok"
    except torch.cuda.OutOfMemoryError as e:
        r.update(status="oom", error=err(e))
    except Exception as e:
        r.update(status="unsupported", error=err(e))
    q.grad = k.grad = v.grad = None
    if dev == "cuda":
        torch.cuda.empty_cache()
    return r


def run_grid(which, dims, seqs, B, dev, iters, warm):
    dtype = torch.float16 if dev == "cuda" else torch.float32
    cands = candidates(which)
    rows = []
    for D in dims:
        for T in seqs:
            torch.manual_seed(1000 * D + T)
            full, slide = synth_masks(B, T, dev, seed=T)
            mask = full if D == 512 else slide
            q = torch.randn(B, HEADS, T, D, device=dev, dtype=dtype, requires_grad=True)
            k = torch.randn(B, KV_HEADS, T, D, device=dev, dtype=dtype, requires_grad=True)
            v = torch.randn(B, KV_HEADS, T, D, device=dev, dtype=dtype, requires_grad=True)
            g = torch.randn(B, HEADS, T, D, device=dev, dtype=dtype)
            scale = D ** -0.5
            ref = reference(q, k, v, mask, scale, g)
            mstats = {"density": round(float(mask.float().mean()), 4),
                      "full_eq_sliding": bool(torch.equal(full, slide))}
            for name, f in cands.items():
                r = bench_one(f, q, k, v, mask, scale, g, ref, dev, iters, warm)
                if name.startswith("a0") and r.get("status") == "ok":
                    try:
                        r["kernel"] = kernel_used(f, q, k, v, mask, scale)
                    except Exception as e:
                        r["kernel"] = err(e)
                row = {"cand": name, "D": D, "T": T, "B": B, **mstats, **r}
                rows.append(row)
                print(json.dumps({k_: v_ for k_, v_ in row.items() if k_ != "ms_all"}), flush=True)
            del q, k, v, g, ref, full, slide, mask
            if dev == "cuda":
                torch.cuda.empty_cache()
    return rows


# ------------------------------------------------- diagnostics: why "No available kernel"?

def eligibility(dev):
    if dev != "cuda":
        return {}
    from torch.backends.cuda import SDPAParams, can_use_efficient_attention, can_use_flash_attention
    from torch.nn.attention import SDPBackend, sdpa_kernel
    out = {}
    T, B = 256, 2
    for dt in (torch.float16, torch.bfloat16):
        for D in (256, 512):
            q = torch.randn(B, HEADS, T, D, device=dev, dtype=dt, requires_grad=True)
            k = torch.randn(B, HEADS, T, D, device=dev, dtype=dt, requires_grad=True)
            for mk, m in {"bool": torch.ones(B, 1, T, T, device=dev, dtype=torch.bool),
                          "float": torch.zeros(B, 1, T, T, device=dev, dtype=dt)}.items():
                key = f"{str(dt).split('.')[-1]}/D{D}/{mk}"
                res = {}
                for kn, fn in (("flash", can_use_flash_attention), ("efficient", can_use_efficient_attention)):
                    try:
                        p = SDPAParams(q, k, k, m, 0.0, False, False)
                        with warnings.catch_warnings(record=True) as w:
                            warnings.simplefilter("always")
                            ok = bool(fn(p, True))
                        res[kn] = True if ok else "no: " + " | ".join(str(x.message) for x in w)[:400]
                    except Exception as e:
                        res[kn] = err(e)
                try:                                       # the legacy call: forced EFFICIENT, fwd+bwd
                    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION), warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        o = F.scaled_dot_product_attention(q, k, k, attn_mask=m)
                        o.sum().backward()
                    res["forced_efficient_fwd_bwd"] = "ok"
                except Exception as e:
                    res["forced_efficient_fwd_bwd"] = err(e)
                    res["warnings"] = " | ".join(str(x.message) for x in w)[:600]
                q.grad = k.grad = None
                out[key] = res
                print("ELIG", key, json.dumps(res), flush=True)
    # GQA in SDPA (enable_gqa=True, unrepeated kv) with a mask: which kernel accepts it?
    q = torch.randn(B, HEADS, T, 256, device=dev, dtype=torch.float16)
    k1 = torch.randn(B, 1, T, 256, device=dev, dtype=torch.float16)
    m = torch.ones(B, 1, T, T, device=dev, dtype=torch.bool)
    for be in ("EFFICIENT_ATTENTION", "MATH"):
        try:
            with sdpa_kernel(getattr(SDPBackend, be)):
                F.scaled_dot_product_attention(q, k1, k1, attn_mask=m, enable_gqa=True)
            out[f"gqa_enable_gqa/{be}"] = "ok"
        except Exception as e:
            out[f"gqa_enable_gqa/{be}"] = err(e)
    print("ELIG gqa", {k_: v_ for k_, v_ in out.items() if k_.startswith("gqa")}, flush=True)
    return out


# ------------------------------------------------- xformers (isolated in a subprocess)

def pip(*args):
    p = subprocess.run([sys.executable, "-m", "pip", *args], capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)[-1500:]


XF_CHECK = ("import torch, xformers, xformers.ops as xo; "
            "q=torch.randn(1,64,8,256,device='cuda',dtype=torch.half,requires_grad=True); "
            "b=torch.zeros(1,8,64,64,device='cuda',dtype=torch.half); "
            "o=xo.memory_efficient_attention(q,q,q,attn_bias=b); o.sum().backward(); "
            "print('XF_OK', xformers.__version__, torch.__version__)")


def install_xformers(torch_ver):
    """Try xformers wheels with --no-deps (never touch torch); keep the first one whose
    fused kernel runs fwd+bwd on this torch. Returns (version or None, attempt log)."""
    log = []
    cuda_tag = "cu" + torch.version.cuda.replace(".", "") if torch.version.cuda else "cpu"
    sources = [["--index-url", f"https://download.pytorch.org/whl/{cuda_tag}"], []]
    for src in sources:
        rc, txt = pip("index", "versions", "xformers", *src)
        vers = []
        for line in txt.splitlines():
            if "Available versions:" in line:
                vers = [s.strip() for s in line.split(":", 1)[1].split(",")]
        log.append({"source": src or ["pypi"], "versions": vers[:8], "rc": rc})
        for ver in vers[:4]:
            rc, txt = pip("install", "-q", "--no-deps", "--force-reinstall", f"xformers=={ver}", *src)
            if rc:
                log.append({"try": ver, "install": "failed", "tail": txt[-400:]})
                continue
            p = subprocess.run([sys.executable, "-c", XF_CHECK], capture_output=True, text=True)
            tv = subprocess.run([sys.executable, "-c", "import torch;print(torch.__version__)"],
                                capture_output=True, text=True).stdout.strip()
            ok = "XF_OK" in p.stdout and tv == torch_ver
            log.append({"try": ver, "source": src or ["pypi"], "ok": ok, "torch_after": tv,
                        "check_tail": (p.stdout + p.stderr)[-600:]})
            if ok:
                return ver, log
    return None, log


# ------------------------------------------------- report (local)

BUCKETS = [128, 256, 512, 1024, 2048]
WEIGHTS = [0.022, 0.163, 0.552, 0.197, 0.066]               # r7 share of real steps
S_PER_SAMPLE = [1 / 6.65, 1 / 3.37, 1 / 1.67, 1 / 0.85, 1 / 0.55]   # r7 T0 (1xT4, sdpa_bool)
ATTN_SHARE = [0.025 * (0.20 / 0.025) ** (i / 4) for i in range(5)]  # 2.5% -> 20%, log-linear
N_LAYERS = {256: 28, 512: 7}


def report(path):
    res = json.load(open(path))
    rows = res["rows"]
    t = {(r["cand"], r["D"], r["T"]): r for r in rows}
    cands = sorted({r["cand"] for r in rows}, key=lambda c: c.replace("a0", "a1"))
    # "current" = median of the four variants that run the same mem-efficient kernel (one of
    # them alone can carry a per-candidate offset, e.g. a0_default_bool at D256/T256).
    same = ("a0_default_bool", "a0_default_float", "b_efficient_float", "b_efficient_bool")
    for D in (256, 512):
        for T in BUCKETS:
            ms = [t[(c, D, T)]["ms"] for c in same if t.get((c, D, T), {}).get("ms")]
            if ms:
                t[("current", D, T)] = {"ms": statistics.median(ms)}
    L = ["| candidate | head_dim | seq | fwd+bwd ms | vs math | vs current | peak MB | max diff out / dq,dk,dv | status |",
         "|---|---|---|---|---|---|---|---|---|"]
    for c in cands:
        for D in (256, 512):
            for T in BUCKETS:
                r = t.get((c, D, T))
                if not r:
                    continue
                m, a0 = t.get(("a_math_bool", D, T), {}), t.get(("current", D, T), {})
                if r.get("status") == "ok":
                    sm = f"{m['ms'] / r['ms']:.2f}x" if m.get("ms") else "-"
                    sa = f"{a0['ms'] / r['ms']:.2f}x" if a0.get("ms") else "-"
                    dg = max(r["diff_dq"], r["diff_dk"], r["diff_dv"])
                    kn = r.get("kernel")
                    if isinstance(kn, list):
                        kn = next((n for n in ("efficient", "flash", "cudnn", "math")
                                   if any(n in x for x in kn)), ",".join(kn))
                    st = "ok" + (f", ran {kn}" if kn else "")
                    L.append(f"| {c} | {D} | {T} | {r['ms']:.2f} | {sm} | {sa} | {r.get('peak_mb', '-')} | "
                             f"{r['diff_out']:.1e} / {dg:.1e} | {st} |")
                else:
                    L.append(f"| {c} | {D} | {T} | - | - | - | - | - | {r['status']}: `{r.get('error', '')[:120]}` |")
    print("\n".join(L))
    print("\n### End-to-end estimate (vs current = median of the 4 mem-efficient variants)\n")
    print("| candidate | " + " | ".join(f"<={b}" for b in BUCKETS) + " | weighted speedup |")
    print("|---|" + "---|" * (len(BUCKETS) + 1))
    for c in cands:
        ratios, ok = [], True
        for T in BUCKETS:
            num = sum(N_LAYERS[D] * t.get((c, D, T), {}).get("ms", math.nan) for D in N_LAYERS)
            den = sum(N_LAYERS[D] * t.get(("current", D, T), {}).get("ms", math.nan) for D in N_LAYERS)
            r_ = num / den
            ok &= not math.isnan(r_)
            ratios.append(r_)
        if not ok:
            # sliding layers only (head_dim 256), global layers stay on a0
            ratios = []
            for T in BUCKETS:
                c256 = t.get((c, 256, T), {}).get("ms", math.nan)
                a256 = t.get(("current", 256, T), {}).get("ms", math.nan)
                a512 = t.get(("current", 512, T), {}).get("ms", math.nan)
                ratios.append((28 * c256 + 7 * a512) / (28 * a256 + 7 * a512))
        old = sum(w * s for w, s in zip(WEIGHTS, S_PER_SAMPLE))
        new = sum(w * s * (1 - a * (1 - r_)) for w, s, a, r_ in zip(WEIGHTS, S_PER_SAMPLE, ATTN_SHARE, ratios))
        cells = " | ".join(f"{r_:.2f}" for r_ in ratios)
        tag = "" if ok else " (sliding only)"
        print(f"| {c}{tag} | {cells} | {old / new:.3f}x |")
    print("\nattention share per bucket:", [round(a, 3) for a in ATTN_SHARE])


# ------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--report")
    ap.add_argument("--cands", default="sdpa")
    ap.add_argument("--json-out")
    a = ap.parse_args()
    if a.report:
        return report(a.report)
    dev = "cuda" if torch.cuda.is_available() and not a.smoke else "cpu"
    if a.smoke:
        rows = run_grid(a.cands.split(","), [256, 512], [128, 256], 2, dev, 2, 1)
        print("smoke ok", len(rows))
        return
    if a.cands == "xformers":                              # subprocess mode
        rows = run_grid(["xformers"], [256, 512], BUCKETS, 8, dev, 10, 3)
        json.dump(rows, open(a.json_out, "w"))
        return
    name = torch.cuda.get_device_name()
    assert "T4" in name, f"expected a Tesla T4, got {name}"
    env = {"gpu": name, "capability": torch.cuda.get_device_capability(), "torch": torch.__version__,
           "cuda": torch.version.cuda, "python": sys.version.split()[0]}
    print("ENV", env, flush=True)
    res = {"env": env, "eligibility": eligibility(dev)}
    res["rows"] = run_grid(["sdpa"], [256, 512], BUCKETS, 8, dev, 10, 3)
    json.dump(res, open(f"{OUT_DIR}/attn_spike.json", "w"), indent=1)
    ver, log = install_xformers(torch.__version__)
    res["xformers"] = {"version": ver, "attempts": log}
    print("XFORMERS", json.dumps(res["xformers"])[:3000], flush=True)
    if ver:
        xo = f"{OUT_DIR}/xf_rows.json"
        p = subprocess.run([sys.executable, __file__, "--cands", "xformers", "--json-out", xo],
                           capture_output=True, text=True)
        print(p.stdout[-6000:], p.stderr[-3000:], flush=True)
        if os.path.exists(xo):
            res["rows"] += json.load(open(xo))
        else:
            res["xformers"]["run_error"] = (p.stderr or p.stdout)[-1500:]
    json.dump(res, open(f"{OUT_DIR}/attn_spike.json", "w"), indent=1)
    print("=== ATTN SPIKE DONE ===", flush=True)


if __name__ == "__main__":
    main()
