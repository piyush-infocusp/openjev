"""int8 matmul spike (Kaggle T4, sm75): do int8 GEMMs speed up the frozen base linears of
Gemma 4 E2B during LoRA training, and at what numerical cost?

No model load. Random fp16 weights (std 0.02) at the exact (out, in) shapes of the 205
LoRA-target decoder linears of gemma-4-E2B (checked with openjev.prism.quant_targets on the
HF config: q/k/v/o, gate/up/down; kv-shared layers 15-34 have no k/v and a 12288 MLP).
M = tokens per micro-batch {1024 .. 16384} (mb 8 x bucket bound 128 .. 2048).

The base weight is frozen, so a training step needs Y = X W^T (forward, run twice with grad
ckpt) and dX = dY W (backward; no dW). LoRA stays fp16 and is not part of this bench.
Per candidate: forward ms and backward(dX) ms (CUDA events, 3 warm-up, median of 10), and the
error of Y and dX vs an fp32 reference on Gaussian X and on X with outlier channels.

Candidates
  fp16            F.linear / dY @ W (current path), timed again at the end of each shape
  i8_row          torch._int_mm, X and dY per-row dynamic scales, W per-channel (one int8 copy
                  per role: rows for Y, columns for dX), eager quantize / dequant
  i8_tensor       same with per-tensor scales everywhere
  i8_row_compiled the i8_row fwd / bwd functions under torch.compile (quantize + dequant fused)
  i8_row_fused    torch.compile with inductor force_fuse_int_mm_with_mul (Triton int8 GEMM)
  i8_gemm_only    _int_mm alone on pre-quantized inputs (ceiling, no quant / dequant)
  q_x / q_dy / dq_y / dq_dx   the eager quantize and dequant pieces of i8_row, timed alone
  torchao_*       (subprocess) prototype int8 mixed-precision training, int8 dyn act + int8 weight
  bnb_*           (subprocess) bitsandbytes Linear8bitLt, threshold 6.0 (our quant=int8) and 0

  python bench_int8.py                    # full run (Kaggle kernel)
  python bench_int8.py --smoke            # tiny CPU run
  python bench_int8.py --report res.json  # markdown tables + end-to-end estimate
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
# (name, out N, in K, count in E2B): sums to the 205 LoRA-target linears
SHAPES = [("q_s", 2048, 1536, 28), ("q_g", 4096, 1536, 7),
          ("kv_s", 256, 1536, 24), ("kv_g", 512, 1536, 6),
          ("o_s", 1536, 2048, 28), ("o_g", 1536, 4096, 7),
          ("gu_6k", 6144, 1536, 30), ("gu_12k", 12288, 1536, 40),
          ("down_6k", 1536, 6144, 15), ("down_12k", 1536, 12288, 20)]
MS = [1024, 2048, 4096, 8192, 16384]
M_ERR = 4096
WARM, ITERS = 3, 10


def err(e):
    return f"{type(e).__name__}: {str(e)[:600]}"


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def bench(fn, dev, warm=WARM, iters=ITERS):
    for _ in range(warm):
        fn()
    sync(dev)
    ts = []
    for _ in range(iters):
        if dev == "cuda":
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b))
        else:
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def gpu_state():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.draw",
                               "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except Exception as e:
        return err(e)


def make(M, N, K, kind, dev):
    """Deterministic X (M,K), dY (M,N), W (N,K) fp16; same in every process. kind 'outlier'
    scales 0.5% of the input channels by 30 (Gemma-like outlier features)."""
    g = torch.Generator(device=dev).manual_seed(1000003 * N + 7919 * K + M)
    w = (torch.randn(N, K, generator=g, device=dev) * 0.02).half()
    x = torch.randn(M, K, generator=g, device=dev)
    if kind == "outlier":
        cols = torch.randperm(K, generator=g, device=dev)[:max(1, K // 200)]
        x[:, cols] *= 30
    dy = torch.randn(M, N, generator=g, device=dev)
    return x.half(), dy.half(), w


def errors(y, dx, x, dy, w):
    yr = x.float() @ w.float().t()
    dr = dy.float() @ w.float()

    def m(a, r):
        a = a.float()
        return {"rel_fro": ((a - r).norm() / r.norm()).item(),
                "max_rel": ((a - r).abs().max() / r.abs().max()).item()}
    return {"y": m(y, yr), "dx": m(dx, dr)}


# ------------------------------------------------- int8 building blocks (torch._int_mm)

def q_rows(x):
    """int8 per-row symmetric: q (M,K) int8, scale (M,1) fp32."""
    s = x.abs().amax(dim=1, keepdim=True).float().clamp_min(1e-12) / 127
    return (x / s).round_().clamp_(-127, 127).to(torch.int8), s


def q_tensor(x):
    s = x.abs().amax().float().clamp_min(1e-12) / 127
    return (x / s).round_().clamp_(-127, 127).to(torch.int8), s


def dequant(yi, sa, sb):
    y = yi.to(torch.float32)
    return y.mul_(sa).mul_(sb).half()


class Int8Weight:
    """A frozen W (N,K) quantized once: rows (per out-channel) for Y = X W^T, columns (per
    in-channel) for dX = dY W, so each scale sits outside its GEMM's reduction dim. Two int8
    copies = the bytes of one fp16 copy. `cm`: pass mat2 as a column-major .t() view (True) or
    as a contiguous row-major tensor (False)."""

    def __init__(self, w, per_tensor=False, cm_f=True, cm_b=True):
        q = q_tensor if per_tensor else q_rows
        wr, sr = q(w)                                  # (N,K), (N,1) or scalar
        wc, sc = q(w.t().contiguous())                 # (K,N), (K,1) or scalar
        self.f = wr.t() if cm_f else wr.t().contiguous()          # (K,N) mat2 for fwd
        self.b = wc.t() if cm_b else wc.t().contiguous()          # (N,K) mat2 for bwd
        self.sf = sr.reshape(1, -1) if sr.dim() else sr
        self.sb = sc.reshape(1, -1) if sc.dim() else sc
        self.per_tensor = per_tensor


def i8_fwd(x, W: Int8Weight):
    xq, sx = (q_tensor if W.per_tensor else q_rows)(x)
    return dequant(torch._int_mm(xq, W.f), sx, W.sf)


def i8_bwd(dy, W: Int8Weight):
    gq, sg = (q_tensor if W.per_tensor else q_rows)(dy)
    return dequant(torch._int_mm(gq, W.b), sg, W.sb)


def i8_fwd_fusable(x, wf, sf):
    xq, sx = q_rows(x)
    return (torch._int_mm(xq, wf) * sx * sf).half()


class Int8LinearFn(torch.autograd.Function):
    """Y = X W^T with a frozen int8 W; backward returns dX only (W gets no grad)."""

    @staticmethod
    def forward(ctx, x, W):
        ctx.W = W
        y = i8_fwd(x.reshape(-1, x.shape[-1]), W)
        return y.view(*x.shape[:-1], -1)

    @staticmethod
    def backward(ctx, dy):
        dx = i8_bwd(dy.reshape(-1, dy.shape[-1]).contiguous(), ctx.W)
        return dx.view(*dy.shape[:-1], -1), None


# ------------------------------------------------- main-process grid

def probe_layout(dev):
    """_int_mm on this GPU: does it run, and is mat2 faster column-major (.t() view) or
    contiguous row-major? Exact errors recorded."""
    res = {}
    M, N, K = (4096, 6144, 1536) if dev == "cuda" else (32, 64, 48)
    x, dy, w = make(M, N, K, "gauss", dev)
    for cm in (True, False):
        tag = "colmajor_view" if cm else "rowmajor_contig"
        try:
            W = Int8Weight(w, cm_f=cm, cm_b=cm)
            xq, _ = q_rows(x)
            gq, _ = q_rows(dy)
            res[f"fwd_{tag}"] = round(bench(lambda: torch._int_mm(xq, W.f), dev), 4)
            res[f"bwd_{tag}"] = round(bench(lambda: torch._int_mm(gq, W.b), dev), 4)
        except Exception as e:
            res[f"fwd_{tag}"] = res[f"bwd_{tag}"] = err(e)
    num = {k: v for k, v in res.items() if isinstance(v, float)}
    pick = {}
    for role in ("fwd", "bwd"):
        c = {k: v for k, v in num.items() if k.startswith(role)}
        pick[role] = (min(c, key=c.get).endswith("colmajor_view")) if c else True
    res["pick_colmajor"] = pick
    # autograd check of the custom Function
    try:
        W = Int8Weight(w, cm_f=pick["fwd"], cm_b=pick["bwd"])
        xx = x.clone().requires_grad_(True)
        y = Int8LinearFn.apply(xx, W)
        y.backward(dy)
        res["autograd_check"] = {"y_eq_direct": bool(torch.equal(y, i8_fwd(x, W))),
                                 "dx_eq_direct": bool(torch.equal(xx.grad, i8_bwd(dy, W)))}
    except Exception as e:
        res["autograd_check"] = err(e)
    print("PROBE", json.dumps(res), flush=True)
    return res


def main_variants(x, dy, w, pick, dev, compiled):
    fp = w
    V = {"fp16": (lambda: F.linear(x, fp), lambda: dy @ fp)}
    Wr = Int8Weight(w, cm_f=pick["fwd"], cm_b=pick["bwd"])
    V["i8_row"] = (lambda: i8_fwd(x, Wr), lambda: i8_bwd(dy, Wr))
    Wt = Int8Weight(w, per_tensor=True, cm_f=pick["fwd"], cm_b=pick["bwd"])
    V["i8_tensor"] = (lambda: i8_fwd(x, Wt), lambda: i8_bwd(dy, Wt))
    if compiled.get("c"):
        cf, cb = compiled["c"]
        V["i8_row_compiled"] = (lambda: cf(x, Wr.f, Wr.sf), lambda: cb(dy, Wr.b, Wr.sb))
    if compiled.get("fused"):
        ff, fb = compiled["fused"]
        V["i8_row_fused"] = (lambda: ff(x, Wr.f, Wr.sf), lambda: fb(dy, Wr.b, Wr.sb))
    xq, sx = q_rows(x)
    gq, sg = q_rows(dy)
    V["i8_gemm_only"] = (lambda: torch._int_mm(xq, Wr.f), lambda: torch._int_mm(gq, Wr.b))
    yi, di = torch._int_mm(xq, Wr.f), torch._int_mm(gq, Wr.b)
    V["pieces"] = (lambda: q_rows(x), lambda: q_rows(dy))          # q_x, q_dy
    V["pieces_dq"] = (lambda: dequant(yi, sx, Wr.sf), lambda: dequant(di, sg, Wr.sb))   # dq_y, dq_dx
    V["fp16_repeat"] = V["fp16"]
    return V


def make_compiled(dev):
    out, info = {}, {}
    try:
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = 512
        if hasattr(torch._dynamo.config, "recompile_limit"):
            torch._dynamo.config.recompile_limit = 512
    except Exception:
        pass
    out["c"] = (torch.compile(i8_fwd_fusable, dynamic=False), torch.compile(i8_fwd_fusable, dynamic=False))
    try:
        opts = {"force_fuse_int_mm_with_mul": True}
        out["fused"] = (torch.compile(i8_fwd_fusable, dynamic=False, options=opts),
                        torch.compile(i8_fwd_fusable, dynamic=False, options=opts))
    except Exception as e:
        info["fused_setup"] = err(e)
    return out, info


def run_main(dev, shapes, ms, m_err, path, res):
    probe = res["probe"]
    pick = probe["pick_colmajor"]
    compiled, cinfo = make_compiled(dev)
    res["compile_info"] = cinfo
    rows = res.setdefault("rows", [])
    errs = res.setdefault("errors", [])
    for name, N, K, cnt in shapes:
        for M in ms:
            x, dy, w = make(M, N, K, "gauss", dev)
            V = main_variants(x, dy, w, pick, dev, compiled)
            for vn, (f, b) in V.items():
                r = {"cand": vn, "shape": name, "N": N, "K": K, "M": M, "count": cnt}
                try:
                    r["fwd_ms"] = bench(f, dev)
                    r["bwd_ms"] = bench(b, dev)
                    r["status"] = "ok"
                except Exception as e:
                    r["status"], r["error"] = "error", err(e)
                    if vn.endswith(("compiled", "fused")):      # do not retry a broken compile
                        compiled.pop("c" if vn.endswith("compiled") else "fused", None)
                rows.append(r)
                print("ROW", json.dumps(r), flush=True)
            del V, x, dy
            if dev == "cuda":
                torch.cuda.empty_cache()
        # errors at M_ERR, both input kinds
        for kind in ("gauss", "outlier"):
            x, dy, w = make(m_err, N, K, kind, dev)
            V = main_variants(x, dy, w, pick, dev, compiled)
            for vn in ("fp16", "i8_row", "i8_tensor", "i8_row_compiled", "i8_row_fused"):
                if vn not in V:
                    continue
                try:
                    e = errors(V[vn][0](), V[vn][1](), x, dy, w)
                except Exception as ex:
                    e = {"error": err(ex)}
                errs.append({"cand": vn, "shape": name, "kind": kind, **e})
            del V
        print("GPU", name, gpu_state(), flush=True)
        json.dump(res, open(path, "w"), indent=1)


# ------------------------------------------------- subprocess candidates (torchao, bnb)

def ag_bench(mod, x, dy, dev):
    """fwd = no-grad forward; bwd = (autograd fwd+bwd) - fwd, x requires grad (dX)."""
    with torch.no_grad():
        tf = bench(lambda: mod(x), dev)
    xx = x.clone().requires_grad_(True)

    def fb():
        mod(xx).backward(dy)
        xx.grad = None
    tfb = bench(fb, dev)
    return tf, tfb - tf


def ag_outputs(mod, x, dy):
    xx = x.clone().requires_grad_(True)
    y = mod(xx)
    y.backward(dy)
    return y.detach(), xx.grad


def sub_grid(builders, dev, shapes, ms, m_err):
    """builders: name -> fn(w) returning a module (or raising). fp16 nn.Linear baseline
    timed the same way in the same process."""
    rows, errs = [], []

    def lin(w):
        m = nn.Linear(w.shape[1], w.shape[0], bias=False, device=w.device, dtype=w.dtype)
        m.weight.data.copy_(w)
        m.weight.requires_grad_(False)
        return m
    allb = {"fp16_ag": lin, **builders}
    broken = {}
    for name, N, K, cnt in shapes:
        mods = {}
        for M in ms:
            x, dy, w = make(M, N, K, "gauss", dev)
            for bn, fn in allb.items():
                r = {"cand": bn, "shape": name, "N": N, "K": K, "M": M, "count": cnt}
                if bn in broken:
                    r["status"], r["error"] = "error", broken[bn]
                    rows.append(r)
                    continue
                try:
                    if bn not in mods:
                        mods[bn] = fn(w)
                    r["fwd_ms"], r["bwd_ms"] = ag_bench(mods[bn], x, dy, dev)
                    r["status"] = "ok"
                except Exception as e:
                    r["status"], r["error"] = "error", err(e)
                    r["tb"] = traceback.format_exc()[-1500:]
                    try:                                    # fwd-only status
                        with torch.no_grad():
                            r["fwd_ms"] = bench(lambda: mods[bn](x), dev)
                        r["status"] = "fwd_only"
                    except Exception as e2:
                        r["fwd_error"] = err(e2)
                rows.append(r)
                print("ROW", json.dumps({k: v for k, v in r.items() if k != "tb"}), flush=True)
            if dev == "cuda":
                torch.cuda.empty_cache()
        for kind in ("gauss", "outlier"):
            x, dy, w = make(m_err, N, K, kind, dev)
            for bn, fn in allb.items():
                try:
                    y, dx = ag_outputs(fn(w), x, dy)
                    e = errors(y, dx, x, dy, w)
                except Exception as ex:
                    e = {"error": err(ex)}
                errs.append({"cand": bn, "shape": name, "kind": kind, **e})
        del mods
    return rows, errs


def torchao_builders(info):
    import copy
    import torchao
    from torchao.quantization import quantize_
    info["version"] = torchao.__version__
    B = {}
    try:
        import torchao.prototype.quantized_training as qt
        info["qt_names"] = [n for n in dir(qt) if not n.startswith("_")]
    except Exception as e:
        qt = None
        info["qt_import"] = err(e)
    import torchao.quantization as tq
    info["quant_names"] = [n for n in dir(tq) if "int8" in n.lower() or "Int8" in n]

    def base(w):
        m = nn.Linear(w.shape[1], w.shape[0], bias=False, device=w.device, dtype=w.dtype)
        m.weight.data.copy_(w)
        return m

    if qt is not None:
        Cfg = getattr(qt, "Int8MixedPrecisionTrainingConfig", None)
        fn = getattr(qt, "int8_mixed_precision_training", None)

        def mpt(w, frozen=True):
            m = base(w)
            tries = []
            if Cfg is not None:
                tries += [lambda: Cfg(output=True, grad_input=True, grad_weight=False), lambda: Cfg()]
            if fn is not None:
                tries += [lambda: fn(Cfg(output=True, grad_input=True, grad_weight=False)) if Cfg else fn(),
                          lambda: fn()]
            last = None
            for t in tries:
                try:
                    mm = copy.deepcopy(m)
                    quantize_(mm, t())
                    if frozen:
                        mm.weight.requires_grad_(False)
                    return mm
                except Exception as e:
                    last = e
            raise RuntimeError(f"no int8 mixed-precision API worked: {err(last) if last else 'none found'}")
        B["torchao_i8_mpt"] = mpt

        def mpt_swap(w):
            # module_swap=True: an explicit autograd.Function (the tensor-subclass path's
            # F.linear override is not hit on torch 2.11 / 2.14: it ran plain fp16 mm in v1)
            m = base(w)
            quantize_(m, Cfg(output=True, grad_input=True, grad_weight=False, module_swap=True))
            m.weight.requires_grad_(False)
            return m
        if Cfg is not None:
            B["torchao_i8_mpt_swap"] = mpt_swap
    D = getattr(tq, "Int8DynamicActivationInt8WeightConfig", None)
    d = getattr(tq, "int8_dynamic_activation_int8_weight", None)

    def dyn(w):
        m = base(w)
        m.weight.requires_grad_(False)
        quantize_(m, D() if D is not None else d())
        return m
    if D is not None or d is not None:
        B["torchao_i8_dyn"] = dyn
    return B


def bnb_builders(info):
    import bitsandbytes as bnb
    info["version"] = bnb.__version__
    B = {}
    for th in (6.0, 0.0):
        def mk(w, th=th):
            m = bnb.nn.Linear8bitLt(w.shape[1], w.shape[0], bias=False, has_fp16_weights=False,
                                    threshold=th)
            m.weight = bnb.nn.Int8Params(w.cpu(), requires_grad=False, has_fp16_weights=False)
            return m.to(w.device)
        B[f"bnb_th{th:g}"] = mk
    return B


def pip(*args):
    p = subprocess.run([sys.executable, "-m", "pip", *args], capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)[-1500:]


TAO_CHECK = ("import torch, torchao; from torchao.quantization import quantize_; "
             "import torch.nn as nn; m=nn.Linear(64,64,bias=False).half().cuda(); "
             "print('TAO_OK', torchao.__version__, torch.__version__)")


def install_torchao(torch_ver):
    """Keep Kaggle's torch: try the preinstalled torchao, then newest PyPI versions --no-deps."""
    log = []
    p = subprocess.run([sys.executable, "-c", TAO_CHECK], capture_output=True, text=True)
    log.append({"try": "preinstalled", "ok": "TAO_OK" in p.stdout, "tail": (p.stdout + p.stderr)[-500:]})
    rc, txt = pip("index", "versions", "torchao")
    vers = []
    for line in txt.splitlines():
        if "Available versions:" in line:
            vers = [s.strip() for s in line.split(":", 1)[1].split(",")]
    log.append({"pypi_versions": vers[:10]})
    for ver in vers[:5]:
        rc, txt = pip("install", "-q", "--no-deps", "--force-reinstall", f"torchao=={ver}")
        if rc:
            log.append({"try": ver, "install": "failed", "tail": txt[-400:]})
            continue
        p = subprocess.run([sys.executable, "-c", TAO_CHECK], capture_output=True, text=True)
        tv = subprocess.run([sys.executable, "-c", "import torch;print(torch.__version__)"],
                            capture_output=True, text=True).stdout.strip()
        ok = "TAO_OK" in p.stdout and tv == torch_ver
        log.append({"try": ver, "ok": ok, "torch_after": tv, "tail": (p.stdout + p.stderr)[-600:]})
        if ok:
            return ver, log
    if log[0]["ok"]:
        return "preinstalled", log
    return None, log


def run_sub(mode, path, shapes, ms, m_err, dev):
    info = {}
    try:
        B = torchao_builders(info) if mode == "torchao" else bnb_builders(info)
        print("INFO", mode, json.dumps(info)[:4000], flush=True)
        rows, errs = sub_grid(B, dev, shapes, ms, m_err)
    except Exception as e:
        info["fatal"] = err(e)
        info["tb"] = traceback.format_exc()[-2000:]
        rows, errs = [], []
    json.dump({"info": info, "rows": rows, "errors": errs}, open(path, "w"), indent=1)


# ------------------------------------------------- report (local)

BUCKETS = [128, 256, 512, 1024, 2048]
WEIGHTS = [0.022, 0.163, 0.552, 0.197, 0.066]               # r7 share of real steps
S_PER_SAMPLE = [1 / 6.65, 1 / 3.37, 1 / 1.67, 1 / 0.85, 1 / 0.55]   # r7 T0 (1xT4, sdpa_bool)
MM_SHARE = 0.28                                              # aten::mm share, r4 trial-23 profile


def report(path):
    res = json.load(open(path))
    rows = [r for r in res["rows"]]
    for sub in ("torchao", "bnb"):
        rows += [dict(r, sub=sub) for r in (res.get(sub) or {}).get("rows", [])]
    T = {(r["cand"], r["shape"], r["M"]): r for r in rows}
    # Conservative fp16 baseline: per cell, the faster of fp16 and fp16_repeat (fwd and dX
    # separately). The first fp16 forward of a cell is often slow (clock ramp after the
    # previous cell's error pass), which would flatter int8.
    for (c, s, M), r in list(T.items()):
        if c == "fp16" and (("fp16_repeat", s, M) in T):
            q = T[("fp16_repeat", s, M)]
            T[("fp16min", s, M)] = dict(r, cand="fp16min", fwd_ms=min(r["fwd_ms"], q["fwd_ms"]),
                                        bwd_ms=min(r["bwd_ms"], q["bwd_ms"]))
    # Selective: int8 on the MLP linears (gate/up/down) only and only at M >= 2048; fp16 else.
    mlp = ("gu_6k", "gu_12k", "down_6k", "down_12k")
    for c in ("i8_row_compiled", "i8_row_fused"):
        for (cc, s, M), r in list(T.items()):
            if cc == "fp16min":
                src = T.get((c, s, M)) if (s in mlp and M >= 2048) else r
                if src:
                    T[("sel_mlp_" + c.split("_")[-1], s, M)] = dict(src, cand="sel_mlp_" + c.split("_")[-1])
    rows = rows + [v for (c, _, _), v in T.items() if c.startswith("sel_mlp_")]
    base_of = lambda r: "fp16_ag" if r.get("sub") else "fp16min"
    cands = [c for c in dict.fromkeys(r["cand"] for r in rows)
             if c not in ("fp16", "fp16_ag", "pieces", "pieces_dq")]
    counts = {s[0]: s[3] for s in SHAPES}
    shapes = list(dict.fromkeys(r["shape"] for r in rows))

    def ok(r):
        return r and r.get("status") == "ok"

    def wsum(c, M, key):
        """count-weighted ms over the 205 linears; key 'fwd'|'bwd'|'step' (2 fwd + 1 bwd)."""
        tot = 0.0
        for s in shapes:
            r = T.get((c, s, M))
            if not ok(r):
                return math.nan
            f, b = r["fwd_ms"], r["bwd_ms"]
            tot += counts[s] * {"fwd": f, "bwd": b, "step": 2 * f + b}[key]
        return tot

    print("### Count-weighted speedup over the 205 linears (fp16min time / candidate time)\n")
    print("step = 2 fwd + 1 dX (grad ckpt recomputes the forward).\n")
    print("| candidate | M | fwd | dX | step | step ms (fp16 → cand) |")
    print("|---|---|---|---|---|---|")
    gstep = {}
    for c in cands:
        sub = next((r.get("sub") for r in rows if r["cand"] == c), None)
        b = "fp16_ag" if sub else "fp16min"
        for M in sorted({r["M"] for r in rows}):
            vals = []
            for k in ("fwd", "bwd", "step"):
                bb, cc = wsum(b, M, k), wsum(c, M, k)
                vals.append(bb / cc if cc == cc and bb == bb else math.nan)
            gstep[(c, M)] = vals[2]
            bs, cs = wsum(b, M, "step"), wsum(c, M, "step")
            st = " | ".join("-" if v != v else f"{v:.2f}x" for v in vals)
            print(f"| {c} | {M} | {st} | {bs:.0f} → {cs:.0f} |" if cs == cs else f"| {c} | {M} | {st} | - |")
    # fp16 drift
    print("\nfp16 drift (fp16_repeat / fp16, step ms):",
          {M: round(wsum("fp16_repeat", M, "step") / wsum("fp16", M, "step"), 3)
           for M in sorted({r["M"] for r in res["rows"]})})

    print("\n### Per shape at M=4096 (fwd / dX ms; speedup vs fp16min on the step)\n")
    hdr = ["fp16", "fp16min"] + [c for c in cands if not c.startswith("sel_")]
    print("| shape (N×K, count) | " + " | ".join(hdr) + " |")
    print("|---|" + "---|" * len(hdr))
    for s in shapes:
        cells = []
        for c in hdr:
            r = T.get((c, s, 4096))
            if not ok(r):
                cells.append((r or {}).get("status", "-"))
                continue
            b = T.get((base_of(r), s, 4096))
            sp = (2 * b["fwd_ms"] + b["bwd_ms"]) / (2 * r["fwd_ms"] + r["bwd_ms"]) if ok(b) else math.nan
            cells.append(f"{r['fwd_ms']:.2f} / {r['bwd_ms']:.2f}" + ("" if c in ("fp16", "fp16min") else f" ({sp:.2f}x)"))
        N, K = T[("fp16", s, 4096)]["N"], T[("fp16", s, 4096)]["K"]
        print(f"| {s} ({N}×{K}, {counts[s]}) | " + " | ".join(cells) + " |")

    print("\n### Where i8_row's time goes (count-weighted ms, one fwd + one dX)\n")
    print("| M | fp16 fwd+dX | quantize X+dY | int8 GEMMs | dequant Y+dX | i8_row total | compiled total |")
    print("|---|---|---|---|---|---|---|")
    for M in sorted({r["M"] for r in res["rows"]}):
        f = lambda c: wsum(c, M, "fwd") + wsum(c, M, "bwd")
        print(f"| {M} | {f('fp16'):.1f} | {f('pieces'):.1f} | {f('i8_gemm_only'):.1f} | {f('pieces_dq'):.1f} | "
              f"{f('i8_row'):.1f} | {f('i8_row_compiled'):.1f} |")

    print("\n### Errors vs fp32 at M=4096 (worst over the 10 shapes; rel_fro / max_rel)\n")
    errs = list(res.get("errors", []))
    for sub in ("torchao", "bnb"):
        errs += (res.get(sub) or {}).get("errors", [])
    print("| candidate | Y gauss | dX gauss | Y outlier | dX outlier |")
    print("|---|---|---|---|---|")
    for c in dict.fromkeys(e["cand"] for e in errs):
        cells = []
        for kind in ("gauss", "outlier"):
            es = [e for e in errs if e["cand"] == c and e["kind"] == kind]
            for t in ("y", "dx"):
                v = [e[t] for e in es if t in e]
                cells.append(f"{max(x['rel_fro'] for x in v):.1e} / {max(x['max_rel'] for x in v):.1e}"
                             if v else (es[0].get("error", "-")[:80] if es else "-"))
        print(f"| {c} | " + " | ".join(cells[i] for i in (0, 1, 2, 3)) + " |")

    print("\n### End-to-end estimate (Amdahl per bucket, linears = 28% mm share, M = 8 × bucket)\n")
    print("| candidate | " + " | ".join(f"<={b}" for b in BUCKETS) + " | weighted samples/s |")
    print("|---|" + "---|" * (len(BUCKETS) + 1))
    for c in cands:
        g = [gstep.get((c, 8 * b), math.nan) for b in BUCKETS]
        if any(v != v for v in g):
            continue
        old = sum(w * s for w, s in zip(WEIGHTS, S_PER_SAMPLE))
        new = sum(w * s * (1 - MM_SHARE * (1 - 1 / gb)) for w, s, gb in zip(WEIGHTS, S_PER_SAMPLE, g))
        print(f"| {c} | " + " | ".join(f"{v:.2f}x" for v in g) + f" | {old / new:.3f}x |")


# ------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--report")
    ap.add_argument("--sub", choices=["torchao", "bnb"])
    ap.add_argument("--json-out")
    ap.add_argument("--ms", default="", help="comma list of M (default: all)")
    a = ap.parse_args()
    if a.report:
        return report(a.report)
    dev = "cuda" if torch.cuda.is_available() and not a.smoke else "cpu"
    shapes, ms, m_err = (SHAPES, MS, M_ERR) if not a.smoke else (SHAPES[:3], [32, 64], 64)
    if a.ms:
        ms = [int(v) for v in a.ms.split(",")]
    if a.sub:
        return run_sub(a.sub, a.json_out, shapes, ms, m_err, dev)
    path = f"{OUT_DIR}/int8_spike.json"
    if dev == "cuda":
        name = torch.cuda.get_device_name()
        assert "T4" in name, f"expected a Tesla T4, got {name}"
    env = {"gpu": torch.cuda.get_device_name() if dev == "cuda" else "cpu",
           "capability": torch.cuda.get_device_capability() if dev == "cuda" else None,
           "torch": torch.__version__, "cuda": torch.version.cuda, "python": sys.version.split()[0],
           "gpu_state_start": gpu_state()}
    try:
        import triton
        env["triton"] = triton.__version__
    except Exception as e:
        env["triton"] = err(e)
    print("ENV", env, flush=True)
    res = {"env": env, "probe": probe_layout(dev)}
    json.dump(res, open(path, "w"), indent=1)
    run_main(dev, shapes, ms, m_err, path, res)
    res["env"]["gpu_state_after_main"] = gpu_state()
    json.dump(res, open(path, "w"), indent=1)
    if a.smoke:
        print("smoke ok", len(res["rows"]), "rows")
        return
    # torchao (install --no-deps, keep torch) and bnb, each in its own subprocess
    ver, log = install_torchao(torch.__version__)
    res["torchao_install"] = {"version": ver, "attempts": log}
    print("TORCHAO", json.dumps(res["torchao_install"])[:3000], flush=True)
    rc, txt = pip("install", "-q", "--no-deps", "bitsandbytes==0.50.2")
    res["bnb_install"] = {"rc": rc, "tail": txt[-500:]}
    for sub in (["torchao"] if ver else []) + ["bnb"]:
        out = f"{OUT_DIR}/{sub}_rows.json"
        p = subprocess.run([sys.executable, __file__, "--sub", sub, "--json-out", out],
                           capture_output=True, text=True)
        print(p.stdout[-8000:], p.stderr[-4000:], flush=True)
        res[sub] = json.load(open(out)) if os.path.exists(out) else {"run_error": (p.stderr or p.stdout)[-2000:]}
        res["env"][f"gpu_state_after_{sub}"] = gpu_state()
        json.dump(res, open(path, "w"), indent=1)
    print("=== INT8 SPIKE DONE ===", flush=True)


if __name__ == "__main__":
    main()
