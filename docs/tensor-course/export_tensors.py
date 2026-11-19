"""Export real tensors of tiny-Qwen3ForCausalLM (layer 0 + final norm + top-5 logits) to tensors.json.

Run:  /home/piyush.sar/openjev/.venv/bin/python export_tensors.py
The model is re-implemented by hand from its weights (no hooks needed for the values)
and checked against the Hugging Face forward pass with torch.allclose before saving.
Activations carry a leading batch axis b (size 1). The token axis of keys and values is named s (the token being looked at), the token axis of queries is t. Weights keep the PyTorch layout [out, in].
"""
import json, math, os, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NAME = "trl-internal-testing/tiny-Qwen3ForCausalLM"
TEXT = "Refund my card please"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tensors.json")

torch.manual_seed(0)
model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).eval()
tok = AutoTokenizer.from_pretrained(NAME)
cfg = model.config
print(cfg)
print("parameters:")
for k, v in model.named_parameters():
    if "layers." in k and "layers.0." not in k:
        continue
    print("  ", k, tuple(v.shape))
n_params = sum(p.numel() for p in model.parameters())
print("total parameters:", n_params)

enc = tok(TEXT)
ids_list = enc.input_ids
tokens = [tok.decode([i]) for i in ids_list]
print("tokens:", list(zip(tokens, ids_list)))
ids = torch.tensor([ids_list])

H, N, K, D, I = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.intermediate_size
EPS, THETA = cfg.rms_norm_eps, cfg.rope_parameters["rope_theta"]
B, T = ids.shape
G = N // K  # query heads per kv head


def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def rot_half(x):
    return torch.cat([-x[..., D // 2:], x[..., :D // 2]], -1)


inv_freq = 1.0 / (THETA ** (torch.arange(0, D, 2).float() / D))
pos = torch.arange(T).float()
fr = torch.outer(pos, inv_freq)
emb_ = torch.cat([fr, fr], -1)
COS, SIN = emb_.cos(), emb_.sin()  # [t d]
MASK = torch.triu(torch.full((T, T), float("-inf")), 1)  # [t s]


def layer(x, L, rec=None):
    r = (lambda k, v: rec.__setitem__(k, v)) if rec is not None else (lambda k, v: None)
    a = L.self_attn
    y = rms(x, L.input_layernorm.weight); r("norm1", y)
    q_lin = y @ a.q_proj.weight.T; r("q_lin", q_lin)
    k_lin = y @ a.k_proj.weight.T; r("k_lin", k_lin)
    v_lin = y @ a.v_proj.weight.T; r("v_lin", v_lin)
    q = q_lin.view(B, T, N, D).transpose(1, 2); r("q_heads", q)
    k = k_lin.view(B, T, K, D).transpose(1, 2); r("k_heads", k)
    v = v_lin.view(B, T, K, D).transpose(1, 2); r("v_heads", v)
    q = rms(q, a.q_norm.weight); r("q_norm", q)
    k = rms(k, a.k_norm.weight); r("k_norm", k)
    q = q * COS + rot_half(q) * SIN; r("q_rope", q)
    k = k * COS + rot_half(k) * SIN; r("k_rope", k)
    kk = k.repeat_interleave(G, 1); vv = v.repeat_interleave(G, 1)
    sc = torch.einsum("bntd,bnsd->bnts", q, kk) * D ** -0.5; r("scores", sc)
    w = torch.softmax(sc + MASK, -1); r("weights", w)
    ctx = torch.einsum("bnts,bnsd->bntd", w, vv); r("ctx", ctx)
    flat = ctx.transpose(1, 2).reshape(B, T, N * D); r("ctx_flat", flat)
    ao = flat @ a.o_proj.weight.T; r("attn_out", ao)
    x = x + ao; r("res1", x)
    y2 = rms(x, L.post_attention_layernorm.weight); r("norm2", y2)
    g = y2 @ L.mlp.gate_proj.weight.T; r("gate", g)
    u = y2 @ L.mlp.up_proj.weight.T; r("up", u)
    act = torch.nn.functional.silu(g); r("act", act)
    au = act * u; r("act_up", au)
    dn = au @ L.mlp.down_proj.weight.T; r("down", dn)
    x = x + dn; r("res2", x)
    return x


with torch.no_grad():
    rec = {}
    x0 = model.model.embed_tokens.weight[ids]; rec["x0"] = x0
    x1 = layer(x0, model.model.layers[0], rec)
    x2 = layer(x1, model.model.layers[1])
    rec["layer1_out"] = x2
    fn = rms(x2, model.model.norm.weight); rec["final_norm"] = fn
    logits = fn @ model.lm_head.weight.T

    # ---- verify against Hugging Face ----
    hf = {}
    hooks = []
    L0 = model.model.layers[0]
    def hk(name):
        def f(mod, i, o): hf[name] = (o[0] if isinstance(o, tuple) else o).detach()
        return f
    for nm, mod in [("x0", model.model.embed_tokens), ("norm1", L0.input_layernorm), ("q_lin", L0.self_attn.q_proj),
                    ("k_lin", L0.self_attn.k_proj), ("v_lin", L0.self_attn.v_proj), ("attn_out", L0.self_attn.o_proj),
                    ("norm2", L0.post_attention_layernorm), ("gate", L0.mlp.gate_proj), ("up", L0.mlp.up_proj),
                    ("down", L0.mlp.down_proj), ("res2", L0), ("final_norm", model.model.norm)]:
        hooks.append(mod.register_forward_hook(hk(nm)))
    ref = model(ids).logits
    for h in hooks: h.remove()
    ok = True
    for nm, t in hf.items():
        c = torch.allclose(rec[nm], t, atol=1e-5)
        ok &= c
        print(f"  allclose {nm:11s} {c}  max|diff|={float((rec[nm]-t).abs().max()):.2e}")
    c = torch.allclose(logits, ref, atol=1e-5)
    ok &= c
    print(f"  allclose logits      {c}  max|diff|={float((logits-ref).abs().max()):.2e}")
    print("ALLCLOSE (atol 1e-5):", ok)
    if not ok:
        sys.exit("hand implementation does not match HF")

    last = logits[0, -1]
    top = torch.topk(last, 5)
    top_ids = top.indices.tolist()
    top_tokens = [tok.decode([i]) for i in top_ids]
    print("top-5 next token:", list(zip(top_tokens, top_ids, [round(float(v), 4) for v in top.values])))

# ---------- save ----------
def rnd(t):
    """5 decimals, or 6 significant digits for tensors whose largest value is below 0.5."""
    t = t.detach().double().flatten()
    m = float(t.abs().max()) if t.numel() else 0.0
    dec = 5 if m >= 0.5 else min(10, 5 + math.ceil(-math.log10(max(m, 1e-12))) )
    return [round(float(v), dec) for v in t], dec

TENS = {}
def add(name, t, axes, kind, desc, split=None, squeeze_b=False):
    data, dec = rnd(t)
    shape = list(t.shape)
    assert len(shape) == len(axes), (name, shape, axes)
    TENS[name] = dict(shape=shape, axes=axes, kind=kind, desc=desc, decimals=dec, data=data)
    if split: TENS[name]["split"] = split

A = "activation"; W = "weight"
SPL = {"nd": ["n", D], "kd": ["k", D]}
add("ids", ids.float(), ["b", "t"], A, "token ids")
add("x0", rec["x0"], ["b", "t", "h"], A, "embedding lookup")
add("norm1", rec["norm1"], ["b", "t", "h"], A, "input RMSNorm")
add("q_lin", rec["q_lin"], ["b", "t", "nd"], A, "query projection, flat", {"nd": ["n", D]})
add("k_lin", rec["k_lin"], ["b", "s", "kd"], A, "key projection, flat", {"kd": ["k", D]})
add("v_lin", rec["v_lin"], ["b", "s", "kd"], A, "value projection, flat", {"kd": ["k", D]})
add("q_heads", rec["q_heads"], ["b", "n", "t", "d"], A, "queries split into heads")
add("k_heads", rec["k_heads"], ["b", "k", "s", "d"], A, "keys split into heads")
add("v_heads", rec["v_heads"], ["b", "k", "s", "d"], A, "values split into heads")
add("q_norm", rec["q_norm"], ["b", "n", "t", "d"], A, "q_norm (RMSNorm over d)")
add("k_norm", rec["k_norm"], ["b", "k", "s", "d"], A, "k_norm (RMSNorm over d)")
add("q_rope", rec["q_rope"], ["b", "n", "t", "d"], A, "queries after RoPE")
add("k_rope", rec["k_rope"], ["b", "k", "s", "d"], A, "keys after RoPE")
add("scores", rec["scores"], ["b", "n", "t", "s"], A, "attention scores (Q Kᵀ · d^-1/2)")
mask_store = torch.where(torch.isinf(MASK), torch.tensor(-1e9), MASK)
add("mask", mask_store, ["t", "s"], A, "causal mask: 0 = allowed, -1e9 stands for minus infinity")
add("weights", rec["weights"], ["b", "n", "t", "s"], A, "softmax weights")
add("ctx", rec["ctx"], ["b", "n", "t", "d"], A, "weighted sum of values")
add("ctx_flat", rec["ctx_flat"], ["b", "t", "nd"], A, "heads merged", {"nd": ["n", D]})
add("attn_out", rec["attn_out"], ["b", "t", "h"], A, "o_proj output")
add("res1", rec["res1"], ["b", "t", "h"], A, "residual 1")
add("norm2", rec["norm2"], ["b", "t", "h"], A, "post-attention RMSNorm (input of the MLP)")
add("gate", rec["gate"], ["b", "t", "i"], A, "gate projection")
add("up", rec["up"], ["b", "t", "i"], A, "up projection")
add("act", rec["act"], ["b", "t", "i"], A, "SiLU(gate)")
add("act_up", rec["act_up"], ["b", "t", "i"], A, "SiLU(gate) * up")
add("down", rec["down"], ["b", "t", "h"], A, "down projection")
add("res2", rec["res2"], ["b", "t", "h"], A, "residual 2 = output of layer 0")
add("layer1_out", rec["layer1_out"], ["b", "t", "h"], A, "output of layer 1 (same graph again)")
add("final_norm", rec["final_norm"], ["b", "t", "h"], A, "final RMSNorm")
add("logits_top5", top.values, ["v"], A, "the 5 largest next-token logits (last position)")
# weights
a0 = L0.self_attn
add("E_rows", model.model.embed_tokens.weight[ids[0]], ["t", "h"], W, "rows of the embedding table E for the input ids (the table has 151,936 rows)")
add("Wq", a0.q_proj.weight.view(N, D, H), ["n", "d", "h"], W, "q_proj weight, PyTorch layout [out=(n d), in=h]")
add("Wk", a0.k_proj.weight.view(K, D, H), ["k", "d", "h"], W, "k_proj weight")
add("Wv", a0.v_proj.weight.view(K, D, H), ["k", "d", "h"], W, "v_proj weight")
add("Wo", a0.o_proj.weight.view(H, N, D), ["h", "n", "d"], W, "o_proj weight, layout [out=h, in=(n d)]")
add("g_norm1", L0.input_layernorm.weight, ["h"], W, "input RMSNorm scale")
add("g_qnorm", a0.q_norm.weight, ["d"], W, "q_norm scale")
add("g_knorm", a0.k_norm.weight, ["d"], W, "k_norm scale")
add("g_norm2", L0.post_attention_layernorm.weight, ["h"], W, "post-attention RMSNorm scale")
add("Wgate", L0.mlp.gate_proj.weight, ["i", "h"], W, "gate_proj weight")
add("Wup", L0.mlp.up_proj.weight, ["i", "h"], W, "up_proj weight")
add("Wdown", L0.mlp.down_proj.weight, ["h", "i"], W, "down_proj weight")
add("cos", COS, ["t", "d"], W, "RoPE cos table for positions 0..t-1")
add("sin", SIN, ["t", "d"], W, "RoPE sin table")
add("g_final", model.model.norm.weight, ["h"], W, "final RMSNorm scale")
add("W_lm_top5", model.lm_head.weight[top_ids], ["v", "h"], W, "the 5 rows of lm_head (out of 151,936) for the top-5 tokens")

out = dict(
    meta=dict(model=NAME, text=TEXT, tokens=tokens, ids=ids_list, parameters=n_params,
              config=dict(hidden=H, heads=N, kv_heads=K, head_dim=D, mlp=I, layers=cfg.num_hidden_layers,
                          vocab=cfg.vocab_size, eps=EPS, rope_theta=THETA, act=cfg.hidden_act, scale=D ** -0.5),
              top5=dict(ids=top_ids, tokens=top_tokens, logits=[round(float(v), 5) for v in top.values]),
              note="Activations carry b=1. Weights use PyTorch layout [out,in]. mask uses -1e9 for minus infinity."),
    tensors=TENS)
with open(OUT, "w") as f:
    json.dump(out, f, separators=(",", ":"))
print("wrote", OUT, os.path.getsize(OUT), "bytes")
