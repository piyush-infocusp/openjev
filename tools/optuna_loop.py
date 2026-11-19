"""Ask/tell Optuna loop for OpenJev training speed (the local master study).

The study lives here (SQLite); the trials run on Kaggle with bench_trials.py.
One round:

  .venv/bin/python tools/optuna_loop.py propose --n 8       # study.ask() -> kaggle/prism/trials.json
      [--enqueue-json '{"bs": 2, "profile": 1}' ...]        # + exact configs (missing keys = yaml defaults)
  PROFILE=optuna kaggle/prism/build.sh  (+ push)            # Kaggle runs bench_trials.py
  .venv/bin/python tools/optuna_loop.py ingest optuna_results.json   # study.tell()
  .venv/bin/python tools/optuna_loop.py report              # docs/speed/EXPERIMENTS.md

Objective: maximize samples_per_sec. OOM, NaN, a kernel that does not exist and a
failed quality guard are PRUNED (TPE treats them as bad points, so it moves away
from them); any other error, and a proposed trial with no result, is FAIL.

Quality guard: bench_trials measures the loss on a fixed probe set before training
(loss_start) and after the warm-up steps (loss_guard). Every trial starts from the same
weights and data and has taken the same number of optimizer steps at loss_guard, so
the guard does not punish a config for being slower (fewer steps in the timed loop).
loss_guard and loss_end must be finite, and the guard drop must be at least
    baseline_drop - max(guard_abs, guard_rel * |baseline_drop|).
loss_end (after the timed loop) is only informational.

Rounds: the yaml has `round: N`. Each round gets its own baseline trial (the yaml
defaults, enqueued first by propose), and the guard and the "vs base" column only use
the baseline of the current round (round 1's baseline ran under the old OJ-080 mask
and other defaults). Trials and baselines without a round attr are round 1.

Dimensions with `sample: false` in the yaml (bs, ple_cpu, dtype, profile) are never drawn
by TPE: trials use the default unless an --enqueue-json config sets them (e.g. the ple_cpu
0/1 pair). profile=1 trials still report samples/s as their objective; their profiler table
is listed under "Profiles". propose writes each trial's FULL effective config (the yaml
defaults of the non-sampled dims too) into trials.json, so bench never falls back to a
train_prism CLI default (round 2 ran ple_cpu=1 that way: train_prism --ple-cpu defaults to 1).

Round 3+ (Idea 5): the yaml study settings global_batch / micro_batch fix the micro-batch
(8) and the global batch (32, grad_accum 4) for every trial: samples/s is measured at that
fixed size; a config that runs out of memory at it is PRUNED (infeasible_at_mb8). The
round baseline also records whether micro-batch 16 WOULD fit (study attr
mb16_fits_round_<N>), which never changes the micro-batch. Every trial trains on the
frozen trial stream (`freeze`, runs/optuna/trial_stream.json; sha in the study attr
trial_stream_sha) for warmup_steps + timed_steps optimizer steps, so all trials see the
same records.

Quality checkpoints: `validate-plan` (due every validate_every completed trials, or with
--end-of-round) writes kaggle/prism/validate.json for PROFILE=optuna_validate (best and
baseline config, validate_steps real optimizer steps each, eval on the frozen val set);
`ingest-validate` stores val_* attrs on the best trial and rejects it (quality_reject)
if its val nll is worse than the baseline's by more than 0.05 or its acc by more than
1 point.

  .venv/bin/python tools/optuna_loop.py freeze               # once per study (round 3)
  .venv/bin/python tools/optuna_loop.py validate-plan [--end-of-round]
  .venv/bin/python tools/optuna_loop.py ingest-validate validate_results.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import optuna
import yaml
from optuna_dashboard import get_note, save_note
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.trial import TrialState

ROOT = Path(__file__).resolve().parent.parent
BASELINE_S_PER_SAMPLE = 4.5
STUDY_NAME = "openjev-speed-v2"       # default; the yaml `study_name` wins
# Rounds 1-2 (old mask, variable bs). The original "openjev-speed" was copied to LEGACY_STUDY
# in runs/optuna/study.db; both names are treated as legacy (annotate --backfill).
LEGACY_STUDY = "LEGACY-r1-r2 (old mask, variable bs; not comparable)"
LEGACY_STUDIES = ("openjev-speed", LEGACY_STUDY)
MANUAL_MARKER = "<!-- manual: kept by report -->"
KERNEL = "piyushsar/openjev-prism-optuna"
EXPERIMENTS = ROOT / "docs/speed/EXPERIMENTS.md"
OBJECTIVE = ("maximize training samples/s of Gemma 4 E2B + Prism mask on Kaggle 2xT4 without quality "
             "loss; quality guard = loss drop vs round baseline")
STREAM_PATH = ROOT / "runs/optuna/trial_stream.json"
VAL_SET_PATH = ROOT / "runs/optuna/val_set.json"
# Quality checkpoint acceptance: val nll at most +0.05 and acc at most -1 point vs baseline.
VAL_NLL_TOL = 0.05
VAL_ACC_TOL = 0.01
# Facts the fixed-conditions block states (checked by tests against train_prism).
LORA = {"r": 16, "alpha": 32, "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                          "up_proj", "down_proj"]}
ROUND_HISTORY = (
    "r1: OLD Prism mask (pre OJ-080: flags saw sibling options), emulated bf16, bs sampled "
    "(one optimizer step per micro-batch). r2: fixed mask and fp16, but bs still sampled and, "
    "because sample:false dims were not written into trials.json, every r2 trial ran ple_cpu=1 "
    "(train_prism default), not the yaml default 0. r3: fixed micro-batch 8 / global batch 32, "
    "full config in trials.json, frozen trial data. Rounds are not comparable with each other.")
METRIC_NAME = ("train samples/sec (higher=better; Gemma4-E2B+Prism, Kaggle T4; baseline 0.22 = "
               "4.5 s/sample)")
META_KEYS = ("idea", "hypothesis", "reference")
TPE_IDEA = ("TPE exploration", "model-based search over implemented dims")
BASELINE_IDEA = "round baseline; re-anchors the quality guard"
BASELINE_HYP = "yaml defaults of this round; the quality guard compares the loss drop of later trials to it"


# ---------------------------------------------------------------- search space

def load_space(path) -> list[dict]:
    dims = yaml.safe_load(Path(path).read_text())["dimensions"]
    seen = set()
    for d in dims:
        for k in ("name", "type", "default", "requires_reload", "implemented"):
            if k not in d:
                raise ValueError(f"search space entry {d.get('name')!r} misses {k!r}")
        if d["name"] in seen:
            raise ValueError(f"duplicate dimension {d['name']!r}")
        seen.add(d["name"])
        if d["type"] == "categorical" and d["default"] not in d["choices"]:
            raise ValueError(f"{d['name']}: default {d['default']!r} not in choices {d['choices']}")
        d.setdefault("sample", True)
    return dims


def load_round(path) -> int:
    return int(yaml.safe_load(Path(path).read_text()).get("round", 1))


SETTING_KEYS = ("global_batch", "micro_batch", "warmup_steps", "timed_steps", "diag_mb",
                "trial_stream_len", "probe_n", "validate_every", "validate_steps", "val_per_source")


def load_settings(path) -> dict:
    """Study-level constants (not sampled). {} for a yaml without them (rounds 1-2)."""
    raw = yaml.safe_load(Path(path).read_text())
    st = {k: raw[k] for k in SETTING_KEYS if raw.get(k) is not None}
    if bool(st.get("global_batch")) != bool(st.get("micro_batch")):
        raise ValueError("set both global_batch and micro_batch, or neither")
    if st.get("global_batch") and st["global_batch"] % st["micro_batch"]:
        raise ValueError(f"global_batch {st['global_batch']} is not a multiple of micro_batch "
                         f"{st['micro_batch']}")
    return st


def bench_settings(st: dict, stream_sha=None) -> dict:
    """The study settings bench_trials needs, written into every trials.json entry."""
    if not st.get("global_batch"):
        return {}
    out = {k: st[k] for k in ("global_batch", "micro_batch", "warmup_steps", "timed_steps") if k in st}
    if stream_sha:
        out["trial_stream_sha"] = stream_sha
    return out


def read_frozen(path):
    """(sha, data) of a frozen trial stream / val set file, or (None, None) if missing."""
    p = Path(path)
    if not p.exists():
        return None, None
    d = json.loads(p.read_text())
    return d.get("sha"), d


def trial_round(t) -> int:
    return int(t.user_attrs.get("round", 1))


def distribution(d: dict):
    if d["type"] == "categorical":
        return CategoricalDistribution(d["choices"])
    if d["type"] == "int":
        return IntDistribution(d["low"], d["high"], step=d.get("step", 1), log=d.get("log", False))
    if d["type"] == "float":
        return FloatDistribution(d["low"], d["high"], step=d.get("step"), log=d.get("log", False))
    raise ValueError(f"{d['name']}: unknown type {d['type']!r}")


def implemented(dims):
    return [d for d in dims if d["implemented"]]


def parse_enqueue(texts, impl, with_meta=False):
    """--enqueue-json values (an object or a list of objects) -> exact configs: every
    implemented dimension set, missing ones from the yaml default, values checked.
    Optional keys _idea / _hypothesis / _reference are removed before the check; with
    with_meta the result is a list of (config, {idea, hypothesis, reference})."""
    by_name = {d["name"]: d for d in impl}
    out = []
    for text in texts or []:
        obj = json.loads(text)
        for cfg in obj if isinstance(obj, list) else [obj]:
            if not isinstance(cfg, dict):
                raise SystemExit(f"--enqueue-json: expected an object, got {cfg!r}")
            meta = {k: str(cfg[f"_{k}"]) for k in META_KEYS if f"_{k}" in cfg}
            cfg = {k: v for k, v in cfg.items() if k not in {f"_{m}" for m in META_KEYS}}
            bad = sorted(set(cfg) - set(by_name))
            if bad:
                raise SystemExit(f"--enqueue-json: {bad} are not implemented dimensions "
                                 f"({sorted(by_name)})")
            full = {}
            for name, d in by_name.items():
                v = cfg.get(name, d["default"])
                ok = (v in d["choices"]) if d["type"] == "categorical" else (
                    isinstance(v, (int, float)) and d["low"] <= v <= d["high"])
                if not ok:
                    raise SystemExit(f"--enqueue-json: {name}={v!r} is outside {d.get('choices', d)}")
                full[name] = v
            out.append((full, meta) if with_meta else full)
    return out


# ---------------------------------------------------------------- study

def open_study(db: Path, seed: int, name: str = STUDY_NAME) -> optuna.Study:
    db.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{db}"
    # Each CLI call is a new process. A fixed sampler seed would replay the same
    # random start-up points every round, so offset it by the number of trials.
    n = 0
    try:
        n = len(optuna.load_study(study_name=name, storage=storage).trials)
    except KeyError:
        pass
    # Params added in a later round (attn_backend, dtype) are missing from older trials,
    # so multivariate TPE leaves them out of its joint model and samples them on their
    # own; enqueue the interactions that matter (--enqueue-json) instead.
    sampler = optuna.samplers.TPESampler(multivariate=True, constant_liar=True, seed=seed + n)
    return optuna.create_study(study_name=name, storage=storage, direction="maximize",
                               sampler=sampler, load_if_exists=True)


def running_trial(study, number: int):
    """A writable handle for a RUNNING trial (user attrs must be set before tell)."""
    for ft in study.get_trials(deepcopy=False):
        if ft.number == number:
            return ft, optuna.trial.Trial(study, ft._trial_id)
    return None, None


# ---------------------------------------------------------------- self-describing trials

def git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True)
        return r.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def find_trial(study, number: int):
    return next((t for t in study.get_trials(deepcopy=False) if t.number == number), None)


def handle(study, ft):
    """What optuna_dashboard.save_note/get_note need from a trial (works in any state)."""
    return SimpleNamespace(storage=study._storage, study=study, _trial_id=ft._trial_id)


@contextlib.contextmanager
def writable(study):
    """Optuna refuses user attrs on finished trials; annotating a finished trial is only
    metadata, so lift that one check on this storage object for the block."""
    sts = [study._storage]
    if hasattr(sts[0], "_backend"):   # _CachedStorage delegates to the RDB backend
        sts.append(sts[0]._backend)
    for st in sts:
        st.check_trial_is_updatable = lambda *a, **k: None
    try:
        yield
    finally:
        for st in sts:
            del st.check_trial_is_updatable


def set_attrs(study, ft, attrs: dict):
    with writable(study):
        for k, v in attrs.items():
            study._storage.set_trial_user_attr(ft._trial_id, k, v)


def space_snapshot(impl) -> str:
    snap = {}
    for d in impl:
        vals = d["choices"] if d["type"] == "categorical" else [d["low"], d["high"]]
        snap[d["name"]] = {"values": vals, "default": d["default"]}
        if not d["sample"]:
            snap[d["name"]]["sampled"] = False
    return json.dumps(snap, separators=(",", ":"))


def round_history(path=None) -> str:
    """The '## Ideas ...' sections of EXPERIMENTS.md (hand-written part), one level deeper."""
    try:
        text = Path(path or EXPERIMENTS).read_text()
    except OSError:
        return ""
    if MANUAL_MARKER in text:
        text = text[text.index(MANUAL_MARKER):]
    out, keep = [], False
    for line in text.splitlines():
        if line.startswith("## "):
            keep = line.startswith("## Ideas")
            if keep:
                out.append("#" + line)
            continue
        if keep:
            out.append(line)
    return "\n".join(out).strip()


def short_sha(obj) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:10]


def fixed_conditions(dims, st: dict, stream_sha=None, val_sha=None) -> dict:
    """Everything held constant in this round (study note, study attr `fixed_conditions`).
    Its short hash is the per-trial attr fixed_conditions_version."""
    impl = implemented(dims)
    fixed = {d["name"]: d["default"] for d in impl if not d["sample"]}
    sampled = {d["name"]: d.get("choices", [d.get("low"), d.get("high")]) for d in impl if d["sample"]}
    gb, mb = st.get("global_batch"), st.get("micro_batch")
    c = {
        "model": "google/gemma-4-E2B-it, text decoder only (vision and audio towers dropped)",
        "prism_mask": "after the OJ-080 fix (6df0efe): a flag sees the shared tokens, its own option "
                      "and its sibling FLAGS, not the sibling options",
        "dtype": f"{fixed.get('dtype', '?')} base weights + GradScaler (dtype dim fixed); never "
                 "emulated bf16 (OJ-081)",
        "loader": "load in bf16, then convert per tensor (2cf488f, host-RAM fix)",
        "ple": f"PENDING: paired test in round 3 (ple_cpu 0 vs 1, everything else equal, at "
               f"micro-batch {mb}); TPE trials use ple_cpu={fixed.get('ple_cpu', '?')}. CPU path: "
               "pinned memory + non_blocking host->device copy",
        "batch": (f"micro-batch {mb}, global batch {gb}, grad_accum {gb // mb} (OOM at micro-batch "
                  f"{mb} = PRUNED infeasible_at_mb{mb})") if gb else "bs sampled (no global batch)",
        "fixed_dims": fixed,
        "sampled_dims": sampled,
        "max_state_tokens": ("SAMPLED " + str(sampled["max_state_tokens"]) + ": NOT fixed; it changes "
                             "sequence lengths and quality, like bs did") if "max_state_tokens" in sampled
                            else f"fixed {fixed.get('max_state_tokens')}",
        "lora": f"LoRA r{LORA['r']}/alpha{LORA['alpha']} on {','.join(t.replace('_proj', '') for t in LORA['targets'])}; "
                "base weights and embeddings frozen, fp32 template delta for the template tokens; "
                "every trainable tensor fp32",
        "trial_stream_sha": stream_sha,
        "val_set_sha": val_sha,
        "timing": (f"{st.get('warmup_steps', 1)} warm-up optimizer step(s), then {st.get('timed_steps')} timed "
                   "optimizer steps on the same frozen records (TRIAL_SECONDS is not used for timing)"
                   if st.get("timed_steps") else "warm-up steps, then a TRIAL_SECONDS timed loop"),
        "quality_guard": "probe-loss drop after the warm-up steps >= round baseline drop - "
                         "max(0.1, 0.5 x |baseline drop|)",
        "quality_checkpoint": (f"every {st.get('validate_every')} completed trials and at round end: "
                               f"{st.get('validate_steps')} optimizer steps, best vs baseline on the val "
                               f"set; accept if nll <= base + {VAL_NLL_TOL} and acc >= base - "
                               f"{VAL_ACC_TOL * 100:.0f} pt") if st.get("validate_every") else None,
        "hardware": "Kaggle 2x T4 (single-process bench: one GPU until DDP)",
        "rounds": ROUND_HISTORY,
    }
    return c


def conditions_md(c: dict) -> str:
    L = []
    for k, v in c.items():
        if v is None:
            continue
        if isinstance(v, dict):
            v = ", ".join(f"{a}={b}" for a, b in v.items())
        L.append(f"- **{k}**: {v}")
    return "\n".join(L)


def study_note(rnd: int, st=None, conditions=None, name: str = STUDY_NAME) -> str:
    st = st or {}
    legacy = ("" if name in LEGACY_STUDIES else
              f"\n\nThis study (`{name}`) holds only trials under the fixed conditions below. The legacy "
              f"study `{LEGACY_STUDY}` (rounds 1-2: old mask, emulated bf16, sampled bs, r2 ran ple_cpu=1) "
              "is kept untouched in the same study.db for reference; its numbers are not comparable.")
    hist = round_history() or "_No round history found in EXPERIMENTS.md._"
    gb, mb = st.get("global_batch"), st.get("micro_batch")
    if gb:
        what = (f"objective = samples/s at a FIXED micro-batch {mb} and global batch {gb} (grad_accum "
                f"{gb // mb}): each timed step is one optimizer step of {gb} samples. The micro-batch is "
                f"the same for every trial, so a memory-saving technique gets credit only by making a "
                f"config feasible at micro-batch {mb} (out of memory there = PRUNED infeasible_at_mb{mb}). "
                f"Each trial: {st.get('warmup_steps', 1)} warm-up optimizer step(s), then "
                f"{st.get('timed_steps')} timed optimizer steps on the frozen trial stream (same records "
                "for every trial), from the same start weights")
        guard = ("the probe loss on a fixed set is measured before training and after the warm-up "
                 f"optimizer steps (same steps, same {gb}-sample batches for every trial, so guards are "
                 "comparable across trials)")
    else:
        what = ("Each trial runs 2 warm-up steps, then a timed loop of real SFT steps on the real data "
                "mixture from the same start weights; the value is samples per second of that timed loop")
        guard = ("the probe loss on a fixed set is measured before training and after the warm-up "
                 "steps (same optimizer-step count for every trial)")
    cond = f"\n### Fixed conditions\n\n{conditions_md(conditions)}\n" if conditions else ""
    return f"""## OpenJev training-speed study

### Objective

**{METRIC_NAME}**

- **What is measured:** {OBJECTIVE}. {what}.
- **Quality guard:** {guard}. The loss drop must be at least the round baseline's drop minus
  max(0.1, 0.5 x |baseline drop|). A trial that fails the guard, runs out of memory, produces
  NaN or uses an unsupported kernel is PRUNED, not rewarded.
- **Baseline:** {BASELINE_S_PER_SAMPLE} s/sample (v7/v8) = {1 / BASELINE_S_PER_SAMPLE:.2f} samples/s.
- **Reading the speedup:** speedup = samples/s x {BASELINE_S_PER_SAMPLE} (above 1.0x is faster than
  v7/v8). `vs base` in EXPERIMENTS.md is against the measured baseline trial of the same round.
{cond}
Kernel: `{KERNEL}`. Current round: {rnd}.{legacy}

### How to read the trials

- Each trial has a note (idea, hypothesis, reference, params, result) and user attrs
  `source` (baseline | enqueued | tpe), `round`, `git_commit`, `idea`, `hypothesis`, `reference`,
  `fixed_conditions_version`, and after ingest `mb_used`, `grad_accum`.
- Value = samples/s (higher is better). COMPLETE = ran and passed the guard; PRUNED = OOM
  (infeasible at the round micro-batch), NaN, unsupported kernel or failed quality guard
  (`reason` attr); FAIL = error or no result; RUNNING = proposed, not ingested yet.
- Compare trials only within a round: every round has its own baseline (`is_baseline`), and the
  quality guard (loss drop after the warm-up steps) is against that baseline.
- Quality checkpoints (`val_*` attrs, `quality_reject`) come from a separate validation run.
- `search_space_round_<N>` (study attrs) is the snapshot of the implemented dimensions.

### Round history

{hist}

### Links

- [SPEED_LOG.md](docs/speed/SPEED_LOG.md)
- [EXPERIMENTS.md](docs/speed/EXPERIMENTS.md)
"""


def set_study_meta(study, impl, rnd: int, st=None, conditions=None):
    st = st or {}
    study.set_metric_names([METRIC_NAME])
    study.set_user_attr("objective", OBJECTIVE)
    study.set_user_attr("baseline_s_per_sample", BASELINE_S_PER_SAMPLE)
    study.set_user_attr("kernel", KERNEL)
    study.set_user_attr(f"search_space_round_{rnd}", space_snapshot(impl))
    if st.get("global_batch"):
        study.set_user_attr(f"micro_batch_round_{rnd}", st["micro_batch"])
        study.set_user_attr(f"global_batch_round_{rnd}", st["global_batch"])
    if conditions:
        study.set_user_attr("fixed_conditions", conditions)
        study.set_user_attr(f"fixed_conditions_round_{rnd}", short_sha(conditions))
    save_note(study, study_note(rnd, st, conditions, study.study_name))


def link(ref: str) -> str:
    return " ".join(f"<{w}>" if w.startswith("http") else w for w in ref.split(" "))


def trial_note(number: int, attrs: dict, params: dict, extra: str = "") -> str:
    ps = " ".join(f"{k}={v}" for k, v in params.items()) or "(none)"
    L = [f"## Trial {number}: {attrs.get('idea') or '(no idea recorded)'}", "",
         f"- source: **{attrs.get('source', '?')}** | round: {attrs.get('round', '?')} | "
         f"commit: `{attrs.get('git_commit', '?')}`",
         f"- hypothesis: {attrs.get('hypothesis') or '-'}",
         f"- reference: {link(attrs['reference']) if attrs.get('reference') else '-'}",
         f"- params: `{ps}`"]
    if extra:
        L += ["", extra]
    return "\n".join(L) + "\n"


RESULT_MARK = "\n### Result\n"


def result_md(state: str, u: dict) -> str:
    sps = u.get("samples_per_sec")
    ok = isinstance(sps, (int, float)) and sps > 0
    L = [f"- status: {state}" + (f" ({u['status']})" if u.get("status") else "")]
    if u.get("reason"):
        L.append(f"- reason: {u['reason']}")
    if ok:
        L.append(f"- samples/s: {sps:.3f} | s/sample: {1 / sps:.2f} | speedup vs "
                 f"{BASELINE_S_PER_SAMPLE}: {sps * BASELINE_S_PER_SAMPLE:.2f}x")
    if u.get("peak_mem_gb") is not None:
        L.append(f"- peak memory: {u['peak_mem_gb']:.2f} GB")
    if u.get("mb_used") is not None:
        L.append(f"- micro-batch {u['mb_used']} x grad_accum {u.get('grad_accum')} = global batch "
                 f"{u.get('global_batch')}")
    if isinstance(u.get("mb_diag"), dict):
        d = u["mb_diag"]
        L.append(f"- diagnostic: micro-batch {d.get('mb')} would {'fit' if d.get('fits') else 'NOT fit'} "
                 f"(peak {d.get('peak_gb')} GB of {d.get('limit_gb')} GB, oom={d.get('oom')}); not used")
    if u.get("error"):
        L.append(f"- error: `{str(u['error']).replace(chr(10), ' ')[:200]}`")
    p = u.get("profile")
    if isinstance(p, dict):
        for key, title in (("top_cuda", "self CUDA"), ("top_cpu", "self CPU")):
            rows = (p.get(key) or [])[:5]
            if rows:
                L.append(f"- top-5 profile ops by {title}: " + "; ".join(
                    f"`{r[0]}` {r[1]} ms ({pct(r[2])})" for r in rows))
    return RESULT_MARK.lstrip("\n") + "\n".join(L) + "\n"


def write_trial_note(study, ft, extra: str = "", new_attrs=None):
    """(Re)write the trial note from its attrs (+ new_attrs: the cached trial is stale after a
    write to a finished trial); add the result section when it has finished."""
    ft = find_trial(study, ft.number)
    u = {**ft.user_attrs, **(new_attrs or {})}
    note = trial_note(ft.number, u, ft.params, extra)
    if ft.state not in (TrialState.RUNNING, TrialState.WAITING):
        note += "\n" + result_md(ft.state.name, u)
    save_note(handle(study, ft), note)


def append_result(study, number: int):
    """Ingest: append the result to the trial note (replaces an earlier result section)."""
    ft = find_trial(study, number)
    h = handle(study, ft)
    old = get_note(h)
    if not old:
        old = trial_note(number, ft.user_attrs, ft.params)
    old = old.split(RESULT_MARK.strip("\n"))[0].rstrip("\n") + "\n"
    save_note(h, old + "\n" + result_md(ft.state.name, ft.user_attrs))


# ---------------------------------------------------------------- propose

def frozen_shas(a, study, st: dict):
    """(stream_sha, val_sha) for round 3+ settings: the frozen files must exist (`freeze`)
    and match what the study already recorded (frozen once per study)."""
    if not st.get("global_batch"):
        return None, None
    stream_sha, _ = read_frozen(a.stream)
    val_sha, _ = read_frozen(a.val_set)
    if stream_sha is None or val_sha is None:
        raise SystemExit(f"no frozen trial data ({a.stream}, {a.val_set}): run "
                         "`tools/optuna_loop.py freeze` once for this study first")
    for key, sha in (("trial_stream_sha", stream_sha), ("val_set_sha", val_sha)):
        old = study.user_attrs.get(key)
        if old and old != sha:
            raise SystemExit(f"{key}: the study was run on {old}, the file is {sha}. The frozen "
                             "data must not change within a study (restore the file).")
    return stream_sha, val_sha


def cmd_propose(a):
    dims = load_space(a.space)
    st = load_settings(a.space)
    study = open_study(a.db, a.seed, a.study)
    stream_sha, val_sha = frozen_shas(a, study, st)
    running = [t for t in study.get_trials(deepcopy=False) if t.state == TrialState.RUNNING]
    if running:
        nums = [t.number for t in running]
        if not a.abandon_running:
            raise SystemExit(f"trials {nums} were proposed but never ingested. Ingest their "
                             "results first, or pass --abandon-running to mark them FAIL.")
        for t in running:
            study.tell(t.number, state=TrialState.FAIL)
        print(f"marked {nums} FAIL (abandoned)")
    impl = implemented(dims)
    if not impl:
        raise SystemExit("no implemented dimensions in the search space")
    queued = parse_enqueue(a.enqueue_json, impl, with_meta=True)
    dists = {d["name"]: distribution(d) for d in impl}
    sampled = {d["name"] for d in impl if d["sample"]}
    defaults = {d["name"]: d["default"] for d in impl}
    reload_keys = [d["name"] for d in impl if d["requires_reload"]]
    # Trial 0 is the baseline. Re-enqueue it if every baseline so far failed or was abandoned.
    alive = (TrialState.COMPLETE, TrialState.RUNNING, TrialState.WAITING)
    rnd = load_round(a.space)
    conditions = fixed_conditions(dims, st, stream_sha, val_sha) if st.get("global_batch") else None
    cver = short_sha(conditions) if conditions else None
    if stream_sha:
        study.set_user_attr("trial_stream_sha", stream_sha)
        study.set_user_attr("val_set_sha", val_sha)
    set_study_meta(study, impl, rnd, st, conditions)
    need_base = not any(t.user_attrs.get("is_baseline") and t.state in alive
                        and trial_round(t) == rnd for t in study.trials)
    if need_base:
        study.enqueue_trial({n: v for n, v in defaults.items() if n in sampled},
                            user_attrs={"is_baseline": True, "round": rnd, "source": "baseline",
                                        "idea": BASELINE_IDEA, "hypothesis": BASELINE_HYP,
                                        "reference": ""})
    for cfg, meta in queued:   # a sample: false dim at its default stays out of the Optuna params
        study.enqueue_trial({k: v for k, v in cfg.items() if k in sampled or v != defaults.get(k)},
                            user_attrs={"enqueued": True, "round": rnd, "source": "enqueued",
                                        "idea": meta.get("idea", "enqueued config"),
                                        "hypothesis": meta.get("hypothesis", ""),
                                        "reference": meta.get("reference", "")})
    commit = git_commit()
    bset = bench_settings(st, stream_sha)
    out = []
    # The baseline (when it is enqueued) counts toward --n; the --enqueue-json configs do not.
    for _ in range(max(a.n, int(need_base)) + len(queued)):
        trial = study.ask()
        fixed = getattr(trial, "_fixed_params", {}) or {}
        if "round" not in trial.user_attrs:
            trial.set_user_attr("round", rnd)
        if "source" not in trial.user_attrs:
            for k, v in (("source", "tpe"), ("idea", TPE_IDEA[0]), ("hypothesis", TPE_IDEA[1]),
                         ("reference", "")):
                trial.set_user_attr(k, v)
        trial.set_user_attr("git_commit", commit)
        if cver:
            trial.set_user_attr("fixed_conditions_version", cver)
            trial.set_user_attr("global_batch", st["global_batch"])
            trial.set_user_attr("micro_batch", st["micro_batch"])
        params = {}
        for name, dist in dists.items():
            if name in sampled or name in fixed:
                params[name] = dist_suggest(trial, name, dist)
            # sample: false and not enqueued -> not an Optuna param; bench gets the default below
        full = {**defaults, **params}       # the full effective config: bench never guesses
        if st.get("micro_batch") and "bs" not in params:
            full.pop("bs", None)            # bench ignores bs at a fixed micro-batch
        save_note(trial, trial_note(trial.number, {**trial.user_attrs, "git_commit": commit}, params,
                                    f"- fixed: `{' '.join(f'{k}={v}' for k, v in full.items() if k not in params)}`"
                                    + (f" | micro-batch {st['micro_batch']}, global batch {st['global_batch']}"
                                       if st.get("micro_batch") else "")))
        entry = {"trial_number": trial.number, "params": full, "reload_keys": reload_keys}
        if bset:
            entry["bench"] = dict(bset, round=rnd)
            if trial.user_attrs.get("is_baseline") and st.get("diag_mb"):
                entry["bench"]["diag_mb"] = st["diag_mb"]
        out.append(entry)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    for t in out:
        print(json.dumps(t))
    print(f"wrote {len(out)} trials to {a.out}")


def dist_suggest(trial, name, dist):
    # suggest_* (not ask(fixed_distributions)) so an enqueued baseline is applied.
    if isinstance(dist, CategoricalDistribution):
        return trial.suggest_categorical(name, dist.choices)
    if isinstance(dist, IntDistribution):
        return trial.suggest_int(name, dist.low, dist.high, step=dist.step, log=dist.log)
    return trial.suggest_float(name, dist.low, dist.high, step=dist.step, log=dist.log)


# ---------------------------------------------------------------- ingest

ATTRS = ("samples_per_sec", "tokens_per_sec", "padded_tokens_per_sec", "pad_frac", "s_per_step",
         "s_per_sample", "steps", "samples", "seconds", "peak_mem_gb", "loss_start",
         "loss_guard", "guard_steps", "loss_end", "train_loss_first", "train_loss_last",
         "scale_start", "scale_end", "skipped_steps", "n_gpu", "gpu_name", "status",
         "error", "group_load_s", "attn_backend", "warmup_s", "is_profile", "profile",
         "sdpa_kernels", "graphs_in_timed_loop", "dtype", "mb_used", "grad_accum", "global_batch",
         "fixed_mb", "bs_ignored", "mb_diag", "timed_steps", "warmup_ids_sha", "data_ids_sha",
         "records_used", "trial_stream_sha", "stream_wrapped")
PRUNE_STATUS = {"oom", "nan", "unsupported"}


def is_pruned_status(status: str) -> bool:
    """OOM at the round's fixed micro-batch (infeasible_at_mb8) is pruned like OOM."""
    return status in PRUNE_STATUS or status == "quality" or str(status).startswith("infeasible_at_mb")


def finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def guard_loss(r: dict):
    """The fixed-step probe loss (older results only have loss_end)."""
    return r.get("loss_guard", r.get("loss_end"))


def baseline_drop(study, results, rnd: int = 1):
    """Guard-loss drop of this round's baseline: from this results file first, else the
    study. Baselines of earlier rounds are never used (other mask / defaults)."""
    base_nums = {t.number for t in study.get_trials(deepcopy=False)
                 if t.user_attrs.get("is_baseline") and trial_round(t) == rnd}
    for r in results:
        if r.get("trial_number") in base_nums and r.get("status") == "ok" \
                and finite(r.get("loss_start")) and finite(guard_loss(r)):
            return r["loss_start"] - guard_loss(r)
    for t in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)):
        if t.number in base_nums:
            s, e = t.user_attrs.get("loss_start"), guard_loss(t.user_attrs)
            if finite(s) and finite(e):
                return s - e
    return None


def stats_state_tokens(path=None) -> dict:
    """data/STATS.md "State token length": {file stem: (p50, p95)} for the train files."""
    try:
        text = Path(path or ROOT / "data/STATS.md").read_text()
    except OSError:
        return {}
    out, on = {}, False
    for line in text.splitlines():
        if line.startswith("## "):
            on = line.strip() == "## State token length"
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if on and len(cells) == 5 and cells[0].endswith(".jsonl"):
            try:
                out[cells[0][:-len(".jsonl")]] = (float(cells[2]), float(cells[3]))
            except ValueError:
                pass
    return out


def cmd_ingest(a):
    study = open_study(a.db, a.seed, a.study)
    raw = json.loads(Path(a.results).read_text())
    results = raw["results"] if isinstance(raw, dict) else raw
    meta = raw.get("meta", {}) if isinstance(raw, dict) else {}
    # The guard uses the baseline of each trial's OWN round (not the yaml round), so a
    # round can be ingested after the yaml moved on to the next one.
    bases = {}

    def base_for(rnd):
        if rnd not in bases:
            bases[rnd] = baseline_drop(study, results, rnd)
            if bases[rnd] is None:
                print(f"warning: no round-{rnd} baseline in the results or the study: quality guard off")
        return bases[rnd]
    if isinstance(meta.get("stream_state_tokens"), dict):
        ref = stats_state_tokens()
        study.set_user_attr("trial_stream_state_tokens", {
            s: {**v, "stats_p50": ref.get(s, (None, None))[0], "stats_p95": ref.get(s, (None, None))[1]}
            for s, v in meta["stream_state_tokens"].items()})
    told = set()
    for r in sorted(results, key=lambda r: r["trial_number"]):
        n = r["trial_number"]
        ft, trial = running_trial(study, n)
        if ft is None:
            print(f"warning: trial {n} is not in the study; skipped")
            continue
        if ft.state != TrialState.RUNNING:
            print(f"warning: trial {n} is already {ft.state.name}; skipped (stale results?)")
            continue
        if ft.params and r.get("params") and any(ft.params.get(k) != v for k, v in r["params"].items()
                                                 if k in ft.params):
            print(f"warning: trial {n} ran with params {r['params']} but the study has "
                  f"{ft.params}; skipped (stale trials.json?)")
            continue
        rnd = trial_round(ft)
        base = base_for(rnd)
        for k in ATTRS:
            if k in r and r[k] is not None:
                trial.set_user_attr(k, r[k])
        d = r.get("mb_diag")
        if isinstance(d, dict) and d.get("mb"):
            # Recorded only: whether micro-batch diag_mb WOULD fit on the round baseline.
            study.set_user_attr(f"mb{d['mb']}_fits_round_{rnd}",
                                {"fits": d.get("fits"), "peak_gb": d.get("peak_gb"),
                                 "limit_gb": d.get("limit_gb"), "oom": d.get("oom"), "trial": n})
        if a.tag:
            trial.set_user_attr("tag", a.tag)
        if meta.get("gpu_name") and "gpu_name" not in r:
            trial.set_user_attr("gpu_name", meta["gpu_name"])
        status, why = r.get("status", "error"), ""
        sps = r.get("samples_per_sec")
        if status == "ok" and not (finite(r.get("loss_end")) and finite(guard_loss(r))):
            status, why = "nan", "probe loss not finite"
        if status == "ok" and not (finite(sps) and sps > 0):
            status, why = "error", "no samples_per_sec"
        if status == "ok" and base is not None and finite(r.get("loss_start")):
            drop = r["loss_start"] - guard_loss(r)
            need = base - max(a.guard_abs, a.guard_rel * abs(base))
            trial.set_user_attr("loss_drop", round(drop, 4))
            if drop < need:
                status, why = "quality", f"loss drop {drop:.3f} < {need:.3f} (baseline {base:.3f})"
        if why:
            trial.set_user_attr("reason", why)
        if status == "ok":
            study.tell(n, sps)
        elif is_pruned_status(status):
            study.tell(n, state=TrialState.PRUNED)
        else:
            study.tell(n, state=TrialState.FAIL)
        append_result(study, n)
        told.add(n)
        print(f"trial {n}: {status} samples/s={sps} {why}".rstrip())
    # Proposed but missing from this results file: the kernel died or timed out.
    if a.fail_missing:
        for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
            if t.number not in told:
                optuna.trial.Trial(study, t._trial_id).set_user_attr("reason", "no result")
                study.tell(t.number, state=TrialState.FAIL)
                append_result(study, t.number)
                print(f"trial {t.number}: FAIL (no result)")


def cmd_abandon(a):
    """Mark the RUNNING trials of --study FAIL (reason 'abandoned'); nothing else changes."""
    study = open_study(a.db, a.seed, a.study)
    nums = []
    for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        optuna.trial.Trial(study, t._trial_id).set_user_attr("reason", "abandoned (no result)")
        study.tell(t.number, state=TrialState.FAIL)
        nums.append(t.number)
    print(f"study {study.study_name}: marked {nums} FAIL (abandoned)")


# ---------------------------------------------------------------- frozen data

def cmd_freeze(a):
    """Build the frozen trial stream + probe set and the val set (once per study)."""
    import glob
    sys.path.insert(0, str(ROOT))
    from openjev import trial_data as td
    st = load_settings(a.space)
    for f in (a.stream, a.val_set):
        if Path(f).exists() and not a.force:
            raise SystemExit(f"{f} exists. The frozen data is built once per study; pass --force "
                             "only for a NEW study (propose refuses a sha that differs from the study's).")
    paths, vpaths = sorted(glob.glob(a.train)), sorted(glob.glob(a.val))
    if not paths or not vpaths:
        raise SystemExit(f"no files match {a.train} / {a.val}")
    n = int(st.get("trial_stream_len", 640))
    need = (int(st.get("warmup_steps", 1)) + int(st.get("timed_steps") or 0)) * int(st.get("global_batch", 1))
    if n < need:
        raise SystemExit(f"trial_stream_len {n} < (warmup_steps + timed_steps) x global_batch = {need}")
    stream = td.freeze_stream(paths, n, int(st.get("probe_n", 8)), alpha=a.alpha, seed=a.data_seed)
    val = td.freeze_val(vpaths, int(st.get("val_per_source", 50)), seed=a.data_seed)
    td.write_json(a.stream, stream)
    td.write_json(a.val_set, val)
    print(f"trial stream: {len(stream['stream'])} records (needs >= {need}), probe {len(stream['probe'])}, "
          f"sha {stream['sha']} -> {a.stream}")
    print("| source | n | chars p50 / p95 (stream) | chars p50 / p95 (whole file) |")
    for s, c in stream["length_check"].items():
        print(f"| {s} | {c['n']} | {c['chars_p50']:.0f} / {c['chars_p95']:.0f} | "
              f"{c['file_chars_p50']:.0f} / {c['file_chars_p95']:.0f} |")
    print(f"val set: {len(val['records'])} records, sha {val['sha']} -> {a.val_set}")
    for s, k in val["strata"].items():
        print(f"  {s}: " + ", ".join(f"{a_}={b}" for a_, b in k.items()))


# ---------------------------------------------------------------- quality checkpoints

def full_config(t, dims, st) -> dict:
    full = {**{d["name"]: d["default"] for d in implemented(dims)}, **t.params}
    if st.get("micro_batch") and "bs" not in t.params:
        full.pop("bs", None)
    return full


def cmd_validate_plan(a):
    dims, st, rnd = load_space(a.space), load_settings(a.space), load_round(a.space)
    if not st.get("global_batch"):
        raise SystemExit("validate-plan needs global_batch / micro_batch in the yaml")
    study = open_study(a.db, a.seed, a.study)
    trials = study.get_trials(deepcopy=False)
    n_complete = sum(t.state == TrialState.COMPLETE for t in trials)
    last = (study.user_attrs.get("last_validation") or {}).get("n_complete", 0)
    every = int(st.get("validate_every", 30))
    if not a.end_of_round and n_complete - last < every:
        print(f"not due: {n_complete - last} new complete trials since the last checkpoint (< {every}); "
              "pass --end-of-round to run it anyway")
        return
    done = [t for t in trials if t.state == TrialState.COMPLETE and trial_round(t) == rnd]
    base = next((t for t in done if t.user_attrs.get("is_baseline")), None)
    if base is None or not done:
        raise SystemExit(f"round {rnd} has no complete baseline yet")
    best = max(done, key=lambda t: t.value)
    if best.number == base.number:
        print(f"the round-{rnd} best is the baseline (trial {base.number}): nothing to compare")
        return
    _, vs = read_frozen(a.val_set)
    plan = {"settings": {"steps": a.steps or int(st.get("validate_steps", 150)),
                         "global_batch": st["global_batch"], "micro_batch": st["micro_batch"],
                         "val_set_sha": vs["sha"] if vs else None, "round": rnd, "n_complete": n_complete},
            "configs": [{"name": "best", "trial_number": best.number, "params": full_config(best, dims, st)},
                        {"name": "baseline", "trial_number": base.number,
                         "params": full_config(base, dims, st)}]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(plan, indent=1) + "\n")
    print(json.dumps(plan))
    print(f"wrote {a.out}: best trial {best.number} vs baseline {base.number}, "
          f"{plan['settings']['steps']} steps each. Run: PROFILE=optuna_validate kaggle/prism/build.sh")


QC_MARK = "### Quality checkpoint"


def qc_md(u: dict) -> str:
    v = u.get("val_vs_baseline") or {}
    b = v.get("baseline") or {}
    L = [QC_MARK, "", f"- verdict: **{'REJECT (quality_reject)' if u.get('quality_reject') else 'accepted'}** "
         f"(nll <= base + {VAL_NLL_TOL}, acc >= base - {VAL_ACC_TOL * 100:.0f} pt)",
         f"- val acc {fmt(u.get('val_acc'), 4)} (base {fmt(b.get('acc'), 4)}), nll {fmt(u.get('val_nll'), 4)} "
         f"(base {fmt(b.get('nll'), 4)}), ece {fmt(u.get('val_ece'), 4)} (base {fmt(b.get('ece'), 4)})",
         f"- {v.get('steps')} optimizer steps at global batch {v.get('global_batch')}, micro-batch "
         f"{v.get('micro_batch')}; val set `{v.get('val_set_sha')}` ({v.get('val_n')} records); "
         f"baseline trial {v.get('baseline_trial')}"]
    return "\n".join(L) + "\n"


def cmd_ingest_validate(a):
    study = open_study(a.db, a.seed, a.study)
    raw = json.loads(Path(a.results).read_text())
    meta, res = raw.get("meta", {}), raw["results"]
    by = {r["name"]: r for r in res}
    best, base = by.get("best"), by.get("baseline")
    if not best or not base:
        raise SystemExit(f"need a 'best' and a 'baseline' result, got {sorted(by)}")
    plan = meta.get("plan") or {}
    ok = lambda r: r.get("status") == "ok" and isinstance(r.get("val"), dict)   # noqa: E731
    attrs_b, attrs_base = {"val_status": best.get("status")}, {"val_status": base.get("status")}
    for r, at in ((best, attrs_b), (base, attrs_base)):
        if ok(r):
            m = r["val"]
            at.update(val_acc=m["acc"], val_nll=m["nll"], val_ece=m["ece"], val_per_source=m["per_source"])
        elif r.get("error"):
            at["val_error"] = str(r["error"])[:300]
    if ok(best) and ok(base):
        mb_, mbase = best["val"], base["val"]
        d_nll, d_acc = mb_["nll"] - mbase["nll"], mb_["acc"] - mbase["acc"]
        reject = d_nll > VAL_NLL_TOL or d_acc < -VAL_ACC_TOL
        attrs_b.update(val_vs_baseline={
            "d_nll": round(d_nll, 4), "d_acc": round(d_acc, 4), "baseline_trial": base.get("trial_number"),
            "baseline": {k: mbase[k] for k in ("acc", "nll", "ece", "per_source")},
            "steps": meta.get("steps"), "global_batch": meta.get("global_batch"),
            "micro_batch": meta.get("micro_batch"), "val_set_sha": meta.get("val_set_sha"),
            "val_n": meta.get("val_n")}, quality_reject=reject, quality_accept=not reject)
    else:
        attrs_b.update(quality_reject=None)        # unknown: a run failed
    for r, at in ((best, attrs_b), (base, attrs_base)):
        ft = find_trial(study, r.get("trial_number"))
        if ft is None:
            print(f"warning: trial {r.get('trial_number')} is not in the study; skipped")
            continue
        set_attrs(study, ft, at)
        if r is best:
            h = handle(study, ft)
            old = (get_note(h) or trial_note(ft.number, ft.user_attrs, ft.params)).split(QC_MARK)[0].rstrip("\n")
            save_note(h, old + "\n\n" + qc_md({**ft.user_attrs, **at}))
        print(f"trial {ft.number} ({r['name']}): {r.get('status')} " +
              (f"acc {at.get('val_acc')} nll {at.get('val_nll')} ece {at.get('val_ece')}" if "val_acc" in at else ""))
    study.set_user_attr("last_validation", {"n_complete": plan.get("n_complete"), "round": plan.get("round"),
                                            "best": best.get("trial_number"),
                                            "baseline": base.get("trial_number"),
                                            "quality_reject": attrs_b.get("quality_reject")})
    print(f"quality_reject={attrs_b.get('quality_reject')}")


# ---------------------------------------------------------------- annotate

R1_CAVEAT = ("> Round 1: measured under the OLD Prism mask (pre-OJ-080) and emulated bf16; "
             "numbers are not comparable to round 2.")
PROFILE_URL = "https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html"
SDPA_URL = "https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html"
_TPE = ("tpe", *TPE_IDEA, "")
# trial -> (round, source, idea, hypothesis, reference)
BACKFILL = {0: (1, "baseline", BASELINE_IDEA, BASELINE_HYP, "")}
BACKFILL.update({n: (1, *_TPE) for n in range(1, 6)})
BACKFILL.update({
    6: (2, "baseline", BASELINE_IDEA, BASELINE_HYP, ""),
    7: (2, "enqueued", "profile of the round-1 best config in fp16",
        "shows the top ops", f"OJ-082; {PROFILE_URL}"),
    8: (2, "enqueued", "sdpa_bool",
        "a boolean mask may enable a faster SDPA path", SDPA_URL),
    9: (2, "enqueued", "chunked_math", "lower peak memory, O(chunk*T)", ""),
    10: (2, "enqueued", "chunked_math with no grad-ckpt",
         "the lower memory may allow dropping ckpt, about 1.3x compute saved",
         "Chen et al. 2016, arXiv:1604.06174"),
    11: (2, "enqueued", "flex", "block-sparse mask kernel", "Dong et al. 2024, arXiv:2412.05496"),
    12: (2, "enqueued", "bf16_emulated vs t7 fp16",
         "OJ-081: measures the cost of emulated bf16 on sm75", "OJ-081"),
})
BACKFILL.update({n: (2, *_TPE) for n in range(13, 20)})


def annotate_trial(study, number, **attrs):
    ft = find_trial(study, number)
    if ft is None:
        print(f"trial {number}: not in the study; skipped")
        return
    set_attrs(study, ft, attrs)
    rnd = int(attrs.get("round", ft.user_attrs.get("round", 1)))
    write_trial_note(study, ft, R1_CAVEAT if rnd == 1 and number in BACKFILL else "", attrs)


def cmd_annotate(a):
    study = open_study(a.db, a.seed, a.study)
    if a.backfill:
        dims = load_space(a.space)
        impl = implemented(dims)
        rnd = load_round(a.space)
        st = load_settings(a.space)
        cond = None
        if st.get("global_batch"):
            cond = fixed_conditions(dims, st, study.user_attrs.get("trial_stream_sha") or read_frozen(a.stream)[0],
                                    study.user_attrs.get("val_set_sha") or read_frozen(a.val_set)[0])
        set_study_meta(study, impl, rnd, st, cond)
        if rnd >= 2 and study.study_name in LEGACY_STUDIES and "search_space_round_1" not in study.user_attrs:
            old = [dict(d, default=1) if d["name"] == "ple_cpu" else d for d in impl
                   if d["name"] not in ("attn_backend", "profile", "dtype")]
            study.set_user_attr("search_space_round_1", space_snapshot(old))
        # BACKFILL describes the trial numbers of the legacy study only.
        for n, (r, src, idea, hyp, ref) in sorted(BACKFILL.items() if study.study_name in LEGACY_STUDIES else []):
            annotate_trial(study, n, round=r, source=src, idea=idea, hypothesis=hyp, reference=ref)
            print(f"annotated trial {n}: {src} r{r} {idea}")
    if a.trial is not None:
        given = {k: getattr(a, k) for k in META_KEYS if getattr(a, k) is not None}
        if a.source:
            given["source"] = a.source
        if not given:
            raise SystemExit("annotate --trial N needs --idea / --hypothesis / --reference / --source")
        annotate_trial(study, a.trial, **given)
        print(f"annotated trial {a.trial}: {given}")
    if not a.backfill and a.trial is None:
        raise SystemExit("annotate: pass --backfill and/or --trial N")


# ---------------------------------------------------------------- report

def fmt(x, nd=3):
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def importances(study) -> tuple[dict, str]:
    try:
        return optuna.importance.get_param_importances(study), "fANOVA"
    except Exception:
        ev = optuna.importance.PedAnovaImportanceEvaluator()
        return optuna.importance.get_param_importances(study, evaluator=ev), "PED-ANOVA"


def pct(x):
    return "" if x is None else f"{100 * x:.1f}%"


def kernel_section(trials) -> list[str]:
    """'## SDPA kernel eligibility' from the newest trial that recorded it."""
    t = next((t for t in reversed(trials) if isinstance(t.user_attrs.get("sdpa_kernels"), dict)
              and t.user_attrs["sdpa_kernels"]), None)
    if t is None:
        return []
    k = t.user_attrs["sdpa_kernels"]
    L = ["## SDPA kernel eligibility", "",
         f"From trial {t.number} ({t.user_attrs.get('gpu_name') or '?'}): can each SDPA kernel take "
         "our case (per head dim, boolean vs additive float mask)? `no: ...` is the reason torch gives.", ""]
    if "error" in k:
        return L + [f"Not available: {k['error']}", ""]
    L += ["| head dim | mask | kernel | eligible |", "|---|---|---|---|"]
    for dim, row in k.items():
        for mask, res in row.items():
            for kern, ok in res.items():
                L.append(f"| {dim} | {mask} | {kern} | {str(ok).replace('|', '/')} |")
    return L + [""]


def profile_section(trials) -> list[str]:
    """'## Profiles': the profiler table of every profile=1 trial."""
    prof = [t for t in trials if t.user_attrs.get("is_profile") and t.user_attrs.get("profile")]
    if not prof:
        return []
    L = ["## Profiles", "",
         "torch.profiler over a few training steps after the timed loop (CPU + CUDA). "
         "`oj::` ranges are inclusive and cover the forward only (attention_fwd, masks, "
         "ple_lookup) or the collate call; CPU share is of the wall time. `*_ops` rows are "
         "op-name buckets by self time over forward + backward; CPU share is of all self CPU "
         "time. Ranges and buckets overlap (math attention's bmm counts in both).", ""]
    for t in prof:
        p = t.user_attrs["profile"]
        L += [f"### Trial {t.number} ({t.user_attrs.get('attn_backend', 'sdpa_default')}, "
              f"{t.user_attrs.get('gpu_name') or 'cpu'})", "",
              f"Params: `{json.dumps(t.params)}`. {p.get('steps')} steps, wall {p.get('wall_ms')} ms, "
              f"kernel time {p.get('cuda_total_ms')} ms, self CPU {p.get('cpu_self_total_ms')} ms.", "",
              "| part | CPU ms | CPU share | CUDA ms | CUDA share |", "|---|---|---|---|---|"]
        for k, v in p.get("shares", {}).items():
            L.append(f"| {k} | {v.get('cpu_ms')} | {pct(v.get('cpu'))} | {v.get('cuda_ms')} | "
                     f"{pct(v.get('cuda'))} |")
        L.append("")
        for key, title in (("top_cuda", "self CUDA"), ("top_cpu", "self CPU")):
            rows = p.get(key) or []
            if not rows:
                continue
            L += [f"Top {len(rows)} ops by {title} time:", "", "| op | ms | share | calls |",
                  "|---|---|---|---|"]
            L += [f"| `{r[0]}` | {r[1]} | {pct(r[2])} | {r[3]} |" for r in rows]
            L.append("")
    return L


def quality_section(trials) -> list[str]:
    """'## Quality checkpoints': trials with val_* attrs (validate run)."""
    qc = [t for t in trials if t.user_attrs.get("val_acc") is not None or t.user_attrs.get("val_status")]
    if not qc:
        return []
    L = ["## Quality checkpoints", "",
         f"Separate validation runs (PROFILE=optuna_validate): best and baseline config, the same real "
         f"optimizer steps from the same start, then the frozen val set. Accepted if val nll <= baseline "
         f"+ {VAL_NLL_TOL} and acc >= baseline - {VAL_ACC_TOL * 100:.0f} pt (with ~400 val records the "
         "standard error of acc is about 2 pt, so the acc rule is inside the noise).", "",
         "| # | round | val acc | val nll | val ece | Δ nll vs base | Δ acc vs base | verdict | per-source acc |",
         "|---|---|---|---|---|---|---|---|---|"]
    for t in qc:
        u = t.user_attrs
        v = u.get("val_vs_baseline") or {}
        verdict = ("baseline" if u.get("is_baseline") else
                   "quality_reject" if u.get("quality_reject") else
                   "accepted" if u.get("quality_accept") else str(u.get("val_status") or ""))
        ps = ", ".join(f"{k}={x}" for k, x in (u.get("val_per_source") or {}).items())
        L.append(f"| {t.number} | {trial_round(t)} | {fmt(u.get('val_acc'), 4)} | {fmt(u.get('val_nll'), 4)} | "
                 f"{fmt(u.get('val_ece'), 4)} | {fmt(v.get('d_nll'), 4)} | {fmt(v.get('d_acc'), 4)} | "
                 f"{verdict} | {ps} |")
    return L + [""]


def cmd_report(a):
    dims = load_space(a.space)
    study = open_study(a.db, a.seed, a.study)
    trials = study.get_trials(deepcopy=False)
    names = [d["name"] for d in dims if any(d["name"] in t.params for t in trials)]
    names += sorted({k for t in trials for k in t.params} - set(names))
    complete = [t for t in trials if t.state == TrialState.COMPLETE]
    rnd = load_round(a.space)
    st = load_settings(a.space)
    base = next((t for t in complete if t.user_attrs.get("is_baseline") and trial_round(t) == rnd), None)
    base_sps = base.value if base else None
    base_by_round = {trial_round(t): t.value for t in complete if t.user_attrs.get("is_baseline")}
    how = (f"Round {rnd}: every trial runs at a FIXED micro-batch {st['micro_batch']} and global batch "
           f"{st['global_batch']} (grad_accum {st['global_batch'] // st['micro_batch']}); a step is one "
           f"optimizer step of {st['global_batch']} samples. {st.get('warmup_steps', 1)} warm-up step(s), then "
           f"{st.get('timed_steps')} timed steps on the frozen trial stream (same records for every trial), "
           "from the same start weights. OOM at that micro-batch = pruned (infeasible_at_mb). The quality "
           "guard uses the probe loss after the warm-up steps (same steps and batches for every trial, so "
           "guards are comparable across trials)." if st.get("global_batch") else
           "Each trial: 2 warm-up steps, then a timed loop of real SFT steps on the real mixture, from the "
           "same start weights. The quality guard uses the probe loss after the warm-up steps.")
    L = ["# OpenJev speed experiments (Optuna)", "",
         f"Generated by `tools/optuna_loop.py report` from `{a.db.name}` (study `{study.study_name}`)"
         + (f"; the legacy study `{LEGACY_STUDY}` (rounds 1-2, not comparable) is in the same db."
            if study.study_name not in LEGACY_STUDIES else "."),
         f"Objective: maximize training samples/s. Speedup is against the "
         f"{BASELINE_S_PER_SAMPLE} s/sample v7/v8 baseline. {how} Rounds 1-2 did one optimizer step per "
         "micro-batch of a sampled `bs`; compare trials only within a round.",
         "", f"Trials: {len(trials)} total, {len(complete)} complete, "
         f"{sum(t.state == TrialState.PRUNED for t in trials)} pruned (OOM / NaN / kernel / "
         f"quality guard), {sum(t.state == TrialState.FAIL for t in trials)} failed.", ""]
    if complete:
        b = study.best_trial
        sps = b.value
        L += ["## Best so far", "",
              f"Trial **{b.number}**: **{sps:.3f} samples/s** = {1 / sps:.2f} s/sample = "
              f"**{sps * BASELINE_S_PER_SAMPLE:.2f}x** vs {BASELINE_S_PER_SAMPLE} s/sample; "
              + (f"{sps / base_sps:.2f}x vs the measured baseline (trial {base.number}); " if base else "")
              + f"peak {fmt(b.user_attrs.get('peak_mem_gb'), 2) or 'n/a'} GB.", "",
              "```json", json.dumps(b.params, indent=1), "```", ""]
    L += ["## All trials", "",
          "`speedup` is vs 4.5 s/sample; `vs base` is vs the measured baseline trial of the trial's own "
          "round. `mb` / `accum`: micro-batch and gradient-accumulation steps it ran with (rounds 1-2: "
          "mb = bs, accum 1).", "",
          "| # | status | idea | source | " + " | ".join(names) + " | samples/s | s/sample | speedup | vs base | "
          "tokens/s | peak GB | mb_used | grad_accum | probe loss start→guard→end | note |",
          "|" + "---|" * (len(names) + 14)]
    for t in trials:
        u = t.user_attrs
        sps = t.value if t.state == TrialState.COMPLETE else u.get("samples_per_sec")
        status = (t.state.name.lower()
                  + (f" (base r{trial_round(t)})" if u.get("is_baseline") else "")
                  + (" (profile)" if u.get("is_profile") else ""))
        note = u.get("reason") or u.get("error") or ""
        note = str(note).replace("|", "/").replace("\n", " ")[:120]
        loss = (f"{fmt(u.get('loss_start'))}→{fmt(u.get('loss_guard'))}→{fmt(u.get('loss_end'))}"
                if u.get("loss_start") is not None else "")
        idea = str(u.get("idea", "")).replace("|", "/").replace("\n", " ")[:70]
        bsps = base_by_round.get(trial_round(t))
        mb_used = u.get("mb_used", t.params.get("bs") if trial_round(t) < 3 else None)
        accum = u.get("grad_accum", 1 if mb_used is not None and trial_round(t) < 3 else None)
        L.append("| " + " | ".join([str(t.number), status, idea, str(u.get("source", "")),
                                    *[fmt(t.params.get(n)) for n in names],
                                    fmt(sps), fmt(1 / sps, 2) if sps else "",
                                    f"{sps * BASELINE_S_PER_SAMPLE:.2f}x" if sps else "",
                                    f"{sps / bsps:.2f}x" if sps and bsps else "",
                                    fmt(u.get("tokens_per_sec"), 1), fmt(u.get("peak_mem_gb"), 2),
                                    fmt(mb_used), fmt(accum), loss, note]) + " |")
    L.append("")
    if len(complete) >= 10:
        try:
            imp, how = importances(study)
            L += [f"## Parameter importances ({how}, {len(complete)} complete trials)", "",
                  "| param | importance |", "|---|---|"]
            L += [f"| {k} | {v:.3f} |" for k, v in imp.items()]
        except Exception as e:  # e.g. only one distinct value per param so far
            L += ["## Parameter importances", "", f"Not available: {type(e).__name__}: {e}"]
        L.append("")
    else:
        L += [f"Parameter importances appear after 10 complete trials ({len(complete)} now).", ""]
    L += quality_section(trials)
    L += kernel_section(trials)
    L += profile_section(trials)
    todo = [d for d in dims if not d["implemented"]]
    if todo:
        L += ["## Ideas not implemented yet (not sampled)", ""]
        L += [f"- `{d['name']}` {d.get('choices', '')}"
              f"{' (reload)' if d['requires_reload'] else ''}: {d.get('idea', '').strip()}"
              for d in todo]
        L.append("")
    # Hand-written notes after the marker survive regeneration.
    if a.out.exists() and MANUAL_MARKER in a.out.read_text():
        L += [a.out.read_text()[a.out.read_text().index(MANUAL_MARKER):].rstrip("\n"), ""]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L))
    print(f"wrote {a.out} ({len(trials)} trials)")


# ---------------------------------------------------------------- cli

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=ROOT / "runs/optuna/study.db")
    ap.add_argument("--space", type=Path, default=ROOT / "tools/search_space.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stream", type=Path, default=STREAM_PATH, help="frozen trial stream (round 3+)")
    ap.add_argument("--val-set", type=Path, default=VAL_SET_PATH, help="frozen val set (round 3+)")
    ap.add_argument("--study", default="", help="study name (default: the yaml study_name, else "
                    f"{STUDY_NAME})")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose", help="study.ask() K trials and write trials.json")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--out", type=Path, default=ROOT / "kaggle/prism/trials.json")
    p.add_argument("--abandon-running", action="store_true",
                   help="mark trials proposed earlier but never ingested as FAIL")
    p.add_argument("--enqueue-json", action="append", default=[], metavar="JSON",
                   help="enqueue an exact config (JSON object or list; missing keys = yaml "
                        "defaults), asked before the --n TPE trials; repeatable")
    p.set_defaults(fn=cmd_propose)
    p = sub.add_parser("ingest", help="study.tell() the results of bench_trials.py")
    p.add_argument("results")
    p.add_argument("--guard-abs", type=float, default=0.1)
    p.add_argument("--guard-rel", type=float, default=0.5)
    p.add_argument("--tag", default="", help="e.g. the Kaggle kernel version")
    p.add_argument("--fail-missing", action=argparse.BooleanOptionalAction, default=True,
                   help="mark RUNNING trials with no result as FAIL")
    p.set_defaults(fn=cmd_ingest)
    p = sub.add_parser("annotate", help="set idea/hypothesis/reference attrs + notes on trials")
    p.add_argument("--backfill", action="store_true", help="annotate the known trials (idempotent)")
    p.add_argument("--trial", type=int)
    p.add_argument("--idea")
    p.add_argument("--hypothesis")
    p.add_argument("--reference")
    p.add_argument("--source", choices=["baseline", "enqueued", "tpe"])
    p.set_defaults(fn=cmd_annotate)
    p = sub.add_parser("abandon", help="mark RUNNING trials of --study FAIL (e.g. a crashed kernel)")
    p.set_defaults(fn=cmd_abandon)
    p = sub.add_parser("freeze", help="build the frozen trial stream + val set (once per study)")
    p.add_argument("--train", default=str(ROOT / "data/*.train.jsonl"))
    p.add_argument("--val", default=str(ROOT / "data/*.val.jsonl"))
    p.add_argument("--alpha", type=float, default=0.5, help="Mixture temperature (train_prism --alpha)")
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="overwrite (only for a NEW study)")
    p.set_defaults(fn=cmd_freeze)
    p = sub.add_parser("validate-plan", help="write validate.json if a quality checkpoint is due")
    p.add_argument("--end-of-round", action="store_true", help="run it now (round end)")
    p.add_argument("--steps", type=int, default=0, help="default: yaml validate_steps")
    p.add_argument("--out", type=Path, default=ROOT / "kaggle/prism/validate.json")
    p.set_defaults(fn=cmd_validate_plan)
    p = sub.add_parser("ingest-validate", help="store a quality checkpoint on the best trial")
    p.add_argument("results")
    p.set_defaults(fn=cmd_ingest_validate)
    p = sub.add_parser("report", help="write docs/speed/EXPERIMENTS.md")
    p.add_argument("--out", type=Path, default=ROOT / "docs/speed/EXPERIMENTS.md")
    p.set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    a.study = a.study or yaml.safe_load(Path(a.space).read_text()).get("study_name") or STUDY_NAME
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
