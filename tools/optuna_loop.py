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

Dimensions with `sample: false` in the yaml (profile) are never drawn by TPE: trials use
the default unless an --enqueue-json config sets them. profile=1 trials still report
samples/s as their objective; their profiler table is listed under "Profiles".
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import optuna
import yaml
import warnings
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.trial import TrialState

ROOT = Path(__file__).resolve().parent.parent
BASELINE_S_PER_SAMPLE = 4.5
STUDY_NAME = "openjev-speed"


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


def parse_enqueue(texts, impl) -> list[dict]:
    """--enqueue-json values (an object or a list of objects) -> exact configs: every
    implemented dimension set, missing ones from the yaml default, values checked."""
    by_name = {d["name"]: d for d in impl}
    out = []
    for text in texts or []:
        obj = json.loads(text)
        for cfg in obj if isinstance(obj, list) else [obj]:
            if not isinstance(cfg, dict):
                raise SystemExit(f"--enqueue-json: expected an object, got {cfg!r}")
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
            out.append(full)
    return out


# ---------------------------------------------------------------- study

def open_study(db: Path, seed: int) -> optuna.Study:
    db.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{db}"
    # Each CLI call is a new process. A fixed sampler seed would replay the same
    # random start-up points every round, so offset it by the number of trials.
    n = 0
    try:
        n = len(optuna.load_study(study_name=STUDY_NAME, storage=storage).trials)
    except KeyError:
        pass
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    # group=True: round-2 trials have params (attn_backend) that round-1 trials lack;
    # grouped multivariate TPE models each co-occurring param set jointly instead of
    # dropping the new params to independent sampling.
    sampler = optuna.samplers.TPESampler(multivariate=True, group=True, constant_liar=True,
                                         seed=seed + n)
    return optuna.create_study(study_name=STUDY_NAME, storage=storage, direction="maximize",
                               sampler=sampler, load_if_exists=True)


def running_trial(study, number: int):
    """A writable handle for a RUNNING trial (user attrs must be set before tell)."""
    for ft in study.get_trials(deepcopy=False):
        if ft.number == number:
            return ft, optuna.trial.Trial(study, ft._trial_id)
    return None, None


# ---------------------------------------------------------------- propose

def cmd_propose(a):
    dims = load_space(a.space)
    study = open_study(a.db, a.seed)
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
    queued = parse_enqueue(a.enqueue_json, impl)
    dists = {d["name"]: distribution(d) for d in impl}
    sampled = {d["name"] for d in impl if d["sample"]}
    defaults = {d["name"]: d["default"] for d in impl}
    reload_keys = [d["name"] for d in impl if d["requires_reload"]]
    # Trial 0 is the baseline. Re-enqueue it if every baseline so far failed or was abandoned.
    alive = (TrialState.COMPLETE, TrialState.RUNNING, TrialState.WAITING)
    need_base = not any(t.user_attrs.get("is_baseline") and t.state in alive for t in study.trials)
    if need_base:
        study.enqueue_trial({n: v for n, v in defaults.items() if n in sampled},
                            user_attrs={"is_baseline": True})
    for cfg in queued:   # a sample: false dim at its default stays out of the Optuna params
        study.enqueue_trial({k: v for k, v in cfg.items() if k in sampled or v != defaults[k]},
                            user_attrs={"enqueued": True})
    out = []
    # The baseline (when it is enqueued) counts toward --n; the --enqueue-json configs do not.
    for _ in range(max(a.n, int(need_base)) + len(queued)):
        trial = study.ask()
        fixed = getattr(trial, "_fixed_params", {}) or {}
        params = {}
        for name, dist in dists.items():
            if name in sampled or name in fixed:
                params[name] = dist_suggest(trial, name, dist)
            # sample: false and not enqueued -> not an Optuna param; bench uses the default
        out.append({"trial_number": trial.number, "params": params, "reload_keys": reload_keys})
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
         "sdpa_kernels")
PRUNE_STATUS = {"oom", "nan", "unsupported"}


def finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def guard_loss(r: dict):
    """The fixed-step probe loss (older results only have loss_end)."""
    return r.get("loss_guard", r.get("loss_end"))


def baseline_drop(study, results):
    """Guard-loss drop of the baseline: from this results file first, else the study."""
    base_nums = {t.number for t in study.get_trials(deepcopy=False) if t.user_attrs.get("is_baseline")}
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


def cmd_ingest(a):
    study = open_study(a.db, a.seed)
    raw = json.loads(Path(a.results).read_text())
    results = raw["results"] if isinstance(raw, dict) else raw
    meta = raw.get("meta", {}) if isinstance(raw, dict) else {}
    base = baseline_drop(study, results)
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
        for k in ATTRS:
            if k in r and r[k] is not None:
                trial.set_user_attr(k, r[k])
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
        elif status in PRUNE_STATUS or status == "quality":
            study.tell(n, state=TrialState.PRUNED)
        else:
            study.tell(n, state=TrialState.FAIL)
        told.add(n)
        print(f"trial {n}: {status} samples/s={sps} {why}".rstrip())
    # Proposed but missing from this results file: the kernel died or timed out.
    if a.fail_missing:
        for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
            if t.number not in told:
                optuna.trial.Trial(study, t._trial_id).set_user_attr("reason", "no result")
                study.tell(t.number, state=TrialState.FAIL)
                print(f"trial {t.number}: FAIL (no result)")


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


def cmd_report(a):
    dims = load_space(a.space)
    study = open_study(a.db, a.seed)
    trials = study.get_trials(deepcopy=False)
    names = [d["name"] for d in dims if any(d["name"] in t.params for t in trials)]
    names += sorted({k for t in trials for k in t.params} - set(names))
    complete = [t for t in trials if t.state == TrialState.COMPLETE]
    base = next((t for t in complete if t.user_attrs.get("is_baseline")), None)
    base_sps = base.value if base else None
    L = ["# OpenJev speed experiments (Optuna)", "",
         f"Generated by `tools/optuna_loop.py report` from `{a.db.name}` (study `{STUDY_NAME}`).",
         f"Objective: maximize training samples/s. Speedup is against the "
         f"{BASELINE_S_PER_SAMPLE} s/sample v7/v8 baseline. Each trial: 2 warm-up steps, then a "
         "timed loop of real SFT steps on the real mixture, from the same start weights. "
         "The quality guard uses the probe loss after the warm-up steps (same step count "
         "for every trial).",
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
          "`speedup` is vs 4.5 s/sample; `vs base` is vs the measured baseline trial of this "
          "study (the bench does one optimizer step per micro-batch, so it is not exactly v8).", "",
          "| # | status | " + " | ".join(names) + " | samples/s | s/sample | speedup | vs base | "
          "tokens/s | peak GB | probe loss start→guard→end | note |",
          "|" + "---|" * (len(names) + 10)]
    for t in trials:
        u = t.user_attrs
        sps = t.value if t.state == TrialState.COMPLETE else u.get("samples_per_sec")
        status = (t.state.name.lower() + (" (base)" if u.get("is_baseline") else "")
                  + (" (profile)" if u.get("is_profile") else ""))
        note = u.get("reason") or u.get("error") or ""
        note = str(note).replace("|", "/").replace("\n", " ")[:120]
        loss = (f"{fmt(u.get('loss_start'))}→{fmt(u.get('loss_guard'))}→{fmt(u.get('loss_end'))}"
                if u.get("loss_start") is not None else "")
        L.append("| " + " | ".join([str(t.number), status, *[fmt(t.params.get(n)) for n in names],
                                    fmt(sps), fmt(1 / sps, 2) if sps else "",
                                    f"{sps * BASELINE_S_PER_SAMPLE:.2f}x" if sps else "",
                                    f"{sps / base_sps:.2f}x" if sps and base_sps else "",
                                    fmt(u.get("tokens_per_sec"), 1), fmt(u.get("peak_mem_gb"), 2),
                                    loss, note]) + " |")
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
    L += profile_section(trials)
    todo = [d for d in dims if not d["implemented"]]
    if todo:
        L += ["## Ideas not implemented yet (not sampled)", ""]
        L += [f"- `{d['name']}` {d.get('choices', '')}"
              f"{' (reload)' if d['requires_reload'] else ''}: {d.get('idea', '').strip()}"
              for d in todo]
        L.append("")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L))
    print(f"wrote {a.out} ({len(trials)} trials)")


# ---------------------------------------------------------------- cli

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=ROOT / "runs/optuna/study.db")
    ap.add_argument("--space", type=Path, default=ROOT / "tools/search_space.yaml")
    ap.add_argument("--seed", type=int, default=0)
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
    p = sub.add_parser("report", help="write docs/speed/EXPERIMENTS.md")
    p.add_argument("--out", type=Path, default=ROOT / "docs/speed/EXPERIMENTS.md")
    p.set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
