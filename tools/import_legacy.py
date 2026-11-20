"""Copy the COMPLETE and PRUNED trials of the legacy study (rounds 1-2) into
openjev-speed-v2 for display only (user, 2026-11-20). Optuna API only, idempotent.

Each copy: source=legacy_r1_r2, comparable=False, the legacy trial number and study,
is_baseline renamed to legacy_is_baseline (so no round logic picks it up), and the
dashboard note with a NOT-comparable header. optuna_loop.py keeps them out of TPE and
the importances (is_comparable).

  .venv/bin/python tools/import_legacy.py [--db runs/optuna/study.db]
"""
import argparse
import re
import sys
from types import SimpleNamespace
from pathlib import Path

import optuna
from optuna.trial import TrialState
from optuna_dashboard import get_note, save_note

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from optuna_loop import handle  # noqa: E402

LEGACY = "LEGACY-r1-r2 (old mask, variable bs; not comparable)"
TARGET = "openjev-speed-v2"
NOTE = "**old mask / emulated bf16 / variable bs: NOT comparable**"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=ROOT / "runs/optuna/study.db")
    ap.add_argument("--legacy", default=LEGACY)
    ap.add_argument("--target", default=TARGET)
    a = ap.parse_args(argv)
    storage = f"sqlite:///{a.db}"
    old = optuna.load_study(study_name=a.legacy, storage=storage)
    new = optuna.load_study(study_name=a.target, storage=storage)
    done = {t.user_attrs.get("legacy_trial_number") for t in new.trials
            if t.user_attrs.get("source") == "legacy_r1_r2"}
    # The legacy study is a copy: its notes are keyed by the ORIGINAL trial ids, so find
    # each trial's note by its "## Trial N:" header.
    notes = {}
    for k in old._storage.get_study_system_attrs(old._study_id):
        m = re.match(r"dashboard:(\d+):note_ver$", k)
        if m:
            txt = get_note(SimpleNamespace(storage=old._storage, study=old, _trial_id=int(m.group(1)))) or ""
            n = re.match(r"## Trial (\d+):", txt)
            if n:
                notes[int(n.group(1))] = txt
    added = []
    for t in old.get_trials(states=(TrialState.COMPLETE, TrialState.PRUNED)):
        if t.number in done:
            continue
        attrs = dict(t.user_attrs)
        attrs["legacy_source"] = attrs.pop("source", None)
        attrs["legacy_is_baseline"] = bool(attrs.pop("is_baseline", False))
        attrs.update(source="legacy_r1_r2", comparable=False, legacy_trial_number=t.number,
                     legacy_study=a.legacy, note="old mask / emulated bf16 / variable bs: NOT comparable")
        new.add_trial(optuna.trial.create_trial(
            state=t.state, values=t.values if t.state == TrialState.COMPLETE else None,
            params=t.params, distributions=t.distributions, user_attrs=attrs,
            intermediate_values=t.intermediate_values))
        ft = new.trials[-1]
        note = notes.get(t.number) or get_note(handle(old, t)) or ""
        save_note(handle(new, ft), f"{NOTE} (legacy trial {t.number})\n\n{note}")
        added.append((t.number, ft.number, t.state.name))
    print(f"added {len(added)} legacy trials (legacy -> v2, state): {added}")


if __name__ == "__main__":
    main()
