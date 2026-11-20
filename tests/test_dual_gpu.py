"""CPU-only: dual-GPU shard split and result merge in bench_trials.py (no GPU, no model)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import bench_trials as bt  # noqa: E402


def tr(n, ple=1, dtype="fp16"):
    return {"trial_number": n, "params": {"ple_cpu": ple, "dtype": dtype}, "reload_keys": ["ple_cpu", "dtype"]}


def test_split_single_group_deals_round_robin():
    trials = [tr(n) for n in range(10, 17)]
    a, b = bt.split_shards(trials)
    assert sorted(t["trial_number"] for t in a + b) == list(range(10, 17))
    assert abs(len(a) - len(b)) <= 1 and a and b
    assert [t["trial_number"] for t in a] == sorted(t["trial_number"] for t in a)


def test_split_keeps_groups_whole_and_balanced():
    trials = [tr(1), tr(2), tr(3), tr(4, ple=0), tr(5, ple=0)]
    a, b = bt.split_shards(trials)
    ga = {bt._group_key(t) for t in a}
    gb = {bt._group_key(t) for t in b}
    assert not ga & gb and len(a) + len(b) == 5 and a and b


def test_split_edge_cases():
    assert bt.split_shards([]) == [[], []]
    a, b = bt.split_shards([tr(1)])
    assert len(a) + len(b) == 1


def test_merge_results(tmp_path):
    for i, nums in enumerate(([3, 1], [2])):
        (tmp_path / f"s{i}.json").write_text(json.dumps(
            {"meta": {"device": "cuda", "gpu_name": "T4"},
             "results": [{"trial_number": n, "status": "ok", "gpu_index": i} for n in nums]}))
    out = tmp_path / "optuna_results.json"
    doc = bt.merge_results([tmp_path / "s0.json", tmp_path / "missing.json", tmp_path / "s1.json"], out,
                           {"dual_gpu": True})
    d = json.loads(out.read_text())
    assert d == doc and set(d) == {"meta", "results"}
    assert [r["trial_number"] for r in d["results"]] == [1, 2, 3]
    assert [r["gpu_index"] for r in d["results"]] == [0, 1, 0]
    assert d["meta"]["gpu_name"] == "T4" and d["meta"]["dual_gpu"] is True
