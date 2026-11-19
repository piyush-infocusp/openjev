import json
import subprocess
import sys
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).parent.parent


def test_train_smoke(tmp_path):
    save_dir = tmp_path / "run"
    cmd = [
        sys.executable, str(ROOT / "train.py"),
        "--model", "trl-internal-testing/tiny-Qwen3ForCausalLM",
        "--train", str(FIXTURES / "*.jsonl"),
        "--val", str(FIXTURES / "*.jsonl"),
        "--steps", "6", "--bs", "4", "--eval-every", "3",
        "--cpu", "--save-dir", str(save_dir), "--seed", "0",
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    log_path = save_dir / "log.jsonl"
    assert log_path.exists()
    lines = [l for l in log_path.read_text().splitlines() if l.strip()]
    assert len(lines) > 0
    json.loads(lines[0])

    eval_files = sorted(save_dir.glob("eval_step*.json"))
    assert eval_files
    summary = json.loads(eval_files[-1].read_text())
    assert "all" in summary
    assert any(k in summary for k in ("noul", "choice", "score"))

    final_dir = save_dir / "final"
    assert (final_dir / "heads.pt").exists()
    assert (final_dir / "openjev_config.json").exists()
