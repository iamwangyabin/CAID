from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_train_launcher_runs_from_outside_repo_without_install(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, str(repo_root / "train.py"), "--help"],
        cwd=tmp_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--config CONFIG" in result.stdout
