"""Check launcher paths without starting training or importing GPU packages."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "scripts/experiments"


def test_straight_search_help_outside_repo(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_straight_search.py"), "--help"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    assert "--baseline-checkpoint" in result.stdout
    assert "--flight-run-dir" in result.stdout
    assert "--output-dir" in result.stdout


@pytest.mark.skipif(os.name == "nt" or not shutil.which("bash"), reason="requires Linux Bash")
def test_evaluation_launcher_outside_repo(tmp_path):
    # Stub uv: exercise the actual shell path/argument plumbing, without CUDA.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text('#!/bin/sh\nprintf "%s\\n" "$PWD" "$UV_PROJECT_ENVIRONMENT" "$@"\n')
    uv.chmod(0o755)
    artifacts = tmp_path / "artifacts with spaces"
    environment = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
                   "MICRODUCK_ARTIFACTS": str(artifacts),
                   "UV_PROJECT_ENVIRONMENT": str(tmp_path / "dedicated-env")}
    for script in EXPERIMENTS.glob("*.sh"):
        subprocess.run(["bash", "-n", str(script)], check=True)
    subprocess.run(
        ["bash", str(EXPERIMENTS / "run_speed2_eval.sh"), "/external/checkpoint.pt", "probe"],
        cwd=tmp_path, env=environment, check=True,
    )
    lines = (artifacts / "speed2/eval/probe_v2.5_t10_s123.log").read_text().splitlines()
    assert lines[:4] == [str(ROOT), str(tmp_path / "dedicated-env"), "run", "--locked"]
    assert "/external/checkpoint.pt" in lines
    assert str(artifacts / "speed2/eval/probe_v2.5_t10_s123.json") in lines
