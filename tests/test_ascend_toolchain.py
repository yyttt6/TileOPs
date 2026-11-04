"""The compiler contract must diagnose environment drift without touching an NPU."""

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/ci/check_ascend_toolchain.py"


def run_check(tmp_path, *, changed_header=False, missing_cann=False):
    compiler = tmp_path / "cxx"
    compiler.write_text("#!/bin/sh\necho 12.3.0\n")
    compiler.chmod(0o700)
    header = tmp_path / "cube.h"
    header.write_text("compatible cube interface\n")
    digest = hashlib.sha256(header.read_bytes()).hexdigest()
    lock = {
        "profile": "test-profile",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "packages": {},
        "cxx_version": "12.3.0",
        "files": [{"name": "Cube header", "path": "cube.h", "sha256": digest}],
    }
    if changed_header:
        header.write_text("different cube interface\n")
    (tmp_path / "lock.json").write_text(json.dumps(lock))
    # Any accidental runtime import turns this CPU-only preflight into a failure.
    (tmp_path / "sitecustomize.py").write_text("""import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torch_npu', 'tilelang'}:
        raise RuntimeError('preflight imported device runtime')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
""")
    env = dict(
        os.environ, CXX=str(compiler), PYTHONPATH=str(tmp_path), TORCH_DEVICE_BACKEND_AUTOLOAD="0"
    )
    env.pop("ASCEND_HOME_PATH", None)
    if not missing_cann:
        env["ASCEND_HOME_PATH"] = str(tmp_path)
    output = tmp_path / "result.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--lock",
            str(tmp_path / "lock.json"),
            "--output",
            str(output),
        ],
        env=env,
        text=True,
        capture_output=True,
    )
    return result, json.loads(output.read_text())


def test_matching_profile_uses_no_device_imports(tmp_path):
    result, receipt = run_check(tmp_path)
    assert result.returncode == 0, result.stderr
    assert receipt["status"] == "passed"


def test_same_version_different_header_is_rejected(tmp_path):
    result, receipt = run_check(tmp_path, changed_header=True)
    assert result.returncode != 0
    assert any("fingerprint differs" in e for e in receipt["errors"])
    assert "before debugging the operator" in result.stderr


def test_unconfigured_cann_reports_setup_problem(tmp_path):
    result, receipt = run_check(tmp_path, missing_cann=True)
    assert result.returncode != 0
    assert any("ASCEND_HOME_PATH is unset" in e for e in receipt["errors"])
