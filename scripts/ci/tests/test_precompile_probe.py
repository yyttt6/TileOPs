"""Diagnostics must not accept partial or failed cache experiments."""

import copy
import json
import subprocess

import pytest

from scripts.ci import precompile_probe as probe

pytestmark = pytest.mark.smoke


def valid_stages():
    return {
        "enumerate": {"status": "passed", "exit_code": 0, "compilers": 0, "programs": 3},
        "compile": {
            "status": "passed",
            "exit_code": 0,
            "compilers": 6,
            "receipt": {
                "operators": {
                    "ReLUFwdOp": {
                        "status": "compiled",
                        "compiled": 3,
                        "skipped_cases": [],
                    }
                }
            },
        },
        "call": {
            "status": "passed",
            "exit_code": 0,
            "compilers": 0,
            "cases": [
                {
                    "id": str(i),
                    "correctness": {
                        "checked_outputs": 1,
                        "total_outputs": 1,
                        "unchecked_reason": None,
                    },
                }
                for i in range(3)
            ],
        },
    }


def test_complete_experiment_passes():
    assert all(probe.evaluate(valid_stages()).values())


@pytest.mark.parametrize(
    "fault", ["missing", "exit", "recompile", "warm", "partial", "wrong", "duplicate"]
)
def test_incomplete_evidence_fails(fault):
    stages = copy.deepcopy(valid_stages())
    if fault == "missing":
        del stages["enumerate"]
    elif fault == "exit":
        stages["call"]["exit_code"] = 1
    elif fault == "recompile":
        stages["call"]["compilers"] = 2
    elif fault == "warm":
        stages["compile"]["compilers"] = 0
    elif fault == "partial":
        stages["call"]["cases"].pop()
    elif fault == "wrong":
        stages["call"]["cases"][0]["correctness"]["checked_outputs"] = 0
    else:
        stages["call"]["cases"][1]["id"] = "0"
    assert not all(probe.evaluate(stages).values())


def test_counter_counts_launches_and_restores_after_error(monkeypatch):
    launched = []

    def popen(args, **kwargs):
        launched.append(args)
        return object()

    monkeypatch.setattr(subprocess, "Popen", popen)
    with pytest.raises(RuntimeError), probe.compiler_counter() as evidence:
        subprocess.Popen(["/compiler/bisheng", "-c", "kernel.cpp"])
        subprocess.Popen(["/compiler/ld.lld", "kernel.o"])
        subprocess.Popen(["python", "child.py"])
        raise RuntimeError("compile failed")
    assert len(evidence) == 2
    assert len(launched) == 3
    assert subprocess.Popen is popen


def test_child_rejects_absent_guard_before_compilation(tmp_path, monkeypatch):
    def refuse(_):
        raise RuntimeError("no reservation")

    monkeypatch.setattr(probe, "require_guard", refuse)
    output = tmp_path / "stage.json"
    assert probe.stage(tmp_path, "compile", output, 4) == 1
    receipt = json.loads(output.read_text())
    assert receipt["status"] == "failed"
    assert "no reservation" in receipt["error"]
    assert "compilers" not in receipt
