import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.smoke
SPEC = importlib.util.spec_from_file_location(
    "batch_test", Path(__file__).resolve().parents[1] / "device_batch.py"
)
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


def test_bounded_groups_keep_every_operator_once():
    ops = [f"Op{i}" for i in range(19)]
    chunks = batch.groups(ops)
    assert [len(c) for c in chunks] == [8, 8, 3]
    assert [o for c in chunks for o in c] == ops
    for size in [0, 9]:
        with pytest.raises(ValueError):
            batch.groups(ops, size)
    with pytest.raises(ValueError):
        batch.groups(["Op", "Op"])


@pytest.mark.parametrize("fault", ["none", "unknown", "conflict", "owner", "incomplete", "missing"])
def test_guard_success_requires_complete_private_receipt(monkeypatch, tmp_path, fault):
    def run(command, **kwargs):
        receipt = Path(command[command.index("--receipt") + 1])
        meta = dict(owner="owner", exit_code=0, performance_valid=True, workload_completed=True)
        if fault == "unknown":
            meta["performance_valid"] = False
        if fault == "conflict":
            meta["exit_code"] = 76
        if fault == "owner":
            meta["owner"] = "other"
        if fault == "incomplete":
            meta["workload_completed"] = False
        if fault != "missing":
            receipt.write_text(json.dumps(meta))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(batch.subprocess, "run", run)
    result = batch.run(
        tmp_path, [dict(op="AbsFwdOp", output=str(tmp_path / "op"))], tmp_path, "batch-0", "owner"
    )
    assert result["code"] == (0 if fault == "none" else 78)
