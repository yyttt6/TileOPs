"""Selection must keep a single-kernel PR small and cover helper-only changes."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "pr_perf", Path(__file__).parents[1] / "pr_performance.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


pytestmark = pytest.mark.smoke


def test_changed_kernel_and_helper(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()

    git("init")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "test")
    family = tmp_path / "src/tileops/_kernels/ascend/gemm"
    family.mkdir(parents=True)
    (family / "bmm.py").write_text('register("BmmFwdOp")\n')
    (family / "helper.py").write_text("X=1\n")
    git("add", ".")
    git("commit", "-m", "base")
    base = git("rev-parse", "HEAD")
    (family / "gemm.py").write_text('register("GemmFwdOp")\n')
    git("add", ".")
    git("commit", "-m", "new kernel")
    assert MODULE.select(tmp_path, base)["operators"] == ["GemmFwdOp"]
    base = git("rev-parse", "HEAD")
    (family / "helper.py").write_text("X=2\n")
    git("add", ".")
    git("commit", "-m", "shared helper")
    assert MODULE.select(tmp_path, base)["operators"] == ["BmmFwdOp", "GemmFwdOp"]


def test_changed_delegate_retests_supported_composites():
    registered = {
        "operators": ["LeafOp", "InnerOp", "OuterOp", "UnrelatedOp"],
        "delegates": {"InnerOp": ["LeafOp"], "OuterOp": ["InnerOp"]},
    }
    assert MODULE.affected_composites(["LeafOp"], registered) == ["InnerOp", "LeafOp", "OuterOp"]
    assert MODULE.affected_composites(["UnrelatedOp"], registered) == ["UnrelatedOp"]


@pytest.mark.parametrize(
    "path,op",
    [
        ("convolution/conv1d.py", "Conv1dFwdOp"),
        ("convolution/conv1d_shallow.py", "Conv1dFwdOp"),
        ("transform/hadamard.py", "HadamardTransformFwdOp"),
    ],
)
def test_builtin_ascend_without_registration_literal_is_selected(tmp_path, path, op):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()

    git("init")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "test")
    source = tmp_path / "src/tileops/_kernels/ascend" / path
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n")
    git("add", ".")
    git("commit", "-m", "base")
    base = git("rev-parse", "HEAD")
    source.write_text("VALUE = 2\n")
    git("add", ".")
    git("commit", "-m", "builtin changed")
    assert op in MODULE.select(tmp_path, base)["operators"]


def test_retired_native_history_cannot_override_a_qualified_alternative():
    params = {"x": [[8], "float32"]}
    shared = dict(op="ExampleOp", case="case", workload={"params": params})
    native = dict(shared, library="provider", latency_us=10, measurement_id="native")
    alternative = dict(
        shared,
        library="old (via provider)",
        actual_library="provider",
        requested_library="old",
        latency_us=10,
        measurement_id="reuse",
    )
    retired = dict(shared, library="old", latency_us=1, measurement_id="retired")
    snapshot = dict(
        rows=[alternative, native], historical_rows=[retired], retired_measurement_ids=["retired"]
    )
    assert MODULE.matching_baselines(snapshot, "ExampleOp", "case", params) == [native]


def test_legacy_snapshot_without_retirement_metadata_keeps_admitted_history():
    saved = dict(
        library="old",
        op="ExampleOp",
        case="case",
        workload={"params": {}},
        latency_us=1,
        measurement_id="legacy",
    )
    assert MODULE.matching_baselines(
        dict(rows=[], historical_rows=[saved]), "ExampleOp", "case", {}
    ) == [saved]
