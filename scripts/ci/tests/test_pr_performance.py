"""Selection must keep a single-kernel PR small and cover helper-only changes."""

import importlib.util
import subprocess
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "pr_perf", Path(__file__).parents[1] / "pr_performance.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_changed_kernel_and_helper(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()

    git("init")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "test")
    family = tmp_path / "src/tileops/kernels/ascend/gemm"
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
