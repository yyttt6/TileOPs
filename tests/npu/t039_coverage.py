"""Generate correctness-only coverage for T039 binary operators."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from harnesslib import (
    THRESHOLD,
    git_revision,
    npu_smi_info,
    validate_result,
    visible_devices,
)

import tileops.kernels  # noqa: F401
from tileops.kernels._registry import REGISTERED
from tileops.ops.elementwise.arithmetic import FloorDivideFwdOp, RemainderFwdOp
from tileops.ops.elementwise.comparison import (
    EqFwdOp,
    GeFwdOp,
    GtFwdOp,
    LeFwdOp,
    LtFwdOp,
    NeFwdOp,
)
from tileops.ops.elementwise.logical import LogicalAndFwdOp, LogicalOrFwdOp

ROOT = Path(__file__).resolve().parents[3]
FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
OPS = {
    "FloorDivideFwdOp": (FloorDivideFwdOp, torch.floor_divide, "numeric"),
    "RemainderFwdOp": (RemainderFwdOp, torch.remainder, "numeric"),
    "EqFwdOp": (EqFwdOp, torch.eq, "predicate"),
    "NeFwdOp": (NeFwdOp, torch.ne, "predicate"),
    "GtFwdOp": (GtFwdOp, torch.gt, "predicate"),
    "GeFwdOp": (GeFwdOp, torch.ge, "predicate"),
    "LtFwdOp": (LtFwdOp, torch.lt, "predicate"),
    "LeFwdOp": (LeFwdOp, torch.le, "predicate"),
    "LogicalAndFwdOp": (LogicalAndFwdOp, torch.logical_and, "logical"),
    "LogicalOrFwdOp": (LogicalOrFwdOp, torch.logical_or, "logical"),
}


def _revision(path):
    proc = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return proc.stdout.strip() or "unversioned"


def _inputs(a_shape, b_shape, dtype, kind):
    if kind == "numeric":
        a = torch.rand(a_shape, dtype=dtype) * 16 - 8
        b = torch.rand(b_shape, dtype=dtype) * 3 + 0.5
        b.reshape(-1)[::2].neg_()
    elif dtype == torch.bool:
        a = torch.randint(0, 2, a_shape, dtype=torch.bool)
        b = torch.randint(0, 2, b_shape, dtype=torch.bool)
    else:
        a = torch.randint(-3, 4, a_shape, dtype=torch.int32).to(dtype)
        b = torch.randint(-3, 4, b_shape, dtype=torch.int32).to(dtype)
    return a, b


def _case(name, cls, ref_fn, kind, a_shape, b_shape, dtype, label):
    cpu_a, cpu_b = _inputs(a_shape, b_shape, dtype, kind)
    ref = ref_fn(cpu_a, cpu_b)
    a, b = cpu_a.npu(), cpu_b.npu()
    got_npu = cls(target="ascend")(a, b)
    torch.npu.synchronize()
    got = got_npu.cpu()
    if got.dtype == torch.bool:
        passed = torch.equal(got, ref)
        max_abs = 0.0 if passed else 1.0
        finite = True
        tolerance = {"atol": 0.0, "rtol": 0.0}
    else:
        tolerance = {
            "atol": 1.6e-2 if dtype == torch.bfloat16 else 1e-3,
            "rtol": 1.6e-2 if dtype == torch.bfloat16 else 1e-3,
        }
        delta = (got.float() - ref.float()).abs()
        max_abs = float(delta.max())
        finite = bool(torch.isfinite(got).all())
        passed = finite and torch.allclose(got.float(), ref.float(), **tolerance)
    return {
        "shape": {
            "input": list(a_shape),
            "other": list(b_shape),
            "out": list(got.shape),
        },
        "dtype": str(dtype).replace("torch.", ""),
        "label": label,
        "params": {},
        "max_abs_err": max_abs,
        "tolerance": tolerance,
        "finite": finite,
        "passed": bool(passed),
    }


def run_op(name, out_dir, report_dir):
    if name not in REGISTERED:
        raise RuntimeError(f"{name} is not registered")
    cls, ref_fn, kind = OPS[name]
    dtypes = (torch.bool, *FLOAT_DTYPES) if kind == "logical" else FLOAT_DTYPES
    cases = []
    for a_shape, b_shape, label in (
        ((2048, 4096), (2048, 4096), "hidden-state-prefill"),
        ((16, 256, 56, 56), (256, 1, 1), "cnn-feat-broadcast"),
    ):
        for dtype in dtypes:
            cases.append(_case(name, cls, ref_fn, kind, a_shape, b_shape, dtype, label))
    cases.append(
        _case(
            name, cls, ref_fn, kind, (17, 257), (17, 257), torch.float16, "nondiv-extra"
        )
    )
    passed = all(case["passed"] for case in cases)
    result = {
        "schema_version": "1.0.0",
        "op": name,
        "target": "ascend",
        "status": "correct" if passed else "blocked",
        "device": {
            "soc": "ascend910b1",
            "visible_devices": visible_devices(),
            "npu_smi": npu_smi_info(),
        },
        "correctness": {
            "passed": passed,
            "cmd": "python -u TileOPs/tests/npu/t039_coverage.py --out tileops-ascend-harness/coverage --report-out docs/reports/R039-data/coverage",
            "per_case": cases,
        },
        "perf": {
            "timing": "not measured: T039 correctness/root-cause task",
            "warmup": 0,
            "repeats": 0,
            "statistic": "none",
            "l2_flushed": False,
            "cases": [],
            "ratio_min": None,
        },
        "baseline": {
            "tier": "vendor",
            "library": "CPU PyTorch reference (correctness only)",
            "repo_commit": _revision(ROOT / "TileOPs"),
            "source_path": str(Path(torch.__file__).resolve()),
            "build_cmd": "not applicable",
            "semantic_match": "exact",
            "fusion_match": "exact",
        },
        "verdict": {
            "perf_ok": False,
            "threshold": THRESHOLD,
            "reason": "correctness/root-cause task; performance not measured",
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "harness_commit": git_revision(ROOT / "tileops-ascend-harness"),
        "implementation_commit": _revision(ROOT / "tileops-ascend"),
    }
    validate_result(result)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(payload, encoding="utf-8")
    (report_dir / f"{name}.json").write_text(payload, encoding="utf-8")
    print(f"{name}: correctness={passed} cases={len(cases)}")
    if not passed:
        raise AssertionError(f"{name} correctness failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ops", nargs="*", default=list(OPS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(39)
    for name in args.ops:
        run_op(name, args.out, args.report_out)


if __name__ == "__main__":
    main()
