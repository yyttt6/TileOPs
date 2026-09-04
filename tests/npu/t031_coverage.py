"""Generate correctness-only canonical coverage for T031 supported operators."""

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

import tileops.kernels  # noqa: F401 - registers the Ascend builders
from tileops.kernels._registry import REGISTERED
from tileops.ops.elementwise.arithmetic import (
    DivFwdOp,
    LerpFwdOp,
    MaximumFwdOp,
    MinimumFwdOp,
    MulFwdOp,
    PowFwdOp,
    SubFwdOp,
)

ROOT = Path(__file__).resolve().parents[3]
DTYPES = (torch.float16, torch.bfloat16, torch.float32)
OPS = {
    "SubFwdOp": (SubFwdOp, torch.sub),
    "MulFwdOp": (MulFwdOp, torch.mul),
    "DivFwdOp": (DivFwdOp, torch.div),
    "PowFwdOp": (PowFwdOp, torch.pow),
    "LerpFwdOp": (LerpFwdOp, torch.lerp),
    "MaximumFwdOp": (MaximumFwdOp, torch.maximum),
    "MinimumFwdOp": (MinimumFwdOp, torch.minimum),
}


def _revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return result.stdout.strip() or "unversioned"


def _inputs(a_shape, b_shape, dtype):
    a = torch.rand(a_shape, dtype=dtype) * 2 + 0.25
    b = torch.rand(b_shape, dtype=dtype) * 2 + 0.25
    return a, b


def _case(op_name, op_cls, ref_fn, a_shape, b_shape, dtype, label, params):
    cpu_a, cpu_b = _inputs(a_shape, b_shape, dtype)
    if op_name == "SubFwdOp":
        ref = ref_fn(cpu_a, cpu_b, alpha=params.get("alpha", 1))
    elif op_name == "LerpFwdOp":
        ref = ref_fn(cpu_a, cpu_b, params.get("weight", 0.5))
    else:
        ref = ref_fn(cpu_a, cpu_b)
    a, b = cpu_a.npu(), cpu_b.npu()
    out = op_cls(target="ascend", **params)(a, b)
    torch.npu.synchronize()
    got = out.cpu()
    atol = rtol = 1.6e-2 if dtype == torch.bfloat16 else 1e-3
    delta = (got.float() - ref.float()).abs()
    passed = bool(torch.allclose(got.float(), ref.float(), atol=atol, rtol=rtol))
    finite = bool(torch.isfinite(got).all())
    return {
        "shape": {
            "input": list(a_shape),
            "other": list(b_shape),
            "out": list(got.shape),
        },
        "dtype": str(dtype).replace("torch.", ""),
        "label": label,
        "params": params,
        "max_abs_err": float(delta.max()),
        "tolerance": {"atol": atol, "rtol": rtol},
        "finite": finite,
        "passed": passed and finite,
    }


def run_op(op_name, out_dir: Path, report_dir: Path):
    if op_name not in REGISTERED:
        raise RuntimeError(f"{op_name} is not registered for target ascend")
    op_cls, ref_fn = OPS[op_name]
    cases = []
    for a_shape, b_shape, label in (
        ((2048, 4096), (2048, 4096), "hidden-state-prefill"),
        ((16, 256, 56, 56), (256, 1, 1), "cnn-feat-broadcast"),
    ):
        for dtype in DTYPES:
            cases.append(
                _case(op_name, op_cls, ref_fn, a_shape, b_shape, dtype, label, {})
            )
    extra_params = (
        {"alpha": 2.5}
        if op_name == "SubFwdOp"
        else ({"weight": 0.25} if op_name == "LerpFwdOp" else {})
    )
    cases.append(
        _case(
            op_name,
            op_cls,
            ref_fn,
            (17, 257),
            (17, 257),
            torch.float16,
            "nondiv-extra",
            extra_params,
        )
    )
    passed = all(case["passed"] for case in cases)
    result = {
        "schema_version": "1.0.0",
        "op": op_name,
        "target": "ascend",
        "status": "correct" if passed else "blocked",
        "device": {
            "soc": "ascend910b1",
            "visible_devices": visible_devices(),
            "npu_smi": npu_smi_info(),
        },
        "correctness": {
            "passed": passed,
            "cmd": "python -u TileOPs/tests/npu/t031_coverage.py --out tileops-ascend-harness/coverage --report-out docs/reports/R031-data/coverage",
            "per_case": cases,
        },
        "perf": {
            "timing": "not measured: T031 correctness-only",
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
            "reason": "correctness-only task; performance adapter unavailable",
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "harness_commit": git_revision(ROOT / "tileops-ascend-harness"),
        "implementation_commit": _revision(ROOT / "tileops-ascend"),
    }
    validate_result(result)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    (out_dir / f"{op_name}.json").write_text(payload, encoding="utf-8")
    (report_dir / f"{op_name}.json").write_text(payload, encoding="utf-8")
    print(f"{op_name}: correctness={passed} cases={len(cases)}")
    if not passed:
        raise AssertionError(f"{op_name} correctness failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ops", nargs="*", default=list(OPS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(31)
    for op_name in args.ops:
        run_op(op_name, args.out, args.report_out)


if __name__ == "__main__":
    main()
