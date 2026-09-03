"""Correctness coverage for the verified row-wise T067 normalization builders."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import tilelang
import torch

from harnesslib import THRESHOLD, git_revision, npu_smi_info, validate_result, visible_devices

from tileops.ops.norm.layer_norm import LayerNormFwdOp
from tileops.ops.norm.rms_norm import RMSNormFwdOp
import tileops_ascend.families.two_pass  # noqa: F401
from tileops_ascend.families.two_pass import _compile_norm


ROOT = Path(__file__).resolve().parents[2]
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}
WORKLOADS = {
    "RMSNormFwdOp": [
        ((2048, 4096), (4096,), ("float16", "bfloat16"), "llama-8b-prefill"),
        ((1, 4096), (4096,), ("bfloat16",), "llama-8b-decode"),
        ((2048, 8192), (8192,), ("float16", "bfloat16"), "llama-70b-prefill"),
        ((1, 8192), (8192,), ("bfloat16",), "llama-70b-decode"),
        ((2048, 16384), (16384,), ("float16", "bfloat16"), "llama-405b-prefill"),
        ((1, 16384), (16384,), ("bfloat16",), "llama-405b-decode"),
        ((1025, 300), (300,), ("float16",), "nondiv-tail"),
    ],
    "LayerNormFwdOp": [
        ((2048, 4096), (4096,), ("float16", "bfloat16"), "llama-8b-prefill"),
        ((1, 4096), (4096,), ("bfloat16",), "llama-8b-decode"),
        ((2048, 8192), (8192,), ("float16", "bfloat16"), "llama-70b-prefill"),
        ((1, 8192), (8192,), ("bfloat16",), "llama-70b-decode"),
        ((2048, 16384), (16384,), ("float16", "bfloat16"), "llama-405b-prefill"),
        ((1, 16384), (16384,), ("bfloat16",), "llama-405b-decode"),
        ((1025, 300), (300,), ("float16",), "nondiv-tail"),
    ],
}


def _tol(name: str, dtype: str) -> tuple[float, float]:
    if name == "RMSNormFwdOp":
        value = 1e-2 if dtype == "float16" else 1.6e-2
    else:
        value = 1e-3 if dtype == "float16" else 1e-2
    return value, value


def _run_case(name: str, shape: tuple[int, ...], dtype_name: str, label: str, index: int) -> dict:
    dtype = DTYPES[dtype_name]
    torch.manual_seed(6700 + index)
    x_cpu = torch.randn(shape, dtype=dtype)
    w_cpu = torch.randn(shape[-1], dtype=dtype)
    b_cpu = torch.randn(shape[-1], dtype=dtype) if name == "LayerNormFwdOp" else None
    x, w = x_cpu.npu(), w_cpu.npu()
    b = b_cpu.npu() if b_cpu is not None else None
    # Two launches are intentional: a single pass cannot detect F013-style tails.
    op = (RMSNormFwdOp if name == "RMSNormFwdOp" else LayerNormFwdOp)((shape[-1],), target="ascend")
    out1 = op(x, w) if b is None else op(x, w, b)
    torch.npu.synchronize()
    out2 = op(x, w) if b is None else op(x, w, b)
    torch.npu.synchronize()
    if name == "RMSNormFwdOp":
        ref = torch.nn.functional.rms_norm(x_cpu.float(), (shape[-1],), w_cpu.float(), 1e-6).to(dtype)
    else:
        ref = torch.nn.functional.layer_norm(x_cpu.float(), (shape[-1],), w_cpu.float(), b_cpu.float(), 1e-5).to(dtype)
    got1, got2 = out1.cpu(), out2.cpu()
    atol, rtol = _tol(name, dtype_name)
    delta = (got1.float() - ref.float()).abs()
    max_abs = float(delta.max())
    stable = bool(torch.equal(got1, got2))
    finite = bool(torch.isfinite(got1).all() and torch.isfinite(got2).all())
    passed = finite and stable and bool(torch.allclose(got1, ref, atol=atol, rtol=rtol))
    return {
        "shape": {"input": list(shape), "out": list(shape)},
        "dtype": dtype_name,
        "label": label,
        "params": {"normalized_shape": [shape[-1]]},
        "max_abs_err": max_abs,
        "tolerance": {"atol": atol, "rtol": rtol},
        "finite": finite,
        "double_sentinel_recheck": stable,
        "passed": passed,
    }


def run(name: str, out_dir: Path, report_dir: Path) -> None:
    cases = []
    seen_dtype = set()
    index = 0
    for shape, _normalized, dtypes, label in WORKLOADS[name]:
        for dtype_name in dtypes:
            if dtype_name not in seen_dtype:
                _compile_norm.cache_clear()
                tilelang.cache.clear_cache()
                seen_dtype.add(dtype_name)
            case = _run_case(name, shape, dtype_name, label, index)
            index += 1
            print(f"{name} {label} {dtype_name}: passed={case['passed']} max_abs={case['max_abs_err']}", flush=True)
            cases.append(case)
    passed = all(case["passed"] for case in cases)
    result = {
        "schema_version": "1.0.0",
        "op": name,
        "target": "ascend",
        "status": "correct" if passed else "blocked",
        "device": {"soc": "ascend910b1", "visible_devices": visible_devices(), "npu_smi": npu_smi_info()},
        "correctness": {"passed": passed, "cmd": f"python -u tileops-ascend/tests/t067_coverage.py {name} --out tileops-ascend-harness/coverage --report-out docs/reports/R067-data/coverage", "per_case": cases},
        "perf": {"timing": "not measured: T067 correctness batch", "warmup": 0, "repeats": 0, "statistic": "none", "l2_flushed": False, "cases": [], "ratio_min": None},
        "baseline": {"tier": "vendor", "library": "CPU PyTorch reference (correctness only)", "repo_commit": "unversioned", "source_path": str(Path(torch.__file__).resolve()), "build_cmd": "not applicable", "semantic_match": "exact", "fusion_match": "exact"},
        "verdict": {"perf_ok": False, "threshold": THRESHOLD, "reason": "correctness batch; performance not measured"},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "harness_commit": git_revision(ROOT / "tileops-ascend-harness"),
    }
    validate_result(result)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(payload, encoding="utf-8")
    (report_dir / f"{name}.json").write_text(payload, encoding="utf-8")
    if not passed:
        raise AssertionError(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("op", choices=tuple(WORKLOADS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.op, args.out, args.report_out)
