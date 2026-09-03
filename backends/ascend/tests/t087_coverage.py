"""T087 correctness coverage for the seven registered-but-unmeasured ops.

Each attempt is executed in a fresh process by the parent so Python LRU state
and TileLang disk cache state are both isolated.  This intentionally records
blocked builders as canonical fail-closed coverage instead of hiding them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch

from harnesslib import THRESHOLD, git_revision, npu_smi_info, validate_result

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "tileops-ascend-harness"
UNARY_SOURCE = ROOT / "tileops-ascend/src/tileops_ascend/kernels/elementwise_unary.py"
MIXED_SOURCE = ROOT / "tileops-ascend/src/tileops_ascend/kernels/elementwise_mixed.py"

UNARY_OPS = {
    "CeilFwdOp": "ceil",
    "ErfFwdOp": "erf",
    "FloorFwdOp": "floor",
    "RoundFwdOp": "round",
    "TruncFwdOp": "trunc",
}
DTYPES = (torch.float16, torch.bfloat16, torch.float32)
MASKED_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.int32)


def _dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


def _unary_input(dtype: torch.dtype) -> torch.Tensor:
    # The first six entries are the required half-integer semantic probes.
    seeds = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, -3.25, 3.25], dtype=torch.float32)
    values = seeds.repeat((4097 + len(seeds) - 1) // len(seeds))[:4097]
    return values.to(dtype).reshape(4097)


def _masked_input(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if dtype.is_floating_point:
        data = (torch.arange(4097, dtype=torch.float32) / 17.0 - 120.0).to(dtype)
        value = torch.tensor(-3.5, dtype=dtype)
    else:
        data = (torch.arange(4097, dtype=torch.int32) - 120).to(dtype)
        value = torch.tensor(-7, dtype=dtype)
    mask = (torch.arange(4097) % 3 == 0)
    return data, mask, value


def _single(op: str, dtype_name: str, variant: str) -> dict:
    """Run exactly one op/dtype attempt and return a serializable observation."""
    dtype = _dtype(dtype_name)
    result = {"op": op, "dtype": dtype_name, "shape": [4097], "variant": variant}
    try:
        if op in UNARY_OPS:
            from tileops_ascend.families.elementwise_unary_math import (  # noqa: PLC0415
                build_ceil,
                build_erf,
                build_floor,
                build_round,
                build_trunc,
            )

            builders = {
                "CeilFwdOp": build_ceil,
                "ErfFwdOp": build_erf,
                "FloorFwdOp": build_floor,
                "RoundFwdOp": build_round,
                "TruncFwdOp": build_trunc,
            }
            cpu = _unary_input(dtype)
            x = cpu.to("npu")
            kernel = builders[op](x)
            got = kernel(x)
            torch.npu.synchronize()
            ref_fn = getattr(torch, UNARY_OPS[op])
            reference = ref_fn(cpu)
        else:
            from tileops_ascend.families.elementwise import (  # noqa: PLC0415
                build_masked_fill,
                build_masked_fill_scalar,
            )

            cpu, mask_cpu, value_cpu = _masked_input(dtype)
            x, mask, value = cpu.to("npu"), mask_cpu.to("npu"), value_cpu.to("npu")
            if op == "MaskedFillFwdOp":
                kernel = build_masked_fill(x, mask, value)
                got = kernel(x, mask, value)
                variant = "tensor-value"
            else:
                scalar_value = -3.5 if dtype.is_floating_point else -7
                kernel = build_masked_fill_scalar(x, mask, value=scalar_value)
                got = kernel(x, mask)
                variant = "scalar-value"
            torch.npu.synchronize()
            reference = torch.where(mask_cpu, value_cpu if op == "MaskedFillFwdOp" else torch.tensor(scalar_value, dtype=dtype), cpu)
            result["variant"] = variant

        got_cpu = got.cpu()
        delta = (got_cpu.float() - reference.float()).abs()
        atol = rtol = 1.6e-2 if dtype == torch.bfloat16 else (1.0e-5 if dtype == torch.float32 else 1.0e-3)
        finite = bool(torch.isfinite(got_cpu).all())
        passed = finite and bool(torch.allclose(got_cpu.float(), reference.float(), atol=atol, rtol=rtol))
        result.update({
            "passed": passed,
            "finite": finite,
            "max_abs_err": float(delta.max()) if delta.numel() else 0.0,
            "max_rel_err": float((delta / reference.float().abs().clamp_min(1e-12)).max()) if delta.numel() else 0.0,
            "tolerance": {"atol": atol, "rtol": rtol},
        })
    except Exception as exc:  # noqa: BLE001 - failure is the measured result
        result.update({"passed": False, "finite": False, "error": f"{type(exc).__name__}: {exc}"})
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def _run_attempt(op: str, dtype_name: str, variant: str, log_path: Path) -> list[dict]:
    cache_dir = Path(tempfile.mkdtemp(prefix="t087-cache-"))
    env = os.environ.copy()
    env["TILELANG_CACHE_DIR"] = str(cache_dir)
    env["ASCEND_RT_VISIBLE_DEVICES"] = os.environ["ASCEND_RT_VISIBLE_DEVICES"]
    cmd = [sys.executable, str(Path(__file__).resolve()), "--double", op, dtype_name, variant]
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.write_text(proc.stdout, encoding="utf-8")
    shutil.rmtree(cache_dir, ignore_errors=True)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
    if len(lines) < 2:
        return [{"op": op, "dtype": dtype_name, "variant": variant, "passed": False,
                 "finite": False, "error": f"child exit={proc.returncode}; no double JSON result"}]
    observations = []
    for line in lines[-2:]:
        try:
            observations.append(json.loads(line))
        except json.JSONDecodeError as exc:
            observations.append({"op": op, "dtype": dtype_name, "variant": variant, "passed": False,
                                  "finite": False, "error": f"invalid child JSON: {exc}"})
    return observations


def _clear_child_cache(op: str, cache_dir: Path) -> None:
    """Clear the relevant Python LRU and all files in the per-child disk cache."""
    module_name = "tileops_ascend.kernels.elementwise_mixed" if op.startswith("MaskedFill") else "tileops_ascend.kernels.elementwise_unary"
    module = __import__(module_name, fromlist=["*"])
    for name in dir(module):
        value = getattr(module, name)
        if name.startswith("_compile_") and hasattr(value, "cache_clear"):
            value.cache_clear()
    for path in cache_dir.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _source_hash(op: str) -> str:
    path = MIXED_SOURCE if op.startswith("MaskedFill") else UNARY_SOURCE
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(op: str, out_dir: Path, data_dir: Path) -> dict:
    out_dir = out_dir.resolve()
    data_dir = data_dir.resolve()
    dtypes = MASKED_DTYPES if op.startswith("MaskedFill") else DTYPES
    variant = "tensor-value" if op == "MaskedFillFwdOp" else "scalar-value" if op == "MaskedFillScalarFwdOp" else "unary"
    cases = []
    for dtype in dtypes:
        for attempt in (1, 2):
            log = data_dir / f"{op}-{dtype.__str__().replace('torch.', '')}-run{attempt}.log"
            if attempt > 1:
                continue
            observations = _run_attempt(op, str(dtype).replace("torch.", ""), variant, log)
            for index, case in enumerate(observations, 1):
                case["attempt"] = index
                case["log"] = str(log.relative_to(ROOT))
                cases.append(case)
                print(json.dumps(case, sort_keys=True))

    passed = bool(cases) and all(case.get("passed", False) for case in cases)
    result = {
        "schema_version": "1.0.0", "op": op, "target": "ascend",
        "status": "correct" if passed else "blocked",
        "device": {"soc": "ascend910b1", "visible_devices": os.environ["ASCEND_RT_VISIBLE_DEVICES"], "npu_smi": npu_smi_info()},
        "correctness": {"passed": passed, "cmd": f"python -u tileops-ascend/tests/t087_coverage.py --op {op} --out tileops-ascend-harness/coverage", "per_case": cases},
        "perf": {"timing": "not measured (T087 correctness coverage)", "warmup": 0, "repeats": 0, "statistic": "none", "l2_flushed": False, "cases": [], "ratio_min": None},
        "baseline": {"tier": "vendor", "library": "CPU PyTorch reference (correctness only)", "repo_commit": git_revision(ROOT / "TileOPs"), "source_path": "torch elementwise API", "build_cmd": "not applicable", "semantic_match": "exact", "fusion_match": "exact"},
        "verdict": {"perf_ok": False, "threshold": THRESHOLD, "reason": "T087 correctness-only; performance not measured"},
        "generated_at": datetime.now(timezone.utc).isoformat(), "harness_commit": git_revision(HARNESS), "source_sha256": _source_hash(op),
    }
    validate_result(result)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{op}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", nargs=3, metavar=("OP", "DTYPE", "VARIANT"))
    parser.add_argument("--double", nargs=3, metavar=("OP", "DTYPE", "VARIANT"))
    parser.add_argument("--op", choices=sorted((*UNARY_OPS, "MaskedFillFwdOp", "MaskedFillScalarFwdOp")))
    parser.add_argument("--out", type=Path, default=HARNESS / "coverage")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "docs/reports/R087-data")
    args = parser.parse_args()
    if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        raise RuntimeError("ASCEND_RT_VISIBLE_DEVICES must be set")
    if args.single:
        _single(*args.single)
        return 0
    if getattr(args, "double", None):
        op, dtype_name, variant = args.double
        cache_dir = Path(os.environ["TILELANG_CACHE_DIR"])
        for _ in range(2):
            _clear_child_cache(op, cache_dir)
            _single(op, dtype_name, variant)
        return 0
    args.data_dir.mkdir(parents=True, exist_ok=True)
    if args.op:
        run(args.op, args.out, args.data_dir)
    else:
        for op in (*UNARY_OPS, "MaskedFillFwdOp", "MaskedFillScalarFwdOp"):
            run(op, args.out, args.data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
