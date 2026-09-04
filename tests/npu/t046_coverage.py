"""Fresh-process correctness coverage for the T046 activation batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from tileops.kernels.families import elementwise_activation as family

ROOT = Path(__file__).resolve().parents[3]
HARNESS = ROOT / "tileops-ascend-harness"
MANIFEST = ROOT / "TileOPs" / "src" / "tileops" / "manifest"
KERNEL_SOURCE = (
    ROOT
    / "tileops-ascend"
    / "src"
    / "tileops"
    / "kernels"
    / "elementwise_activation.py"
)

SUPPORTED = {
    "ClampFwdOp",
    "ClampScalarFwdOp",
    "EluFwdOp",
    "GeluAndMulFwdOp",
    "GeluFwdOp",
    "GeluTanhAndMulFwdOp",
    "HardsigmoidFwdOp",
    "HardswishFwdOp",
    "HardtanhFwdOp",
    "LeakyReluFwdOp",
    "LerpTensorFwdOp",
    "MishFwdOp",
    "ReluFwdOp",
    "SeluFwdOp",
    "SiluAndMulFwdOp",
    "SiluFwdOp",
    "SoftplusFwdOp",
}

BUILDERS = {
    "ClampFwdOp": family.build_clamp,
    "ClampScalarFwdOp": family.build_clamp_scalar,
    "EluFwdOp": family.build_elu,
    "GeluAndMulFwdOp": family.build_gelu_and_mul,
    "GeluFwdOp": family.build_gelu,
    "GeluTanhAndMulFwdOp": family.build_gelu_tanh_and_mul,
    "HardsigmoidFwdOp": family.build_hardsigmoid,
    "HardswishFwdOp": family.build_hardswish,
    "HardtanhFwdOp": family.build_hardtanh,
    "LeakyReluFwdOp": family.build_leaky_relu,
    "LerpTensorFwdOp": family.build_lerp_tensor,
    "MishFwdOp": family.build_mish,
    "ReluFwdOp": family.build_relu,
    "SeluFwdOp": family.build_selu,
    "SiluAndMulFwdOp": family.build_silu_and_mul,
    "SiluFwdOp": family.build_silu,
    "SoftplusFwdOp": family.build_softplus,
}

UNARY_REFS = {
    "EluFwdOp": lambda x, _: F.elu(x.float(), 1.0).to(x.dtype),
    "GeluFwdOp": lambda x, kw: F.gelu(x, approximate=kw["approximate"]),
    "HardsigmoidFwdOp": lambda x, _: F.hardsigmoid(x),
    "HardswishFwdOp": lambda x, _: F.hardswish(x),
    "HardtanhFwdOp": lambda x, _: F.hardtanh(x.float(), -1.0, 1.0).to(x.dtype),
    "LeakyReluFwdOp": lambda x, _: F.leaky_relu(x.float(), 0.01).to(x.dtype),
    "MishFwdOp": lambda x, _: F.mish(x),
    "ReluFwdOp": lambda x, _: F.relu(x),
    "SeluFwdOp": lambda x, _: F.selu(x.float()).to(x.dtype),
    "SiluFwdOp": lambda x, _: F.silu(x),
    "SoftplusFwdOp": lambda x, _: F.softplus(x.float(), 1.0, 20.0).to(x.dtype),
}


def _git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return result.stdout.strip() or "uncommitted"


def _npu_smi() -> str:
    result = subprocess.run(
        ["npu-smi", "info"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout


def _manifest_entry(op_name: str) -> dict:
    for path in sorted(MANIFEST.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if op_name in data:
            return data[op_name]
    raise KeyError(op_name)


def _dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


def _tolerance(op_name: str, dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1.0e-5, 1.0e-5
    if dtype == torch.float16:
        if op_name in {
            "GeluAndMulFwdOp",
            "GeluTanhAndMulFwdOp",
            "SiluAndMulFwdOp",
        }:
            return 1.0e-2, 1.0e-2
        return 1.0e-3, 1.0e-3
    return 1.6e-2, 1.6e-2


def _cpu_input(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    count = 1
    for extent in shape:
        count *= extent
    base = torch.arange(count, dtype=torch.int32)
    values = ((base % 8191).float() / 1365.0 - 3.0).to(dtype).reshape(shape)
    return values


def _case_specs(
    op_name: str, shape_set: str
) -> list[tuple[tuple[int, ...], torch.dtype, str, dict]]:
    gated = op_name in {
        "GeluAndMulFwdOp",
        "GeluTanhAndMulFwdOp",
        "SiluAndMulFwdOp",
    }
    tail_shape = (17, 514) if gated else (8195,)
    specs = [(tail_shape, torch.float16, "tail-fp16", {})]
    if op_name == "GeluFwdOp":
        specs.append(
            (tail_shape, torch.float16, "tail-fp16-tanh", {"approximate": "tanh"})
        )
    if shape_set == "smoke":
        return specs

    workload = _manifest_entry(op_name)["workloads"][0]
    shape_key = "x_shape" if gated else "input_shape"
    shape = tuple(workload[shape_key])
    for dtype_name in workload["dtypes"]:
        specs.append(
            (shape, _dtype(dtype_name), f"{workload['label']}-{dtype_name}", {})
        )
    return specs


def _run_case(
    op_name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    label: str,
    kwargs: dict,
) -> dict:
    builder = BUILDERS[op_name]
    x_cpu = _cpu_input(shape, dtype)
    x = x_cpu.to("npu")

    if op_name == "ClampFwdOp":
        lo = torch.full(shape, -0.75, dtype=dtype).to("npu")
        hi = torch.full(shape, 0.75, dtype=dtype).to("npu")
        tested = builder(x, min=lo, max=hi)(x, lo, hi)
        reference = torch.clamp(x, min=lo, max=hi)
    elif op_name == "ClampScalarFwdOp":
        tested = builder(x, min=-0.75, max=0.75)(x)
        reference = torch.clamp(x, min=-0.75, max=0.75)
    elif op_name == "LerpTensorFwdOp":
        end = (-x_cpu).to("npu")
        weight = torch.full(shape, 0.25, dtype=dtype).to("npu")
        tested = builder(x, end, weight)(x, end, weight)
        reference = torch.lerp(x, end, weight)
    elif op_name in {"GeluAndMulFwdOp", "GeluTanhAndMulFwdOp", "SiluAndMulFwdOp"}:
        tested = builder(x)(x)
        gate, value = x_cpu.float().chunk(2, dim=1)
        if op_name == "SiluAndMulFwdOp":
            reference = (F.silu(gate) * value).to(dtype)
        elif op_name == "GeluAndMulFwdOp":
            reference = (F.gelu(gate, approximate="none") * value).to(dtype)
        else:
            reference = (F.gelu(gate, approximate="tanh") * value).to(dtype)
    else:
        if op_name == "GeluFwdOp" and "approximate" not in kwargs:
            kwargs = {"approximate": "none"}
        tested = builder(x, **kwargs)(x)
        reference = UNARY_REFS[op_name](x, kwargs)

    torch.npu.synchronize()
    tested_cpu = tested.cpu()
    reference_cpu = reference if reference.device.type == "cpu" else reference.cpu()
    finite = bool(torch.isfinite(tested_cpu).all())
    delta = (tested_cpu.float() - reference_cpu.float()).abs()
    max_abs = float(delta.max()) if delta.numel() else 0.0
    denom = reference_cpu.float().abs().clamp_min(1.0e-12)
    max_rel = float((delta / denom).max()) if delta.numel() else 0.0
    atol, rtol = _tolerance(op_name, dtype)
    passed = finite and bool(
        torch.allclose(tested_cpu, reference_cpu, atol=atol, rtol=rtol)
    )
    return {
        "shape": list(shape),
        "dtype": str(dtype).replace("torch.", ""),
        "label": label,
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "tolerance": {"atol": atol, "rtol": rtol},
        "finite": finite,
        "passed": passed,
    }


def _result(op_name: str, cases: list[dict], command: str, npu_smi: str) -> dict:
    passed = bool(cases) and all(case["passed"] for case in cases)
    return {
        "schema_version": "1.0.0",
        "op": op_name,
        "target": "ascend",
        "status": "correct" if passed else "blocked",
        "device": {
            "soc": "ascend910b1",
            "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""),
            "npu_smi": npu_smi,
        },
        "correctness": {"passed": passed, "cmd": command, "per_case": cases},
        "perf": {
            "timing": "not measured (T046 correctness coverage)",
            "warmup": 0,
            "repeats": 0,
            "statistic": "none",
            "l2_flushed": False,
            "cases": [],
            "ratio_min": None,
        },
        "baseline": {
            "tier": "vendor",
            "library": "Torch-NPU eager reference matching TileOPs tests",
            "repo_commit": _git_revision(ROOT / "ascend_baselines" / "pytorch"),
            "source_path": "torch.nn.functional / torch elementwise API",
            "build_cmd": "preinstalled torch_npu in conda environment tlx",
            "semantic_match": "exact",
            "fusion_match": "exact",
        },
        "verdict": {
            "perf_ok": False,
            "threshold": 0.7,
            "reason": "T046 correctness-only; performance not measured",
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "harness_commit": _git_revision(HARNESS),
        "source_sha256": hashlib.sha256(KERNEL_SOURCE.read_bytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", required=True, choices=sorted(SUPPORTED))
    parser.add_argument("--shape-set", choices=("smoke", "full"), default="full")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--canonical-out", type=Path)
    args = parser.parse_args()
    if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        raise RuntimeError("ASCEND_RT_VISIBLE_DEVICES must be set")

    cases = []
    for shape, dtype, label, kwargs in _case_specs(args.op, args.shape_set):
        case = _run_case(args.op, shape, dtype, label, kwargs)
        cases.append(case)
        print(json.dumps(case, sort_keys=True))

    command = (
        "source /home/dyq/miniconda3/etc/profile.d/conda.sh; conda activate tlx; "
        "source /usr/local/Ascend/cann-8.5.0/set_env.sh; "
        "source /home/dyq/workspace/tilelang-ascend/set_env.sh; "
        f"cd {ROOT}; export ASCEND_RT_VISIBLE_DEVICES="
        f"{os.environ['ASCEND_RT_VISIBLE_DEVICES']} && python -u "
        f"TileOPs/tests/npu/t046_coverage.py --op {args.op} "
        f"--shape-set {args.shape_set} --out {args.out}"
    )
    result = _result(args.op, cases, command, _npu_smi())
    sys.path.insert(0, str(HARNESS))
    from harnesslib import validate_result

    validate_result(result)
    args.out.mkdir(parents=True, exist_ok=True)
    result_path = args.out / f"{args.op}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if result["correctness"]["passed"] and args.canonical_out is not None:
        args.canonical_out.mkdir(parents=True, exist_ok=True)
        canonical_path = args.canonical_out / f"{args.op}.json"
        canonical_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"op": args.op, "status": result["status"], "result": str(result_path)}
        )
    )
    return 0 if result["correctness"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
