#!/usr/bin/env python3
"""Run every declared Conv2d workload in an independently spawned process."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "TileOPs/src/tileops/manifest/convolution.yaml"
DATA = ROOT / "docs/reports/R148-data"
sys.path.insert(0, str(ROOT / "TileOPs"))


def _workloads() -> list[dict]:
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["Conv2dFwdOp"][
        "workloads"
    ]


def _worker(index: int, dtype_name: str) -> dict:
    import torch
    import torch.nn.functional as functional

    import tileops.kernels.families.convolution  # noqa: F401
    from tileops.backend import TensorSpec
    from tileops.kernels._registry import REGISTERED

    workload = _workloads()[index]
    dtype = getattr(torch, dtype_name)
    input_shape = tuple(workload["input_shape"])
    groups = int(workload.get("groups", 1))
    weight_shape = (
        int(workload["C_out"]),
        input_shape[1] // groups,
        int(workload["kH"]),
        int(workload["kW"]),
    )
    params = {
        key: tuple(workload[key]) if isinstance(workload[key], list) else workload[key]
        for key in ("stride", "padding", "dilation", "groups")
        if key in workload
    }
    torch.manual_seed(13900 + index)
    x = torch.randn(input_shape, device="npu", dtype=dtype) * 0.25
    weight = torch.randn(weight_shape, device="npu", dtype=dtype) * 0.25
    bias = (
        torch.randn(weight_shape[0], device="npu", dtype=dtype) * 0.25
        if workload.get("bias_shape")
        else None
    )
    builder = REGISTERED["Conv2dFwdOp"]
    kernel = builder(
        TensorSpec.of(x),
        TensorSpec.of(weight),
        TensorSpec.of(bias) if bias is not None else None,
        **params,
    )
    got = kernel(x, weight, bias)
    ref = functional.conv2d(x, weight, bias, **params)
    torch.npu.synchronize()
    tolerance = 1.6e-2 if dtype == torch.bfloat16 else 1e-3
    delta = (got.float() - ref.float()).abs()
    return {
        "index": index,
        "label": workload["label"],
        "dtype": dtype_name,
        "path": kernel.path_kind,
        "max_abs_err": float(delta.max()),
        "finite": bool(torch.isfinite(got).all()),
        "atol": tolerance,
        "rtol": tolerance,
        "passed": bool(torch.allclose(got, ref, atol=tolerance, rtol=tolerance)),
        "status": "ok",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int)
    parser.add_argument("--dtype")
    parser.add_argument("--output", type=Path, default=DATA / "manifest-full.json")
    args = parser.parse_args()
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "5":
        raise RuntimeError("T148 requires ASCEND_RT_VISIBLE_DEVICES=5")
    if args.worker is not None:
        print(json.dumps(_worker(args.worker, args.dtype), sort_keys=True), flush=True)
        return

    log_dir = DATA / "manifest-full-cases"
    log_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, workload in enumerate(_workloads()):
        for dtype_name in workload["dtypes"]:
            command = [
                sys.executable,
                __file__,
                "--worker",
                str(index),
                "--dtype",
                dtype_name,
            ]
            try:
                completed = subprocess.run(
                    command, text=True, capture_output=True, timeout=300
                )
                raw = completed.stdout + completed.stderr
                (log_dir / f"{index:02d}-{dtype_name}.log").write_text(
                    raw, encoding="utf-8"
                )
                if completed.returncode == 0:
                    row = json.loads(completed.stdout.strip().splitlines()[-1])
                else:
                    row = {
                        "index": index,
                        "label": workload["label"],
                        "dtype": dtype_name,
                        "passed": None,
                        "status": "blocked",
                        "reason": f"worker exit {completed.returncode}",
                    }
            except subprocess.TimeoutExpired as exc:
                raw = (exc.stdout or "") + (exc.stderr or "")
                (log_dir / f"{index:02d}-{dtype_name}.log").write_text(
                    raw, encoding="utf-8"
                )
                row = {
                    "index": index,
                    "label": workload["label"],
                    "dtype": dtype_name,
                    "passed": None,
                    "status": "timeout",
                    "reason": "worker exceeded 300 seconds",
                }
            rows.append(row)
            args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(row, sort_keys=True), flush=True)

    summary = {
        "declared_cases": len(rows),
        "passed": sum(row.get("passed") is True for row in rows),
        "failed": sum(row.get("passed") is False for row in rows),
        "blocked": sum(row.get("passed") is None for row in rows),
    }
    (DATA / "manifest-full-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"summary": summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
