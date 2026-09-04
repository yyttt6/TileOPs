#!/usr/bin/env python3
"""Fresh-process correctness and sentinel evidence for T148 convolution."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "docs/reports/R148-data/regression.json"
sys.path.insert(0, str(ROOT / "TileOPs"))

CASES = {
    "resnet-fp16": ((2, 64, 56, 56), (64, 64, 3, 3), "float16", {"padding": (1, 1)}),
    "resnet-bf16": ((2, 64, 56, 56), (64, 64, 3, 3), "bfloat16", {"padding": (1, 1)}),
    "bottleneck-reduce": ((2, 512, 28, 28), (128, 512, 1, 1), "float16", {}),
    "classifier": ((1, 512, 7, 7), (2048, 512, 1, 1), "float16", {}),
    "deeplab-rate12": (
        (1, 2048, 32, 32),
        (256, 2048, 3, 3),
        "float16",
        {"padding": (12, 12), "dilation": (12, 12)},
    ),
}


def _worker(name: str, sentinel: bool) -> dict:
    import torch
    import torch.nn.functional as functional

    import tileops.kernels.families.convolution  # noqa: F401
    from tileops.kernels.convolution import build_conv2d_kernel
    from tileops.ops.convolution import Conv2dFwdOp

    input_shape, weight_shape, dtype_name, params = CASES[name]
    dtype = getattr(torch, dtype_name)
    torch.manual_seed(14800 + list(CASES).index(name))
    x = torch.randn(input_shape, device="npu", dtype=dtype) * 0.25
    weight = torch.randn(weight_shape, device="npu", dtype=dtype) * 0.25
    tolerance = 1.6e-2 if dtype == torch.bfloat16 else 1e-3

    if sentinel:
        kernel = build_conv2d_kernel(
            input_shape,
            weight_shape,
            dtype,
            stride=params.get("stride", (1, 1)),
            padding=params.get("padding", (0, 0)),
            dilation=params.get("dilation", (1, 1)),
            groups=params.get("groups", 1),
            has_bias=False,
        )
        workspace = torch.empty(kernel.workspace_shape, device="npu", dtype=dtype)
        output = torch.full(kernel.output_shape, 12345.0, device="npu", dtype=dtype)
        stream = torch.npu.current_stream().npu_stream
        kernel.compiled.adapter._forward_from_prebuild_lib(
            x, weight, workspace, output, stream=stream
        )
        torch.npu.synchronize()
        remaining = int(
            (output == torch.tensor(12345.0, device="npu", dtype=dtype)).sum()
        )
        return {
            "case": name,
            "sentinel": 12345.0,
            "remaining": remaining,
            "numel": output.numel(),
            "passed": remaining == 0,
        }

    op = Conv2dFwdOp(target="ascend", **params)
    got = op(x, weight, None)
    ref = functional.conv2d(x, weight, None, **params)
    torch.npu.synchronize()
    delta = (got.float() - ref.float()).abs()
    return {
        "case": name,
        "dtype": dtype_name,
        "path": op.kernel.path_kind,
        "max_abs_err": float(delta.max()),
        "finite": bool(torch.isfinite(got).all()),
        "atol": tolerance,
        "rtol": tolerance,
        "passed": bool(torch.allclose(got, ref, atol=tolerance, rtol=tolerance)),
        "logical_blocks": op.kernel.logical_blocks,
        "launch_blocks": op.kernel.launch_blocks,
        "grid_repeats": op.kernel.grid_repeats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=CASES)
    parser.add_argument("--sentinel", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "5":
        raise RuntimeError("T148 requires ASCEND_RT_VISIBLE_DEVICES=5")
    if args.worker:
        print(
            json.dumps(_worker(args.worker, args.sentinel), sort_keys=True), flush=True
        )
        return

    rows = []
    for name in CASES:
        command = [sys.executable, __file__, "--worker", name]
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        row = json.loads(completed.stdout.strip().splitlines()[-1])
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    command = [sys.executable, __file__, "--worker", "resnet-fp16", "--sentinel"]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    sentinel = json.loads(completed.stdout.strip().splitlines()[-1])
    print(json.dumps(sentinel, sort_keys=True), flush=True)
    payload = {"device": "5", "cases": rows, "sentinel": sentinel}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not all(row["passed"] for row in rows) or not sentinel["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
