"""T092 profiling and tail-artifact evidence for the three-launch scan."""

import argparse
import json
import os
import statistics
import time

import torch
import torch_npu  # noqa: F401

from tileops.kernels.scan import build_scan


def profile(repeats: int) -> None:
    x = torch.full((2048, 4096), 0.001, dtype=torch.float16, device="npu")
    scan = build_scan(x, op_kind="sum")
    flat = scan._scan_prepare(x)
    local_kernel, totals_kernel, carry_kernel = scan._scan_kernels
    scan(x)
    torch.npu.synchronize()

    samples = []
    for _ in range(repeats):
        events = [torch.npu.Event(enable_timing=True) for _ in range(6)]
        torch.npu.synchronize()
        wall_start = time.perf_counter_ns()
        events[0].record()
        local = local_kernel(flat)
        events[1].record()
        events[2].record()
        totals = totals_kernel(local)
        events[3].record()
        events[4].record()
        output = carry_kernel(local, totals)
        events[5].record()
        torch.npu.synchronize()
        wall_us = (time.perf_counter_ns() - wall_start) / 1000.0
        stage_us = [events[i].elapsed_time(events[i + 1]) * 1000.0 for i in (0, 2, 4)]
        samples.append({
            "stage_us": stage_us,
            "device_total_us": sum(stage_us),
            "wall_total_us": wall_us,
            "host_and_sync_us": wall_us - sum(stage_us),
            "finite": bool(torch.isfinite(output).all()),
        })

    print(json.dumps({
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "geometry": scan._scan_geometry,
        "repeats": repeats,
        "samples": samples,
        "median": {
            "stage_us": [statistics.median(s["stage_us"][i] for s in samples) for i in range(3)],
            "device_total_us": statistics.median(s["device_total_us"] for s in samples),
            "wall_total_us": statistics.median(s["wall_total_us"] for s in samples),
            "host_and_sync_us": statistics.median(s["host_and_sync_us"] for s in samples),
        },
    }, indent=2))


def probe(length: int) -> None:
    x = torch.full((length,), 0.001, dtype=torch.float16, device="npu")
    scan = build_scan(x, op_kind="sum")
    output = scan(x)
    expected = x.float().cumsum(-1).to(x.dtype)
    torch.npu.synchronize()
    print(json.dumps({
        "length": length,
        "geometry": scan._scan_geometry,
        "finite": bool(torch.isfinite(output).all()),
        "allclose": bool(torch.allclose(output, expected, atol=1e-2, rtol=1e-2)),
        "max_abs_err": float((output - expected).abs().max().cpu()),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--length", type=int)
    args = parser.parse_args()
    if args.profile:
        profile(args.repeats)
    elif args.length is not None:
        probe(args.length)
    else:
        parser.error("select --profile or --length")


if __name__ == "__main__":
    main()
