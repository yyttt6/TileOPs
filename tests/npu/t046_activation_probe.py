"""Focused correctness probes for the T046 activation kernel."""

from __future__ import annotations

import argparse
import math
import os

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

from tileops.kernels.elementwise_activation import (
    _compile_activation,
    _shape_info,
    build_activation_kernel,
)


def _sentinel_kernel(n_total: int):
    tile = 2048
    block_total = tile * 2
    block_count = max(1, math.ceil(n_total / block_total))

    @tilelang.jit()
    def kernel():
        @T.prim_func
        def main(
            A: T.Tensor((n_total,), "float16"),
            Y: T.Tensor((n_total,), "float16"),
        ):
            with T.Kernel(block_count, is_npu=True) as (cid, vid):
                start = cid * block_total + vid * tile
                full = start + tile <= n_total
                x_ub = T.alloc_ub((tile,), "float16")
                y_ub = T.alloc_ub((tile,), "float16")
                T.tile.fill(x_ub, 0)
                if full:
                    T.copy(A[start : start + tile], x_ub)
                else:
                    for lane in T.serial(tile):
                        idx = start + lane
                        if idx < n_total:
                            x_ub[lane] = A[idx]
                T.tile.add(y_ub, x_ub, 1.0)
                if full:
                    T.copy(y_ub, Y[start : start + tile])
                else:
                    if start < n_total:
                        T.copy(y_ub[0 : n_total - start], Y[start:n_total])

        return main

    return kernel()


def _cpu_input(n_total: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.linspace(-3.0, 3.0, n_total, dtype=torch.float32).to(dtype)


def run_three_stage(n_total: int) -> int:
    print(f"visible={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')}")
    x_cpu = _cpu_input(n_total, torch.float16)
    x = x_cpu.to("npu")

    sentinel = torch.full((n_total,), 123.0, dtype=torch.float16, device="npu")
    torch.npu.synchronize()
    _sentinel_kernel(n_total)(x, sentinel)
    torch.npu.synchronize()
    sentinel_cpu = sentinel.cpu()
    sentinel_ref = (x_cpu + 1.0).to(torch.float16)
    print(
        "stage_a",
        f"sentinel_remaining={int((sentinel_cpu == 123).sum())}",
        f"max_abs_err={float((sentinel_cpu.float() - sentinel_ref.float()).abs().max())}",
        f"finite={bool(torch.isfinite(sentinel_cpu).all())}",
    )

    relu = build_activation_kernel(
        ((n_total,),), torch.float16, op_kind="relu", op_name="ReluFwdOp"
    )
    relu_out = relu(x)
    torch.npu.synchronize()
    relu_cpu = relu_out.cpu()
    print(
        "stage_b",
        f"shape={tuple(relu_out.shape)}",
        f"input_ptr={x.data_ptr()}",
        f"return_ptr={relu_out.data_ptr()}",
        f"distinct={x.data_ptr() != relu_out.data_ptr()}",
        f"max_abs_err={float((relu_cpu.float() - torch.relu(x_cpu).float()).abs().max())}",
        f"finite={bool(torch.isfinite(relu_cpu).all())}",
    )

    out_shape, strides = _shape_info(((n_total,),))
    dummy = torch.empty((1,), dtype=torch.float16, device="npu")
    refs = {
        "silu": F.silu,
        "gelu_none": lambda value: F.gelu(value, approximate="none"),
        "gelu_tanh": lambda value: F.gelu(value, approximate="tanh"),
    }
    ok = True
    for kind, ref_fn in refs.items():
        compiled = _compile_activation(
            (n_total,), out_shape, strides, "float16", kind, ()
        )
        output = compiled(x, dummy, dummy)
        torch.npu.synchronize()
        output_cpu = output.cpu()
        expected = ref_fn(x_cpu)
        delta = (output_cpu.float() - expected.float()).abs()
        max_index = int(delta.argmax())
        passed = bool(torch.allclose(output_cpu, expected, atol=1e-3, rtol=1e-3))
        ok &= passed
        print(
            "stage_c",
            f"kind={kind}",
            f"finite={bool(torch.isfinite(output_cpu).all())}",
            f"max_abs_err={float(delta.max())}",
            f"max_index={max_index}",
            f"input_at_max={float(x_cpu[max_index])}",
            f"actual_at_max={float(output_cpu[max_index])}",
            f"expected_at_max={float(expected[max_index])}",
            f"over_1e-3={int((delta > 1e-3).sum())}",
            f"allclose_1e-3={passed}",
        )
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("three-stage",), default="three-stage")
    parser.add_argument("--n-total", type=int, default=8195)
    args = parser.parse_args()
    tilelang.disable_cache()
    return run_three_stage(args.n_total)


if __name__ == "__main__":
    raise SystemExit(main())
