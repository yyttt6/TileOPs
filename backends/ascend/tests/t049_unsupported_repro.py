"""Minimal template-boundary evidence for deferred T049 reductions."""

from __future__ import annotations

import tilelang.language as T
import torch

from tileops_ascend.kernels.reduction import build_reduction_kernel


def main() -> None:
    print("has_reduce_prod", hasattr(T, "reduce_prod"))
    print("has_welford", hasattr(T, "welford"))
    print("has_reduce_welford", hasattr(T, "reduce_welford"))
    for op_kind, op_name in (
        ("prod", "ProdFwdOp"),
        ("var", "VarFwdOp"),
        ("var_mean", "VarMeanFwdOp"),
    ):
        try:
            build_reduction_kernel(
                (17, 257),
                torch.float16,
                -1,
                False,
                op_kind=op_kind,
                op_name=op_name,
            )
        except Exception as exc:  # noqa: BLE001 - exact fail-closed evidence
            print(op_name, type(exc).__name__, str(exc))
        else:
            raise AssertionError(f"{op_name} unexpectedly entered the shared template")


if __name__ == "__main__":
    main()
