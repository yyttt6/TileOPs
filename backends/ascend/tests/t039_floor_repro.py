"""Minimal fp32 counterexample for div-then-floor lowering on Ascend."""

import torch

from tileops_ascend.kernels.elementwise_binary_batch import build_batch_binary


def main():
    a_cpu = torch.tensor([-4.783865928649902], dtype=torch.float32)
    b_cpu = torch.tensor([-1.594622015953064], dtype=torch.float32)
    a, b = a_cpu.npu(), b_cpu.npu()
    floor_kernel = build_batch_binary(
        a.shape,
        b.shape,
        a.dtype,
        op_kind="floor_divide",
        op_name="floor-divide-minimal-repro",
    )
    remainder_kernel = build_batch_binary(
        a.shape,
        b.shape,
        a.dtype,
        op_kind="remainder",
        op_name="remainder-minimal-repro",
    )
    floor_got = floor_kernel(a, b)
    remainder_got = remainder_kernel(a, b)
    torch.npu.synchronize()
    print("a", a_cpu.item(), "b", b_cpu.item())
    print("fp32_div", (a_cpu / b_cpu).item(), "fp32_3b", (3 * b_cpu).item())
    print(
        "floor",
        floor_got.cpu().item(),
        "reference",
        torch.floor_divide(a_cpu, b_cpu).item(),
    )
    print(
        "remainder",
        remainder_got.cpu().item(),
        "reference",
        torch.remainder(a_cpu, b_cpu).item(),
    )


if __name__ == "__main__":
    main()
