"""Three-stage output-write diagnostic for the T049 reduction template."""

from __future__ import annotations

import torch

from tileops.kernels.reduction import build_reduction_kernel


def main() -> None:
    shape = (129, 300)

    numeric_host = (
        torch.arange(shape[0] * shape[1], dtype=torch.float32)
        .reshape(shape)
        .remainder(13)
        .sub(6)
        .to(torch.float16)
    )
    numeric_kernel = build_reduction_kernel(
        shape,
        torch.float16,
        -1,
        False,
        op_kind="sum",
        op_name="SumFwdOp",
        diagnostic_sentinel=True,
    )
    numeric_input = numeric_host.npu()
    numeric_output = numeric_kernel(numeric_input)
    torch.npu.synchronize()
    numeric_got = numeric_output.cpu()
    numeric_ref = numeric_host.float().sum(-1).to(torch.float16)
    print(
        "numeric.returned_distinct",
        numeric_output.data_ptr() != numeric_input.data_ptr(),
    )
    print("numeric.sentinel_remaining", int((numeric_got == 123).sum()))
    print("numeric.reference_equal", torch.equal(numeric_got, numeric_ref))

    logical_host = torch.ones(shape, dtype=torch.bool)
    logical_kernel = build_reduction_kernel(
        shape,
        torch.bool,
        -1,
        False,
        op_kind="all",
        op_name="AllFwdOp",
        diagnostic_sentinel=True,
    )
    logical_input = logical_host.npu()
    logical_output = logical_kernel(logical_input)
    torch.npu.synchronize()
    logical_got = logical_output.cpu()
    logical_ref = torch.all(logical_host, dim=-1)
    print(
        "logical.returned_distinct",
        logical_output.data_ptr() != logical_input.data_ptr(),
    )
    print("logical.false_sentinel_remaining", int((~logical_got).sum()))
    print("logical.reference_equal", torch.equal(logical_got, logical_ref))

    passed = (
        numeric_output.data_ptr() != numeric_input.data_ptr()
        and not bool((numeric_got == 123).any())
        and torch.equal(numeric_got, numeric_ref)
        and logical_output.data_ptr() != logical_input.data_ptr()
        and bool(logical_got.all())
        and torch.equal(logical_got, logical_ref)
    )
    print("sentinel_probe_passed", passed)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
