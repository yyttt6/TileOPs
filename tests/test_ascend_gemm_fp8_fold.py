"""FP8 scale folding: preserve arithmetic, scale grids, and row-tail boundaries."""

import pytest
import torch

pytest.importorskip("torch_npu")
pytestmark = pytest.mark.full


@pytest.mark.parametrize(
    "rows,k,block_rows",
    [
        (1, 128, 1),
        (3, 384, 1),
        (3, 384, 128),
        (17, 384, 1),
        (129, 384, 128),
        (129, 512, 1),
        (256, 1536, 128),
        (128, 7168, 1),
        (129, 7168, 128),
        (4096, 512, 1),
        (4097, 512, 128),
        (32768, 512, 128),
    ],
)
def test_ascend_fp8_fold_scale_grid_and_tail(rows: int, k: int, block_rows: int) -> None:
    from tileops._kernels.ascend.gemm.fp8 import _make_fold

    torch.manual_seed(187)
    scale_rows = (rows + block_rows - 1) // block_rows
    x = (torch.randn(rows, k, device="npu") * 0.25).to(torch.float8_e4m3fn)
    scales = torch.randn(scale_rows, k // 128, device="npu", dtype=torch.float32)
    expanded = scales.repeat_interleave(block_rows, 0)[:rows].repeat_interleave(128, 1)
    expected = (x.half() * expanded.half()).half()
    x_bits = x.view(torch.uint8).clone()
    scale_snapshot = scales.clone()

    result = _make_fold(rows, k, scale_rows, block_rows)(x, scales)
    torch.npu.synchronize()

    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert torch.equal(x.view(torch.uint8), x_bits)
    assert torch.equal(scales, scale_snapshot)


@pytest.mark.parametrize("mode", ["per_tensor", "block128", "block128x128"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("n,k", [(16, 128), (144, 384)])
def test_ascend_fp8_single_row_public_contract(
    mode: str, dtype: torch.dtype, bias: bool, n: int, k: int
) -> None:
    from workloads.device import set_run_device
    from workloads.gemm import GEMMFP8Workload
    from workloads.numerics import Request, verify

    from tileops._kernels.ascend.gemm import fp8  # noqa: F401 -- register this checkout
    from tileops.gemm import GEMMFP8FwdOp

    set_run_device("npu")
    torch.manual_seed(287)
    workload = GEMMFP8Workload(1, n, k, torch.float8_e4m3fn, mode, out_dtype=dtype, bias=bias)
    inputs = workload.gen_inputs()
    op = GEMMFP8FwdOp(out_dtype=dtype)
    checks = verify(
        workload.ref_program,
        inputs,
        evidence=workload.verification(*inputs),
        requests={"candidate": Request(op, inputs, preserve_inputs=True)},
    )
    assert checks["candidate"].checked_outputs == checks["candidate"].total_outputs == 1


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("large_a", [True, False])
def test_ascend_fp8_single_row_scale_order(dtype: torch.dtype, large_a: bool) -> None:
    from workloads.device import set_run_device
    from workloads.gemm import GEMMFP8Workload
    from workloads.numerics import Request, verify

    from tileops._kernels.ascend.gemm import fp8  # noqa: F401 -- register this checkout
    from tileops.gemm import GEMMFP8FwdOp

    set_run_device("npu")
    workload = GEMMFP8Workload(1, 16, 128, torch.float8_e4m3fn, "per_tensor", out_dtype=dtype)
    a_value, b_value = (1.0, 16.0) if large_a else (16.0, 1.0)
    a_scale, b_scale = (1e38, 1e-37) if large_a else (1e-37, 1e38)
    inputs = (
        torch.full((1, 128), a_value, device="npu").to(torch.float8_e4m3fn),
        torch.full((16, 128), b_value, device="npu").to(torch.float8_e4m3fn),
        torch.full((1, 1), a_scale, device="npu", dtype=torch.float32),
        torch.full((1, 1), b_scale, device="npu", dtype=torch.float32),
    )
    op = GEMMFP8FwdOp(out_dtype=dtype)
    checks = verify(
        workload.ref_program,
        inputs,
        evidence=workload.verification(*inputs),
        requests={"candidate": Request(op, inputs, preserve_inputs=True)},
    )
    assert checks["candidate"].checked_outputs == checks["candidate"].total_outputs == 1
    assert torch.isfinite(op(*inputs)).all()


@pytest.mark.parametrize("k", [128, 2048, 7168, 16384, 32768, 65536])
@pytest.mark.parametrize("per_tensor,per_row", [(True, False), (False, True), (False, False)])
def test_ascend_fp8_single_row_tile_budget(k: int, per_tensor: bool, per_row: bool) -> None:
    from tileops._kernels.ascend.gemm.fp8 import _gemv_tile

    tile = _gemv_tile(7168, k, per_tensor, per_row)
    columns = 1 if per_tensor else k // 128
    if tile is None:
        assert 2 * (k + 4 * columns + 8) + k + 4 * columns > 176 * 1024
    else:
        rows = tile if per_row else 1
        assert 128 % tile == 0 and 7168 % tile == 0
        assert 2 * (tile * k + 4 * rows * columns + 8 * tile) + k + 4 * columns <= 176 * 1024
