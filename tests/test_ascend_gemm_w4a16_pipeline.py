"""W4A16 producer ownership, persistent tails, and physical tiling constraints."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu")
pytestmark = pytest.mark.full


@pytest.mark.parametrize("m", [1, 2, 3, 16, 32, 65, 128, 129, 512])
@pytest.mark.parametrize("n", [16, 32, 48, 64, 128, 256, 496, 8192])
def test_ascend_w4a16_tile_grid_budget(m: int, n: int) -> None:
    from tileops._kernels.ascend.gemm.w4a16 import _tiles

    tile_m, tile_n, tile_k, stages, blocks = _tiles(m, n, 384)
    assert 16 <= tile_m <= 256 and tile_m % 16 == 0
    assert 16 <= tile_n <= min(256, n) and n % tile_n == 0
    assert tile_k == 128 and stages == 2 and blocks == 32
    assert 2 * (tile_m + tile_n) * tile_k * stages <= 512 * 1024
    assert 4 * tile_m * tile_n <= 256 * 1024
    atom = 16
    if ((m + 15) // 16) * (n // atom) >= blocks:
        assert ((m + tile_m - 1) // tile_m) * (n // tile_n) >= blocks


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 16, 128),
        (3, 128, 384),
        (65, 128, 384),
        (129, 8192, 384),
        (128, 6400, 384),
        (1, 16, 8192),
        (1, 2048, 384),
        (1, 8192, 32256),
        (1, 8192, 32384),
        (1, 8192, 32512),
        (1, 8192, 32768),
        (1, 12288, 8192),
        (129, 8192, 512),
        (1, 4096, 14336),
        (1, 4096, 14464),
        (129, 8192, 256),
        (129, 8192, 4864),
        (129, 8192, 5120),
        (512, 8192, 512),
        (129, 7680, 512),
    ],
)
def test_ascend_w4a16_producer_pipeline_boundaries(m: int, n: int, k: int) -> None:
    from tileops._kernels.ascend.gemm import w4a16  # noqa: F401 -- register this checkout
    from tileops.gemm import GEMMW4A16FwdOp
    from workloads.device import set_run_device
    from workloads.gemm import GEMMW4A16Workload
    from workloads.numerics import Request, verify

    set_run_device("npu")
    torch.manual_seed(328)
    workload = GEMMW4A16Workload(m, n, k, torch.float16)
    inputs = workload.gen_inputs()
    op = GEMMW4A16FwdOp()
    checks = verify(
        workload.ref_program,
        inputs,
        evidence=workload.verification(*inputs),
        requests={"candidate": Request(op, inputs, preserve_inputs=True)},
    )
    assert checks["candidate"].checked_outputs == checks["candidate"].total_outputs == 1
    zeroed = (torch.zeros_like(inputs[0]), *inputs[1:])
    assert torch.count_nonzero(op(*zeroed)) == 0


@pytest.mark.parametrize("k", [128, 256, 384, 8192, 81920])
@pytest.mark.parametrize("tile_m,tile_n", [(16, 16), (32, 32), (128, 128), (256, 256)])
def test_ascend_w4a16_resident_capacity(k: int, tile_m: int, tile_n: int) -> None:
    from tileops._kernels.ascend.gemm.w4a16_resident import fits

    expected = (
        2 * (tile_m + tile_n) * k <= 512 * 1024
        and 4 * tile_m * tile_n <= 256 * 1024
        and tile_n * (k // 2 + 6 * k + 3 * (k // 128)) + 512 <= 176 * 1024
    )
    assert fits(tile_m + 1, tile_n * 2, k, tile_m, tile_n) == expected
    assert not fits(tile_m, tile_n + 1, k, tile_m, tile_n)


@pytest.mark.parametrize("k", [128, 384, 8192, 16384, 81920])
@pytest.mark.parametrize("tile_n", [16, 32, 128, 256])
def test_ascend_w4a16_metadata_residency_budget(k: int, tile_n: int) -> None:
    from tileops._kernels.ascend.gemm.w4a16 import _metadata_resident, _producer_rows

    _, rows = _producer_rows(tile_n)
    expected = 2 * ((rows + 3) * 64 + 2 * (rows + 1) * 128) + 3 * rows * (k // 128) + 512
    assert _metadata_resident(tile_n, k, 2) == (expected <= 176 * 1024)


@pytest.mark.parametrize(
    "k,expected", [(128, 128), (256, 256), (384, 128), (4864, 256), (5120, 128)]
)
def test_ascend_w4a16_paired_group_capacity(k: int, expected: int) -> None:
    from tileops._kernels.ascend.gemm.w4a16 import _fused_program, _metadata_resident, _tiles

    assert _tiles(512, 8192, k)[2] == expected
    assert _metadata_resident(256, 4864, 2, 256)
    assert not _metadata_resident(256, 5120, 2, 256)
    assert not _metadata_resident(240, 256, 2, 256)
    assert _metadata_resident(256, 32256, 2, 128)
    assert not _metadata_resident(256, 32384, 2, 128)
    # A forced pair beyond the metadata bound must refuse rather than stream one group.
    with pytest.raises(ValueError, match="full metadata"):
        _fused_program(512, 8192, 5120, 256, 256, 256, 2, 32)


@pytest.mark.parametrize("k", [256, 512, 4864])
def test_ascend_w4a16_paired_groups_distinct_metadata(k: int) -> None:
    from tileops._kernels.ascend.gemm.w4a16 import _make_fused, _make_permute, _permute_index
    from workloads.device import set_run_device
    from workloads.gemm import GEMMW4A16Workload
    from workloads.numerics import Request, verify

    set_run_device("npu")
    torch.manual_seed(328)
    workload = GEMMW4A16Workload(129, 8192, k, torch.float16)
    activation, packed, scales, zeros = workload.gen_inputs()
    scales[:, 0::2].fill_(0.03125)
    scales[:, 1::2].fill_(0.0625)
    zeros[:, 0::2].fill_(3)
    zeros[:, 1::2].fill_(11)
    order = torch.tensor(_permute_index(), dtype=torch.uint16, device="npu")

    def candidate(a, p, s, z):
        return _make_fused(129, 8192, k)(_make_permute(129, k)(a, order), p, s, z)

    inputs = activation, packed, scales, zeros
    checks = verify(
        workload.ref_program,
        inputs,
        evidence=workload.verification(*inputs),
        requests={"candidate": Request(candidate, inputs, preserve_inputs=True)},
    )
    assert checks["candidate"].checked_outputs == checks["candidate"].total_outputs == 1


@pytest.mark.parametrize("k", [128, 8192])
@pytest.mark.parametrize("zero", [0, 15, 127, 128, 255])
def test_ascend_w4a16_full_uint8_zero_point(k: int, zero: int) -> None:
    from tileops._kernels.ascend.gemm import w4a16  # noqa: F401 -- register this checkout
    from tileops.gemm import GEMMW4A16FwdOp
    from workloads.device import set_run_device
    from workloads.gemm import GEMMW4A16Workload
    from workloads.numerics import Request, verify

    set_run_device("npu")
    workload = GEMMW4A16Workload(1, 32, k, torch.float16)
    activation, packed, scales, zeros = workload.gen_inputs()
    activation.fill_(0.125)
    scales.fill_(0.03125)
    zeros.fill_(zero)
    inputs = activation, packed, scales, zeros
    checks = verify(
        workload.ref_program,
        inputs,
        evidence=workload.verification(*inputs),
        requests={"candidate": Request(GEMMW4A16FwdOp(), inputs, preserve_inputs=True)},
    )
    assert checks["candidate"].checked_outputs == 1


@pytest.mark.parametrize("n,k", [(1, 128), (7, 384), (32, 8192)])
def test_ascend_w4a16_public_repack_matches_contract(n: int, k: int) -> None:
    from tileops._kernels.ascend.gemm import w4a16  # noqa: F401 -- register this checkout
    from tileops.gemm import GEMMW4A16FwdOp
    from workloads.gemm import repack_w4a16_weight

    source = torch.randint(0, 256, (n, k // 2), device="npu", dtype=torch.uint8)
    snapshot = source.clone()
    expected = repack_w4a16_weight(source)
    result = GEMMW4A16FwdOp().repack(source)
    assert torch.equal(source, snapshot)
    assert torch.equal(result, expected)
    assert result.data_ptr() != source.data_ptr()


def test_ascend_w4a16_empty_reduction_is_refused() -> None:
    from tileops._kernels.ascend.gemm.w4a16 import AscendGEMMW4A16Kernel

    call = SimpleNamespace(group_size=128, dtype=torch.float16, m=1, n=32, k=0)
    assert "positive dimensions" in AscendGEMMW4A16Kernel.refusal(call)
