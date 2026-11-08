"""Kernel-level grid configuration coverage."""

import pytest
import torch

import tileops.kernels.ascend.moe.permute_align as permute_align
from tileops.kernels.ascend.moe.permute_align import AscendMoEPermuteAlignKernel
from tileops.kernels.moe.call_spec import PermuteAlignCall
from workloads.device import run_device, run_device_available


@pytest.fixture(scope="session", autouse=True)
def _configure_device_limit():
    """This module must run in an isolated pytest process before NPU work."""
    device = torch.device(run_device())
    if device.type != "npu":
        return
    index = device.index if device.index is not None else 0
    torch.npu.set_device(index)
    try:
        torch.npu.set_device_limit(index, cube_num=32, vector_num=64)
    except RuntimeError as exc:
        if "already been called" not in str(exc):
            raise
    assert torch.npu.get_device_limit(index) == {"cube_core_num": 32, "vector_core_num": 64}


@pytest.mark.smoke
@pytest.mark.parametrize("n,grid,expected", [(256, None, 1), (264, None, 64), (32768, 1, 1)])
def test_grid_selection_cpu(monkeypatch, n, grid, expected):
    monkeypatch.setattr(AscendMoEPermuteAlignKernel, "refusal", lambda self, call: None)
    monkeypatch.setattr(AscendMoEPermuteAlignKernel, "_check_arch", lambda self: None)
    monkeypatch.setattr(permute_align, "_vector_capacity", lambda device: 64)
    compiled = []
    monkeypatch.setattr(permute_align, "_compile", lambda e, g: compiled.append((e, g)))
    entry = AscendMoEPermuteAlignKernel(_call(torch.empty(n), 128), grid=grid)
    assert entry.grid == expected
    assert compiled == [(128, expected)]


@pytest.mark.smoke
@pytest.mark.parametrize("grid", [0, -1, 65, True, 1.0, "1"])
def test_invalid_explicit_grid_cpu(monkeypatch, grid):
    monkeypatch.setattr(AscendMoEPermuteAlignKernel, "refusal", lambda self, call: None)
    monkeypatch.setattr(AscendMoEPermuteAlignKernel, "_check_arch", lambda self: None)
    monkeypatch.setattr(permute_align, "_vector_capacity", lambda device: 64)
    compiled = []
    monkeypatch.setattr(permute_align, "_compile", lambda e, g: compiled.append((e, g)))
    with pytest.raises(ValueError, match="effective vector_core_num"):
        AscendMoEPermuteAlignKernel(_call(torch.empty(264), 128), grid=grid)
    assert not compiled


def _require_npu():
    if torch.device(run_device()).type != "npu" or not run_device_available():
        pytest.skip("requires an Ascend NPU and --tileops-device=npu")


def _call(ids, experts):
    return PermuteAlignCall(
        num_routes=ids.numel(), num_experts=experts, block_size=64, device=ids.device
    )


def _check(ids, results, experts):
    ids = ids.cpu().reshape(-1)
    out, owner, count = (x.cpu() for x in results)
    n = ids.numel()
    p = n + (experts + 1) * 63
    assert out.shape == (p,)
    assert owner.shape == ((p + 63) // 64,)
    assert count.shape == (1,)
    assert all(x.dtype == torch.int32 for x in (out, owner, count))
    cursor = 0
    for e in range(experts):
        expected = torch.nonzero(ids == e).reshape(-1).to(torch.int32)
        padded = ((expected.numel() + 63) // 64) * 64
        torch.testing.assert_close(
            torch.sort(out[cursor : cursor + expected.numel()]).values, expected, rtol=0, atol=0
        )
        assert torch.all(out[cursor + expected.numel() : cursor + padded] == n)
        assert torch.all(owner[cursor // 64 : (cursor + padded) // 64] == e)
        cursor += padded
    assert count.item() == cursor
    assert torch.all(out[cursor:] == n)
    assert torch.all(owner[cursor // 64 :] == -1)


@pytest.mark.smoke
def test_kernel_explicit_grid_one_large_n():
    _require_npu()
    ids = (torch.arange(32768, dtype=torch.int32, device=run_device()) % 128).reshape(-1, 8)
    entry = AscendMoEPermuteAlignKernel(_call(ids, 128), grid=1)
    assert entry.grid == 1
    _check(ids, entry.forward(ids), 128)


@pytest.mark.smoke
def test_kernel_grid_above_device_limit():
    _require_npu()
    ids = torch.zeros(264, dtype=torch.int32, device=run_device())
    with pytest.raises(ValueError, match="effective vector_core_num"):
        AscendMoEPermuteAlignKernel(_call(ids, 128), grid=72)
