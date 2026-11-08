"""Public Ascend permute-align dispatch, dynamic cache and result coverage."""

from types import SimpleNamespace

import pytest
import torch

from tileops.kernels.ascend.moe.permute_align import AscendMoEPermuteAlignKernel, _vector_capacity
from tileops.kernels.moe.call_spec import PermuteAlignCall
from tileops.ops.moe.permute_align import MoEPermuteAlignFwdOp
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
@pytest.mark.parametrize("limit,expected", [(64, 64), (80, 72)])
def test_effective_vector_capacity_cpu(monkeypatch, limit, expected):
    npu = SimpleNamespace(
        get_device_properties=lambda device: SimpleNamespace(vector_core_num=72),
        get_device_limit=lambda device: {"vector_core_num": limit},
    )
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    assert _vector_capacity(0) == expected


@pytest.mark.smoke
@pytest.mark.parametrize("limit", [0, -1, None, True, "64"])
def test_invalid_vector_limit_is_not_ignored_cpu(monkeypatch, limit):
    npu = SimpleNamespace(
        get_device_properties=lambda device: SimpleNamespace(vector_core_num=72),
        get_device_limit=lambda device: {"vector_core_num": limit},
    )
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    with pytest.raises(RuntimeError, match="invalid vector_core_num"):
        _vector_capacity(0)


@pytest.mark.smoke
def test_legacy_no_limit_getter_cpu(monkeypatch):
    npu = SimpleNamespace(get_device_properties=lambda device: SimpleNamespace(vector_core_num=72))
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    assert _vector_capacity(0) == 72


@pytest.mark.smoke
def test_cached_grid_rechecked_cpu(monkeypatch):
    npu = SimpleNamespace(
        get_device_properties=lambda device: SimpleNamespace(vector_core_num=72),
        get_device_limit=lambda device: {"vector_core_num": 64},
    )
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    monkeypatch.setattr(AscendMoEPermuteAlignKernel, "_require_devices", lambda *args, **kw: None)
    entry = object.__new__(AscendMoEPermuteAlignKernel)
    entry.grid = 72
    entry.device_index = 0
    with pytest.raises(RuntimeError, match="cached grid exceeds"):
        entry.forward(torch.empty(8, dtype=torch.int32))


@pytest.mark.smoke
def test_limit_getter_errors_propagate_cpu(monkeypatch):
    def fail(device):
        raise RuntimeError("getter failed")

    npu = SimpleNamespace(
        get_device_properties=lambda device: SimpleNamespace(vector_core_num=72),
        get_device_limit=fail,
    )
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    with pytest.raises(RuntimeError, match="getter failed"):
        _vector_capacity(0)


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
def test_public_op_dynamic_n_and_compiled_cache():
    _require_npu()
    experts = 384
    ops = [MoEPermuteAlignFwdOp(experts), MoEPermuteAlignFwdOp(experts)]
    entries, retained = [], []
    for i, n in enumerate((8, 248, 256, 264, 32768, 65536, 264)):
        ids = (torch.arange(n, dtype=torch.int32, device=run_device()) * 17 % experts).reshape(
            -1, 8
        )
        before = ids.clone()
        op = ops[i % 2]
        outputs = op(ids)
        assert all(x.data_ptr() != ids.data_ptr() for x in outputs)
        entry = op.kernel_for("permute_align", _call(ids, experts))
        assert isinstance(entry, AscendMoEPermuteAlignKernel)
        entries.append(entry)
        _check(ids, outputs, experts)
        torch.testing.assert_close(ids, before, rtol=0, atol=0)
        retained.append(outputs)
    for entry in entries:
        assert entry.grid == (1 if entry.call.num_routes <= 256 else 64)
        assert all(other.kernels is entry.kernels for other in entries if other.grid == entry.grid)
    assert len({x.data_ptr() for result in retained for x in result}) == 3 * len(retained)
    # Later calls must not alter earlier output allocations.
    for i, n in enumerate((8, 248, 256, 264, 32768, 65536, 264)):
        ids = (torch.arange(n, dtype=torch.int32) * 17 % experts).reshape(-1, 8)
        _check(ids, retained[i], experts)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "n,experts,block", [(0, 8, 64), (7, 8, 64), (8, 7, 64), (8, 392, 64), (8, 8, 32)]
)
def test_public_op_rejects_unsupported_shapes(n, experts, block):
    _require_npu()
    shape = (n // 8, 8) if n % 8 == 0 else (1, n)
    ids = torch.zeros(shape, dtype=torch.int32, device=run_device())
    with pytest.raises(ValueError, match="no implementation serves"):
        MoEPermuteAlignFwdOp(experts, block)(ids)


@pytest.mark.smoke
@pytest.mark.parametrize("bad", ["dtype", "stride"])
def test_public_op_rejects_bad_input(bad):
    _require_npu()
    ids = torch.zeros((8, 16), dtype=torch.int32, device=run_device())
    ids = ids.to(torch.int64) if bad == "dtype" else ids[:, ::2]
    expected_message = "dtype is not int32" if bad == "dtype" else "contiguous int32"
    with pytest.raises(ValueError, match=expected_message):
        MoEPermuteAlignFwdOp(8)(ids)


@pytest.mark.smoke
def test_public_default_grid_limit_and_stream():
    _require_npu()
    op = MoEPermuteAlignFwdOp(128)
    small = (torch.arange(256, dtype=torch.int32, device=run_device()) % 128).reshape(-1, 8)
    _check(small, op(small), 128)
    assert op.kernel_for("permute_align", _call(small, 128)).grid == 1

    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        ids = (torch.arange(264, dtype=torch.int32, device=run_device()) % 128).reshape(-1, 8)
        before = ids.clone()
        cold = op(ids)
        warm = op(ids)
        ids2 = (torch.arange(520, dtype=torch.int32, device=run_device()) % 128).reshape(-1, 8)
        second_n = op(ids2)
        consumed = tuple(x.clone() for x in warm)
    stream.synchronize()
    entry = op.kernel_for("permute_align", _call(ids, 128))
    assert entry.grid == 64
    _check(ids, cold, 128)
    _check(ids, warm, 128)
    _check(ids2, second_n, 128)
    assert op.kernel_for("permute_align", _call(ids2, 128)).kernels is entry.kernels
    for x, y in zip(consumed, warm, strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    torch.testing.assert_close(ids, before, rtol=0, atol=0)
