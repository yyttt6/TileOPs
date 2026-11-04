"""Comparison contracts across full vector blocks and exact storage boundaries."""

import pytest
import torch

from tileops.ops import elementwise
from workloads.device import run_device

_NAMES = ("Eq", "Ne", "Gt", "Ge", "Lt", "Le")


def _check(name, a_cpu, b_cpu):
    a, b = a_cpu.to(run_device()), b_cpu.to(run_device())
    expected = getattr(torch, name.lower())(a_cpu, b_cpu)
    result = getattr(elementwise, name + "FwdOp")()(a, b)
    assert result.dtype == torch.bool
    assert result.shape == expected.shape
    assert torch.equal(result.cpu(), expected)
    assert torch.equal(a.cpu().view(torch.uint8), a_cpu.view(torch.uint8))
    assert torch.equal(b.cpu().view(torch.uint8), b_cpu.view(torch.uint8))
    assert result.untyped_storage().data_ptr() not in {
        a.untyped_storage().data_ptr(), b.untyped_storage().data_ptr()
    }


@pytest.mark.smoke
@pytest.mark.parametrize("name", _NAMES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_comparison_special_values_across_vector_blocks(name, dtype):
    tiny = torch.finfo(dtype).tiny
    a = torch.tensor([
        float("nan"), float("nan"), 1., 1., 2., float("inf"), float("-inf"),
        0., -0., tiny, -tiny, tiny / 2, -tiny / 2,
    ], dtype=dtype).repeat(384)[:4096]
    b = torch.tensor([
        float("nan"), 1., float("nan"), 1., 3., float("inf"), 0.,
        -0., 0., 0., 0., 0., 0.,
    ], dtype=dtype).repeat(384)[:4096]
    _check(name, a, b)


@pytest.mark.smoke
@pytest.mark.parametrize("name", _NAMES)
def test_comparison_int64_extrema(name):
    limits = torch.iinfo(torch.int64)
    a = torch.tensor([
        limits.min, limits.min + 1, -1, 0, 1, limits.max - 1, limits.max,
    ], dtype=torch.int64).repeat(600)[:4096]
    _check(name, a, a.roll(1))


@pytest.mark.smoke
@pytest.mark.parametrize("name", _NAMES)
def test_comparison_noncanonical_boolean_storage(name):
    a = torch.tensor([0, 1, 2, 127, 128, 255], dtype=torch.uint8).repeat(700)[:4096]
    b = torch.tensor([0, 255, 128, 0, 1, 2], dtype=torch.uint8).repeat(700)[:4096]
    _check(name, a.view(torch.bool), b.view(torch.bool))


@pytest.mark.smoke
@pytest.mark.parametrize("dtype,n", [
    (torch.int64, 8192), (torch.int64, 8256),
    (torch.float16, 24576), (torch.float16, 32704),
    (torch.float32, 12288), (torch.float32, 16320),
])
def test_comparison_full_tiles_and_large_tails(dtype, n):
    a = torch.arange(n).remainder(97).to(dtype)
    b = a.roll(1)
    b[::3] = a[::3]
    _check("Eq", a, b)


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_comparison_channel_exceeding_shared_memory(dtype):
    a = torch.arange(2 * 8 * 8192).remainder(97).to(dtype).reshape(2, 8, 8192)
    b = torch.arange(8).to(dtype).reshape(8, 1)
    _check("Eq", a, b)
