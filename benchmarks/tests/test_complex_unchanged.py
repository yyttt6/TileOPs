"""The NPU complex input guard retains exact, full-complex comparisons."""

from types import SimpleNamespace

import pytest
import torch

from workloads.numerics import _unchanged


class ComplexNpuView:
    def __init__(self, tensor):
        self.tensor = tensor
        self.dtype = tensor.dtype
        self.device = SimpleNamespace(type="npu", index=0)

    def cpu(self):
        return self.tensor


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_complex_input_guard_detects_imaginary_mutation(dtype):
    original = torch.tensor([1 + 2j, complex(float("nan"), 3)], dtype=dtype)
    current = original.clone()
    assert _unchanged([ComplexNpuView(current)], [ComplexNpuView(original)])
    current[0] = 1 + 4j
    assert not _unchanged([ComplexNpuView(current)], [ComplexNpuView(original)])
