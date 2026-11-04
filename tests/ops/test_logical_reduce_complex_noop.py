"""Ascend complex truth must include the imaginary component on bypass routes."""

import pytest
import torch

from tileops.ops.reduction import AllFwdOp
from workloads.device import run_device


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("mode", ["scalar", "noop"])
@pytest.mark.parametrize("case", ["zero", "imaginary", "tiny-imaginary", "tiny-real", "nan"])
def test_all_complex_bypass_truth(dtype, mode, case):
    if torch.device(run_device()).type != "npu":
        pytest.skip("Ascend complex conversion regression")
    tiny = 1e-40 if dtype == torch.complex64 else 1e-300
    value = {
        "zero": 0j,
        "imaginary": 1j,
        "tiny-imaginary": complex(0, tiny),
        "tiny-real": complex(tiny, 0),
        "nan": complex(float("nan"), 0),
    }[case]
    cpu = torch.tensor(value if mode == "scalar" else [value, 0, 1], dtype=dtype)
    x = cpu.to(run_device())
    before = x.reshape(-1).view(torch.int32).cpu().clone()
    output = AllFwdOp(dim=None if mode == "scalar" else [])(x)

    assert output.dtype == torch.bool
    assert output.shape == cpu.shape
    assert torch.equal(output.cpu(), cpu != 0)
    assert output.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
    assert torch.equal(x.reshape(-1).view(torch.int32).cpu(), before)
