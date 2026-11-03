"""Pool references preserve the public output contract across backends."""

import pytest
import torch
import torch.nn.functional as F

from workloads.pool import MaxPoolWorkload


@pytest.mark.smoke
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("return_indices", [False, True])
def test_max_pool_reference_normalizes_backend_indices(monkeypatch, ndim, return_indices):
    """An int32 backend index must not change the reference's int64 contract."""
    workload = MaxPoolWorkload(
        ndim=ndim,
        kernel_size=(2,) * ndim,
        stride=(2,) * ndim,
        padding=(0,) * ndim,
        dilation=(1,) * ndim,
        ceil_mode=False,
        dtype=torch.float32,
        return_indices=return_indices,
    )
    inputs = workload.gen_inputs(1, 1, *((4,) * ndim))
    expected = workload.ref_program(*inputs)
    pool = getattr(F, f"max_pool{ndim}d")

    def backend_pool(*args, **kwargs):
        result = pool(*args, **kwargs)
        if kwargs.get("return_indices"):
            values, indices = result
            return values, indices.to(torch.int32)
        return result

    monkeypatch.setattr(F, f"max_pool{ndim}d", backend_pool)
    actual = workload.ref_program(*inputs)
    if return_indices:
        assert actual[1].dtype == torch.int64
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])
    else:
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(actual, expected)
