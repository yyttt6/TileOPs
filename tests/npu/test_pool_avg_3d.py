"""Static contracts for shared fixed-window avg pooling and MaxPool3d."""

import pytest
import torch

from tileops.kernels.pool_avg import (
    _normalize_fixed_pool,
    _spatial_tuple,
    build_avg_pool_kernel,
    build_max_pool3d_kernel,
)


def test_spatial_tuple_accepts_manifest_forms():
    assert _spatial_tuple("kernel_size", 3, 3) == (3, 3, 3)
    assert _spatial_tuple("kernel_size", [2, 3], 2) == (2, 3)


@pytest.mark.parametrize("value", [True, (1,), (1, 2, 3, 4), (1, 2.0)])
def test_spatial_tuple_rejects_invalid_forms(value):
    with pytest.raises(TypeError):
        _spatial_tuple("kernel_size", value, 2)


def test_normalize_rejects_zero_divisor():
    with pytest.raises(ValueError, match="must not be zero"):
        _normalize_fixed_pool(2, 3, 2, 1, 1, False, divisor_override=0)


@pytest.mark.parametrize(
    ("ndim", "shape", "expected"),
    [
        (1, (2, 3, 17), (2, 3, 9)),
        (2, (2, 3, 17, 19), (2, 3, 9, 10)),
        (3, (2, 3, 9, 17, 19), (2, 3, 5, 9, 10)),
    ],
)
def test_avg_builder_shapes_and_shared_path(monkeypatch, ndim, shape, expected):
    marker = lambda x: x  # noqa: E731
    monkeypatch.setattr(
        "tileops.kernels.pool_avg._compile_fixed_pool", lambda *args: marker
    )
    kernel = build_avg_pool_kernel(
        shape,
        torch.float16,
        ndim=ndim,
        kernel_size=3,
        stride=2,
        padding=1,
        ceil_mode=True,
        count_include_pad=False,
    )
    assert kernel.output_shape == expected
    assert kernel.path_kind == f"avg{ndim}d_generic"


def test_max3d_builder_grid_stride_metadata(monkeypatch):
    marker = lambda x: x  # noqa: E731
    monkeypatch.setattr(
        "tileops.kernels.pool_avg._compile_fixed_pool", lambda *args: marker
    )
    kernel = build_max_pool3d_kernel(
        (1, 1, 258, 256, 128),
        torch.float16,
        kernel_size=1,
    )
    assert kernel.logical_blocks == 66_048
    assert kernel.launch_blocks == 65_535
    assert kernel.output_shape == (1, 1, 258, 256, 128)
