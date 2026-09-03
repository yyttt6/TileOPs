"""Static contract tests for the Ascend MaxPool2dFwdOp builder."""

from __future__ import annotations

import pytest
import torch

from tileops_ascend.kernels.pool import (
    _normalize_params,
    _output_dim,
    build_max_pool2d_kernel,
)


@pytest.mark.parametrize(
    ("size", "kernel", "stride", "padding", "dilation", "ceil_mode", "expected"),
    [
        (112, 3, 2, 1, 1, False, 56),
        (56, 2, 2, 0, 1, False, 28),
        (55, 3, 2, 0, 1, True, 27),
        (17, 3, 2, 1, 2, True, 8),
    ],
)
def test_output_dim_matches_manifest_formula(
    size, kernel, stride, padding, dilation, ceil_mode, expected
):
    assert _output_dim(size, kernel, stride, padding, dilation, ceil_mode) == expected


def test_params_accept_int_and_pairs():
    assert _normalize_params(3, None, 1, 1, False) == (
        3,
        3,
        3,
        3,
        1,
        1,
        1,
        1,
        False,
    )
    assert _normalize_params((3, 5), (2, 3), (1, 2), (2, 1), True) == (
        3,
        5,
        2,
        3,
        1,
        2,
        2,
        1,
        True,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kernel_size": 0},
        {"kernel_size": 3, "stride": 0},
        {"kernel_size": 3, "padding": 2},
        {"kernel_size": 3, "dilation": 0},
        {"kernel_size": 3, "ceil_mode": 1},
    ],
)
def test_invalid_params_are_rejected(kwargs):
    with pytest.raises((TypeError, ValueError)):
        _normalize_params(
            kwargs.get("kernel_size", 3),
            kwargs.get("stride"),
            kwargs.get("padding", 0),
            kwargs.get("dilation", 1),
            kwargs.get("ceil_mode", False),
        )


def test_declared_dtypes_only():
    with pytest.raises(TypeError, match="supports float16"):
        build_max_pool2d_kernel(
            (1, 1, 8, 8),
            torch.int32,
            kernel_size=2,
            stride=2,
        )


def test_row_reuse_and_generic_dispatch(monkeypatch):
    marker = object()
    monkeypatch.setattr(
        "tileops_ascend.kernels.pool._compile_row_reuse", lambda *args: marker
    )
    monkeypatch.setattr(
        "tileops_ascend.kernels.pool._compile_generic", lambda *args: marker
    )

    row = build_max_pool2d_kernel(
        (1, 1, 32, 32), torch.float16, kernel_size=3, stride=2, padding=1
    )
    wide = build_max_pool2d_kernel(
        (1, 1, 8, 257), torch.float16, kernel_size=1, stride=1
    )
    assert row.path_kind == "row_reuse"
    assert row.output_shape == (1, 1, 16, 16)
    assert wide.path_kind == "generic"
    assert wide.output_shape == (1, 1, 8, 257)
