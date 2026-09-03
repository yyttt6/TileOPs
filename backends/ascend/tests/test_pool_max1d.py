"""Static contract tests for the Ascend MaxPool1dFwdOp builder."""

import pytest
import torch

from tileops_ascend.kernels.pool_max1d import _single, build_max_pool1d_kernel


def test_single_accepts_manifest_forms():
    assert _single("kernel_size", 3) == 3
    assert _single("kernel_size", (3,)) == 3
    assert _single("kernel_size", [3]) == 3


@pytest.mark.parametrize("value", [True, 0.5, (), (1, 2)])
def test_single_rejects_non_manifest_forms(value):
    with pytest.raises(TypeError):
        _single("kernel_size", value)


def test_builder_output_shape_and_metadata(monkeypatch):
    marker = lambda x: x  # noqa: E731
    monkeypatch.setattr(
        "tileops_ascend.kernels.pool_max1d._compile_max_pool1d",
        lambda *args: marker,
    )
    kernel = build_max_pool1d_kernel(
        (3, 5, 4097), torch.float16,
        kernel_size=7, stride=3, padding=2, dilation=2, ceil_mode=True,
    )
    assert kernel.output_shape == (3, 5, 1364)
    assert kernel.path_kind == "max1d_generic"


def test_builder_rejects_invalid_params():
    with pytest.raises(ValueError, match="padding"):
        build_max_pool1d_kernel((1, 1, 16), torch.float16, kernel_size=3, padding=2)
