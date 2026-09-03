"""Static contracts for the Ascend Conv2d builder."""

from types import SimpleNamespace

import pytest
import torch

from tileops_ascend.families.convolution import _padding_pair, _pair, build_conv2d
from tileops_ascend.kernels.convolution import build_conv2d_kernel


def _spec(shape, dtype=torch.float16, device=torch.device("npu:0")):
    return SimpleNamespace(shape=shape, dtype=dtype, device=device)


def test_pair_and_string_padding_contract():
    assert _pair("stride", 2) == (2, 2)
    assert _pair("dilation", (2, 3)) == (2, 3)
    assert _padding_pair("valid", (2, 2), (3, 3), (1, 1)) == (0, 0)
    assert _padding_pair("same", (1, 1), (3, 5), (2, 1)) == (2, 2)


@pytest.mark.parametrize(
    "args",
    [
        ("stride", 0),
        ("stride", (1,)),
        ("padding", (1, -1)),
    ],
)
def test_invalid_metadata_is_rejected(args):
    name, value = args
    kwargs = {"stride": 1, "padding": 0, "dilation": 1, "groups": 1}
    kwargs[name] = value
    with pytest.raises((TypeError, ValueError)):
        build_conv2d(
            _spec((1, 4, 8, 8)),
            _spec((8, 4, 3, 3)),
            None,
            **kwargs,
        )


def test_group_shape_contract_is_rejected():
    with pytest.raises(ValueError, match=r"weight\.shape\[1\]"):
        build_conv2d(
            _spec((1, 8, 8, 8)),
            _spec((8, 3, 3, 3)),
            None,
            groups=2,
        )


def test_cube_and_direct_dispatch(monkeypatch):
    marker = object()
    monkeypatch.setattr(
        "tileops_ascend.kernels.convolution._compile_cube_conv2d",
        lambda *args: marker,
    )
    monkeypatch.setattr(
        "tileops_ascend.kernels.convolution._compile_direct_conv2d",
        lambda *args: marker,
    )

    cube = build_conv2d_kernel(
        (1, 16, 8, 9),
        (32, 16, 3, 3),
        torch.float16,
        stride=(2, 2),
        padding=(1, 1),
        dilation=(1, 1),
        groups=1,
        has_bias=False,
    )
    depthwise = build_conv2d_kernel(
        (1, 8, 8, 9),
        (8, 1, 3, 3),
        torch.float16,
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        groups=8,
        has_bias=False,
    )
    biased = build_conv2d_kernel(
        (1, 16, 8, 9),
        (32, 16, 3, 3),
        torch.float16,
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        groups=1,
        has_bias=True,
    )
    fp32 = build_conv2d_kernel(
        (1, 2, 5, 6),
        (3, 2, 3, 3),
        torch.float32,
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        groups=1,
        has_bias=False,
    )

    assert cube.path_kind == "implicit_im2col_cube"
    assert cube.output_shape == (1, 32, 4, 5)
    assert cube.logical_blocks == 2
    assert cube.launch_blocks == 2
    assert cube.grid_repeats == 1
    assert depthwise.path_kind == "direct_window"
    assert biased.path_kind == "direct_window"
    assert fp32.path_kind == "direct_window"


def test_large_depthwise_dispatch_uses_grid_stride(monkeypatch):
    monkeypatch.setattr(
        "tileops_ascend.kernels.convolution._compile_direct_conv2d",
        lambda *args: object(),
    )
    kernel = build_conv2d_kernel(
        (16, 256, 56, 56),
        (256, 1, 1, 1),
        torch.float16,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=256,
        has_bias=False,
    )

    assert kernel.path_kind == "direct_window"
    assert kernel.logical_blocks == 100352
    assert kernel.launch_blocks == 65535
    assert kernel.grid_repeats == 2
