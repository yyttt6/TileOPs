from pathlib import Path

import pytest

from tileops_ascend.kernels.pool_indices import (
    _adaptive_output_size,
    _max_adaptive_extent,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, (7, 9)),
        (3, (3, 3)),
        ((None, 4), (7, 4)),
        ([5, None], (5, 9)),
    ],
)
def test_adaptive_output_size(value, expected):
    assert _adaptive_output_size(value, 7, 9) == expected


@pytest.mark.parametrize("value", [0, (3,), (2, 0), (2, "3"), True])
def test_adaptive_output_size_rejects_invalid(value):
    with pytest.raises((TypeError, ValueError)):
        _adaptive_output_size(value, 7, 9)


def test_adaptive_extent_covers_nondivisible_bins():
    assert _max_adaptive_extent(55, 7) == 9
    assert _max_adaptive_extent(57, 7) == 9


def test_shared_template_uses_explicit_dual_outputs_and_int64():
    source = (
        Path(__file__).parents[1] / "src/tileops_ascend/kernels/pool_indices.py"
    ).read_text()
    assert "out_idx=[]" in source
    assert 'indices_gm: T.Tensor((padded_total,), "int64")' in source
    assert "T.copy(\n                                best_indices," in source
    assert "torch.empty((padded_total,), dtype=torch.int64" in source
