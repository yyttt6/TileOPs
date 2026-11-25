"""Tests for Rotary Position Embedding (RoPE) ops — 5 variants x 2 layouts.

Variants:
- neox: GPT-NeoX interleaved rotation (ref: GPT-NeoX / HuggingFace transformers)
- non_neox: original RoFormer adjacent-pair rotation (ref: RoFormer paper)
- rope_llama31: Llama 3.1 with frequency scaling (ref: Meta Llama 3.1)
- yarn_rope: YaRN with attention-factor scaling (ref: YaRN paper)
- longrope: LongRoPE with per-dimension rescale factors (ref: LongRoPE paper)

Each variant supports 1D layout (seq_len, head_dim) and
2D layout (batch, seq_len, num_heads, head_dim).

The op computes cos/sin internally from variant parameters; tests call
``op(x)`` directly and compare against a pure-PyTorch reference that
independently computes the same frequency tables.
"""

import math

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase

# Test fixtures
from tileops.rope import RoPECosSinFwdOp
from workloads.device import run_device
from workloads.numerics import compare_outputs
from workloads.rope import (
    RoPECase,
    RoPECosSinCall,
    RoPECosSinViewCall,
    ref_rope_neox_position_ids,
    rope_frequency_tables,
    rope_verification,
)
from workloads.workload_base import manifest_call


class RoPETest(RoPECase, TestBase):
    pass


class RoPEBasicFixture(FixtureBase):
    """Basic RoPE fixture: shapes x dtypes."""

    PARAMS = [
        (
            "batch, seq_len, num_heads, head_dim, dtype",
            [
                pytest.param(
                    2,
                    128,
                    8,
                    64,
                    torch.float16,
                    marks=[pytest.mark.smoke, pytest.mark.packaging(family="rope")],
                ),
                pytest.param(2, 128, 8, 64, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(2, 128, 8, 64, torch.float32, marks=pytest.mark.smoke),
                pytest.param(1, 256, 4, 128, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


class RoPEEdgeFixture(FixtureBase):
    """Edge cases: one position, a small head_dim, and a half-row of 12 float16 elements.

    12 is not a whole number of 8-element vectors, so 2D neox takes its per-pair path.
    """

    PARAMS = [
        (
            "batch, seq_len, num_heads, head_dim, dtype",
            [
                pytest.param(1, 1, 1, 16, torch.float32, marks=pytest.mark.smoke),
                pytest.param(1, 1, 1, 16, torch.float16, marks=pytest.mark.smoke),
                pytest.param(1, 4, 2, 24, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 512, 8, 64, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


# Base-frequency RoPE tests, both rotation conventions


@pytest.mark.parametrize("rope_layout, variant", [("neox", "neox"), ("interleaved", "non_neox")])
@RoPEBasicFixture
def test_rope_1d(
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    rope_layout: str,
    variant: str,
) -> None:
    from tileops._ops.rope import RoPEFwdOp

    test = RoPETest(variant, "1d", batch, seq_len, num_heads, head_dim, dtype)
    op = RoPEFwdOp(rope_layout=rope_layout, input_layout="1d")
    test.check(op, *test.gen_inputs())


@pytest.mark.parametrize("rope_layout, variant", [("neox", "neox"), ("interleaved", "non_neox")])
@RoPEBasicFixture
def test_rope_2d(
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    rope_layout: str,
    variant: str,
) -> None:
    from tileops._ops.rope import RoPEFwdOp

    test = RoPETest(variant, "2d", batch, seq_len, num_heads, head_dim, dtype)
    op = RoPEFwdOp(rope_layout=rope_layout, input_layout="2d")
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize(
    "rotary_dim, dtype",
    [
        (None, torch.float16),
        (32, torch.float16),
        (None, torch.bfloat16),
        (None, torch.float32),
    ],
)
def test_rope_neox_position_ids_thd(rotary_dim: int | None, dtype: torch.dtype) -> None:
    from tileops._ops.rope import RoPENeoxPositionIdsFwdOp

    num_tokens, num_heads, head_dim, max_position = 96, 8, 64, 512
    table_dim = head_dim if rotary_dim is None else rotary_dim
    x = torch.randn(num_tokens, num_heads, head_dim, device=run_device(), dtype=dtype)
    position_ids = (
        torch.arange(num_tokens, device=run_device(), dtype=torch.int32) * 3 + 17
    ) % max_position
    cos, sin = rope_frequency_tables(table_dim, max_position, dtype=dtype, device=run_device())
    ref = ref_rope_neox_position_ids(x, cos, sin, position_ids.long(), rotary_dim=rotary_dim)

    op = RoPENeoxPositionIdsFwdOp(
        max_position=max_position,
        rotary_dim=rotary_dim,
    )
    output = op(x, position_ids)
    compare_outputs(output, ref, rope_verification())


@pytest.mark.smoke
# 32768 tokens of eight pairs take more than one wave of 512 threads of two pairs.
@pytest.mark.parametrize("num_tokens", [2, 32768], ids=["one-wave", "multi-wave"])
def test_rope_neox_position_ids_validates_range(num_tokens: int) -> None:
    from tileops._ops.rope import RoPENeoxPositionIdsFwdOp

    op = RoPENeoxPositionIdsFwdOp(max_position=8)
    x = torch.randn(num_tokens, 1, 16, device=run_device(), dtype=torch.float16)
    position_ids = torch.zeros(num_tokens, device=run_device(), dtype=torch.int32)
    position_ids[-1] = 8
    with pytest.raises(ValueError, match="positions"):
        op(x, position_ids)


@pytest.mark.smoke
@pytest.mark.parametrize("num_tokens", [96, 2048], ids=["one-wave", "multi-wave"])
def test_rope_neox_position_ids_input_off_a_vector_boundary(num_tokens: int) -> None:
    """A contiguous input starting one element past a 16-byte boundary matches torch."""
    from tileops._ops.rope import RoPENeoxPositionIdsFwdOp

    shape, max_position = (num_tokens, 8, 64), 512
    x = torch.randn(math.prod(shape) + 1, device=run_device(), dtype=torch.float16)[1:]
    x = x.view(shape)
    position_ids = torch.randint(0, max_position, (num_tokens,), device=run_device())
    position_ids = position_ids.to(torch.int32)
    cos, sin = rope_frequency_tables(64, max_position, dtype=x.dtype, device=run_device())
    ref = ref_rope_neox_position_ids(x, cos, sin, position_ids.long())
    op = RoPENeoxPositionIdsFwdOp(max_position=max_position)
    compare_outputs(op(x, position_ids), ref, rope_verification())


@pytest.mark.smoke
@pytest.mark.parametrize("rope_layout, variant", [("neox", "neox"), ("interleaved", "non_neox")])
def test_rope_input_off_a_vector_boundary(rope_layout: str, variant: str) -> None:
    """A contiguous input starting one element past a 16-byte boundary matches the reference."""
    from tileops._ops.rope import RoPEFwdOp

    test = RoPETest(variant, "2d", 2, 64, 4, 128, torch.float16)
    (aligned,) = test.gen_inputs()
    x = torch.empty(aligned.numel() + 1, device=run_device(), dtype=aligned.dtype)[1:]
    x = x.view(aligned.shape).copy_(aligned)
    test.check(RoPEFwdOp(rope_layout=rope_layout, input_layout="2d"), x)


@pytest.mark.smoke
def test_rope_longrope_rejects_a_zero_rescale_factor() -> None:
    from tileops._ops.rope import LongRoPEFwdOp

    rescale = torch.tensor([1.0, 0.0, 2.0, 1.5], device=run_device())
    with pytest.raises(ValueError, match="rescale_factors"):
        LongRoPEFwdOp(rescale_factors=rescale)


@pytest.mark.smoke
def test_rope_longrope_rescale_factors_written_in_inference_mode_recompute_its_tables() -> None:
    """A write in place to a construction tensor is seen by the next call, for an ordinary
    tensor through its version and for an inference tensor, which has none; so is a write
    back to its first values."""
    import contextlib

    from tileops._ops.rope import LongRoPEFwdOp

    for mode in (contextlib.nullcontext, torch.inference_mode):
        with mode():
            rescale = torch.ones(64, device=run_device())
            extra = {"rescale_factors": rescale}
            test = RoPETest("longrope", "1d", 1, 8, 1, 128, torch.float16, extra)
            (x,) = test.gen_inputs()
            op = LongRoPEFwdOp(rescale_factors=rescale)
            op(x)
            for value in (2.0, 1.0):
                rescale.fill_(value)
                test.check(op, x)


@pytest.mark.smoke
def test_rope_neox_position_ids_none_rotary_dim_reinfers_head_dim() -> None:
    from tileops._ops.rope import RoPENeoxPositionIdsFwdOp

    max_position = 64
    position_ids = torch.arange(8, device=run_device(), dtype=torch.int32)
    op = RoPENeoxPositionIdsFwdOp(max_position=max_position, rotary_dim=None)

    x1 = torch.randn(8, 2, 16, device=run_device(), dtype=torch.float16)
    cos1, sin1 = rope_frequency_tables(16, max_position, dtype=x1.dtype, device=run_device())
    ref1 = ref_rope_neox_position_ids(x1, cos1, sin1, position_ids.long(), rotary_dim=None)
    compare_outputs(op(x1, position_ids), ref1, rope_verification())

    x2 = torch.randn(8, 2, 32, device=run_device(), dtype=torch.float16)
    cos2, sin2 = rope_frequency_tables(32, max_position, dtype=x2.dtype, device=run_device())
    ref2 = ref_rope_neox_position_ids(x2, cos2, sin2, position_ids.long(), rotary_dim=None)
    compare_outputs(op(x2, position_ids), ref2, rope_verification())


# Non-neox (RoFormer) RoPE tests


# Llama 3.1 RoPE tests


@RoPEBasicFixture
def test_rope_llama31_1d(
    batch: int, seq_len: int, num_heads: int, head_dim: int, dtype: torch.dtype
) -> None:
    from tileops._ops.rope import RoPELlama31FwdOp

    extra = {
        "scaling_factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position": 8192,
    }
    test = RoPETest(
        "rope_llama31", "1d", batch, seq_len, num_heads, head_dim, dtype, extra_kwargs=extra
    )
    op = RoPELlama31FwdOp(input_layout="1d", **extra)
    test.check(op, *test.gen_inputs())


@RoPEBasicFixture
def test_rope_llama31_2d(
    batch: int, seq_len: int, num_heads: int, head_dim: int, dtype: torch.dtype
) -> None:
    from tileops._ops.rope import RoPELlama31FwdOp

    extra = {
        "scaling_factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position": 8192,
    }
    test = RoPETest(
        "rope_llama31", "2d", batch, seq_len, num_heads, head_dim, dtype, extra_kwargs=extra
    )
    op = RoPELlama31FwdOp(input_layout="2d", **extra)
    test.check(op, *test.gen_inputs())


# YaRN RoPE tests


@RoPEBasicFixture
def test_yarn_1d(
    batch: int, seq_len: int, num_heads: int, head_dim: int, dtype: torch.dtype
) -> None:
    from tileops._ops.rope import YaRNFwdOp

    extra = {
        "scaling_factor": 16.0,
        "original_max_position": 4096,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "attn_factor": 1.0,
    }
    test = RoPETest(
        "yarn_rope", "1d", batch, seq_len, num_heads, head_dim, dtype, extra_kwargs=extra
    )
    op = YaRNFwdOp(input_layout="1d", **extra)
    test.check(op, *test.gen_inputs())


@RoPEBasicFixture
def test_yarn_2d(
    batch: int, seq_len: int, num_heads: int, head_dim: int, dtype: torch.dtype
) -> None:
    from tileops._ops.rope import YaRNFwdOp

    extra = {
        "scaling_factor": 16.0,
        "original_max_position": 4096,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "attn_factor": 1.0,
    }
    test = RoPETest(
        "yarn_rope", "2d", batch, seq_len, num_heads, head_dim, dtype, extra_kwargs=extra
    )
    op = YaRNFwdOp(input_layout="2d", **extra)
    test.check(op, *test.gen_inputs())


# LongRoPE tests


@RoPEBasicFixture
def test_rope_longrope_1d(
    batch: int, seq_len: int, num_heads: int, head_dim: int, dtype: torch.dtype
) -> None:
    from tileops._ops.rope import LongRoPEFwdOp

    half = head_dim // 2
    rescale = torch.linspace(1.0, 2.0, half, device=run_device())
    max_pos = 16384
    orig_max_pos = 4096
    extra = {
        "rescale_factors": rescale,
        "max_position": max_pos,
        "original_max_position": orig_max_pos,
    }
    test = RoPETest(
        "longrope", "1d", batch, seq_len, num_heads, head_dim, dtype, extra_kwargs=extra
    )
    op = LongRoPEFwdOp(
        input_layout="1d",
        rescale_factors=rescale,
        max_position=max_pos,
        original_max_position=orig_max_pos,
    )
    test.check(op, *test.gen_inputs())


@RoPEBasicFixture
def test_rope_longrope_2d(
    batch: int, seq_len: int, num_heads: int, head_dim: int, dtype: torch.dtype
) -> None:
    from tileops._ops.rope import LongRoPEFwdOp

    half = head_dim // 2
    rescale = torch.linspace(1.0, 2.0, half, device=run_device())
    max_pos = 16384
    orig_max_pos = 4096
    extra = {
        "rescale_factors": rescale,
        "max_position": max_pos,
        "original_max_position": orig_max_pos,
    }
    test = RoPETest(
        "longrope", "2d", batch, seq_len, num_heads, head_dim, dtype, extra_kwargs=extra
    )
    op = LongRoPEFwdOp(
        input_layout="2d",
        rescale_factors=rescale,
        max_position=max_pos,
        original_max_position=orig_max_pos,
    )
    test.check(op, *test.gen_inputs())


# Edge case tests


@pytest.mark.parametrize("rope_layout, variant", [("neox", "neox"), ("interleaved", "non_neox")])
@RoPEEdgeFixture
def test_rope_edge(
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    rope_layout: str,
    variant: str,
) -> None:
    """Edge cases: seq_len=1 and longer sequences."""
    from tileops._ops.rope import RoPEFwdOp

    test = RoPETest(variant, "2d", batch, seq_len, num_heads, head_dim, dtype)
    op = RoPEFwdOp(rope_layout=rope_layout, input_layout="2d")
    test.check(op, *test.gen_inputs())


# Layout and dtype regression tests


@pytest.mark.smoke
def test_rope_noncontiguous_1d_works() -> None:
    """A non-contiguous 1D view must produce correct results after contiguity normalization."""
    from tileops._ops.rope import RoPEFwdOp

    seq_len, head_dim = 4, 8
    op = RoPEFwdOp(input_layout="1d")

    # Create a non-contiguous view: transpose makes it non-contiguous
    base = torch.randn(head_dim, seq_len, device=run_device(), dtype=torch.float32)
    x_nc = base.t()  # shape (seq_len, head_dim), non-contiguous
    assert not x_nc.is_contiguous()

    # Reference with contiguous copy
    x_c = x_nc.contiguous()
    out_nc = op(x_nc)
    out_c = op(x_c)
    assert torch.equal(out_nc, out_c)


@pytest.mark.smoke
@pytest.mark.parametrize("invalid_position", [-1, 8])
def test_rope_position_ids_multihead_range_recovery(invalid_position: int) -> None:
    """Multi-head rotation rejects either range boundary and accepts the next valid call."""
    from tileops._ops.rope import RoPENeoxPositionIdsFwdOp

    max_position = 8
    x = torch.randn(2, 32, 128, device=run_device(), dtype=torch.bfloat16)
    positions = torch.tensor([0, invalid_position], device=run_device(), dtype=torch.int64)
    op = RoPENeoxPositionIdsFwdOp(max_position=max_position)
    with pytest.raises(ValueError, match="position_ids"):
        op(x, positions)

    positions[-1] = max_position - 1
    cos, sin = rope_frequency_tables(128, max_position, dtype=x.dtype, device=run_device())
    ref = ref_rope_neox_position_ids(x, cos, sin, positions)
    compare_outputs(op(x, positions), ref, rope_verification())


class RoPECosSinTest(RoPECosSinCall, TestBase):
    pass


_SMOKE, _FULL = pytest.mark.smoke, pytest.mark.full
_F32_TABLES = {"C": "float32", "Cs": "float32"}


def _dense(B, S, H, D, Rh, P=None, **row):
    return {"B": B, "S": S, "H": H, "D": D, "Rh": Rh, "P": P or S, **row}


def _packed(seq_lens, H, D, Rh, P, **row):
    some = row.pop("some", [])
    return {
        "Tt": sum(seq_lens),
        "seq_lens": seq_lens,
        "max_seqlen": max(seq_lens),
        "H": H,
        "D": D,
        "Rh": Rh,
        "P": P,
        "some": ["cu_seqlens", *some],
        **row,
    }


_COS_SIN_CASES = (
    # Qwen3.6 vision: a full 72-wide NeoX head, float32 tables.
    pytest.param(
        _dense(1, 37, 4, 72, 36), {"T": "bfloat16", **_F32_TABLES}, marks=_SMOKE, id="bf16"
    ),
    pytest.param(
        _dense(2, 19, 3, 128, 64, P=32),
        {"T": "float16", "C": "float16", "Cs": "float16"},
        marks=_SMOKE,
        id="fp16",
    ),
    pytest.param(
        _dense(2, 19, 3, 64, 32), {"T": "float32", **_F32_TABLES}, marks=_SMOKE, id="fp32"
    ),
    # Each table converts from its own dtype.
    pytest.param(
        _dense(1, 21, 2, 64, 32),
        {"T": "bfloat16", "C": "float16", "Cs": "float32"},
        marks=_FULL,
        id="mixed-tables",
    ),
    pytest.param(
        _dense(2, 17, 3, 128, 32, P=40, interleaved=True, seqlen_offsets=9),
        {"T": "bfloat16", **_F32_TABLES},
        marks=_FULL,
        id="interleaved-partial-offset",
    ),
    # Decode: one token per sequence, each at its own offset.
    pytest.param(
        _dense(6, 1, 4, 128, 64, P=64, some=["seqlen_offsets"]),
        {"T": "bfloat16", **_F32_TABLES, "O": "int32"},
        marks=_FULL,
        id="decode-offsets",
    ),
    # A token's 256 units fill two whole blocks: the grid holds exactly the units.
    pytest.param(
        _dense(3, 1, 32, 128, 64, P=16, some=["seqlen_offsets"]),
        {"T": "bfloat16", **_F32_TABLES, "O": "int64"},
        marks=_FULL,
        id="decode-whole-blocks",
    ),
    # An empty sequence among packed ones, each sequence at its own offset.
    pytest.param(
        _packed([5, 0, 9], 3, 64, 32, P=24, some=["seqlen_offsets"]),
        {"T": "bfloat16", "C": "bfloat16", "Cs": "bfloat16", "I": "int32", "O": "int64"},
        marks=_FULL,
        id="packed-offsets",
    ),
    # An odd head leaves no column offset on a vector boundary: scalar accesses.
    pytest.param(
        _dense(1, 9, 3, 65, 16), {"T": "float16", **_F32_TABLES}, marks=_FULL, id="odd-head"
    ),
    pytest.param(
        _dense(1, 9, 3, 65, 16, interleaved=True),
        {"T": "float16", **_F32_TABLES},
        marks=_FULL,
        id="odd-head-interleaved",
    ),
    pytest.param(
        _packed([3, 8], 2, 72, 36, P=16, seqlen_offsets=4, interleaved=True),
        {"T": "float16", **_F32_TABLES, "I": "int64"},
        marks=_FULL,
        id="packed-int-offset",
    ),
    # In place with unrotated columns: the head stride exceeds the rotated width.
    pytest.param(
        _dense(1, 37, 4, 136, 16, inplace=True),
        {"T": "bfloat16", **_F32_TABLES},
        marks=_FULL,
        id="inplace-unrotated-columns",
    ),
    # Calls whose work fills more than half the device's resident threads give each thread
    # several heads, an odd head count leaving the last thread of a token one.
    pytest.param(
        _dense(1, 4096, 5, 128, 32),
        {"T": "bfloat16", **_F32_TABLES},
        marks=_FULL,
        id="many-units-partial",
    ),
    # Regression: more dense sequences than CUDA's grid has rows, each at its own offset.
    pytest.param(
        _dense(65537, 1, 1, 2, 1, P=4, some=["seqlen_offsets"]),
        {"T": "bfloat16", **_F32_TABLES, "O": "int64"},
        marks=_FULL,
        id="more-sequences-than-grid-rows",
    ),
)


@pytest.mark.parametrize("row, dtype_case", _COS_SIN_CASES)
def test_rope_cos_sin_matches_the_reference(row, dtype_case) -> None:
    test = RoPECosSinTest(manifest_call("RoPECosSinFwdOp", dtype_case, **row))
    test.check(RoPECosSinFwdOp(**test.arguments()), *test.gen_inputs())


_VIEWS = pytest.mark.parametrize(
    "interleaved, columns",
    [
        pytest.param(False, slice(64, 136), marks=_SMOKE, id="neox-tail"),
        pytest.param(True, slice(64, 136), marks=_FULL, id="interleaved-tail"),
        pytest.param(False, slice(8, 136, 2), marks=_FULL, id="neox-every-2nd"),
        pytest.param(True, slice(8, 136, 2), marks=_FULL, id="interleaved-every-2nd"),
    ],
)


class RoPECosSinViewTest(RoPECosSinViewCall, TestBase):
    pass


def _strided_view_call(interleaved, inplace, columns, width):
    """The op and its inputs on a column view of heads *width* columns wide."""
    row = _dense(1, 37, 4, len(range(width)[columns]), 16, interleaved=interleaved, inplace=inplace)
    call = manifest_call("RoPECosSinFwdOp", {"T": "bfloat16", **_F32_TABLES}, **row)
    test = RoPECosSinViewTest(call, width, columns)
    return test, RoPECosSinFwdOp(**test.arguments()), test.gen_inputs()


@_VIEWS
def test_rope_cos_sin_rotates_a_strided_view_in_place(interleaved, columns) -> None:
    """A column view of wider heads rotated in place, as DeepSeek-V4-Flash rotates the
    trailing columns of each head: x is returned, and only the view's columns change."""
    test, op, inputs = _strided_view_call(interleaved, True, columns, 136)
    test.check(op, *inputs)
    x, wide = inputs[0], inputs[0]._base
    before = wide.clone()
    assert op(*inputs) is x
    kept = torch.ones_like(wide, dtype=torch.bool)
    kept[..., columns] = False
    assert torch.equal(wide[kept], before[kept])


@pytest.mark.parametrize(
    "columns",
    [
        pytest.param(slice(64, 136), marks=_SMOKE, id="neox-tail"),
        pytest.param(slice(8, 136, 2), marks=_FULL, id="neox-every-2nd"),
    ],
)
def test_rope_cos_sin_rotates_a_strided_view_into_a_new_tensor(columns) -> None:
    """A column view of wider heads read out of place: the wider heads are left as they were."""
    test, op, inputs = _strided_view_call(False, False, columns, 136)
    test.check(op, *inputs)
    wide = inputs[0]._base
    before = wide.clone()
    op(*inputs)
    assert torch.equal(wide, before)


_OFFSETS = _dense(2, 19, 3, 128, 64, P=32, some=["seqlen_offsets"])


@pytest.mark.parametrize(
    "row, dtypes, offsets",
    [
        pytest.param(_OFFSETS, {"O": "int64"}, [20, -3], marks=_SMOKE, id="int64"),
        # An int32 offset at its extremes, widened before it is added to a position, reads
        # the nearest row.
        pytest.param(
            _OFFSETS,
            {"O": "int32"},
            [2**31 - 1, -(2**31)],
            marks=_SMOKE,
            id="int32-extremes",
        ),
        pytest.param(
            _packed([5, 0, 9], 3, 64, 32, P=24, some=["seqlen_offsets"]),
            {"I": "int32", "O": "int32"},
            [2**31 - 1, 0, -(2**31)],
            marks=_FULL,
            id="packed-int32-extremes",
        ),
    ],
)
def test_rope_cos_sin_unchecked_row_outside_the_tables_reads_its_nearest_row(
    row, dtypes, offsets
) -> None:
    call = manifest_call("RoPECosSinFwdOp", {"T": "float16", **_F32_TABLES, **dtypes}, **row)
    test = RoPECosSinTest(call)
    x, cos, sin, cu_seqlens, seqlen_offsets = test.gen_inputs()
    seqlen_offsets.copy_(torch.tensor(offsets))
    test.check(RoPECosSinFwdOp(**test.arguments()), x, cos, sin, cu_seqlens, seqlen_offsets)


@pytest.mark.parametrize(
    "row",
    [
        pytest.param(_OFFSETS, marks=_SMOKE, id="one-head"),
        # Units past half the resident threads: two heads a thread.
        pytest.param(
            _dense(2, 2048, 8, 128, 64, P=4096, some=["seqlen_offsets"]),
            marks=_FULL,
            id="two-heads",
        ),
    ],
)
def test_rope_cos_sin_int64_offset_at_its_maximum_reads_the_last_row(row) -> None:
    """Offset 2**63 - 1 plus a position past 0 would wrap below row 0 before its clamp. The op
    on the int64 extremes reads the rows the reference reads on P - 1 and -S, which reach the
    same rows without the wrapping sum the reference makes on the extremes."""
    call = manifest_call("RoPECosSinFwdOp", {"T": "float16", **_F32_TABLES, "O": "int64"}, **row)
    test = RoPECosSinTest(call)
    x, cos, sin, cu_seqlens, _ = test.gen_inputs()
    op = RoPECosSinFwdOp(**test.arguments())
    extreme = torch.tensor([2**63 - 1, -(2**63)], device=x.device)
    nearest = torch.tensor([cos.shape[0] - 1, -x.shape[1]], device=x.device)

    def on_extremes(x, cos, sin, cu_seqlens=None, seqlen_offsets=0):
        return op(x, cos, sin, cu_seqlens, extreme)

    test.check(op, x, cos, sin, cu_seqlens, nearest, runs=on_extremes)


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "extreme",
    [
        pytest.param(2**63 - 1, marks=_SMOKE, id="max"),
        pytest.param(-(2**63), marks=_FULL, id="min"),
    ],
)
def test_rope_cos_sin_kernel_int_offset_at_its_extremes_reads_the_end_rows(extreme) -> None:
    """The op takes an int offset in [0, P - S]; the kernel clamps any other to the rows it
    reaches, so an int64 extreme reads the rows P - 1 or -S read, without a wrapping sum."""
    row = _dense(1, 37, 4, 72, 36)
    call = manifest_call("RoPECosSinFwdOp", {"T": "float16", **_F32_TABLES}, **row)
    test = RoPECosSinTest(call)
    x, cos, sin, cu_seqlens, seqlen_offsets = test.gen_inputs()
    op = RoPECosSinFwdOp(**test.arguments())
    op(x, cos, sin, cu_seqlens, seqlen_offsets)
    (kernel,) = op.iter_kernels()
    nearest = cos.shape[0] - 1 if extreme > 0 else -x.shape[1]
    want = kernel.forward(x, cos, sin, cu_seqlens, nearest)
    got = kernel.forward(x, cos, sin, cu_seqlens, extreme)
    torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.smoke
def test_rope_cos_sin_in_place_leaves_the_memory_past_x_untouched() -> None:
    """The 1332 units of 37 tokens fill 52 threads of the last block; its other threads touch
    nothing, here the part of the buffer past x."""
    row = _dense(1, 37, 4, 72, 36, inplace=True)
    call = manifest_call("RoPECosSinFwdOp", {"T": "bfloat16", **_F32_TABLES}, **row)
    test = RoPECosSinTest(call)
    x, cos, sin, cu_seqlens, seqlen_offsets = test.gen_inputs()
    op = RoPECosSinFwdOp(**test.arguments())
    buffer = torch.randn(x.numel() + 128 * 4 * 72, dtype=x.dtype, device=x.device)
    head = buffer[: x.numel()].view(x.shape)
    rest = buffer[x.numel() :].clone()
    assert op(head, cos, sin, cu_seqlens, seqlen_offsets) is head
    assert torch.equal(buffer[x.numel() :], rest)


@pytest.mark.parametrize(
    "apart",
    [
        pytest.param(19 * 3 * 256, marks=_SMOKE, id="adjacent"),
        # Element offsets past int32: the program takes 64-bit indexes.
        pytest.param(2**31, marks=_FULL, id="past-int32"),
    ],
)
def test_rope_cos_sin_rotates_sequences_far_apart(apart) -> None:
    """Two sequences *apart* elements apart, every second column of 256-column heads:
    one-column accesses."""
    row = _dense(2, 19, 3, 128, 64, P=32)
    call = manifest_call(
        "RoPECosSinFwdOp", {"T": "float16", "C": "float16", "Cs": "float16"}, **row
    )
    test = RoPECosSinTest(call)
    x, cos, sin, cu_seqlens, seqlen_offsets = test.gen_inputs()
    storage = torch.zeros(apart + 19 * 3 * 256, dtype=x.dtype, device=x.device)
    view = storage.as_strided(x.shape, (apart, 3 * 256, 256, 2))
    view.copy_(x)
    test.check(RoPECosSinFwdOp(), view, cos, sin, cu_seqlens, seqlen_offsets)
