"""Contract tests for ``Op.compute_roof`` (docs/design/roofline.md §1.4)."""

import pytest
import torch

from tileops.perf.profile import cube_roof

pytestmark = pytest.mark.smoke


class TestCubeRoof:
    def test_maps_torch_dtypes_to_profile_keys(self):
        assert cube_roof(torch.float16) == "cube.fp16"
        assert cube_roof(torch.bfloat16) == "cube.bf16"
        assert cube_roof(torch.float32) == "cube.fp32"

    def test_unbound_or_unknown_dtype_raises(self):
        with pytest.raises(ValueError, match="no Cube roof"):
            cube_roof(None)
        with pytest.raises(ValueError, match="no Cube roof"):
            cube_roof(torch.int32)

    def test_fp8_has_no_cube_section(self):
        """910B1's Cube unit stops at 16 bits, so an fp8 contraction has no roof of its own.

        The two ops that ask for one price at ``cube.fp16`` and say why; the mapping
        must not invent a key the profile cannot resolve.
        """
        with pytest.raises(ValueError, match="no Cube roof"):
            cube_roof(torch.float8_e4m3fn)
        with pytest.raises(ValueError, match="no Cube roof"):
            cube_roof(torch.float8_e5m2)


class TestComputeRoofContract:
    # __new__ bypasses kernel construction so the smoke needs no card;
    # only the state each override reads is bound.

    def test_base_default_is_vector_fp32(self):
        from tileops.ops.elementwise.arithmetic import AddFwdOp

        op = AddFwdOp.__new__(AddFwdOp)
        assert op.compute_roof() == "vector.fp32"

    def test_matmul_op_prices_on_the_bound_dtype(self):
        from tileops.ops.gemm.gemm import GemmFwdOp

        op = GemmFwdOp.__new__(GemmFwdOp)
        op.dtype = torch.bfloat16
        assert op.compute_roof() == "cube.bf16"

    def test_gqa_prefill_prices_the_call_that_ran(self):
        from tileops.ops.attention.gqa import GroupedQueryAttentionPrefillFwdOp

        op = GroupedQueryAttentionPrefillFwdOp.__new__(GroupedQueryAttentionPrefillFwdOp)
        op.dtype = torch.float16
        # An fp8 contraction runs on the same 16-bit Cube path, so it is priced there.
        op.backend = "fp8"
        assert op.compute_roof() == "cube.fp16"
        op.backend = "dense"
        assert op.compute_roof() == "cube.fp16"
        # backend="auto" dispatches by the tensors the call passed, so the
        # recorded call dtype outranks the constructed one.
        op.backend = "auto"
        op._roofline_kwargs = {"dtype": torch.bfloat16}
        assert op.compute_roof() == "cube.bf16"
