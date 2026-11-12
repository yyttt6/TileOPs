"""FP8 storage and untouched-slot controls, CPU only."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nightly_support"))
from cache_write_reference import reference

pytestmark = pytest.mark.smoke


def test_paged_fp8_scatter_keeps_skipped_and_untouched_rows():
    dtype = torch.float8_e4m3fn
    pages = torch.arange(16).reshape(2, 2, 1, 4).float().to(dtype)
    original = pages.view(torch.uint8).clone()
    k = torch.tensor([[[1.0, 2.0, 3.0, 4.0]], [[9.0, 9.0, 9.0, 9.0]]], dtype=torch.float16)
    v = k * 0.5
    slots = torch.tensor([2, -1])
    ks = torch.tensor([1.25])
    vs = torch.tensor([0.75])
    got = reference("PagedKVCacheWriteFwdOp", None, (k, v, pages, pages, slots, ks, vs))
    assert torch.equal(pages.view(torch.uint8), original)
    for x, new, scale in [(got[0], k, ks), (got[1], v, vs)]:
        torch.testing.assert_close(
            x.reshape(4, 1, 4)[2], (new[0].float() / scale[0]).to(dtype), atol=0, rtol=0
        )
        assert torch.equal(
            x.reshape(4, 1, 4).view(torch.uint8)[[0, 1, 3]], original.reshape(4, 1, 4)[[0, 1, 3]]
        )


@pytest.mark.parametrize("layout", ["neox", "interleaved"])
def test_mla_fp8_rope_preserves_original_rounding_contract(layout):
    from workloads.attention.mla_cache_write import MLAKVCacheWriteWorkload
    from workloads.numerics import compare_outputs

    dtype = torch.float8_e4m3fn
    workload = SimpleNamespace(
        rope_layout=layout, latent_dim=2, dtype=torch.float16, cache_dtype=dtype
    )
    workload.verification = lambda *args: MLAKVCacheWriteWorkload.verification(workload, *args)
    kv = torch.tensor([[1.0, 2.0], [4.0, 5.0]], dtype=torch.float16)
    pe = torch.tensor([[0.25, 0.5, 0.75, 1.0], [0.5, 0.75, 1.0, 1.25]], dtype=torch.float16)
    cache = torch.arange(24).reshape(2, 2, 6).float().to(dtype)
    original = cache.view(torch.uint8).clone()
    slots = torch.tensor([1, -1])
    scale = torch.tensor([0.75])
    positions = torch.tensor([0, 0])
    table = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float16)
    args = (kv, pe, cache, slots, scale, positions, table)
    expected = reference("MLAKVCacheWriteFwdOp", workload, args)
    evidence = workload.verification(*args)
    assert compare_outputs(expected, expected, evidence).checked_outputs == 1
    assert torch.equal(cache.view(torch.uint8), original)
    bad = expected.clone()
    bad.view(torch.uint8)[0, 0, 0] ^= 1
    with pytest.raises(AssertionError):
        compare_outputs(bad, expected, evidence)
