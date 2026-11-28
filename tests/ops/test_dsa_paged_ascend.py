"""Byte layout, repeated selected positions and empty rows in paged sparse MLA."""

import pytest
import torch

from tileops.attention import DSAPagedFwdOp
from workloads.device import run_device


@pytest.mark.smoke
def test_ascend_dsa_paged_reads_live_page_table_and_preserves_duplicate_slots():
    device = torch.device(run_device())
    if device.type != "npu":
        pytest.skip("Ascend paged sparse attention regression")
    torch.manual_seed(45)
    pages, page_size, heads = 6, 3, 5
    codes = torch.randn(pages, page_size, 512).to(torch.float8_e4m3fn)
    multiplier = torch.tensor([0.25, 0.5, 1.0, 2.0]).expand(pages, page_size, 4).contiguous()
    rope = torch.randn(pages, page_size, 64).to(torch.bfloat16)
    cache = torch.cat(
        (codes.view(torch.uint8), multiplier.view(torch.uint8), rope.view(torch.uint8)),
        -1,
    )
    latent = (codes.float().reshape(pages, page_size, 4, 128) * multiplier[..., None]).flatten(-2)
    key = torch.cat((latent, rope.float()), dim=-1)
    q = torch.randn(2, 2, heads, 576).to(torch.bfloat16)
    ids = torch.tensor(
        [[[0, 0, 3, -1], [-1, -1, -1, -1]], [[1, 2, 1, 3], [0, 3, -1, -1]]],
        dtype=torch.int32,
    )
    lengths = torch.tensor([5, 4], dtype=torch.int32)
    qd, cached, idx, lens = [v.to(device) for v in (q, cache, ids, lengths)]
    table = torch.empty(2, 2, dtype=torch.int32, device=device)
    op = DSAPagedFwdOp(sm_scale=0.036)
    for page_order in ([[3, 0], [5, 1]], [[1, 4], [2, 3]]):
        table.copy_(torch.tensor(page_order, dtype=torch.int32))
        out, lse = op(qd, cached, table, lens, idx)
        expected = torch.zeros(2, 2, heads, 512)
        expected_lse = torch.full((2, heads, 2), -torch.inf)
        for b in range(2):
            for t in range(2):
                pos = ids[b, t].long()
                pos = pos[(pos >= 0) & (pos < lengths[b])]
                if pos.numel():
                    page = torch.tensor(page_order[b])[pos // page_size]
                    row = pos % page_size
                    score = (q[b, t].float() @ key[page, row].T) * 0.036
                    expected[b, t] = score.softmax(-1) @ latent[page, row]
                    expected_lse[b, :, t] = score.logsumexp(-1)
        torch.testing.assert_close(out.cpu().float(), expected, rtol=0.015, atol=0.015)
        torch.testing.assert_close(lse.cpu(), expected_lse, rtol=0.002, atol=0.002)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("queries", "expected_28", "expected_32"), [(64, 2, 2), (32, 4, 2)]
)
def test_ascend_dsa_paged_split_schedule_uses_cube_count(queries, expected_28, expected_32):
    pytest.importorskip("tilelang.ascend.language")
    from tileops._kernels.ascend.attention.dsa_paged import choose_split_count

    assert choose_split_count(queries, 2048, 28, 512) == expected_28
    assert choose_split_count(queries, 2048, 32, 512) == expected_32
    assert choose_split_count(queries, 384, 32, 512) == 0


@pytest.mark.smoke
@pytest.mark.parametrize(("queries", "aivs"), [(48, 56), (48, 64), (47, 64), (65, 56)])
def test_ascend_dsa_paged_gather_schedule_covers_live_tiles(queries, aivs):
    pytest.importorskip("tilelang.ascend.language")
    from tileops._kernels.ascend.attention.dsa_paged_h64 import gather_schedule

    tiles = 64
    blocks, base, shared = gather_schedule(queries, tiles, aivs)
    assert 0 < blocks <= aivs
    if not shared:
        assert sorted(row for core in range(blocks) for row in range(core, queries, blocks)) == list(range(queries))
        return

    helpers = blocks - queries
    rows = [row for core in range(helpers)
            for row in range(core * queries // helpers, (core + 1) * queries // helpers)]
    assert rows == list(range(queries))
    for live in range(tiles + 1):
        primary = list(range(min(live, base))) + list(range(base + shared, live))
        helper = list(range(base, min(live, base + shared)))
        assert sorted(primary + helper) == list(range(live))


@pytest.mark.smoke
@pytest.mark.parametrize("heads", [64, 128])
def test_ascend_dsa_paged_tiled_attention_with_repeated_and_empty_slots(heads):
    device = torch.device(run_device())
    if device.type != "npu":
        pytest.skip("Ascend paged optimization")
    latent = torch.randn(4, 64, 512).to(torch.float8_e4m3fn)
    scale = torch.ones((4, 64, 4), dtype=torch.float32)
    rope = torch.randn(4, 64, 64).bfloat16()
    cache = torch.cat((latent.view(torch.uint8), scale.view(torch.uint8), rope.view(torch.uint8)), -1)
    q = torch.randn(1, 2, heads, 576).bfloat16()
    table = torch.tensor([[3, 1, 0, 2]], dtype=torch.int32)
    limits = torch.tensor([187], dtype=torch.int32)
    ids = torch.full((1, 2, 256), -1, dtype=torch.int32)
    ids[0, 0, :4] = torch.tensor([1, 65, 190, 1], dtype=torch.int32)
    ids[0, 0, 128:131] = torch.tensor([126, 3, 7], dtype=torch.int32)
    inputs = (q, cache, table, limits, ids)
    out, lse = DSAPagedFwdOp()(*(tensor.to(device) for tensor in inputs))
    logical = ids[0, 0]
    logical = logical[(logical >= 0) & (logical < limits[0])].long()
    page = table[0, logical // 64].long()
    offset = logical % 64
    value = latent.float()[page, offset]
    key = torch.cat((value, rope.float()[page, offset]), -1)
    score = (q[0, 0].float() @ key.T) * 576**-0.5
    expected = (score.softmax(-1) @ value).bfloat16()
    torch.testing.assert_close(out[0, 0].cpu(), expected, rtol=0.02, atol=0.02)
    torch.testing.assert_close(lse[0, :, 0].cpu(), score.logsumexp(-1), rtol=0.002, atol=0.002)
    assert torch.count_nonzero(out[:, 1].cpu()) == 0
    assert torch.isneginf(lse[:, :, 1].cpu()).all()
