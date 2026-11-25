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
