"""FP8 key scales and small normal FP32 head weights over mutable key windows."""

import pytest
import torch

from tileops.attention import FP8LightningIndexerFwdOp
from workloads.device import run_device


@pytest.mark.smoke
def test_ascend_indexer_small_normal_weights_and_mutable_grouped_windows():
    device = torch.device(run_device())
    if device.type != "npu":
        pytest.skip("Ascend FP8 indexer regression")
    batch, queries, keys, heads, groups, dim = 1, 4, 137, 10, 2, 64
    q = torch.full((batch, queries, heads, dim), 448.0).to(torch.float8_e4m3fn)
    k = torch.full((batch, keys, groups, dim), 448.0).to(torch.float8_e4m3fn)
    scales = torch.linspace(1.0e29, 1.0e30, keys * groups).reshape(batch, keys, groups)
    weights = torch.full((queries, heads), 1.5 * torch.finfo(torch.float32).tiny)
    weights[:, heads // groups :] *= 2
    expected = ((448.0 * 448.0 * dim) * scales[:, None]) * weights.reshape(
        1, queries, 1, groups, heads // groups
    ).sum(-1)
    token = torch.arange(keys)[None, :]
    qs, ks, ws, sc = [t.to(device) for t in (q, k, weights, scales)]
    starts, ends = (torch.empty(queries, device=device, dtype=torch.int32) for _ in range(2))
    clean_op = FP8LightningIndexerFwdOp(clean_logits=True)
    raw_op = FP8LightningIndexerFwdOp(clean_logits=False)
    for begin, end in (
        ([0, 3, 117, 12], [137, 3, 131, 2]),
        ([71, 0, 5, 136], [88, 1, 3, 137]),
    ):
        ls, rs = [torch.tensor(a, dtype=torch.int32) for a in (begin, end)]
        starts.copy_(ls)
        ends.copy_(rs)
        live = (token >= ls[:, None]) & (token < rs[:, None])
        clean = clean_op(qs, ks, ws, starts, ends, sc).cpu()
        raw = raw_op(qs, ks, ws, starts, ends, sc).cpu()
        mask = live[None, :, :, None].expand_as(expected)
        assert torch.isneginf(clean[~mask]).all()
        torch.testing.assert_close(clean[mask], expected[mask], rtol=2e-5, atol=0)
        torch.testing.assert_close(raw[mask], clean[mask], rtol=0, atol=0)
