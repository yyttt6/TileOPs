"""Dense DSA indexes a set; repeated or padded slots cannot change normalization."""

import pytest
import torch

from tileops.attention import DSADenseFwdOp
from workloads.attention.dsa import DSADecodeWorkload
from workloads.device import run_device
from workloads.numerics import Request, verify


@pytest.mark.smoke
def test_ascend_dsa_dense_duplicate_keys_and_compressed_causal_limit():
    device = torch.device(run_device())
    if device.type != "npu":
        pytest.skip("Ascend sparse attention")
    batch, sq, sk, heads, groups, value, tail = 1, 3, 7, 4, 2, 64, 16
    stride, first = 2, 5
    q = torch.randn(batch, sq, heads, value + tail, dtype=torch.float16, device=device)
    kv = torch.randn(batch, sk, groups, value + tail, dtype=torch.float16, device=device)
    ids = torch.tensor(
        [
            [
                [[1, 1, 7, 6], [7, 7, 7, 7]],
                [[7, 7, 7, 7], [1, 1, 2, 6]],
                [[0, 3, 3, 7], [2, 4, 2, 4]],
            ]
        ],
        dtype=torch.int32,
    )
    op = DSADenseFwdOp(dim_tail=tail, stride_kv=stride, q_start_index_s=first)
    got = op(q, kv, ids.to(device)).cpu()
    expected = torch.zeros_like(got)
    qf, kf = q.cpu().float(), kv.cpu().float()
    for qi in range(sq):
        for head in range(heads):
            g = head // (heads // groups)
            keep = {
                int(pos)
                for pos in ids[0, qi, g]
                if pos < sk and (pos + 1) * stride - 1 <= qi + first
            }
            if qi < stride - 1:
                keep.add(0)
            if keep:
                keys = kf[0, sorted(keep), g, :]
                scores = (keys @ qf[0, qi, head]) * (value + tail) ** -0.5
                expected[0, qi, head] = scores.softmax(0) @ keys[:, :value]
    torch.testing.assert_close(got, expected, rtol=3e-3, atol=2e-3)

    empty_slots = ids[..., :0].to(device).contiguous()
    empty = op(q, kv, empty_slots).cpu()
    zero_selection = torch.zeros_like(empty)
    zero_selection[:, 0] = kv[:, 0, :, :value].cpu().repeat_interleave(heads // groups, dim=1)
    torch.testing.assert_close(empty, zero_selection, rtol=0, atol=0)


@pytest.mark.smoke
def test_ascend_dsa_dense_membership_tiles_follow_changed_slots():
    device = torch.device(run_device())
    if device.type != "npu":
        pytest.skip("Ascend dense sparse-index membership regression")
    torch.manual_seed(119)
    sq, sk, h, hk, value, tail = 3, 256, 4, 2, 64, 16
    q = torch.randn(1, sq, h, value + tail, dtype=torch.float16)
    kv = torch.randn(1, sk, hk, value + tail, dtype=torch.float16)
    qd, kd = q.to(device), kv.to(device)
    idx = torch.full((1, sq, hk, 128), sk, dtype=torch.int32)
    idx[0, 0, 0, :64] = torch.arange(64)
    idx[0, 0, 0, 64:96] = 63
    idx[0, 0, 1, :6] = torch.tensor([2, 2, 11, 127, 200, 255])
    idx[0, 1, 1, :64] = torch.arange(64, 128)
    idx[0, 1, 1, 64:68] = torch.tensor([128, 129, 127, 127])
    idx[0, 2, :, :] = torch.randint(0, sk, (hk, 128), dtype=torch.int32)
    ids = idx.to(device)
    op = DSADenseFwdOp(dim_tail=tail, stride_kv=1, q_start_index_s=127)
    # The dense membership route covers complete, partial and empty tiles.
    for order in ((0, 1, 2), (1, 2, 0)):
        current = idx[:, list(order)]
        ids.copy_(current)
        result = op(qd, kd, ids).cpu().float()
        ref = torch.zeros_like(result)
        for row in range(sq):
            for head in range(h):
                g = head // (h // hk)
                pos = current[0, row, g].long()
                keep = pos[(pos < sk) & (pos <= 127 + row)].unique()
                if keep.numel():
                    keys = kv[0, keep, g, :].float()
                    p = (keys @ q[0, row, head].float() * (value + tail) ** -0.5).softmax(0)
                    ref[0, row, head] = p @ keys[:, :value]
        torch.testing.assert_close(result, ref, atol=0.003, rtol=0.005)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("heads", "topk", "seq"),
    [(64, 128, 3), (128, 128, 3), (64, 384, 4), (128, 1024, 8)],
    ids=[
        "h64-single-tile",
        "h128-single-tile",
        "h64-non-power-of-two",
        "h128-specialized-compaction",
    ],
)
def test_ascend_dsa_dense_tiled_attention_at_variable_query_length(heads, topk, seq):
    device = torch.device(run_device())
    if device.type != "npu":
        pytest.skip("Ascend D512 optimization")
    generator = torch.Generator().manual_seed(173)
    kv_len = topk + 64
    start = kv_len - seq - 2
    scale = 0.04
    workload = DSADecodeWorkload(
        1, heads, seq, kv_len, 512, 0, topk, 1, 1, start, sm_scale=scale, dtype=torch.bfloat16
    )
    q = torch.randn(1, seq, heads, 512, generator=generator).bfloat16()
    kv = torch.randn(1, kv_len, 1, 512, generator=generator).bfloat16()
    ids = torch.arange(topk, dtype=torch.int32).repeat(1, seq, 1, 1)
    # Duplicate across lanes 63 and 64, then test both sides of each row's cutoff.
    ids[..., 64] = ids[..., 63]
    cutoff = torch.arange(start, start + seq, dtype=torch.int32)
    ids[0, :, 0, -4] = cutoff - 1
    ids[0, :, 0, -3] = cutoff
    ids[0, :, 0, -2] = cutoff + 1
    ids[..., -1] = kv_len
    ids[:, 1] = kv_len
    inputs = tuple(tensor.to(device) for tensor in (q, kv, ids))
    op = DSADenseFwdOp(dim_tail=0, stride_kv=1, q_start_index_s=start, sm_scale=scale)
    checks = verify(
        workload.ref_program,
        inputs,
        evidence=workload.verification(*inputs),
        requests={"tileops": Request(op, inputs)},
    )
    result = checks["tileops"]
    assert result.checked_outputs == result.total_outputs == 1, result.unchecked_reason
