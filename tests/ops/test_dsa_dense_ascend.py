"""Dense DSA indexes a set; repeated or padded slots cannot change normalization."""

import pytest
import torch

from tileops.attention import DSADenseFwdOp
from workloads.device import run_device


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
def test_ascend_dsa_dense_schedule_accepts_variable_shapes():
    from types import SimpleNamespace

    pytest.importorskip("tilelang.ascend.language")
    from tileops._kernels.ascend.attention.dsa_dense import uses_d512_schedule

    call = SimpleNamespace(kv_group=1, kv_stride=1, is_causal=True,
                           dim=512, tail_dim=0, q_start_index_s=3000)
    q = torch.empty((2, 17, 128, 512), device="meta", dtype=torch.bfloat16)
    kv = torch.empty((2, 3072, 1, 512), device="meta", dtype=torch.bfloat16)
    idx = torch.empty((2, 17, 1, 384), device="meta", dtype=torch.int32)
    assert uses_d512_schedule(call, q, kv, idx)
    assert not uses_d512_schedule(call, q, kv, idx[..., :383])
    call.kv_stride = 2
    assert not uses_d512_schedule(call, q, kv, idx)


@pytest.mark.smoke
def test_ascend_dsa_schedules_respect_device_core_limits(monkeypatch):
    from types import SimpleNamespace

    pytest.importorskip("tilelang.ascend.language")
    from tileops._kernels.ascend.attention.dsa_dense import device_core_counts

    npu = SimpleNamespace(
        get_device_properties=lambda device: SimpleNamespace(cube_core_num=36, vector_core_num=72),
        get_device_limit=lambda device: {"cube_core_num": 32, "vector_core_num": 64},
    )
    # Restore the runtime before the suite's teardown hook accesses its allocator.
    with monkeypatch.context() as runtime_patch:
        runtime_patch.setattr(torch, "npu", npu, raising=False)
        assert device_core_counts(SimpleNamespace(index=1)) == (32, 64)
        del npu.get_device_limit
        assert device_core_counts(SimpleNamespace(index=1)) == (36, 72)


@pytest.mark.smoke
@pytest.mark.parametrize("heads", [64, 128])
def test_ascend_dsa_dense_tiled_attention_at_variable_query_length(heads):
    device = torch.device(run_device())
    if device.type != "npu":
        pytest.skip("Ascend D512 optimization")
    q = torch.randn(2, 3, heads, 512).bfloat16()
    kv = torch.randn(2, 256, 1, 512).bfloat16()
    ids = torch.arange(128, dtype=torch.int32).expand(2, 3, 1, 128).clone()
    ids[..., 2] = 0
    ids[:, 1] = -1
    ids[..., -1] = 256
    op = DSADenseFwdOp(dim_tail=0, stride_kv=1, q_start_index_s=128)
    out = op(q.to(device), kv.to(device), ids.to(device)).cpu()
    selected = torch.cat((kv[:, :2, 0], kv[:, 3:127, 0]), dim=1).float()
    scores = (q.float() @ selected.transpose(-1, -2).unsqueeze(1)) * 512**-0.5
    expected = (scores.softmax(-1) @ selected.unsqueeze(1)).bfloat16()
    expected[:, 1] = 0
    torch.testing.assert_close(out, expected, rtol=0.015, atol=0.015)
