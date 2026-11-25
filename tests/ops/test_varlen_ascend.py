"""Keep packed request boundaries and denominator-only sink logits live on Ascend."""

import pytest
import torch

from tileops.attention import GQAVarlenFwdOp
from workloads.attention.gqa.dense import dense_gqa_ref
from workloads.device import run_device


@pytest.mark.smoke
@pytest.mark.parametrize("options", [{"window_size_left": 2}, {"sm_scale": 0.13, "softcap": 3.0}])
def test_ascend_varlen_reuses_changed_offsets_with_empty_requests(options):
    dev = torch.device(run_device())
    if dev.type != "npu":
        pytest.skip("Ascend packed attention regression")
    q = torch.randn(11, 6, 64, dtype=torch.float16, device=dev)
    k = torch.randn(19, 2, 64, dtype=torch.float16, device=dev)
    v = torch.randn_like(k)
    sinks = torch.tensor([-float("inf"), 0.0, float("inf"), 1.25, -2.0, 0.5], device=dev)
    cq = torch.empty(4, dtype=torch.int32, device=dev)
    ck = torch.empty_like(cq)
    op = GQAVarlenFwdOp(is_causal=True, **options)
    for q_offsets, k_offsets in (
        ([0, 0, 3, 11], [0, 4, 8, 19]),
        ([0, 4, 11, 11], [0, 5, 14, 19]),
    ):
        cq.copy_(torch.tensor(q_offsets, dtype=torch.int32))
        ck.copy_(torch.tensor(k_offsets, dtype=torch.int32))
        got = op(q, k, v, cq, ck, sinks=sinks)
        expected = []
        for qa, qb, ka, kb in zip(
            q_offsets[:-1], q_offsets[1:], k_offsets[:-1], k_offsets[1:], strict=True
        ):
            if qb > qa:
                expected.append(
                    dense_gqa_ref(
                        q[None, qa:qb].cpu(),
                        k[None, ka:kb].cpu(),
                        v[None, ka:kb].cpu(),
                        heads=6,
                        heads_kv=2,
                        is_causal=True,
                        **options,
                        sinks=sinks.cpu(),
                    )[0]
                )
        torch.testing.assert_close(
            got.cpu().float(), torch.cat(expected).float(), rtol=0.02, atol=0.005
        )


@pytest.mark.smoke
def test_ascend_varlen_mixes_aligned_and_partial_requests():
    dev = torch.device(run_device())
    if dev.type != "npu":
        pytest.skip("Ascend packed attention regression")
    torch.manual_seed(124)
    q = torch.randn(257, 4, 64, dtype=torch.float16)
    k = torch.randn(389, 2, 64, dtype=torch.float16)
    v = torch.randn_like(k)
    qd, kd, vd = (x.to(dev) for x in (q, k, v))
    cq, ck = (torch.empty(3, dtype=torch.int32, device=dev) for _ in range(2))
    op = GQAVarlenFwdOp(is_causal=True, window_size_left=256)
    for qs, ks in ((128, 256), (129, 133)):
        # The aligned request switches places while all tensor addresses are fixed.
        cq.copy_(torch.tensor([0, qs, len(q)], dtype=torch.int32))
        ck.copy_(torch.tensor([0, ks, len(k)], dtype=torch.int32))
        actual = op(qd, kd, vd, cq, ck)
        expected = []
        for qr, kr in ((slice(None, qs), slice(None, ks)), (slice(qs, None), slice(ks, None))):
            expected.append(
                dense_gqa_ref(
                    q[None, qr],
                    k[None, kr],
                    v[None, kr],
                    heads=4,
                    heads_kv=2,
                    is_causal=True,
                    window_size_left=256,
                )[0]
            )
        torch.testing.assert_close(actual.cpu(), torch.cat(expected), atol=0.005, rtol=0.015)
