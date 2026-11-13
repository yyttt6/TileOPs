"""Public-interface regressions for Ascend TopK selection."""

import pytest
import torch

from tileops.attention import TopKSelectFwdOp
from workloads.attention.topk_select import TopKSelectWorkload
from workloads.device import run_device, run_device_available
from workloads.numerics import Request, verify

pytestmark = pytest.mark.in_tree_kernels


@pytest.mark.parametrize(
    "pattern,start,end",
    [
        pytest.param("random", 0, 16384, id="manifest-window", marks=pytest.mark.smoke),
        pytest.param("random", 16384, 16384, id="empty-end", marks=pytest.mark.full),
        pytest.param("random", 127, 128, id="single-at-vector-edge", marks=pytest.mark.full),
        pytest.param("random", 129, 2176, id="short-k-minus-one", marks=pytest.mark.full),
        pytest.param("random", 127, 2175, id="exact-k", marks=pytest.mark.full),
        pytest.param("random", 127, 2176, id="refine-k-plus-one", marks=pytest.mark.full),
        pytest.param("random", 14208, 16384, id="nonaligned-window-at-end", marks=pytest.mark.full),
        pytest.param("equal", 0, 16384, id="all-tied-overflow", marks=pytest.mark.full),
        pytest.param("quantized", 0, 16384, id="threshold-ties", marks=pytest.mark.full),
        pytest.param("adjacent", 0, 16384, id="fp32-boundary-precision", marks=pytest.mark.full),
        pytest.param("negative", 0, 16384, id="negative-key-order", marks=pytest.mark.full),
        pytest.param("zero", 0, 16384, id="signed-zero-ties", marks=pytest.mark.full),
    ],
)
def test_topk_select_ascend_decode_boundaries(pattern, start, end):
    """Guard short-window padding, refinement, full-key order and reproducible tie sets."""
    if torch.device(run_device()).type != "npu" or not run_device_available():
        pytest.skip("Ascend decode integration regression")
    generator = torch.Generator().manual_seed(20261105)
    values = torch.randn(16384, generator=generator)
    if pattern == "equal":
        values.fill_(1.0)
    elif pattern == "quantized":
        values = (torch.arange(16384) % 4).float()
    elif pattern == "adjacent":
        # All values round to the same coarse FP16 bucket, but differ in FP32.
        values = (torch.arange(16384, dtype=torch.int32) + 0x3F800000).view(torch.float32)
    elif pattern == "negative":
        values = -torch.arange(1, 16385, dtype=torch.float32)
    elif pattern == "zero":
        values = torch.zeros(16384)
        values[::2] = -0.0
    inputs = (
        values.reshape(1, 1, 16384, 1).to(run_device()),
        torch.tensor([[start]], dtype=torch.int32, device=run_device()),
        torch.tensor([[end]], dtype=torch.int32, device=run_device()),
    )
    workload = TopKSelectWorkload(1, 1, 16384, 1, 2048, torch.float32, torch.int32)
    op = TopKSelectFwdOp(top_k=2048)
    selected = []

    def run(*args):
        score_bits = args[0].view(torch.int32).clone()
        output = op(*args)
        assert torch.equal(args[0].view(torch.int32), score_bits), "score bits modified"
        selected.append(output.detach().cpu().sort(-1).values)
        return output

    for _ in range(3):
        checks = verify(
            workload.ref_program,
            inputs,
            evidence=workload.verification(*inputs),
            requests={"tileops": Request(run, inputs, preserve_inputs=True)},
        )
        assert checks["tileops"].checked_outputs == checks["tileops"].total_outputs == 1
        assert checks["tileops"].unchecked_reason is None
    # The public contract permits output permutations, but requires the same selected set.
    assert len(selected) == 3
    assert torch.equal(selected[0], selected[1]) and torch.equal(selected[0], selected[2])


@pytest.mark.parametrize(
    "batch,queries,length,groups,topk,layout",
    [
        pytest.param(1, 1, 1, 1, 1, "contiguous", id="single-key", marks=pytest.mark.smoke),
        pytest.param(2, 3, 257, 3, 17, "contiguous", id="batch-and-groups", marks=pytest.mark.full),
        pytest.param(
            1, 29, 129, 1, 33, "contiguous", id="worker-stride-tail", marks=pytest.mark.full
        ),
        pytest.param(
            2, 5, 513, 3, 65, "contiguous", id="group-row-stride-tail", marks=pytest.mark.full
        ),
        pytest.param(
            1, 1, 127, 1, 126, "contiguous", id="almost-whole-row", marks=pytest.mark.full
        ),
        pytest.param(1, 1, 128, 1, 128, "contiguous", id="whole-row", marks=pytest.mark.full),
        pytest.param(1, 1, 129, 1, 1, "contiguous", id="maximum-only", marks=pytest.mark.full),
        pytest.param(
            1, 2, 4097, 1, 3072, "contiguous", id="k-beyond-cache", marks=pytest.mark.full
        ),
        pytest.param(
            1, 2, 65536, 2, 1024, "contiguous", id="prefill-axis-k1024", marks=pytest.mark.full
        ),
        pytest.param(
            1, 3, 65536, 1, 2048, "contiguous", id="prefill-axis-k2048", marks=pytest.mark.full
        ),
        pytest.param(
            2, 3, 257, 2, 17, "strided", id="noncontiguous-inputs", marks=pytest.mark.full
        ),
        pytest.param(0, 3, 129, 2, 32, "contiguous", id="empty-batch", marks=pytest.mark.full),
        pytest.param(2, 0, 129, 2, 32, "contiguous", id="empty-queries", marks=pytest.mark.full),
        pytest.param(2, 3, 129, 0, 32, "contiguous", id="empty-groups", marks=pytest.mark.full),
    ],
)
def test_topk_select_ascend_public_shapes(batch, queries, length, groups, topk, layout):
    """Guard B/S/G row mapping, worker tail, legal K, empty axes and contiguous copies."""
    if torch.device(run_device()).type != "npu" or not run_device_available():
        pytest.skip("Ascend public-shape regression")
    device = run_device()
    storage_groups = groups * 2 if layout == "strided" else groups
    scores = torch.randn(batch, queries, length, storage_groups, device=device)
    if layout == "strided":
        scores = scores[..., ::2]
    windows = torch.empty(batch, queries, 2, dtype=torch.int32, device=device)
    for b in range(batch):
        for s in range(queries):
            mode = (b * queries + s) % 6
            if mode == 0:
                start, end = 0, length
            elif mode == 1:
                start, end = length, length
            elif mode == 2:
                start = min(1, length)
                end = min(length, start + topk - 1)
            elif mode == 3:
                start = (length - topk) // 2
                end = start + topk
            elif mode == 4:
                start, end = 0, min(length, topk + 1)
            else:
                # Both bounds satisfy the manifest; its interval/reference is empty.
                start, end = min(1, length), 0
            windows[b, s, 0], windows[b, s, 1] = start, end
    starts, ends = windows[..., 0], windows[..., 1]
    inputs = scores, starts, ends
    workload = TopKSelectWorkload(batch, queries, length, groups, topk, torch.float32, torch.int32)
    op = TopKSelectFwdOp(top_k=topk)
    selected = []

    def run(*args):
        bits = args[0].cpu().contiguous().view(torch.int32).clone()
        output = op(*args)
        assert torch.equal(args[0].cpu().contiguous().view(torch.int32), bits)
        selected.append(output.cpu().sort(-1).values)
        return output

    for _ in range(3):
        checks = verify(
            workload.ref_program,
            inputs,
            evidence=workload.verification(*inputs),
            requests={"tileops": Request(run, inputs, preserve_inputs=True)},
        )
        assert checks["tileops"].checked_outputs == checks["tileops"].total_outputs == 1
        assert checks["tileops"].unchecked_reason is None
    assert len(selected) == 3
    assert torch.equal(selected[0], selected[1]) and torch.equal(selected[0], selected[2])
