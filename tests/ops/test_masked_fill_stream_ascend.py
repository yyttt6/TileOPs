"""Bit-exact mask, storage-offset and tail regressions for streaming masked fill."""

import pytest
import torch

from tileops._ops.elementwise import MaskedFillScalarFwdOp
from workloads.device import run_device


@pytest.mark.smoke
@pytest.mark.parametrize(
    "size,offset,dtype,fill",
    [
        (1 << 20, 0, torch.float32, -0.0),
        ((1 << 20) + 16384, 1, torch.float32, -7.25),
        (1 << 20, 1, torch.float32, float("nan")),
        ((1 << 20) + 1, 1, torch.float32, -0.0),
        (1 << 20, 1, torch.int32, -7),
    ],
)
def test_scalar_stream_storage_and_mask_bytes(size, offset, dtype, fill):
    if torch.device(run_device()).type != "npu":
        pytest.skip("Ascend streaming-path regression")
    storage = torch.arange(size + offset, device=run_device(), dtype=dtype)
    x = storage[offset:]
    if dtype == torch.float32:
        x[:6] = torch.tensor(
            [float("nan"), -0.0, float("inf"), -float("inf"), 0.0, 1.0],
            device=run_device(),
            dtype=dtype,
        )
    raw = (torch.arange(size + offset, device=run_device(), dtype=torch.int32) % 6).to(torch.uint8)
    palette = torch.tensor([0, 1, 2, 127, 128, 255], device=run_device(), dtype=torch.uint8)
    raw = palette[raw.to(torch.int64)]
    mask = raw[offset:].view(torch.bool)
    before, mask_before = x.cpu(), raw.cpu()
    expected = before.masked_fill(mask_before[offset:] != 0, fill)
    op = MaskedFillScalarFwdOp(value=fill)
    actual = op(x, mask).cpu()
    torch.testing.assert_close(actual.view(torch.int32), expected.view(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(x.cpu().view(torch.int32), before.view(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(raw.cpu(), mask_before, rtol=0, atol=0)

    x[8:16].fill_(11)
    raw[offset : offset + 16].fill_(0)
    updated = x.cpu().masked_fill(raw[offset:].cpu() != 0, fill)
    torch.testing.assert_close(
        op(x, mask).cpu().view(torch.int32), updated.view(torch.int32), rtol=0, atol=0
    )
