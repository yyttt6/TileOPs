import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nightly_support"))
from ssd_reference import coupling

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_coupling_matches_scalar_dot_and_causal_mask(dtype):
    torch.manual_seed(101)
    C = torch.randn(2, 8, 2, 3).to(dtype)
    B = torch.randn_like(C)
    got = coupling(C, B, 4)
    expected = torch.zeros(2, 2, 2, 4, 4, dtype=dtype)
    for b in range(2):
        for c in range(2):
            for g in range(2):
                for i in range(4):
                    for j in range(i + 1):
                        expected[b, c, g, i, j] = sum(
                            float(C[b, c * 4 + i, g, n]) * float(B[b, c * 4 + j, g, n])
                            for n in range(3)
                        )
    torch.testing.assert_close(got, expected, atol=0, rtol=0)
