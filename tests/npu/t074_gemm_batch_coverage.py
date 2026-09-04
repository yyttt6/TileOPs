import json
import os
from pathlib import Path

import torch

from tileops.kernels.common import MAX_BLOCK_COUNT, launch_block_count
from tileops.kernels.families.gemm import build_bmm

CASES = (
    ("fp16-nondiv-mn", 2, 33, 65, 128, torch.float16),
    ("bf16-small", 3, 64, 96, 128, torch.bfloat16),
    ("fp16-k256", 4, 129, 257, 256, torch.float16),
    ("bf16-grid-stride", 65536, 1, 1, 16, torch.bfloat16),
)


def main() -> None:
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "3":
        raise RuntimeError("T074 must run with ASCEND_RT_VISIBLE_DEVICES=3")
    torch.manual_seed(74)
    results = []
    for label, batch, m, n, k, dtype in CASES:
        a = torch.randn((batch, m, k), device="npu", dtype=dtype)
        b = torch.randn((batch, k, n), device="npu", dtype=dtype)
        output = build_bmm(a, b)(a, b)
        reference = torch.bmm(a, b)
        torch.npu.synchronize()
        delta = (output.float() - reference.float()).abs()
        atol = 1e-3 if dtype == torch.float16 else 1.6e-2
        rtol = atol
        finite = bool(torch.isfinite(output).all())
        passed = finite and bool(torch.allclose(output, reference, atol=atol, rtol=rtol))
        block_m = 32 if m < 128 else 128
        block_n = 64 if n < 256 else 256
        logical_blocks = batch * ((m + block_m - 1) // block_m) * ((n + block_n - 1) // block_n)
        results.append(
            {
                "label": label,
                "shape": {"a": list(a.shape), "b": list(b.shape), "out": list(output.shape)},
                "dtype": str(dtype).removeprefix("torch."),
                "max_abs_err": float(delta.max().cpu()),
                "tolerance": {"atol": atol, "rtol": rtol},
                "finite": finite,
                "passed": passed,
                "logical_blocks": logical_blocks,
                "launch_blocks": launch_block_count(logical_blocks),
            }
        )
        del a, b, output, reference, delta
        torch.npu.empty_cache()

    payload = {
        "op": "BmmFwdOp",
        "visible_devices": os.environ["ASCEND_RT_VISIBLE_DEVICES"],
        "max_block_count": MAX_BLOCK_COUNT,
        "correctness": {"passed": all(row["passed"] for row in results), "per_case": results},
    }
    destination = Path(__file__).resolve().parents[3] / "tileops-ascend-harness/coverage/BmmFwdOp.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not payload["correctness"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
