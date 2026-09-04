"""T105 regression probe for the two existing MaskedFill registrations."""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import torch

sys.modules.setdefault(
    "tileops.kernels.families.two_pass", types.ModuleType("tileops.kernels.families.two_pass")
)
from tileops.ops.elementwise import MaskedFillFwdOp, MaskedFillScalarFwdOp  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        raise RuntimeError("ASCEND_RT_VISIBLE_DEVICES must be set")
    rows = []
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.int32):
        n = 4097
        torch.manual_seed(105 + len(rows))
        if dtype.is_floating_point:
            x = torch.randn((n,), device="npu", dtype=dtype)
            value = torch.tensor(-3.5, device="npu", dtype=dtype)
            scalar = -3.5
        else:
            x = torch.randint(-10, 10, (n,), device="npu", dtype=dtype)
            value = torch.tensor(-7, device="npu", dtype=dtype)
            scalar = -7
        mask = (torch.arange(n, device="npu") % 3 == 0)
        for name, op, args, ref in (
            ("MaskedFillFwdOp", MaskedFillFwdOp(target="ascend"), (x, mask, value), torch.where(mask.cpu(), value.cpu(), x.cpu())),
            ("MaskedFillScalarFwdOp", MaskedFillScalarFwdOp(value=scalar, target="ascend"), (x, mask), torch.where(mask.cpu(), torch.tensor(scalar, dtype=dtype), x.cpu())),
        ):
            try:
                got = op(*args)
                torch.npu.synchronize()
                delta = (got.cpu().float() - ref.float()).abs()
                atol = rtol = 1.6e-2 if dtype == torch.bfloat16 else 1.0e-3 if dtype == torch.float16 else 1e-5 if dtype == torch.float32 else 0
                ok = bool(torch.allclose(got.cpu().float(), ref.float(), atol=atol, rtol=rtol))
                err = float(delta.max())
                error = None
            except Exception as exc:  # noqa: BLE001 - preserve evidence
                ok, err, error = False, None, f"{type(exc).__name__}: {exc}"
            rows.append({"op": name, "dtype": str(dtype).replace("torch.", ""), "shape": [n], "passed": ok, "max_abs_err": err, "error": error})
    path = ROOT / "docs/reports/R105-data/masked-fill-regression.json"
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, sort_keys=True))
    if not all(row["passed"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
