"""Locate the bf16 GQA error and relate it to the local bf16 ULP."""

import math

import torch
import torch.nn.functional as F

from tileops.kernels.attention import build_grouped_query_attention_kernel


def main() -> None:
    torch.manual_seed(123)
    batch, seq_len, heads, heads_kv, dim = 1, 128, 10, 2, 128
    dtype = torch.bfloat16

    q = torch.randn((batch, seq_len, heads, dim), dtype=dtype, device="npu")
    k = torch.randn((batch, seq_len, heads_kv, dim), dtype=dtype, device="npu")
    v = torch.randn_like(k)

    kernel = build_grouped_query_attention_kernel(
        batch, seq_len, heads, heads_kv, dim, True, dtype
    )
    output = kernel(
        q.view(batch * seq_len, heads, dim),
        k.view(batch * seq_len, heads_kv, dim),
        v.view(batch * seq_len, heads_kv, dim),
        None,
        None,
    ).view(batch, seq_len, heads, dim)
    reference = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        is_causal=True,
        enable_gqa=True,
    ).transpose(1, 2)
    reference_fp32 = F.scaled_dot_product_attention(
        q.float().transpose(1, 2),
        k.float().transpose(1, 2),
        v.float().transpose(1, 2),
        is_causal=True,
        enable_gqa=True,
    ).transpose(1, 2)
    torch.npu.synchronize()

    delta = (output.float() - reference.float()).abs()
    flat_idx = int(delta.argmax().cpu())
    coords = []
    remainder = flat_idx
    for extent in reversed(delta.shape):
        coords.append(remainder % extent)
        remainder //= extent
    index = tuple(reversed(coords))
    output_value = float(output[index].float().cpu())
    reference_value = float(reference[index].float().cpu())
    magnitude = abs(reference_value)
    ulp = 2.0 ** (math.floor(math.log2(magnitude)) - 7) if magnitude else 2.0**-133

    reference_round_delta = (reference.float() - reference_fp32.to(dtype).float()).abs()
    print(f"shape={(batch, seq_len, heads, heads_kv, dim)} dtype={dtype}")
    print(f"max_index={index}")
    print(f"kernel={output_value} reference={reference_value}")
    print(f"max_abs_err={float(delta.max().cpu())}")
    print(f"reference_magnitude={magnitude} bf16_ulp={ulp}")
    print(f"error_in_ulp={float(delta.max().cpu()) / ulp}")
    print(f"count_gt_5e_3={int((delta > 5e-3).sum().cpu())}")
    print(
        "bf16_reference_vs_fp32_then_bf16_max_abs_err="
        f"{float(reference_round_delta.max().cpu())}"
    )


if __name__ == "__main__":
    main()
