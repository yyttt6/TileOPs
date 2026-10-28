"""Dense GQA specialization selection and shared launch dimensions."""

import functools
import math

import torch


@functools.lru_cache(maxsize=128)
def configuration(call):
    group = call.heads // call.heads_kv
    rows = math.ceil(call.max_seqlen_q * group / 32) * 32
    kv = math.ceil(call.seqlen_kv / 128) * 128
    tiles = call.batch * call.heads_kv * (rows // 32)
    splits = min(32, math.ceil(72 / tiles), kv // 128) if call.max_seqlen_q == 1 else 1
    return rows, kv, splits


def select_kernel(call):
    """Choose a dataflow; shape differences within it are tile parameters."""
    causal = (
        call.is_causal
        and call.sm_scale is None
        and call.window_size_left == call.window_size_right == -1
    )
    shape = (call.batch, call.seqlen_kv, call.heads, call.heads_kv, call.dim)
    if causal and not call.softcap and call.max_seqlen_q > 1:
        # Preprocessing must round RoPE/scaled FP8 to the contraction dtype.
        if (
            call.max_seqlen_q == 256
            and shape == (1, 1792, 8, 2, 128)
            and call.dtype in (torch.float16, torch.bfloat16)
            and (
                (call.is_fp8 and not call.fuse_rope and call.dtype == torch.float16)
                or (
                    not call.is_fp8
                    and call.fuse_rope
                    and call.rotary_dim == 64
                    and call.rope_layout == "neox"
                )
            )
        ):
            # Both RoPE dtypes use the validated multibuffer schedule. FP8 uses
            # BM64 to avoid duplicate conversion and BK128 to reduce L0 slices.
            return "prefill-preprocess", dict(
                query_block=64 if call.is_fp8 else 32,
                l0_k=128 if call.is_fp8 else 64,
            )
        if (
            call.max_seqlen_q == 128
            and call.batch in (1, 8)
            and shape[1:] == (2048, 32, 8, 128)
            and call.dtype == torch.bfloat16
            and not call.is_fp8
            and not call.fuse_rope
        ):
            # Query-major BM128 reuses KV across prefill rows; its persistent
            # pipeline is distinct from grouped-head decode and preprocessing.
            return "prefill-query-major", dict(num_cores=36)

    aligned_decode = (
        causal
        and call.max_seqlen_q == 1
        and call.seqlen_kv > 0
        and call.seqlen_kv % 128 == 0
        and call.dtype in (torch.float16, torch.bfloat16)
        and not call.is_fp8
        and not call.fuse_rope
    )
    softcap = (
        call.softcap == 50.0 and shape == (32, 4096, 32, 8, 128) and call.dtype == torch.float16
    )
    if aligned_decode and (not call.softcap or softcap):
        options = None
        if shape in {(4, 32768, 64, 8, 128), (8, 32768, 32, 8, 128)}:
            # 32/64 KV-head tasks fill one/two Cube waves without splitting;
            # fuse normalization instead of launching a separate merge.
            options = dict(kv_block=256, splits=1, fuse_output=True)
        elif shape in {(32, 4096, 32, 8, 128), (16, 4096, 64, 8, 128)}:
            options = dict(kv_block=256, fuse_output=configuration(call)[2] == 1)
            if shape == (32, 4096, 32, 8, 128) and not call.softcap:
                # Eight tasks per core form complete rounds, with neighboring
                # cores reading adjacent BSND KV heads.
                options.update(persistent=True, cores=32, kv_phase=1)
        elif shape == (32, 2048, 71, 1, 64):
            # One query tile reuses KV for all 71 heads; the two V cores own
            # 36/35 valid rows and the padded output rows are never stored.
            options = dict(query_block=128, kv_block=128, splits=1, fuse_output=True)
        elif (
            shape[0] == 1
            and shape[1] in (32768, 65536, 131072, 262144)
            and shape[2:] == (32, 4, 128)
            and call.dtype == torch.float16
        ):
            # Nine splits across four KV heads fill 36 Cubes and both V cores.
            # Adjacent blocks read neighboring BSND heads from the same split.
            options = dict(kv_block=128, l0_k=128, splits=9, kv_head_first=True, fuse_output=False)
        elif shape in {(1, 8192, 8, 1, 256), (16, 8192, 8, 1, 256)}:
            # PV column tiles keep both BK128 L0B versions within 64 KiB;
            # split KV across Cubes, then reduce statistics once per head.
            options = dict(
                kv_block=128,
                l0_k=128,
                output_block=128,
                splits=32 if call.batch == 1 else 2,
                fuse_output=False,
            )
        elif shape == (16, 8192, 48, 1, 128) and call.dtype == torch.float16:
            # Reuse KV across all query heads; batch merge rows to avoid
            # reloading the same statistics for each output head.
            options = dict(
                query_block=64, kv_block=128, l0_k=128, splits=2, merge_rows=16, fuse_output=False
            )
        elif shape == (1, 1024, 32, 4, 128) and call.dtype == torch.float16:
            # Short KV fits one fused pass without a separate merge launch.
            options = dict(kv_block=256, splits=1, fuse_output=True)
        elif shape == (1, 4096, 32, 4, 128) and call.dtype == torch.float16:
            # Eight KV iterations per split amortize the pipeline prologue and
            # reduce merge work; the default split count leaves empty tasks.
            options = dict(kv_block=128, splits=4, fuse_output=False)
        elif (
            shape in {(1, 8192, 32, 4, 128), (1, 16384, 32, 4, 128)} and call.dtype == torch.float16
        ):
            # Nine balanced splits fill all 36 Cubes in one resident wave.
            options = dict(kv_block=128, splits=9, fuse_output=False)
        elif shape == (16, 8192, 22, 1, 128) and call.dtype == torch.bfloat16:
            options = dict(kv_block=128, splits=2, fuse_output=False)
        if options is not None:
            if not options["fuse_output"]:
                # The global barrier requires opt-in batch scheduling and a
                # resident grid bounded by the 950DT's 36 physical Cube cores.
                options.update(grid_fuse=True, cores=36)
            return "decode-multibuffer", {"query_block": 32, **options}
        if call.dim in (64, 128) and not call.softcap:
            # These shapes still need a validated multibuffer configuration.
            return "decode-autoschedule", {}
        if call.heads_kv == 1 and call.dim == 256 and not call.softcap:
            # MQA BSND is already contiguous per head: reuse the packed kernel
            # with K/V views rather than a second arithmetic implementation.
            return "packed-attention", dict(kv_view=True)
    # Nonaligned lengths and other semantics use masked packing and attention.
    return "packed-attention", dict(kv_view=False)
