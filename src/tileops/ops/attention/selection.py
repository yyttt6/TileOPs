"""What an attention request may and may not say.

The slot names each attention op asks its target for, and the check that a
request does not contradict the user-visible ``backend`` parameter. Which
implementation then serves the slot is the target's answer, not this module's.
"""

from .call_spec import AttentionCall, fp8_dtype, uses_sliding_window

__all__ = [
    "DECODE_SLOT",
    "DENSE_PREFILL_SLOT",
    "MHA_PAGED_DECODE_SLOT",
    "PACKED_PREFILL_SLOT",
    "PAGED_DECODE_SLOT",
    "PAGED_PREFILL_SLOT",
    "AttentionCall",
    "check_packed_prefill_request",
    "fp8_dtype",
]

#: The slot each attention op asks its target for. One name per computation, not per
#: implementation: which schedule serves a shape is the target's decision, made inside
#: its ``build_kernel`` where the shapes and dtypes are, so the op layer names the
#: computation and stops there.
PACKED_PREFILL_SLOT = "gqa_packed_prefill"
DENSE_PREFILL_SLOT = "gqa_dense_prefill"
PAGED_PREFILL_SLOT = "gqa_paged_prefill"
DECODE_SLOT = "gqa_decode"
PAGED_DECODE_SLOT = "gqa_paged_decode"
MHA_PAGED_DECODE_SLOT = "mha_paged_decode"


def check_packed_prefill_request(call: AttentionCall) -> None:
    """Reject a packed prefill request its ``backend`` knob cannot describe.

    This is the user-visible contract of the parameter, not a statement about
    any implementation: it says what the caller asked for is not what the
    request is. Which implementation then runs is decided by capability.

    Raises:
        ValueError: When ``backend`` contradicts the request.
    """
    if call.is_fp8:
        if call.backend not in ("auto", "fp8"):
            raise ValueError("FP8 prefill requires backend='auto' or backend='fp8'.")
        if call.is_causal:
            raise ValueError("FP8 prefill currently supports non-causal prefill only.")
        if uses_sliding_window(call):
            raise ValueError("FP8 prefill does not support sliding-window dispatch.")
        if call.max_seqlen_q != call.max_seqlen_kv:
            raise ValueError("FP8 prefill requires max_seqlen_q == max_seqlen_kv.")
        if not call.is_uniform:
            raise ValueError("FP8 prefill requires uniform packed cu_seqlens.")
        return
    if call.backend == "fp8":
        raise ValueError("backend='fp8' requires float8_e4m3fn q/k/v.")
    if uses_sliding_window(call):
        if call.backend not in ("auto", "sliding_window"):
            raise ValueError(
                "sliding-window prefill requires backend='auto' or backend='sliding_window'."
            )
        return
    if call.backend == "sliding_window":
        raise ValueError("backend='sliding_window' requires window_size_left or window_size_right.")
    if call.backend == "dense" and not call.is_uniform:
        raise ValueError("backend='dense' requires uniform packed cu_seqlens.")
