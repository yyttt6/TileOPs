"""The facts of one attention call.

``AttentionCall`` is what an attention op states about a call: op state plus
what only the call knows -- the element type, whether the packed ranges are
uniform, whether the inputs are FP8. ``selection`` reads it to reject a request
its ``backend`` knob cannot describe.
"""

import dataclasses
from typing import Optional

import torch

from .._call_spec import CallSpec

__all__ = [
    "ATTENTION_DTYPES",
    "AttentionCall",
    "fp8_dtype",
    "uses_sliding_window",
]

ATTENTION_DTYPES = (torch.float16, torch.bfloat16)


def fp8_dtype() -> Optional[torch.dtype]:
    """Return ``torch.float8_e4m3fn`` when the torch build carries it.

    910B1 has no native FP8 unit, so an FP8 request is a storage format the op
    layer recognises rather than a compute path it promises; whether a target
    serves it is that target's ``build_kernel`` to answer.
    """
    return getattr(torch, "float8_e4m3fn", None)


@dataclasses.dataclass(frozen=True)
class AttentionCall(CallSpec):
    """What one attention call is, as the op knows it."""

    dtype: Optional[torch.dtype] = None
    batch: int = 0
    heads: int = 0
    heads_kv: int = 0
    dim: int = 0
    max_seqlen_q: int = 0
    max_seqlen_kv: int = 0
    seqlen_kv: int = 0
    page_size: int = 0
    max_pages_per_req: int = 0
    is_causal: bool = False
    sm_scale: Optional[float] = None
    softcap: float = 0.0
    window_size_left: int = -1
    window_size_right: int = -1
    backend: str = "auto"
    is_fp8: bool = False
    is_uniform: bool = True
    cache_dtype: Optional[torch.dtype] = None
    fuse_rope: bool = False
    max_position: Optional[int] = None
    rotary_dim: Optional[int] = None
    accum_dtype: torch.dtype = torch.float32
    tune: bool = False


def uses_sliding_window(call: AttentionCall) -> bool:
    """Whether either window bound is set, which narrows what may serve the call."""
    return call.window_size_left != -1 or call.window_size_right != -1
