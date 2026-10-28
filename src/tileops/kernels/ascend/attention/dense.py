"""Ascend dense GQA dispatch over preprocessing, decode and prefill dataflows."""

import tilelang
import torch

import tileops.backend
from tileops.kernels.ascend.attention.common import pass_configs
from tileops.kernels.ascend.attention.config import configuration, select_kernel
from tileops.kernels.ascend.attention.decode_autoschedule import decode_autoschedule
from tileops.kernels.ascend.attention.decode_multibuffer import (
    decode_multibuffer,
    grid_pass_configs,
)
from tileops.kernels.ascend.attention.fallback import (
    attention_merge,
    attention_partial,
    pack_operand,
)
from tileops.kernels.ascend.attention.prefill_preprocess import (
    prefill_preprocess,
    preprocess_pass_configs,
)
from tileops.kernels.ascend.attention.prefill_query_major import prefill_query_major
from tileops.kernels.attention.call_spec import GQADenseFwdInterface
from tileops.kernels.kernel_base import Kernel


class AscendGQADenseKernel(Kernel, GQADenseFwdInterface):
    devices = frozenset({"npu"})

    @classmethod
    def refusal(cls, call):
        if call.dtype not in (torch.float16, torch.bfloat16):
            return "requires float16 or bfloat16 output"
        if call.dim not in (64, 128, 256):
            return "requires head dimension 64, 128 or 256"
        return None

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__()
        self.call = call
        variant, options = select_kernel(call)
        if options.get("grid_fuse"):
            device = call.device if call.device is not None else torch.npu.current_device()
            cube_cores = int(torch.npu.get_device_properties(device).cube_core_num)
            if cube_cores <= 0:
                raise RuntimeError("cross-block GQA requires a positive physical Cube count")
            # A global barrier must not wait for blocks outside the resident
            # grid. Smaller devices keep the same work loop with fewer blocks.
            options = {**options, "cores": min(options["cores"], cube_cores)}
        self.config = {
            **dict(zip(("query_rows", "kv_rows", "kv_splits"), configuration(call), strict=True)),
            "variant": variant,
            **options,
        }
        self.fused = None
        self.grid_fused = options.get("grid_fuse", False)
        self.pack = None
        self.kv_view = options.get("kv_view", False)
        if variant == "prefill-preprocess":
            self.fused = tilelang.compile(
                prefill_preprocess(call, **options),
                target="ascend",
                out_idx=-1,
                pass_configs=preprocess_pass_configs(call),
            )
            return
        if variant == "prefill-query-major":
            self.fused = tilelang.compile(
                prefill_query_major(call, **options),
                target="ascend",
                out_idx=-1,
                pass_configs=pass_configs(),
            )
            return
        if variant == "decode-multibuffer":
            self.config.update(multibuffer=True, balanced_vectors=True)
            if "splits" in options:
                self.config["kv_splits"] = options["splits"]
            assert options["fuse_output"] or self.grid_fused
            self.fused = tilelang.compile(
                decode_multibuffer(call, **options),
                target="ascend",
                # Allocate scratch tensors with the output, then execute one
                # kernel containing both split attention and normalization.
                out_idx=[3, 4, 5] if self.grid_fused else -1,
                pass_configs=grid_pass_configs() if self.grid_fused else pass_configs(),
            )
            return
        elif variant == "decode-autoschedule":
            self.partial = tilelang.compile(
                decode_autoschedule(call), target="ascend", out_idx=[3, 4]
            )
        else:
            operands = ("q",) if self.kv_view else ("q", "k", "v")
            self.pack = tuple(
                tilelang.compile(pack_operand(call, kind), target="ascend", out_idx=-1)
                for kind in operands
            )
            self.partial = tilelang.compile(
                attention_partial(call), target="ascend", out_idx=[3, 4]
            )
        self.merge = tilelang.compile(
            attention_merge(call, splits=self.config["kv_splits"]),
            target="ascend",
            out_idx=-1,
        )

    def forward(
        self, q, k, v, q_scale=None, k_scale=None, v_scale=None, rope_cos=None, rope_sin=None
    ):
        call = self.call
        if self.fused is not None and self.config["variant"] != "prefill-preprocess":
            result = self.fused(q, k, v)
            return result[-1] if self.grid_fused else result
        if self.fused is None and self.pack is None:
            return self.merge(*self.partial(q, k, v))

        dummy_scale = torch.empty((call.batch, call.heads_kv), dtype=torch.float32, device=q.device)
        dummy_rope = torch.empty((1, 1), dtype=call.dtype, device=q.device)
        scales = tuple(dummy_scale if x is None else x for x in (q_scale, k_scale, v_scale))
        cosine = dummy_rope if rope_cos is None else rope_cos
        sine = dummy_rope if rope_sin is None else rope_sin
        if self.fused is not None:
            return self.fused(q, k, v, *scales, cosine, sine)
        if self.kv_view:
            query = self.pack[0](q, dummy_scale, dummy_rope, dummy_rope)
            shape = (call.batch, call.seqlen_kv, call.dim)
            return self.merge(*self.partial(query, k.view(shape), v.view(shape)))
        packed = []
        for j, (fn, x, scale) in enumerate(zip(self.pack, (q, k, v), scales, strict=True)):
            packed.append(
                fn(
                    x.view(torch.uint8) if call.is_fp8 else x,
                    scale,
                    dummy_rope if j == 2 else cosine,
                    dummy_rope if j == 2 else sine,
                )
            )
        return self.merge(*self.partial(*packed))


tileops.backend.register_kernel_type("GQADenseFwdOp", "ascend", AscendGQADenseKernel)
