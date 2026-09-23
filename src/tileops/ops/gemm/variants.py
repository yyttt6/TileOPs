"""T301C GEMM variants; correctness-first compositions without autotuning.

Derived from tileops/kernels/gemm.py (Cube contractions) and the staging ideas
in tilelang-ascend/examples/deepseek_v4/int8_gemm.py (MIT, Copyright (c)
Tile-AI Corporation). See _variants_kernels.py and NOTICE. Examples were read
as text. Tensor views/copies and HCCL collectives are host orchestration;
all matrix contractions and numerical epilogues use TileLang kernels.

FP4 storage uses low-nibble-first E2M1, packed along the last logical axis.
E8M0 is stored as uint8 exponent bits. Sparse metadata stores two ordered
2-bit positions in each nibble, four nibbles per uint16. These conventions
are explicit because op-list-150 specifies types/shapes but not encodings.
"""

from functools import partial
from math import prod

import torch

from ..op_base import Op


def variant_roofline(op):
    """Semantic contractions and tensor I/O, excluding implementation scratch."""
    return op._flops, op._bytes


def _typename(dtype):
    return str(dtype).removeprefix("torch.")


def _matrix_dims(a, b, trans_b):
    if len(a.shape) != 2 or len(b.shape) != 2:
        raise ValueError("GEMM inputs must be matrices")
    m, k = a.shape
    n, kb = b.shape if trans_b else (b.shape[1], b.shape[0])
    if k != kb or min(m, n, k) <= 0:
        raise ValueError("invalid contraction extents")
    return m, n, k


class _Variant(Op):
    def __init__(self, trans_a=False, trans_b=False, target="ascend", tune=False):
        if trans_a:
            raise ValueError("T301C variants require non-transposed A")
        if tune:
            raise ValueError("T301C does not enable tuning")
        self.trans_a = False
        self.trans_b = bool(trans_b)
        self.target = target
        self.tune = False
        _register_variants()
        self.dispatch_kernel()

    def _run(self, inputs, shapes, flops, out_dtype=None):
        self._validate_dtypes(*inputs)
        if len({x.device for x in inputs}) != 1:
            raise ValueError("all inputs must share a device")
        if any(not x.is_contiguous() for x in inputs):
            raise ValueError(
                "input tensors must be contiguous; use explicit batch strides"
            )
        self.dtype = out_dtype or inputs[0].dtype
        self._flops = int(flops)
        self._bytes = (
            sum(x.numel() * x.element_size() for x in inputs)
            + sum(prod(s) for s in shapes)
            * torch.empty((), dtype=self.dtype).element_size()
        )
        return self.get_or_build_kernel("variant", inputs)(*inputs)

    def _cache_key(self, *shapes):
        return shapes

    def compute_roof(self):
        from tileops.perf.profile import cube_roof

        return cube_roof(self.dtype)


class StridedBatchedGemmFwdOp(_Variant):
    """Batched A @ B with explicit element strides; stride_b=0 broadcasts B."""

    def __init__(self, stride_a=None, stride_b=0, target="ascend", tune=False):
        self.stride_a = stride_a
        self.stride_b = stride_b
        super().__init__(target=target, tune=tune)

    def _infer_output_shapes(self, a_shape, b_shape):
        return {"c": (a_shape[0], a_shape[1], b_shape[-1])}

    def forward(self, a, b):
        if a.ndim != 3 or b.ndim not in (2, 3) or a.shape[2] != b.shape[-2]:
            raise ValueError("expected A[B,M,K], B[K,N] (or backing B[B,K,N])")
        if self.stride_a is None:
            self.stride_a = a.shape[1] * a.shape[2]
        batch, m, k = a.shape
        n = b.shape[-1]
        if min(batch, m, n, k) <= 0 or self.stride_a < 0 or self.stride_b < 0:
            raise ValueError("positive dimensions and nonnegative strides required")
        if (batch - 1) * self.stride_a + m * k > a.numel() or (
            batch - 1
        ) * self.stride_b + k * n > b.numel():
            raise ValueError("batch strides exceed the supplied backing tensors")
        return self._run((a, b), [(batch, m, n)], 2 * batch * m * n * k)


class GemvFwdOp(_Variant):
    """Matrix/vector product A[M,K] @ x[K] -> y[M]."""

    def __init__(self, target="ascend", tune=False):
        super().__init__(target=target, tune=tune)

    def _infer_output_shapes(self, a_shape, x_shape):
        return {"y": (a_shape[0],)}

    def forward(self, a, x):
        if a.ndim != 2 or x.ndim != 1 or a.shape[1] != x.shape[0]:
            raise ValueError("expected A[M,K], x[K]")
        return self._run((a, x), [(a.shape[0],)], 2 * a.numel())


class GemmInt8FwdOp(_Variant):
    """INT8 x INT8 -> INT32, fp32 row/column scaling, then output cast."""

    def __init__(
        self,
        trans_a=False,
        trans_b=False,
        out_dtype=torch.bfloat16,
        target="ascend",
        tune=False,
    ):
        if out_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("output must be float16 or bfloat16")
        self.out_dtype = out_dtype
        super().__init__(trans_a, trans_b, target, tune)

    def _infer_output_shapes(self, a_shape, b_shape, a_scale_shape, b_scale_shape):
        return {"c": (a_shape[0], b_shape[0] if self.trans_b else b_shape[1])}

    def forward(self, a, b, a_scale, b_scale):
        m, n, k = _matrix_dims(a, b, self.trans_b)
        if a_scale.shape != (m,) or b_scale.shape != (n,):
            raise ValueError("scales must have shapes [M] and [N]")
        if k > 131071:
            raise ValueError("K exceeds conservative INT32 accumulation bound")
        return self._run(
            (a, b, a_scale, b_scale), [(m, n)], 2 * m * n * k, self.out_dtype
        )

    def compute_roof(self):
        # The current profile has no INT8 rate. Fail explicitly instead of
        # mispricing the native INT8 contraction at its output's bf16 rate.
        from tileops.perf.profile import cube_roof

        return cube_roof(torch.int8)


class GemmA16W8FwdOp(_Variant):
    """Floating A times INT8 B, followed by per-column scale in fp32."""

    def _infer_output_shapes(self, a_shape, b_q_shape, scale_shape):
        return {"c": (a_shape[0], b_q_shape[0] if self.trans_b else b_q_shape[1])}

    def forward(self, a, b_q, scale):
        m, n, k = _matrix_dims(a, b_q, self.trans_b)
        if scale.shape != (n,):
            raise ValueError("scale must have shape [N]")
        return self._run((a, b_q, scale), [(m, n)], 2 * m * n * k)


class GemmSparse2to4FwdOp(_Variant):
    """Expand the declared ordered-index 2:4 layout, then contract densely."""

    def _infer_output_shapes(self, a_c_shape, meta_shape, b_shape):
        return {"c": (a_c_shape[0], b_shape[0] if self.trans_b else b_shape[1])}

    def forward(self, a_c, meta, b):
        if a_c.ndim != 2 or b.ndim != 2:
            raise ValueError("expected compressed matrix and dense matrix")
        m, k = a_c.shape[0], a_c.shape[1] * 2
        n, kb = b.shape if self.trans_b else (b.shape[1], b.shape[0])
        if kb != k or k % 16 or meta.shape != (m, k // 16):
            raise ValueError("requires K multiple of 16 and metadata [M,K/16]")
        return self._run((a_c, meta, b), [(m, n)], m * n * k)


class GemmEpilogueSwiGLUFwdOp(_Variant):
    """Split A@B into gate/up halves; return silu(gate)*up."""

    def _infer_output_shapes(self, a_shape, b_shape):
        return {"c": (a_shape[0], (b_shape[0] if self.trans_b else b_shape[1]) // 2)}

    def forward(self, a, b):
        m, nn, k = _matrix_dims(a, b, self.trans_b)
        if nn % 2:
            raise ValueError("B output dimension must contain two equal halves")
        return self._run((a, b), [(m, nn // 2)], 2 * m * nn * k)


class DualGemmFwdOp(_Variant):
    """Two independent contractions sharing A; returns both products."""

    def _infer_output_shapes(self, a_shape, b0_shape, b1_shape):
        return {
            "c0": (a_shape[0], b0_shape[0] if self.trans_b else b0_shape[1]),
            "c1": (a_shape[0], b1_shape[0] if self.trans_b else b1_shape[1]),
        }

    def forward(self, a, b0, b1):
        m, n, k = _matrix_dims(a, b0, self.trans_b)
        if b0.shape != b1.shape:
            raise ValueError("B0 and B1 must share a shape")
        return self._run((a, b0, b1), [(m, n), (m, n)], 4 * m * n * k)


class GemmReduceScatterFwdOp(_Variant):
    """Local TileLang GEMM then HCCL SUM reduce-scatter along M, real ranks."""

    def __init__(
        self, world_size=2, trans_a=False, trans_b=False, target="ascend", tune=False
    ):
        self.world_size = world_size
        super().__init__(trans_a, trans_b, target, tune)

    def _infer_output_shapes(self, a_shape, b_shape):
        return {
            "c": (
                a_shape[0] // self.world_size,
                b_shape[0] if self.trans_b else b_shape[1],
            )
        }

    def forward(self, a, b):
        _check_group(self.world_size)
        m, n, k = _matrix_dims(a, b, self.trans_b)
        if m % self.world_size:
            raise ValueError("M must be divisible by world_size")
        return self._run((a, b), [(m // self.world_size, n)], 2 * m * n * k)


class AllGatherGemmFwdOp(_Variant):
    """HCCL gather A along M, then local TileLang GEMM on each rank."""

    def __init__(
        self, world_size=2, trans_a=False, trans_b=False, target="ascend", tune=False
    ):
        self.world_size = world_size
        super().__init__(trans_a, trans_b, target, tune)

    def _infer_output_shapes(self, a_shape, b_shape):
        return {
            "c": (
                a_shape[0] * self.world_size,
                b_shape[0] if self.trans_b else b_shape[1],
            )
        }

    def forward(self, a, b):
        _check_group(self.world_size)
        m, n, k = _matrix_dims(a, b, self.trans_b)
        return self._run(
            (a, b), [(m * self.world_size, n)], 2 * m * self.world_size * n * k
        )


class GemmBlockScaledFwdOp(_Variant):
    """Packed FP4 E2M1 with K-block32 E8M0 scales, accumulated in fp32.

    A is [M,K/2] bytes. B is [N,K/2] when trans_b, otherwise [K,N/2].
    Scales are [M,K/32] and [N,K/32] (trans_b) or [K/32,N] (NN).
    Output defaults to bf16, with fp16 as an explicit precision extension.
    """

    def __init__(
        self,
        trans_a=False,
        trans_b=False,
        out_dtype=torch.bfloat16,
        target="ascend",
        tune=False,
    ):
        if out_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("output must be float16 or bfloat16")
        self.out_dtype = out_dtype
        super().__init__(trans_a, trans_b, target, tune)

    def _infer_output_shapes(self, a_q_shape, a_scale_shape, b_q_shape, b_scale_shape):
        return {"c": (a_q_shape[0], b_q_shape[0] if self.trans_b else b_q_shape[1] * 2)}

    def forward(self, a_q, a_scale, b_q, b_scale):
        if a_q.ndim != 2 or b_q.ndim != 2:
            raise ValueError("FP4 backing tensors must be matrices")
        m, k = a_q.shape[0], a_q.shape[1] * 2
        n, kb = (
            (b_q.shape[0], b_q.shape[1] * 2)
            if self.trans_b
            else (b_q.shape[1] * 2, b_q.shape[0])
        )
        bs = (n, k // 32) if self.trans_b else (k // 32, n)
        if k != kb or k % 32 or a_scale.shape != (m, k // 32) or b_scale.shape != bs:
            raise ValueError("invalid FP4 packed shapes or block32 scales")
        return self._run(
            (a_q, a_scale, b_q, b_scale), [(m, n)], 2 * m * n * k, self.out_dtype
        )

    def compute_roof(self):
        # FP4 values are unpacked to exact fp16 before the Cube contraction.
        from tileops.perf.profile import cube_roof

        return cube_roof(torch.float16)


def _check_group(world_size):
    import torch.distributed as dist

    if (
        world_size < 2
        or not dist.is_initialized()
        or dist.get_world_size() != world_size
    ):
        raise ValueError(
            "a real initialized process group matching world_size>=2 is required"
        )


def _build(name, *specs, **params):
    from . import _variants_kernels as vk
    from tileops.kernels.gemm import build_gemm_kernel, build_bmm_kernel

    a, b = specs[:2]
    dtype = _typename(a.dtype)
    tb = params.get("trans_b", False)
    if name == "StridedBatchedGemmFwdOp":
        batch, m, k = a.shape
        n = b.shape[-1]
        kernel = build_bmm_kernel((batch, m, k), (batch, k, n), a.dtype)

        def call(x, w):
            aa = torch.as_strided(
                x, (batch, m, k), (params["stride_a"], k, 1)
            ).contiguous()
            bb = torch.as_strided(
                w, (batch, k, n), (params["stride_b"], n, 1)
            ).contiguous()
            return kernel(aa, bb)

        return call
    if name == "GemvFwdOp":
        m, k = a.shape
        kernel = build_gemm_kernel((m, k), (k, 1), a.dtype, False, False)
        return lambda x, v: kernel(x, v.view(k, 1)).view(m)
    if name == "GemmSparse2to4FwdOp":
        # specs[1] is metadata, specs[2] is dense B.
        b = specs[2]
        m, k = a.shape[0], a.shape[1] * 2
        decode = vk.sparse_expand(m, k, dtype)
        dense = build_gemm_kernel((m, k), tuple(b.shape), a.dtype, False, tb)
        return lambda ac, meta, w: dense(decode(ac, meta), w)
    if name == "GemmBlockScaledFwdOp":
        b = specs[2]
        m, k = a.shape[0], a.shape[1] * 2
        n = b.shape[0] if tb else b.shape[1] * 2
        decode_a = vk.unpack_fp4(m * k)
        decode_b = vk.unpack_fp4(n * k)
        product = vk.dense_accum(m, n, 32, "float16", tb)
        accumulate = vk.accumulate_fp4_block(m, n, k // 32)
        cast = vk.cast_flat(m * n, "float32", _typename(params["out_dtype"]))

        def call(aq, sa, bq, sb):
            aa = decode_a(aq.view(-1)).view(m, k)
            bb = decode_b(bq.view(-1)).view((n, k) if tb else (k, n))
            ss = sb.T.contiguous() if tb else sb
            acc = torch.empty((m, n), dtype=torch.float32, device=aq.device)
            for group in range(k // 32):
                start = group * 32
                x = aa[:, start : start + 32].contiguous()
                w = (
                    bb[:, start : start + 32] if tb else bb[start : start + 32, :]
                ).contiguous()
                accumulate(product(x, w), sa, ss, acc, group)
            return cast(acc.view(-1)).view(m, n)

        return call
    m, n, k = _matrix_dims(a, b, tb)
    if name == "DualGemmFwdOp":
        kernel = build_gemm_kernel(tuple(a.shape), tuple(b.shape), a.dtype, False, tb)
        return lambda x, w0, w1: (kernel(x, w0), kernel(x, w1))
    if name == "GemmEpilogueSwiGLUFwdOp":
        # Uniform compensated accumulation fixes near-zero gates observed
        # against fp64. This intentionally costs extra launches, not tuning.
        chunk = 256
        kernels = {
            length: vk.dense_accum(m, n, length, dtype, tb)
            for length in {min(chunk, k), k % chunk}
            if length
        }
        add = vk.compensated_add(m * n)
        epi = vk.swiglu(m, n // 2, dtype)

        def call(x, w):
            total = torch.empty((m, n), dtype=torch.float32, device=x.device)
            correction = torch.empty_like(total)
            for start in range(0, k, chunk):
                length = min(chunk, k - start)
                xx = x[:, start : start + length].contiguous()
                ww = (
                    w[:, start : start + length] if tb else w[start : start + length, :]
                ).contiguous()
                part = kernels[length](xx, ww)
                add(part.view(-1), total.view(-1), correction.view(-1), int(start == 0))
            return epi(total)

        return call
    if name == "GemmInt8FwdOp":
        kernel = vk.dense_accum(m, n, k, "int8", tb)
        epi = vk.scaled_output(m, n, "int32", _typename(params["out_dtype"]), True)
        return lambda x, w, sa, sb: epi(kernel(x, w), sa, sb)
    if name == "GemmA16W8FwdOp":
        cast = vk.cast_flat(k * n, "int8", dtype)
        scast = vk.cast_flat(n, dtype, "float32")
        kernel = vk.dense_accum(m, n, k, dtype, tb)
        epi = vk.scaled_output(m, n, "float32", dtype, False)

        def call(x, w, s):
            ww = cast(w.view(-1)).view(w.shape)
            ss = scast(s)
            # row-scale input is unused for A16W8; provide an existing fp32 view.
            return epi(
                kernel(x, ww),
                torch.empty((m,), dtype=torch.float32, device=x.device),
                ss,
            )

        return call
    world = params.get("world_size", 2)
    if name == "GemmReduceScatterFwdOp":
        kernel = vk.dense_accum(m, n, k, dtype, tb)
        cast = vk.cast_flat(m // world * n, "float32", dtype)

        def call(x, w):
            import torch.distributed as dist

            local = kernel(x, w)
            out = torch.empty((m // world, n), dtype=torch.float32, device=x.device)
            dist.reduce_scatter_tensor(out, local)
            return cast(out.view(-1)).view(m // world, n)

        return call
    if name == "AllGatherGemmFwdOp":
        kernel = build_gemm_kernel((m * world, k), tuple(b.shape), a.dtype, False, tb)

        def call(x, w):
            import torch.distributed as dist

            gathered = torch.empty((m * world, k), dtype=x.dtype, device=x.device)
            dist.all_gather_into_tensor(gathered, x)
            return kernel(gathered, w)

        return call
    raise ValueError(f"unknown variant {name}")


_REGISTERED = False


def _register_variants():
    global _REGISTERED
    if _REGISTERED:
        return
    from tileops.kernels._registry import register

    for cls in (
        StridedBatchedGemmFwdOp,
        GemvFwdOp,
        GemmInt8FwdOp,
        GemmA16W8FwdOp,
        GemmSparse2to4FwdOp,
        GemmEpilogueSwiGLUFwdOp,
        DualGemmFwdOp,
        GemmReduceScatterFwdOp,
        AllGatherGemmFwdOp,
        GemmBlockScaledFwdOp,
    ):
        register(cls.__name__)(partial(_build, cls.__name__))
    _REGISTERED = True
