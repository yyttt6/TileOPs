"""Ascend 950 implementations for ErfFwdOp."""

import functools

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.carver.arch.driver import get_num_vector_cores

import tileops.backend
from tileops.kernels.elementwise._nan import keep_nan
from tileops.kernels.elementwise.call_spec import ElementwiseCall, UnaryElementwiseFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel


def _floor_pow2(n: int) -> int:
    """Largest power of two ``<= n`` (``n >= 1``)."""
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


@functools.lru_cache(maxsize=1)
def _calibrated_num_blocks(fallback: int = 64) -> int:
    """Persistent-grid block count for the large, bandwidth-bound path."""
    try:
        queried = get_num_vector_cores(device_id=0)
    except Exception:
        queried = None
    if not queried:
        return fallback
    return _floor_pow2(int(queried))


_TILE_BY_DTYPE = {"float16": 16384, "bfloat16": 16384, "float32": 8192}
_NUM_STAGES_BY_DTYPE = {"float16": 2, "bfloat16": 2, "float32": 2}
_CLAMP = 3.6
_FP16_POLY = (
    10.346251487731934,
    -32.11433792114258,
    42.01287841796875,
    -28.760366439819336,
    11.102203369140625,
    -2.2027556896209717,
    0.3365305960178375,
    0.12998349964618683,
    0.27783286571502686,
)
_BF16_POLY = (
    6.515710353851318,
    -14.437643051147461,
    12.623186111450195,
    -4.944900989532471,
    1.0117061138153076,
    0.07986576855182648,
    0.27838456630706787,
)
_LOCAL_POLY = {"float16": _FP16_POLY, "bfloat16": _BF16_POLY}


def _local_polynomial(wide, dtype):
    """Clamped/scaled Horner in fp32 with the selected 16-bit coefficients."""
    coeffs = _LOCAL_POLY[dtype]
    one = T.cast(1.0, "float32")
    clamped = T.min(T.max(wide, T.cast(-_CLAMP, "float32")), T.cast(_CLAMP, "float32"))
    scaled = clamped * T.cast(1.0 / _CLAMP, "float32")
    w = one - scaled * scaled
    acc = T.cast(coeffs[0], "float32")
    for coeff in coeffs[1:]:
        acc = acc * w + T.cast(coeff, "float32")
    return T.min(T.max(clamped * acc, -one), one)


def _local_erf(x, dtype):
    """op-local 16-bit erf, preserving the shared NaN contract."""
    wide = T.cast(x, "float32")
    return keep_nan(wide, lambda wide: _local_polynomial(wide, dtype))


_FP32_P = 0.3275911
_FP32_COEFFS = (1.061405429, -1.453152027, 1.421413741, -0.284496736, 0.254829592)


def _fp32_body(w):
    zero = T.cast(0.0, "float32")
    one = T.cast(1.0, "float32")
    five = T.cast(5.0, "float32")
    a = T.abs(w)
    t = one / (one + T.cast(_FP32_P, "float32") * a)
    acc = T.cast(_FP32_COEFFS[0], "float32")
    for coeff in _FP32_COEFFS[1:]:
        acc = acc * t + T.cast(coeff, "float32")
    residual = acc * t * T.exp(-(a * a))
    signed = T.if_then_else(w >= zero, one, -one) * (one - residual)
    saturated = T.if_then_else(a >= five, T.if_then_else(w >= zero, one, -one), signed)
    return T.if_then_else(w == zero, w, saturated)


def _fp32_erf(x):
    wide = T.cast(x, "float32")
    body = _fp32_body(wide)
    return T.if_then_else(T.isnan(wide), wide, body)


def _tile_for_dtype(dtype_str: str) -> int:
    return _TILE_BY_DTYPE[dtype_str]


def _stages_for_dtype(dtype_str: str) -> int:
    return _NUM_STAGES_BY_DTYPE[dtype_str]


_SINGLE_BLOCK_THRESHOLD = min(_TILE_BY_DTYPE.values())


def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


@T.macro
def _erf_body(x_ub, y_ub, n: int, threads: int, dtype: str) -> None:
    """``y_ub[:n] = erf(x_ub[:n])`` over ``threads`` lanes."""
    with T.SimtVF(threads=threads):
        for i in T.Parallel(n):
            value = _fp32_erf(x_ub[i]) if dtype == "float32" else _local_erf(x_ub[i], dtype)
            y_ub[i] = T.Cast(dtype, value)


@functools.lru_cache(maxsize=64)
def _make_erf_single_block(n_total: int, dtype: str):
    """One block, one tile, no pipelining: correctness-first fallback for a small shape
    that does not divide ``_calibrated_num_blocks()`` evenly."""

    @T.prim_func
    def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), dtype)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((n_total,), dtype)
            y_ub = T.alloc_shared((n_total,), dtype)
            T.copy(X, x_ub)
            _erf_body(x_ub, y_ub, n_total, threads=min(128, n_total), dtype=dtype)
            T.copy(y_ub, Y)

    return tilelang.compile(main, target="ascend", out_idx=-1)


@functools.lru_cache(maxsize=64)
def _make_erf_small_multi_block(n_total: int, dtype: str, num_blocks: int):
    assert n_total % num_blocks == 0, (
        f"_make_erf_small_multi_block requires n_total ({n_total}) to divide num_blocks ({num_blocks}) exactly; the caller routes other shapes elsewhere"
    )
    chunk = n_total // num_blocks

    @T.prim_func
    def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), dtype)):
        with T.Kernel(num_blocks) as bx:
            x_ub = T.alloc_shared((chunk,), dtype)
            y_ub = T.alloc_shared((chunk,), dtype)
            begin = bx * chunk
            T.copy(X[begin : begin + chunk], x_ub)
            _erf_body(x_ub, y_ub, chunk, threads=min(2048, chunk), dtype=dtype)
            T.copy(y_ub, Y[begin : begin + chunk])

    return tilelang.compile(main, target="ascend", out_idx=-1)


@functools.lru_cache(maxsize=64)
def _make_erf_persistent(n_total: int, dtype: str, num_blocks: int, tile: int, num_stages: int):
    block_span = num_blocks * tile
    assert n_total % block_span == 0, (
        f"_make_erf_persistent requires n_total ({n_total}) to divide num_blocks*tile ({block_span}) exactly; the caller routes other shapes elsewhere"
    )
    iterations = n_total // block_span

    @T.prim_func
    def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), dtype)):
        with T.Kernel(num_blocks) as bx:
            x_ub = T.alloc_shared((tile,), dtype)
            y_ub = T.alloc_shared((tile,), dtype)
            T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})
            for step in T.Pipelined(iterations, num_stages=num_stages):
                begin = (step * num_blocks + bx) * tile
                T.copy(X[begin : begin + tile], x_ub)
                _erf_body(x_ub, y_ub, tile, threads=min(2048, tile), dtype=dtype)
                T.copy(y_ub, Y[begin : begin + tile])

    return tilelang.compile(main, target="ascend", out_idx=-1)


@functools.lru_cache(maxsize=64)
def _make_erf_generic_single_block(n_total: int, dtype: str, tile: int, num_stages: int):
    full_tiles, remainder = divmod(n_total, tile)

    @T.prim_func
    def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), dtype)):
        with T.Kernel(1):
            if full_tiles:
                x_ub = T.alloc_shared((tile,), dtype)
                y_ub = T.alloc_shared((tile,), dtype)
                T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})
                for step in T.Pipelined(full_tiles, num_stages=num_stages):
                    begin = step * tile
                    T.copy(X[begin : begin + tile], x_ub)
                    _erf_body(x_ub, y_ub, tile, threads=min(2048, tile), dtype=dtype)
                    T.copy(y_ub, Y[begin : begin + tile])
            if remainder:
                xt_ub = T.alloc_shared((remainder,), dtype)
                yt_ub = T.alloc_shared((remainder,), dtype)
                begin = full_tiles * tile
                T.copy(X[begin : begin + remainder], xt_ub)
                _erf_body(xt_ub, yt_ub, remainder, threads=min(128, remainder), dtype=dtype)
                T.copy(yt_ub, Y[begin : begin + remainder])

    return tilelang.compile(main, target="ascend", out_idx=-1)


def _make_erf_kernel(n_total: int, dtype: str):
    """Pick the tiling for ``n_total`` elements: small multi-block, single-block,
    persistent, or fallback."""
    if n_total <= _SINGLE_BLOCK_THRESHOLD:
        num_blocks = _calibrated_num_blocks()
        if n_total % num_blocks == 0:
            return _make_erf_small_multi_block(n_total, dtype, num_blocks)
        return _make_erf_single_block(n_total, dtype)
    tile = _tile_for_dtype(dtype)
    num_blocks = _calibrated_num_blocks()
    block_span = num_blocks * tile
    if n_total % block_span == 0:
        return _make_erf_persistent(n_total, dtype, num_blocks, tile, _stages_for_dtype(dtype))
    return _make_erf_generic_single_block(n_total, dtype, tile, _stages_for_dtype(dtype))


class AscendErfKernel(Kernel, UnaryElementwiseFwdInterface):
    """Erf on Ascend NPUs: ``y = erf(x)``; op-local fp32 rational and 16-bit fits."""

    devices = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call: ElementwiseCall) -> Entry:
        return (call, lambda: cls(call.n_total, call.dtype))

    def __init__(self, n_total: int, dtype: torch.dtype):
        super().__init__()
        self.n_total = n_total
        self.dtype = dtype
        self._compiled_fn = _make_erf_kernel(n_total, _dtype_str(dtype))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        flat = input.reshape(-1)
        result = self._compiled_fn(flat)
        return result.reshape(input.shape)


tileops.backend.register_kernel_type("ErfFwdOp", "ascend", AscendErfKernel)
