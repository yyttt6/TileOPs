"""Ascend GELU with exact-erf and tanh modes and bounded arbitrary tails."""

import functools

import torch

from tileops.backend import register_kernel_type
from tileops.kernels.elementwise._erf import gelu as _exact_gelu
from tileops.kernels.elementwise.call_spec import GeluCall, UnaryElementwiseFwdInterface
from tileops.kernels.kernel_base import Kernel

__all__ = ["AscendGeluKernel"]

try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:  # pragma: no cover - exercised only where the Ascend toolchain is absent
    pass
else:

    def gelu(x, dtype, approximate):
        if approximate == "none":
            return _exact_gelu(x, dtype)
        wide = T.cast(x, "float32")
        argument = T.cast(0.7978845608028654, "float32") * (
            wide + T.cast(0.044715, "float32") * wide * wide * wide
        )
        return T.cast(0.5, "float32") * wide * (T.cast(1.0, "float32") + T.tanh(argument))

    # Tiling, round_1_schedule.md §3.3: anchored on the one proven-working Ascend elementwise
    # example (``tilelang/examples/ascend/example_vecadd.py``: num_blocks=64, tile=8192,
    # num_stages=2, num_threads=2048), not a cited hardware UB-capacity figure (none was found
    # in docs/ascend950/*.md or the tilelang-ascend skill docs this round).
    #
    # Prefill uses the full physical 64-AIV grid. Decode selects fewer active cores
    # from its workload size to avoid the launch overhead of many tiny blocks.
    _NUM_BLOCKS = 64
    _TILE = 8192
    _NUM_STAGES = 2
    _THREADS = 2048
    # Both prefill cases (29,360,128 elements) divide this exactly (56 iterations, no tail).
    # The decode case (14,336 elements) is far smaller than one pipelined block's tile and
    # is instead split into vector-aligned blocks by ``_build_decode_multicore`` below
    # (round_2_schedule.md §2 -- round 1's single-core ``_build_single_shot`` path left 63 of
    # 64 AIV cores idle and missed its target by 2.02x).

    # Ceiling for the single-shot (no-pipelining, single-core) fallback path: total GM<->UB
    # traffic at 4 bytes/elem (fp16/bf16 in + out) stays a small fraction of the ~192 KiB
    # envelope example_vecadd.py's own (3-buffer, fp32, double-buffered) footprint already
    # demonstrates works. Also used as the upper bound for the decode-multicore path's
    # per-call total (split across cores, so each core's actual footprint is far smaller).
    _SINGLE_SHOT_MAX_ELEMENTS = 65536

    def _mode_for(n_total: int) -> str:
        """Choose pipelined, small multicore, bounded single-shot, or general tail path."""
        if n_total % (_NUM_BLOCKS * _TILE) == 0:
            return "pipelined"
        if n_total <= _SINGLE_SHOT_MAX_ELEMENTS and n_total % _NUM_BLOCKS == 0:
            return "decode_multicore"
        if n_total <= 16384:
            return "single_shot"
        return "tail"

    def _decode_blocks(n_total: int) -> int:
        """Keep complete vectors per core and about 2,048 elements per block."""
        cap = min(_NUM_BLOCKS, max(1, (n_total + 2047) // 2048))
        for cores in range(cap, 0, -1):
            if n_total % (cores * 64) == 0:
                return cores
        return 1

    @functools.lru_cache(maxsize=None)
    def _build_pipelined(
        n_total: int, dtype_str: str, num_blocks: int, tile: int, num_stages: int, approximate: str
    ):
        """Multi-block, ``T.Pipelined`` GELU kernel for the two prefill cases."""
        assert n_total % (num_blocks * tile) == 0, (
            f"_build_pipelined requires n_total={n_total} to divide evenly into "
            f"num_blocks={num_blocks} x tile={tile}"
        )
        iterations = n_total // (num_blocks * tile)
        threads = min(tile, _THREADS)

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
            with T.Kernel(num_blocks) as bx:
                x_ub = T.alloc_shared((tile,), dtype_str)
                y_ub = T.alloc_shared((tile,), dtype_str)
                T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})
                for step in T.Pipelined(iterations, num_stages=num_stages):
                    begin = (step * num_blocks + bx) * tile
                    T.copy(X[begin : begin + tile], x_ub)
                    with T.SimtVF(threads=threads):
                        for i in T.Parallel(tile):
                            # Promote to fp32 inside the per-lane body (code-style.md), not at
                            # the UB-buffer level: _erf.gelu casts to fp32 internally and
                            # returns an fp32 PrimExpr; cast back to storage dtype at the
                            # boundary.
                            y_ub[i] = T.cast(gelu(x_ub[i], dtype_str, approximate), dtype_str)
                    T.copy(y_ub, Y[begin : begin + tile])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=None)
    def _build_single_shot(n_total: int, dtype_str: str, approximate: str):
        """Small single-block path, limited to 16384 elements for the FP32 UB budget."""
        threads = min(n_total, _THREADS)

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
            with T.Kernel(1):
                x_ub = T.alloc_shared((n_total,), dtype_str)
                y_ub = T.alloc_shared((n_total,), dtype_str)
                T.copy(X, x_ub)
                with T.SimtVF(threads=threads):
                    for i in T.Parallel(n_total):
                        y_ub[i] = T.cast(gelu(x_ub[i], dtype_str, approximate), dtype_str)
                T.copy(y_ub, Y)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=None)
    def _build_decode_multicore(n_total: int, dtype_str: str, num_cores: int, approximate: str):
        """Grid-of-``num_cores`` GELU kernel for a small shape that divides evenly by
        ``num_cores``; round 2's fix for the decode case (round_2_schedule.md §2).

        Every core gets exactly one statically-sized ``block = n_total // num_cores``-element
        tile -- no re-looping, hence a plain ``T.Kernel(num_cores) as core_id:`` body rather
        than ``T.Persistent`` (round_2_schedule.md §2 point 2: with exactly one tile per
        core, ``T.Persistent``'s general re-looping machinery degenerates to exactly what
        this direct form already expresses, so the simpler form is used). Structurally this
        ports ``AscendGeluAndMulKernel``'s decode-case grid shape
        (``gelu_and_mul.py``'s ``_gelu_and_mul_program`` with ``m=1``), not its math -- the
        GELU polynomial itself is still the shared, unmodified ``_erf.gelu`` import.
        """
        assert n_total % num_cores == 0, (
            f"_build_decode_multicore requires n_total={n_total} to divide evenly by "
            f"num_cores={num_cores}; _mode_for should not route here otherwise"
        )
        block = n_total // num_cores
        threads = min(block, _THREADS)

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
            with T.Kernel(num_cores) as core_id:
                x_ub = T.alloc_shared((block,), dtype_str)
                y_ub = T.alloc_shared((block,), dtype_str)
                begin = core_id * block
                T.copy(X[begin : begin + block], x_ub)
                with T.SimtVF(threads=threads):
                    for i in T.Parallel(block):
                        y_ub[i] = T.cast(gelu(x_ub[i], dtype_str, approximate), dtype_str)
                T.copy(y_ub, Y[begin : begin + block])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @T.macro
    def _gelu_body(x, y, width: int, dtype: str, approximate: str):
        with T.SimtVF(threads=min(width, 2048)):
            for i in T.Parallel(width):
                y[i] = T.cast(gelu(x[i], dtype, approximate), dtype)

    @functools.lru_cache(maxsize=128)
    def _build_with_tail(n_total: int, dtype_str: str, approximate: str):
        """Bounded UB tail; only valid elements cross the global-memory boundary."""
        tile = 8192
        stages = 2
        # Leave room for both tail buffers alongside pipelined full tiles.
        itemsize = 4 if dtype_str == "float32" else 2
        while (2 * tile * stages + 2 * ((n_total % tile + 63) // 64 * 64)) * itemsize > 248 * 1024:
            tile //= 2
        full, tail = divmod(n_total, tile)
        grid = min(64, full) if full else 1
        padded_tail = (tail + 63) // 64 * 64

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
            with T.Kernel(grid) as bx:
                if full:
                    x = T.alloc_shared((tile,), dtype_str)
                    y = T.alloc_shared((tile,), dtype_str)
                    T.annotate_buffer_versions({x: stages, y: stages})
                    for step in T.Pipelined((full + grid - 1) // grid, num_stages=stages):
                        block = step * grid + bx
                        if block < full:
                            begin = block * tile
                            T.copy(X[begin : begin + tile], x)
                            _gelu_body(x, y, tile, dtype_str, approximate)
                            T.copy(y, Y[begin : begin + tile])
                if tail:  # noqa: SIM102 - preserve validated TileLang DSL structure
                    if bx == 0:
                        xt = T.alloc_shared((padded_tail,), dtype_str)
                        yt = T.alloc_shared((padded_tail,), dtype_str)
                        T.copy(X[full * tile : n_total], xt[:tail])
                        _gelu_body(xt, yt, padded_tail, dtype_str, approximate)
                        T.copy(yt[:tail], Y[full * tile : n_total])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    class AscendGeluKernel(Kernel, UnaryElementwiseFwdInterface):
        """GELU on Ascend, with exact-erf and tanh approximation modes."""

        devices = frozenset({"npu"})

        def __init__(self, call: GeluCall, *, device_index: "int | None" = None) -> None:
            super().__init__(device_index=device_index)
            self.n_total = call.n_total
            self.dtype = call.dtype
            self.approximate = call.approximate
            dtype_str = self.dtype_to_str(call.dtype)
            if not call.n_total:
                self.kernel = None
                return
            mode = _mode_for(call.n_total)
            if mode == "pipelined":
                self.kernel = _build_pipelined(
                    call.n_total, dtype_str, _NUM_BLOCKS, _TILE, _NUM_STAGES, call.approximate
                )
            elif mode == "decode_multicore":
                self.kernel = _build_decode_multicore(
                    call.n_total, dtype_str, _decode_blocks(call.n_total), call.approximate
                )
            elif mode == "single_shot":
                self.kernel = _build_single_shot(call.n_total, dtype_str, call.approximate)
            else:
                self.kernel = _build_with_tail(call.n_total, dtype_str, call.approximate)

        @classmethod
        def refusal(cls, call: GeluCall) -> "str | None":
            if call.approximate not in ("none", "tanh"):
                return "unsupported approximation"
            return super().refusal(call)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            """Return a new tensor, GELU applied element-wise; ``input`` is left untouched."""
            if not self.n_total:
                return torch.empty_like(input)
            flat = input.contiguous().reshape(-1)
            result = self.kernel(flat)
            return result.reshape(input.shape)

    register_kernel_type("GeluFwdOp", "ascend", AscendGeluKernel)
