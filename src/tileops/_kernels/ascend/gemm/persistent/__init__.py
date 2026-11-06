"""Shape-specialized 16-bit GEMM schedules for Ascend 950."""

import importlib

# Every (M, N, K, dtype) the Ascend GemmFwdOp serves, the module holding its schedule,
# and the ``trans_b`` layout it compiles for. Modules are imported on demand: the
# fixed-shape schedules parse their PrimFunc at import, and one call needs one schedule.
SCHEDULES: dict[tuple[int, int, int, str], tuple[str, bool]] = {
    (16, 4096, 7168, "bfloat16"): ("bf16_nt_16_4096_7168", True),
    (32, 4096, 7168, "bfloat16"): ("bf16_nt_32_4096_7168", True),
    (128, 2112, 7168, "bfloat16"): ("bf16_nt_128_2112_7168", True),
    (128, 7168, 2048, "bfloat16"): ("bf16_nt_128_7168_2048", True),
    (4096, 2112, 7168, "bfloat16"): ("bf16_nt_4096_2112_7168", True),
    (1024, 1024, 1024, "bfloat16"): ("dynamic_tail", False),
    (1024, 1024, 1024, "float16"): ("dynamic_tail", False),
    (1, 24576, 4096, "bfloat16"): ("dynamic_tail", True),
    (128, 4096, 12288, "bfloat16"): ("dynamic_tail", True),
    (1, 2112, 7168, "bfloat16"): ("l1_direct_tail", True),
    (2, 2112, 7168, "bfloat16"): ("l1_direct_tail", True),
    (256, 21843, 1024, "bfloat16"): ("l1_direct_tail", True),
    (32, 24576, 4096, "bfloat16"): ("swizzled_persistent", True),
    (64, 7168, 2048, "bfloat16"): ("swizzled_persistent", True),
    (96, 2112, 7168, "bfloat16"): ("swizzled_persistent", True),
    (4096, 4096, 7168, "bfloat16"): ("swizzled_persistent", True),
    (4096, 4096, 7168, "float16"): ("swizzled_persistent", True),
    (4096, 7168, 2048, "bfloat16"): ("swizzled_persistent", True),
    (4096, 7168, 16384, "bfloat16"): ("swizzled_persistent", True),
    (4096, 24576, 1536, "bfloat16"): ("swizzled_persistent", True),
}

__all__ = ["SCHEDULES", "compile_schedule"]


def compile_schedule(m: int, n: int, k: int, dtype: str):
    """Compile the schedule serving ``(m, n, k)`` and ``dtype``.

    Args:
        m: Rows of ``a`` and ``d``.
        n: Columns of ``d``.
        k: Contraction length; ``b`` is ``(n, k)`` or ``(k, n)`` per the table.
        dtype: ``"bfloat16"`` or ``"float16"``, the dtype the schedule compiles for.

    Returns:
        The compiled TileLang kernel, called with ``(a, b)``.

    Raises:
        KeyError: No schedule serves the shape and dtype; :class:`AscendGemmKernel`
            refuses the call before this is reached.
    """
    module_name, trans_b = SCHEDULES[m, n, k, dtype]
    module = importlib.import_module(f"{__name__}.{module_name}")
    return module.compile_kernel(m, n, k, trans_b, dtype)
