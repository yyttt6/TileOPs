"""Exact MedicalNet BF16 adapter using the unchanged native16 implementation."""

import functools

from tileops.kernels.ascend.pool import max_native_spatial

GEOMETRY = (
    128,
    (32, 112, 112),
    (3, 3, 3),
    (2, 2, 2),
    (1, 1, 1),
    (1, 1, 1),
    (17, 57, 57),
    "bfloat16",
)


def native():
    return max_native_spatial


def supports_call(call):
    return (
        call.n == 2
        and call.c_in == 64
        and tuple(call.size) == (32, 112, 112)
        and tuple(call.window) == (3, 3, 3)
        and tuple(call.stride) == (2, 2, 2)
        and tuple(call.pad) == (1, 1, 1)
        and tuple(call.dilation) == (1, 1, 1)
        and call.ceil_mode is True
        and str(call.dtype).split(".")[-1] == "bfloat16"
    )


def program(indices=False):
    return native().program(*GEOMETRY, indices).with_attr("tilelang_callee_allocated_outputs", 0)


@functools.lru_cache(maxsize=2)
def build(indices=False):
    import tilelang

    return tilelang.compile(program(indices), target="ascend", out_idx=[1, 2])
