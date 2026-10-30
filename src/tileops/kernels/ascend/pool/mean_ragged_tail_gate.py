"""Per-invocation eligibility for the isolated aligned07 dispatch.

Effective pointers include storage_offset. No device reads or metadata copies.
32-byte GM gate is deliberately stricter than uint2's 8-byte requirement:
it also admits only fully aligned existing 2D DMA and output DMA addresses.
The proof is scoped to the pinned generated source, not arbitrary compiler output.
"""

PARAMS = (1, 1000, 8192, 32, 34, 4, True, "float16", "float32")


def eligible(
    params,
    *,
    x_ptr,
    x_shape,
    x_stride,
    x_dtype,
    y_ptr,
    y_shape,
    y_stride,
    y_dtype,
    offsets_ptr,
    offsets_shape,
    offsets_stride,
    offsets_dtype,
    indices_ptr,
    indices_shape,
    indices_stride,
    indices_dtype,
):
    """Shapes/strides refer to flattened 3D views passed to the compiled kernel."""
    if tuple(params) != PARAMS:
        return False
    layouts = (
        (x_shape, x_stride, x_dtype, (1, 1000, 8192), (8192000, 8192, 1), "float16"),
        (y_shape, y_stride, y_dtype, (1, 34, 8192), (278528, 8192, 1), "float16"),
        (offsets_shape, offsets_stride, offsets_dtype, (5,), (1,), "int32"),
        (indices_shape, indices_stride, indices_dtype, (34, 2), (2, 1), "int32"),
    )
    for shape, stride, dtype, required_shape, required_stride, required_dtype in layouts:
        if (
            tuple(shape) != required_shape
            or tuple(stride) != required_stride
            or dtype != required_dtype
        ):
            return False
    for pointer, alignment in ((x_ptr, 32), (y_ptr, 32), (offsets_ptr, 4), (indices_ptr, 4)):
        if not isinstance(pointer, int) or pointer <= 0 or pointer % alignment:
            return False
    return True


def eligible_tensors(params, x, offsets, indices, y):
    """Host metadata only; evaluate again for every forward, after allocating Y.

    Caller must use the normal public-op validation/device checks and flattened
    views. Never cache this boolean on MeanPoolingCall or on the kernel instance.
    Live metadata values remain inputs to the unchanged TileLang program.
    """
    args = {}
    for name, tensor in (("x", x), ("y", y), ("offsets", offsets), ("indices", indices)):
        args.update(
            {
                name + "_ptr": tensor.data_ptr(),
                name + "_shape": tuple(tensor.shape),
                name + "_stride": tuple(tensor.stride()),
                name + "_dtype": str(tensor.dtype).split(".")[-1],
            }
        )
    return eligible(params, **args)
