"""Pooling output-extent arithmetic, shared by every pooling op.

Shape rules, not an implementation: an op's ``_infer_output_shapes`` is what the
manifest declares and what the compile boundary's fake reads, so this has to
answer without a device and without building anything.
"""

__all__ = ["pool_output_dim"]


def pool_output_dim(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    ceil_mode: bool,
    dilation: int = 1,
) -> int:
    """Output extent of one pooled axis, matching ``torch.nn.functional``.

    Under ``ceil_mode`` PyTorch drops a final window that starts inside the
    right-hand padding, which is the correction after the division.
    """
    effective_kernel = dilation * (kernel_size - 1) + 1
    if ceil_mode:
        out = (input_size + 2 * padding - effective_kernel + stride - 1) // stride + 1
    else:
        out = (input_size + 2 * padding - effective_kernel) // stride + 1

    if ceil_mode and out > 0 and (out - 1) * stride >= input_size + padding:
        out -= 1

    return max(out, 0)
