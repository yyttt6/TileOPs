"""Stable raw gate formula from fixed v0.1.0 KDA gate.py, using Ascend T."""

import tilelang.ascend.language as T


def log_decay(raw, log_rate, bias, lower_bound):
    """The channel's log-space decay, with an optional bounded sigmoid gate."""
    shifted = T.cast(raw, "float32") + bias
    rate = T.exp(log_rate)
    if lower_bound is not None:
        # exp(A_log) can overflow FP32, and inf * 0 is NaN: a zero shifted gate takes the
        # sigmoid's value at zero directly instead of through rate * shifted.
        return T.if_then_else(
            shifted == 0.0, lower_bound * 0.5, lower_bound / (1.0 + T.exp(-rate * shifted))
        )
    # Below -20, softplus(x)/exp(x) differs from 1 by less than exp(-20)/2 < 2e-9.
    # Combining exponents preserves a small tail before a large rate scales it.
    # Torch softplus uses the identity above 20.
    return T.if_then_else(
        shifted < -20.0,
        -T.exp(log_rate + shifted),
        -rate * T.if_then_else(shifted > 20.0, shifted, T.log1p(T.exp(shifted))),
    )
