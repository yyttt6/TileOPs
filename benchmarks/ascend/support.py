"""Explicit, verified unsupported manifest cases; never infer support from failures."""


def unsupported_reason(op_name, case, kernel_types):
    if op_name != "FFTC2CFwdOp" or case.id not in {
        "throughput-4k-complex128",
        "scientific-1m-complex128",
    }:
        return None
    shape, dtype = case._call.tensors["input"]
    if dtype != "complex128":
        raise RuntimeError("FFT unsupported-case declaration no longer matches the manifest")
    import torch

    from tileops.kernels.fft import FFTC2CCall

    kernel = kernel_types[op_name]["ascend"]
    reason = kernel.refusal(
        FFTC2CCall(n=shape[-1], dtype=torch.complex128, device=torch.device("npu", 0))
    )
    if not reason or "FP64/complex128 is unsupported" not in reason:
        raise RuntimeError("FFT support changed; update the explicit unsupported-case declaration")
    return reason
