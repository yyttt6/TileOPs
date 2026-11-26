"""CUDA properties used only by the typed kernel build cache."""

import functools
from typing import NamedTuple

import torch

from tileops._utils.utils import calibration_key


class CudaFacts(NamedTuple):
    """What kernel selection and builds read about one CUDA device."""

    # ``major * 10 + minor``.
    arch: int
    # :func:`calibration_key` of the device's name.
    calibration: "str | None"
    sm_count: int
    # Shared memory one block may take after opting in, in bytes.
    smem_budget: int
    threads_per_sm: int
    regs_per_sm: int
    # Shared memory of one SM, in bytes.
    smem_per_sm: int
    l2_bytes: int
    # The device's GPU profile (:func:`tileops._perf.find_profile`): the calibrated
    # tensor-core rate of each dtype it covers, in flop/s; the calibrated HBM rate, in
    # bytes/s; and the busy time one more dependent kernel launch adds, in seconds. Empty
    # and zero for a device without a profile.
    tensor_core_rates: "tuple[tuple[str, float], ...]"
    hbm_rate: float
    launch_latency: float


def _profile_rates(device_name: str) -> "tuple[tuple[tuple[str, float], ...], float, float]":
    """The ``tensor_core_rates``, ``hbm_rate`` and ``launch_latency`` of *device_name*'s GPU
    profile; empty and zero without one."""
    from tileops._perf import find_profile

    profile = find_profile(device_name)
    if profile is None:
        return (), 0.0, 0.0
    rates = tuple(
        sorted(
            (dtype, float(section["effective"]))
            for dtype, section in (profile.get("tensor_core") or {}).items()
            if "effective" in section
        )
    )
    hbm = float((profile.get("hbm") or {}).get("effective", 0.0))
    latency = float((profile.get("launch") or {}).get("latency", 0.0))
    return rates, hbm, latency


@functools.lru_cache(maxsize=16)
def _device_facts(index: int) -> CudaFacts:
    major, minor = torch.cuda.get_device_capability(index)
    props = torch.cuda.get_device_properties(index)
    name = torch.cuda.get_device_name(index)
    tensor_core_rates, hbm_rate, launch_latency = _profile_rates(name)
    return CudaFacts(
        arch=major * 10 + minor,
        calibration=calibration_key(name),
        sm_count=props.multi_processor_count,
        smem_budget=props.shared_memory_per_block_optin,
        threads_per_sm=props.max_threads_per_multi_processor,
        regs_per_sm=props.regs_per_multiprocessor,
        smem_per_sm=props.shared_memory_per_multiprocessor,
        l2_bytes=props.L2_cache_size,
        tensor_core_rates=tensor_core_rates,
        hbm_rate=hbm_rate,
        launch_latency=launch_latency,
    )
