"""Verify the last timed result without executing a second ordinary correctness call."""

import time
from dataclasses import asdict, replace


def measure_and_verify(
    case, impl, measure, *, expected_calls, golden_cache=None, timings=None, synchronize=None
):
    """Run the unmodified timer, retain only its final output, then apply workload checks.

    ``measure(run)`` must perform exactly ``expected_calls`` calls (including cold and
    warmup calls). Capture is just a Python reference assignment, never device work.
    The timer must synchronize before returning. Custom probes remain real executions.
    """
    from workloads.numerics import (
        Custom,
        Request,
        _copy_outputs,
        _require_unchanged,
        _restore,
        _snapshot,
        _unique,
        verify,
    )

    if impl.noncomparable_reason:
        raise ValueError(f"tileops: not comparable: {impl.noncomparable_reason}")
    if expected_calls <= 0:
        raise ValueError("measurement must include an execution")
    timings = {} if timings is None else timings
    phase_start = time.monotonic()
    inputs = case.inputs
    args = inputs if impl.args is None else impl.args
    shared = _unique(inputs)
    pristine = _snapshot(shared)
    own = [t for t in _unique(args) if all(t is not s for s in shared)]
    own_pristine = _snapshot(own)
    calls = 0
    final = []

    def capture(*values):
        nonlocal calls
        produced = impl.run(*values)
        calls += 1
        if calls == expected_calls:
            final.append(produced)
        return produced

    try:
        if synchronize is not None:
            synchronize()
        timings["snapshot_seconds"] = time.monotonic() - phase_start
        phase_start = time.monotonic()
        measurement = measure(capture)
        timings["timer_seconds"] = time.monotonic() - phase_start
        phase_start = time.monotonic()
        if calls != expected_calls or len(final) != 1:
            raise AssertionError(f"timer call count changed: {calls} != {expected_calls}")
        _require_unchanged(shared, pristine)
        # Freeze aliased/private output before reference execution or any reset/probe.
        produced = _copy_outputs(final.pop())
        evidence = case.verification
        comparison = replace(evidence, probe=None) if isinstance(evidence, Custom) else evidence
        if synchronize is not None:
            synchronize()
        timings["preservation_and_output_copy_seconds"] = time.monotonic() - phase_start
        reference = (
            golden_cache.wrap(case.reference)
            if golden_cache is not None and case.reference is not None
            else case.reference
        )
        timings["reference_seconds"] = 0.0
        timings["reference_calls"] = 0

        def timed_reference(*values):
            started = time.monotonic()
            try:
                return reference(*values)
            finally:
                if synchronize is not None:
                    synchronize()
                timings["reference_seconds"] += time.monotonic() - started
                timings["reference_calls"] += 1

        phase_start = time.monotonic()
        checks = verify(
            timed_reference if reference is not None else None,
            inputs,
            evidence=comparison,
            requests={"tileops": Request(lambda: produced, ())},
        )
        if synchronize is not None:
            synchronize()
        timings["verification_including_reference_seconds"] = time.monotonic() - phase_start
        timings["verification_excluding_reference_seconds"] = (
            timings["verification_including_reference_seconds"] - timings["reference_seconds"]
        )
        phase_start = time.monotonic()
        checked = checks["tileops"]
        if (
            checked.checked_outputs <= 0
            or checked.checked_outputs != checked.total_outputs
            or checked.unchecked_reason
        ):
            raise AssertionError(f"tileops: incomplete correctness coverage: {checked}")
        if isinstance(evidence, Custom) and evidence.probe is not None:
            _restore(own, own_pristine)
            if impl.reset is not None:
                impl.reset()
            evidence.probe(impl.run, args)
            _require_unchanged(shared, pristine)
        if golden_cache is not None:
            golden_cache.commit()
        if synchronize is not None:
            synchronize()
        timings["custom_probes_and_cache_commit_seconds"] = time.monotonic() - phase_start
        return asdict(checked), measurement
    finally:
        phase_start = time.monotonic()
        if golden_cache is not None:
            golden_cache.discard()
        _restore(shared, pristine)
        _restore(own, own_pristine)
        final.clear()
        pristine.clear()
        own_pristine.clear()
        if synchronize is not None:
            synchronize()
        timings["cleanup_seconds"] = time.monotonic() - phase_start
