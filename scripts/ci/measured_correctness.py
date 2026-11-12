"""Verify the last timed result without executing a second ordinary correctness call."""

from dataclasses import asdict, replace


def measure_and_verify(
    case, impl, measure, *, expected_calls, golden_cache=None, comparison_transform=None
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
        try:
            measurement = measure(capture)
        except Exception:
            _require_unchanged(shared, pristine)
            raise
        if calls != expected_calls or len(final) != 1:
            raise AssertionError(f"timer call count changed: {calls} != {expected_calls}")
        _require_unchanged(shared, pristine)
        # Freeze aliased/private output before reference execution or any reset/probe.
        produced = _copy_outputs(final.pop())
        if comparison_transform is not None:
            produced = comparison_transform(produced)
        evidence = case.verification
        comparison = replace(evidence, probe=None) if isinstance(evidence, Custom) else evidence
        checks = verify(
            (
                golden_cache.wrap(case.reference)
                if golden_cache is not None and case.reference is not None
                else case.reference
            ),
            inputs,
            evidence=comparison,
            requests={"tileops": Request(lambda: produced, ())},
        )
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
        return asdict(checked), measurement
    finally:
        if golden_cache is not None:
            golden_cache.discard()
        _restore(shared, pristine)
        _restore(own, own_pristine)
        final.clear()
        pristine.clear()
        own_pristine.clear()


def measure_profile_and_verify(
    case,
    impl,
    torch,
    torch_npu,
    args,
    evidence_dir,
    *,
    mode="auto",
    golden_cache=None,
    comparison_transform=None,
):
    """Retry only an unsupported captured topology, with restored inputs and a fresh counter."""
    from pathlib import Path

    from complete_profile import make_timer
    from session_profile import UnsupportedSessionError

    selected = "session" if mode == "auto" else mode

    def perform(backend):
        timer = make_timer(Path(evidence_dir) / backend, mode=backend)
        return measure_and_verify(
            case,
            impl,
            lambda run: timer(torch, torch_npu, run, args, 3, 15, reset=impl.reset),
            expected_calls=19,
            golden_cache=golden_cache,
            comparison_transform=comparison_transform,
        )

    try:
        return perform(selected)
    except UnsupportedSessionError as exc:
        if mode != "auto":
            raise
        correctness, measurement = perform("isolated")
        measurement["session_fallback_reason"] = str(exc)
        return correctness, measurement
