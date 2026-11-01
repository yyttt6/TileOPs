"""Qualify an eager timing against the same complete TileOPs accuracy contract."""
import dataclasses


def check_eager(torch, case, inputs, reference):
    from workloads.numerics import Custom, compare_outputs

    def cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, tuple):
            return tuple(cpu(v) for v in value)
        if isinstance(value, list):
            return [cpu(v) for v in value]
        if isinstance(value, dict):
            return {k: cpu(v) for k, v in value.items()}
        return value

    try:
        actual = case.reference(*inputs)
        torch.npu.synchronize()
        evidence = case.verification
        if isinstance(evidence, Custom):
            # Custom validators can close over the supplied inputs. Rebind the
            # unchanged workload policy to the same input values on CPU.
            rebound = case.workload.verification(*cpu(inputs))
            assert isinstance(rebound, Custom) and rebound.reason == evidence.reason
            assert rebound.controls == evidence.controls and rebound.probe == evidence.probe
            evidence = rebound
        check = compare_outputs(cpu(actual), cpu(reference), evidence)
        assert check.checked_outputs == check.total_outputs and check.checked_outputs > 0
        return dict(eligible=True, verification=dataclasses.asdict(check),
                    contract='Same expected outputs and original tolerances as TileOPs verification')
    except Exception as exc:
        return dict(eligible=False, error_type=type(exc).__name__, error=str(exc),
                    contract='Eager timing excluded from baseline comparisons on this case')
