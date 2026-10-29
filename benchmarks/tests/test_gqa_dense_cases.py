"""The benchmark must execute the query length declared by the manifest."""

from benchmarks._cases.attention import _gqa_dense_workload
from tileops.manifest import load_adts, load_manifest
from tileops.manifest.plan import entry_plan
from tileops.manifest.workload import instantiate
from workloads.attention.gqa.dense import GQADensePrefillCall
from workloads.device import run_device, set_run_device


def test_chunked_prefill_uses_manifest_shapes():
    entry = load_manifest()["GQADenseFwdOp"]
    plan = entry_plan("GQADenseFwdOp", entry, load_adts())
    previous = run_device()
    set_run_device("meta")
    try:
        for row in entry["workloads"]:
            for dtype in row.get("dtype_cases", [{}]):
                call = instantiate(plan, row, dtype)
                workload = _gqa_dense_workload(call)
                if call.indices["S_q"] > 1:
                    assert isinstance(workload, GQADensePrefillCall)
                inputs = workload.gen_inputs()
                for name, tensor in zip(("q", "k", "v"), inputs[:3], strict=True):
                    assert tuple(tensor.shape) == call.specs[name].shape
    finally:
        set_run_device(previous)
