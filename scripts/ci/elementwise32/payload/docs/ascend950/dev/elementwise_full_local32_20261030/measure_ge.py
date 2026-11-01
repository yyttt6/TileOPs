import argparse
import dataclasses
from workloads.numerics import compare_outputs, describe
import json
from pathlib import Path
import sys
import time
import hashlib

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT/'TileOPs'), str(ROOT/'TileOPs/src')]
import torch
import torch_npu

def forbidden(*args, **kwargs): raise AssertionError('Event timing is forbidden')
torch.npu.Event = forbidden
import tilelang

from benchmarks.api import cases
from workloads.device import set_run_device
from tools.measure_baselines import _time_reference, _tensor_digest
from tools.baseline_protocol import seed_case
from tools.elementwise_eager_check import check_eager
sys.path.insert(0, str(ROOT/'docs/ascend950/dev/logroots4_20261028'))
from full_digest import tensor_digest as _tensor_digest, PROTOCOL as DIGEST_PROTOCOL

p=argparse.ArgumentParser()
p.add_argument('--op',default='GeFwdOp')
p.add_argument('--regex',default='')
p.add_argument('--output',required=True)
p.add_argument('--repeats',type=int,default=15)
p.add_argument('--no-time',action='store_true')
a=p.parse_args()
module_name = {'GeFwdOp':'comparison'}[a.op]
K=__import__('importlib').import_module('tileops.kernels.ascend.elementwise.'+module_name)
torch.npu.set_device(0);set_run_device('npu')
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
source_paths=[Path(K.__file__),Path(__file__),ROOT/'TileOPs/workloads/numerics.py',ROOT/'TileOPs/workloads/elementwise.py',ROOT/'TileOPs/src/tileops/ops/elementwise/comparison.py',ROOT/'TileOPs/src/tileops/kernels/elementwise/call_spec.py',ROOT/'TileOPs/src/tileops/kernels/elementwise/_dtype.py']
frozen={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
from tileops.ops import elementwise
for case in cases(a.op):
    if a.regex and not __import__('re').search(a.regex,case.id):continue
    start=time.monotonic();seed_case(torch,a.op,case.id)
    xs=case.inputs;before=_tensor_digest(xs)
    ref=case.reference(*(x.cpu() for x in xs))
    op=getattr(elementwise,a.op)(**case.arguments)
    got=op(*xs);torch.npu.synchronize()
    built=[k for entries in op._built_entries.values() for k in entries.values()]
    assert built and all(type(k).__name__=='Ascend'+a.op.removesuffix('FwdOp')+'Kernel' for k in built), [type(k).__name__ for k in built]
    source_dir=out.parent/'generated'/out.stem;source_dir.mkdir(exist_ok=True,parents=True)
    executable = getattr(built[0], 'compiled', None) or getattr(built[0], 'kernel', None) or getattr(built[0], '_compiled_fn', None)
    kernels = ([executable] if hasattr(executable, 'get_kernel_source') else
               [k for k in (getattr(executable, 'main_kernel', None), getattr(executable, 'tail_kernel', None)) if k is not None])
    assert kernels, 'No compiled kernel found for source audit'
    for index, kernel in enumerate(kernels):
        (source_dir/(a.op+'-'+case.id+'-'+str(index)+'.c')).write_text(kernel.get_kernel_source())
    tolerance=0
    check = compare_outputs(got.cpu(), ref, case.verification)
    assert check.checked_outputs
    assert all(got.untyped_storage().data_ptr()!=x.untyped_storage().data_ptr() for x in xs if isinstance(x,torch.Tensor) and x.numel())
    row={'op':a.op,'case_id':case.id,'source_hash':hashlib.sha256(Path(K.__file__).read_bytes()).hexdigest(),
         
         'verification':dataclasses.asdict(check),'correctness_contract':describe(case.verification),
         'max_error':float((got.cpu().float()-ref.float()).abs().nan_to_num().max()),'correct':True,
         'source_manifest':frozen,'runtime_limit':torch.npu.get_device_limit(0),
         'environment':{'torch':torch.__version__,'torch_file':torch.__file__,'tilelang':tilelang.__file__,'host':__import__('platform').node(),'device':torch.npu.get_device_name(0),'timer_sha256':hashlib.sha256((ROOT/'tools/measure_baselines.py').read_bytes()).hexdigest()},
         'time':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}
    row['eager_correctness'] = check_eager(torch, case, xs, ref)
    if not a.no_time:
        row.update(_time_reference(torch,torch_npu,op,tuple(xs),3,a.repeats))
        row['eager'] = _time_reference(torch,torch_npu,case.reference,tuple(xs),3,a.repeats)
    if a.no_time:
        for _ in range(20):op(*xs)
        torch.npu.synchronize()
    assert {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}==frozen, 'Source changed during measurement'
    assert _tensor_digest(xs)==before
    row.update(wall_seconds=time.monotonic()-start,input_unchanged=True,input_digest_protocol=DIGEST_PROTOCOL,input_digest=before)
    with out.open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
    print(json.dumps(row),flush=True)
torch.npu.synchronize()
