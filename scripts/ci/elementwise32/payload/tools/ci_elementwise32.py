"""Run a frozen elementwise export in the isolated physical32-core CI runtime.

The controller is CPU only. Each complete device command gets its own host
reservation, source fingerprint, hardware check and durable guard receipt.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + '\n')


def child(script, output, op):
    sys.path[:0] = [str(ROOT), str(ROOT / 'TileOPs'), str(ROOT / 'TileOPs/src')]
    from benchmarks.ascend.guard import require_reservation
    require_reservation()
    import torch
    import torch_npu
    torch.npu.set_device(0)
    p = torch.npu.get_device_properties(0)
    hardware = dict(device=torch.npu.get_device_name(0), cube=p.cube_core_num,
                    vector=p.vector_core_num, torch=torch.__version__)
    write(output / 'hardware.json', hardware)
    assert hardware['device'] == 'Ascend950DT_9582'
    assert (hardware['cube'], hardware['vector']) == (32, 64)
    torch.npu.set_device_limit(0, cube_num=32, vector_num=64)
    assert torch.npu.get_device_limit(0) == {'cube_core_num': 32, 'vector_core_num': 64}
    os.environ['TILEOPS_PROFILE_DIR'] = str(output / 'profiles')
    try:
        sys.argv = [str(script), '--op', op, '--output', str(output / 'measurements.jsonl')]
        runpy.run_path(str(script), run_name='__main__')
    finally:
        torch.npu.synchronize()


def fingerprint():
    export = json.loads((ROOT / 'ELEMENTWISE_EXPORT.json').read_text())
    result = {str(ROOT / 'TileOPs' / name): digest
              for name, digest in export['source_files'].items()}
    # Include case factories, manifest parameters, dispatch and verification code,
    # in addition to the operator export and measurement helpers.
    for name in ('src/tileops', 'workloads', 'benchmarks', 'tests', 'scripts'):
        for p in (ROOT / 'TileOPs' / name).rglob('*'):
            if p.is_file() and p.suffix in {'.py', '.yaml', '.toml', '.json'}:
                result.setdefault(str(p), sha(p))
    for name in ('tools', 'docs/ascend950/dev'):
        for p in (ROOT / name).rglob('*.py'):
            result[str(p)] = sha(p)
    for p in (ROOT / 'ELEMENTWISE_PLAN.json', ROOT / 'ELEMENTWISE_EXPORT.json',
              ROOT / 'BASELINE_ADMISSION.json'):
        result[str(p)] = sha(p)
    for name, digest in result.items():
        assert sha(name) == digest, 'Frozen source changed: ' + name
    return result


def validate_rows(rows, expected):
    assert len(rows) == len(expected) and {r['case_id'] for r in rows} == set(expected)
    for r in rows:
        assert r['correct'] is True and r['input_unchanged'] is True
        assert r['runtime_limit'] == {'cube_core_num': 32, 'vector_core_num': 64}
        for timing in (r, r['eager']):
            samples = timing['latency_samples_ms']
            assert len(samples) == 15 and all(math.isfinite(x) and x > 0 for x in samples)
            assert math.isclose(timing['latency_ms'], statistics.mean(samples), rel_tol=1e-9)
            assert timing['measurement_protocol'] == 'ascend950-isolated-ffts-v3'
            from kernel_task_profile import parse_profile, trace_digest
            profiles = timing['retained_profiles']
            assert len(profiles) == 15
            for value, trace in zip(samples, profiles, strict=True):
                assert trace_digest(trace['profile_path']) == trace['profile_trace_sha256']
                reparsed = parse_profile(trace['profile_path'], 1)
                assert math.isclose(reparsed['latency_ms'], value, rel_tol=1e-9, abs_tol=1e-12)


def controller(output, selected, attempt_tag=None):
    admission = ROOT / 'BASELINE_ADMISSION.json'
    gate = json.loads(admission.read_text())
    assert gate['status'] == 'passed' and gate['server1_complete'] is True
    assert sha(ROOT / 'BASELINE32_PRIVATE.json') == gate['catalog_sha256']
    plan = json.loads((ROOT / 'ELEMENTWISE_PLAN.json').read_text())
    if selected:
        plan = [p for p in plan if p['op'] in selected]
        assert {p['op'] for p in plan} == set(selected)
    output.mkdir(parents=True, exist_ok=True)
    from baseline_cache_guard_binding import capture_guard
    frozen = fingerprint()
    sources_path = output / 'sources.json'
    if sources_path.exists() and json.loads(sources_path.read_text()) != frozen:
        backup = output / ('sources.' + sha(sources_path)[:16] + '.json')
        if not backup.exists():
            backup.write_bytes(sources_path.read_bytes())
    write(sources_path, frozen)
    write(output / 'baseline-admission.json', gate)
    results = []
    for p in plan:
        primary = output / p['op']
        directory = primary
        pointer = primary / 'LATEST_ATTEMPT.json'
        if pointer.exists():
            directory = output / json.loads(pointer.read_text())['path']
            assert directory.resolve().is_relative_to(output.resolve())
        if directory.exists():
            receipt = directory / 'receipt.json'
            prior = json.loads(receipt.read_text()) if receipt.exists() else {}
            admitted = prior.get('sources') == frozen
            proof = directory / 'READMISSION.json'
            if not admitted and proof.exists():
                proof = json.loads(proof.read_text())
                admitted = (proof.get('status') == 'passed'
                            and proof.get('current_sources') == frozen
                            and proof.get('original_receipt_sha256') == sha(receipt))
            if prior.get('status') == 'passed' and admitted:
                validate_rows([json.loads(l) for l in (directory / 'measurements.jsonl').read_text().splitlines()], p['case_ids'])
                results.append(prior)
                continue
            if not attempt_tag:
                raise RuntimeError('Preserve existing failed/incomplete attempt; select --attempt-tag: ' + str(directory))
            directory = primary / 'attempts' / attempt_tag
        directory.mkdir(parents=True)
        owner = 'ci/elementwise32/' + output.parent.name + '/' + p['op'] + '/' + str(time.time_ns())
        command = ['npu-guard', 'run', '--owner', owner, '--wait', '600', '--timeout', '5400', '--',
                   sys.executable, str(Path(__file__).resolve()), '--device-child', p['script'],
                   '--output', str(directory), '--op', p['op']]
        before = capture_guard()
        started = time.time()
        with (directory / 'run.log').open('x') as log:
            run = subprocess.run(command, cwd=ROOT / 'TileOPs', stdout=log, stderr=subprocess.STDOUT)
        errors = []
        guard = {}
        try:
            guard = json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
            assert guard['owner'] == owner, 'Guard metadata replaced before capture'
            write(directory / 'guard.json', guard)
            assert guard['phase'] == 'finished' and guard['performance_valid'] is True
            assert guard['workload_exit_code'] == 0 and not guard.get('conflict_detected') and not guard.get('visibility_lost')
            assert before['files'] == capture_guard()['files'], 'Guard implementation changed'
            assert run.returncode == 0, 'Device command failed'
            assert fingerprint() == frozen
            rows = [json.loads(l) for l in (directory / 'measurements.jsonl').read_text().splitlines()]
            validate_rows(rows, p['case_ids'])
        except Exception as e:
            errors.append(type(e).__name__ + ': ' + str(e))
        receipt = dict(op=p['op'], status='failed' if errors else 'passed',
                       command=command, exit_code=run.returncode, errors=errors,
                       started_at=started, finished_at=time.time(), sources=frozen,
                       guard_implementation=before['files'], baseline_catalog_sha256=gate['catalog_sha256'])
        write(directory / 'receipt.json', receipt)
        if directory != primary:
            write(pointer, dict(path=str(directory.relative_to(output)), status=receipt['status']))
        results.append(receipt)
        summary = dict(status='running', expected=len(plan), completed=len(results),
                       passed=sum(r['status'] == 'passed' for r in results),
                       failed=[r['op'] for r in results if r['status'] != 'passed'])
        write(output / 'STATE.json', summary)
        print(json.dumps({**summary, 'last_op': p['op']}), flush=True)
    summary['status'] = 'failed' if summary['failed'] else 'passed'
    write(output / 'STATE.json', summary)
    return bool(summary['failed'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device-child')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--op', action='append')
    p.add_argument('--attempt-tag')
    a = p.parse_args()
    if a.device_child:
        child(ROOT / a.device_child, a.output, a.op[0])
    else:
        if a.attempt_tag and not __import__('re').fullmatch(r'[A-Za-z0-9_-]+', a.attempt_tag):
            p.error('Use a simple attempt tag')
        raise SystemExit(controller(a.output, a.op, a.attempt_tag))
