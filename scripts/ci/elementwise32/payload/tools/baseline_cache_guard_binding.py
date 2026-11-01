"""CPU-only binding of a timing cache to its guard receipt and implementation."""
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import time


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def capture_guard():
    launcher = Path(shutil.which('npu-guard')).resolve()
    paths = {launcher}
    for token in shlex.split(launcher.read_text()):
        if token.startswith('/') and token.endswith('/npu_guard.py'):
            paths.add(Path(token).resolve())
    if len(paths) < 2:
        raise ValueError('Cannot resolve installed guard implementation from launcher')
    files, stamps = {}, {}
    for path in sorted(paths):
        before = path.stat()
        files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        after = path.stat()
        stamp = lambda s: [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]
        if stamp(before) != stamp(after):
            raise ValueError('Guard implementation changed during fingerprinting')
        stamps[str(path)] = stamp(after)
    return {'files': files, 'stamps': stamps, 'observed_at': time.time()}


def bind_guard(row, before, after, *, retrospective=False):
    if before['files'] != after['files'] or before['stamps'] != after['stamps']:
        raise ValueError('Guard implementation changed during measurement')
    if retrospective and any(max(s[-2:]) / 1e9 > row['started_at'] for s in after['stamps'].values()):
        raise ValueError('Cannot bind historical measurement to a newer guard file')
    receipt = Path(row['guard_receipt'])
    raw = receipt.read_bytes()
    g = json.loads(raw)
    if not (g.get('phase') == 'finished' and g.get('performance_valid') is True
            and g.get('workload_exit_code') == 0 and not g.get('conflict_detected')
            and not g.get('visibility_lost') and g.get('owner', '').endswith('/' + row['task_id'])):
        raise ValueError('Guard receipt does not certify this complete exclusive measurement')
    binding = {'schema': 1, 'cache_key': row['cache_key'], 'guard_receipt': str(receipt),
               'receipt_sha256': hashlib.sha256(raw).hexdigest(), 'implementation': after['files'],
               'before': before, 'after': after,
               'evidence': 'post_run_observation_files_predate_measurement' if retrospective else 'captured_before_and_after_measurement'}
    binding['bound_cache_key'] = digest(binding)
    return binding


def validate_binding(row, binding, *, check_implementation=True):
    if not binding:
        return ['missing guard provenance binding']
    errors = []
    body = {k: v for k, v in binding.items() if k != 'bound_cache_key'}
    if digest(body) != binding.get('bound_cache_key') or binding.get('cache_key') != row['cache_key']:
        errors.append('guard binding digest/cache key mismatch')
    p = Path(binding['guard_receipt'])
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != binding['receipt_sha256']:
        errors.append('guard receipt changed or missing')
    if check_implementation:
        for name, expected in binding['implementation'].items():
            p = Path(name)
            if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
                errors.append('guard implementation changed: ' + name)
    return errors
