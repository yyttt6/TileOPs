"""Sum distinct kernel task spans; count AIC/AIV roles of one kernel once."""
import collections
import hashlib
from pathlib import Path
import struct


def trace_digest(directory):
    root = Path(directory)
    h = hashlib.sha256()
    for p in sorted(root.rglob('*')):
        if p.is_file() and not p.name.endswith(('.done', '.complete')) and any(
                token in p.name for token in ('ffts_profile', 'hash_dic.slice_', 'task_track.slice_')):
            h.update(str(p.relative_to(root)).encode())
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def parse_profile(prof_path, repeats, *, seq_to_name=None, exclude_name_marker=None,
                  exclude_names=frozenset(), require_marker_seen=False):
    root = Path(prof_path)
    if repeats <= 0:
        raise ValueError('Positive repeat count required')
    names = seq_to_name or {}
    streams = collections.defaultdict(set)
    for p in root.rglob('*task_track.slice_*'):
        if p.name.endswith(('.done', '.complete')):
            continue
        raw = p.read_bytes()
        if len(raw) % 64:
            raise ValueError('Truncated task track: ' + str(p))
        for offset in range(0, len(raw), 64):
            v = struct.unpack_from('<8q', raw, offset)
            streams[(v[3] >> 32) & 65535].add((v[3] >> 16) & 65535)
    groups = collections.defaultdict(list)
    aic_cycles = aiv_cycles = raw_count = 0
    for p in root.rglob('ffts_profile*'):
        if p.name.endswith(('.done', '.complete')):
            continue
        raw = p.read_bytes()
        if len(raw) % 128:
            raise ValueError('Truncated FFTS record: ' + str(p))
        for offset in range(0, len(raw), 128):
            v = struct.unpack_from('<16q', raw, offset)
            if v[1] == 0:
                continue
            kind = v[0] & 255
            if kind not in (0x2A, 0x6A):
                raise ValueError('Unsupported FFTS record kind: ' + hex(kind))
            if v[15] < v[14] or v[14] <= 0:
                raise ValueError('Invalid kernel timestamp')
            role = 'AIV' if kind == 0x6A else 'AIC'
            raw_count += 1
            if role == 'AIV':
                aiv_cycles += v[1]
            else:
                aic_cycles += v[1]
            # 0x40 distinguishes the AIC/AIV record kind. Preserve every other
            # header bit and the FFTS context; unlike a global interval union,
            # independent task IDs remain separate even when they overlap.
            key = (v[0] & ~0x40, (v[2] >> 16) & 65535)
            groups[key].append((role, v[14], v[15]))
    if not raw_count:
        raise ValueError('No valid FFTS activity')
    tasks = []
    excluded_count = excluded_ns = 0
    marker_seen = False
    for (header, context), records in groups.items():
        seq = (header >> 32) & 65535
        roles = {r[0] for r in records}
        if len(roles) > 1 and len(streams[seq]) > 1:
            raise ValueError('Ambiguous multi-stream task ID; retain trace for full decoding')
        for role in roles:
            intervals = {(start, end) for r, start, end in records if r == role}
            if len(intervals) > 1:
                raise ValueError('Repeated task/context identity; do not merge unrelated launches')
        start = min(r[1] for r in records)
        end = max(r[2] for r in records)
        name = names.get(seq, '')
        is_marker = bool(exclude_name_marker) and exclude_name_marker in name
        marker_seen |= is_marker
        if is_marker or name in exclude_names:
            excluded_count += 1
            excluded_ns += end - start
            continue
        tasks.append(dict(task_id=seq, context_id=context, stream_ids=sorted(streams[seq]),
                          name=name, core_roles=sorted(roles), start_ns=start, end_ns=end,
                          duration_us=(end-start)/1000))
    if require_marker_seen and exclude_name_marker and not marker_seen:
        raise ValueError('Flush marker not resolved')
    total_ns = sum(t['end_ns']-t['start_ns'] for t in tasks)
    return dict(latency_ms=total_ns/repeats/1e6, profile_kernel_us=total_ns/repeats/1000,
                profile_kernel_count=len(tasks)/repeats, profile_raw_record_count=raw_count/repeats,
                profile_aic_cycles=aic_cycles/repeats, profile_aiv_cycles=aiv_cycles/repeats,
                profile_excluded_kernel_count=excluded_count/repeats,
                profile_excluded_kernel_us=excluded_ns/repeats/1000,
                profile_kernel_tasks=tasks)
