"""CPU regression tests for mixed-core and overlapping kernel accounting."""
import struct

import pytest

from scripts.ci.elementwise32.payload.tools.kernel_task_profile import parse_profile


def record(task, role, start, end, context=0):
    values = [0] * 16
    values[0] = (task << 32) | (0x6A if role == 'AIV' else 0x2A)
    values[1] = 100
    values[2] = (context << 16) | (1 << 32 if role == 'AIV' else 0)
    values[14:16] = [start, end]
    return struct.pack('<16q', *values)


def parse(tmp_path, *records):
    (tmp_path / 'ffts_profile.slice_0').write_bytes(b''.join(records))
    return parse_profile(tmp_path, 1)


@pytest.mark.smoke
def test_mixed_roles_of_one_kernel_are_counted_once(tmp_path):
    result = parse(tmp_path, record(1, 'AIC', 1000, 2000), record(1, 'AIV', 999, 2001))
    assert result['profile_kernel_us'] == pytest.approx(1.002)
    assert result['profile_kernel_count'] == 1
    assert result['profile_raw_record_count'] == 2


@pytest.mark.smoke
def test_distinct_overlapping_kernels_keep_their_sum(tmp_path):
    result = parse(tmp_path, record(1, 'AIV', 1000, 2000), record(2, 'AIV', 1500, 2500))
    assert result['profile_kernel_us'] == 2
    assert result['profile_kernel_count'] == 2


@pytest.mark.smoke
def test_distinct_ffts_contexts_remain_separate(tmp_path):
    result = parse(tmp_path, record(1, 'AIV', 1000, 2000, 1), record(1, 'AIV', 1500, 2500, 2))
    assert result['profile_kernel_us'] == 2


@pytest.mark.smoke
def test_reused_task_identity_cannot_merge_unrelated_launches(tmp_path):
    with pytest.raises(ValueError, match='Repeated task'):
        parse(tmp_path, record(1, 'AIV', 1000, 2000), record(1, 'AIV', 3000, 4000))


@pytest.mark.smoke
def test_ambiguous_multistream_pair_is_rejected(tmp_path):
    tasks = []
    for stream in (61, 62):
        values = [0] * 8
        values[3] = (1 << 32) | (stream << 16)
        tasks.append(struct.pack('<8q', *values))
    (tmp_path / 'task_track.slice_0').write_bytes(b''.join(tasks))
    with pytest.raises(ValueError, match='multi-stream'):
        parse(tmp_path, record(1, 'AIC', 1000, 2000), record(1, 'AIV', 999, 2001))


@pytest.mark.smoke
def test_truncated_raw_data_is_not_admitted(tmp_path):
    (tmp_path / 'ffts_profile.slice_0').write_bytes(b'incomplete')
    with pytest.raises(ValueError, match='Truncated'):
        parse_profile(tmp_path, 1)


@pytest.mark.smoke
def test_packet_flags_do_not_split_one_mixed_task(tmp_path):
    records = []
    for role, flags, start, end in [('AIC', 18, 1000, 2000), ('AIV', 19, 999, 2001)]:
        values = list(struct.unpack('<16q', record(1, role, start, end)))
        values[0] = (1 << 32) | 0x2A | (flags << 6)
        records.append(struct.pack('<16q', *values))
    result = parse(tmp_path, *records)
    assert result['profile_kernel_count'] == 1
    assert result['profile_kernel_us'] == pytest.approx(1.002)
