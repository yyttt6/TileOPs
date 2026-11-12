"""Select qualified reference adapters before executing or comparing an operator."""

from cache_write_reference import prepare_cache_write_reference
from complete_reference import prepare_complete_reference
from packed_reference import prepare_packed_reference
from reference_placement import prepare_reference as prepare_existing_reference
from ssd_reference import prepare_ssd_reference


def prepare_reference(op, case):
    ssd = prepare_ssd_reference(op, case)
    if ssd is not None:
        return ssd
    cache_write = prepare_cache_write_reference(op, case)
    if cache_write is not None:
        return cache_write
    completed = prepare_complete_reference(op, case)
    if completed is not None:
        return completed
    packed = prepare_packed_reference(op, case)
    return packed if packed is not None else prepare_existing_reference(op, case)
