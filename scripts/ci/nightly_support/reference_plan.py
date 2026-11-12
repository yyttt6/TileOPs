"""Select qualified reference adapters before executing or comparing an operator."""

from complete_reference import prepare_complete_reference
from packed_reference import prepare_packed_reference
from reference_placement import prepare_reference as prepare_existing_reference


def prepare_reference(op, case):
    completed = prepare_complete_reference(op, case)
    if completed is not None:
        return completed
    packed = prepare_packed_reference(op, case)
    return packed if packed is not None else prepare_existing_reference(op, case)
