"""Hash a sequence against its original context, then commit the updated context rows."""

from functools import lru_cache

import tilelang
import tilelang.ascend.language as T
import torch

from tileops._kernels.kernel_base import Kernel
from tileops._kernels.sequence_modeling.ngram_hash import NGramHashFwdInterface
from tileops.backend._registry import register_kernel_type


@lru_cache(maxsize=64)
def _programs(width, order, heads, eos, packed, slotted, rollback, batch, length, total, slots):
    block = 128

    @T.prim_func
    def hash_ids(
        Tokens: T.Tensor((total,) if packed else (batch, length), "int64"),
        Context: T.Tensor((slots, width), "int64"),
        Multipliers: T.Tensor((order,), "int64"),
        Primes: T.Tensor((heads,), "int64"),
        Offsets: T.Tensor((heads,), "int64"),
        Cu: T.Tensor((batch + 1,) if packed else (total,), "int32" if packed else "int64"),
        Slots: T.Tensor((batch,) if slotted else (total,), "int32" if slotted else "int64"),
        Accepted: T.Tensor((batch,) if rollback else (total,), "int32" if rollback else "int64"),
        Output: T.Tensor((total, heads), "int64"),
    ):
        with T.Kernel(T.ceildiv(total * heads, block)) as bx:  # noqa: SIM117 - separate TileLang block and vector scopes
            with T.SimtVF(threads=128):
                for x in T.Parallel(block):
                    linear = bx * block + x
                    if linear < total * heads:
                        token = linear // heads
                        head = linear % heads
                        seq = T.alloc_var("int32", init=0)
                        begin = T.alloc_var("int32", init=0)
                        if packed:
                            hi = T.alloc_var("int32", init=batch)
                            while seq < hi:
                                mid = (seq + hi) // 2
                                if Cu[mid + 1] <= token:
                                    seq = mid + 1
                                else:
                                    hi = mid
                            begin = Cu[seq]
                        else:
                            seq = token // length
                            begin = seq * length
                        slot = Slots[seq] if slotted else seq
                        start = Accepted[seq] - 1 if rollback else 0
                        mixed = T.alloc_var("uint64", init=0)
                        blocked = T.alloc_var("bool", init=False)
                        for i in T.unroll(order):
                            value = T.alloc_var("int64", init=eos)
                            if i == 0 or not blocked:
                                if token - begin >= i:
                                    if packed:
                                        value = Tokens[token - i]
                                    else:
                                        value = Tokens[seq, token - begin - i]
                                else:
                                    value = Context[slot, start + order - 1 + token - begin - i]
                            # Only a previous EOS blocks history; the current EOS still hashes.
                            if i > 0 and value == eos:
                                blocked = True
                            if i < 2 + head // (heads // (order - 1)):
                                mixed = mixed ^ (
                                    T.cast(value, "uint64") * T.cast(Multipliers[i], "uint64")
                                )
                        signed_hash = T.cast(mixed, "int64")
                        remainder = T.truncmod(signed_hash, Primes[head])
                        positive = T.if_then_else(
                            remainder < 0, remainder + Primes[head], remainder
                        )
                        Output[token, head] = T.cast(
                            T.cast(positive, "uint64") + T.cast(Offsets[head], "uint64"), "int64"
                        )

    @T.prim_func
    def update(
        Tokens: T.Tensor((total,) if packed else (batch, length), "int64"),
        Context: T.Tensor((slots, width), "int64"),
        Cu: T.Tensor((batch + 1,) if packed else (total,), "int32" if packed else "int64"),
        Slots: T.Tensor((batch,) if slotted else (total,), "int32" if slotted else "int64"),
        Accepted: T.Tensor((batch,) if rollback else (total,), "int32" if rollback else "int64"),
    ):
        with T.Kernel(batch) as b:
            begin = Cu[b] if packed else b * length
            end = Cu[b + 1] if packed else (b + 1) * length
            slot = Slots[b] if slotted else b
            start = Accepted[b] - 1 if rollback else 0
            old = T.alloc_shared((order - 1,), "int64")
            if end > begin:
                for i in T.serial(order - 1):
                    old[i] = Context[slot, start + i]
                # A shifted read and write may overlap in the same row.
                for j in T.serial(width if rollback else order - 1):
                    source = j + 1 if rollback else end - begin + j
                    if source < order - 1 + end - begin:
                        if source < order - 1:
                            Context[slot, j] = old[source]
                        else:
                            if packed:
                                Context[slot, j] = Tokens[begin + source - (order - 1)]
                            else:
                                Context[slot, j] = Tokens[b, source - (order - 1)]

    return (
        tilelang.compile(hash_ids, target="ascend", out_idx=[]) if heads else None,
        tilelang.compile(update, target="ascend", out_idx=[]),
    )


class AscendNGramHashKernel(Kernel, NGramHashFwdInterface):
    devices = frozenset({"npu"})
    supported_archs = None

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self.call = call

    def forward(
        self,
        input_ids,
        context,
        multipliers,
        primes,
        offsets,
        cu_seqlens=None,
        context_indices=None,
        num_accepted_tokens=None,
    ):
        call = self.call
        total = input_ids.numel()
        output = torch.empty(
            (*input_ids.shape, call.heads), dtype=torch.int64, device=input_ids.device
        )
        if not total:
            return output
        tokens = input_ids.view(-1)
        length = 0 if call.packed else input_ids.shape[1]
        batch = cu_seqlens.numel() - 1 if call.packed else input_ids.shape[0]
        table = context.contiguous()
        cu = tokens if cu_seqlens is None else cu_seqlens
        slots = tokens if context_indices is None else context_indices
        accepted = tokens if num_accepted_tokens is None else num_accepted_tokens
        hash_ids, update = _programs(
            call.width,
            call.order,
            call.heads,
            call.eos,
            call.packed,
            call.slotted,
            call.rollback,
            batch,
            length,
            total,
            context.shape[0],
        )
        if call.heads:
            hash_ids(
                input_ids,
                table,
                multipliers,
                primes,
                offsets,
                cu,
                slots,
                accepted,
                output.view(total, call.heads),
            )
        update(input_ids, table, cu, slots, accepted)
        if table is not context:
            context.copy_(table)
        return output


register_kernel_type("NGramHashFwdOp", "ascend", AscendNGramHashKernel)
