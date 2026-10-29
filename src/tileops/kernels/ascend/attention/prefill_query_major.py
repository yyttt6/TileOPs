"""Fused BF16 GQA prefill with explicit Cube/Vector scheduling.

Adapted from the user's FIA aligned pipeline (MIT, Copyright Tile-AI).
Dispatch is restricted to the shapes validated on Ascend 950DT.
"""

import functools
import math

import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
from tilelang.layout import make_ascend_compact_nz_layout


@functools.lru_cache(maxsize=128)
def prefill_query_major(call, num_cores=36):
    q_len, kv_len = call.max_seqlen_q, call.seqlen_kv
    batch_size, num_heads, num_kv_heads = call.batch, call.heads, call.heads_kv
    group = num_heads // num_kv_heads
    out_dtype = str(call.dtype).split(".")[-1]
    softmax_scale = call.sm_scale
    assert call.dim == 128 and out_dtype == "bfloat16"
    assert q_len % 128 == 0 and kv_len % 128 == 0 and kv_len >= q_len + 128
    assert call.is_causal and call.softcap == 0
    assert not call.is_fp8 and not call.fuse_rope
    assert call.window_size_left == -1 and call.window_size_right == -1
    assert num_cores > 0
    BR, BC, ROWS, VL, D, D0 = 128, 128, 64, 64, 128, 128
    M_TILES = q_len // BR
    TASKS = batch_size * num_heads * M_TILES
    # Every launched block must own a Q tile: the bounds assumptions below
    # allow lowering to remove the zero-task path for otherwise idle blocks.
    num_cores = min(num_cores, TASKS)
    PREFIX = (kv_len - q_len) // BC + 1
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(D)

    @T.macro
    def softmax(
        score,
        prob,
        maximum,
        denominator,
        alpha,
        old_max,
        row_sum,
        task,
        sequence,
        sid,
        first,
        masked,
    ):
        with T.SimdVF():
            full = S.pset(32, "PAT_ALL")
            one = S.pset(32, "PAT_VL1")
            b16 = S.pset(16, "PAT_VL128")
            ss = sequence % 3
            ts = task % 3
            dst = S.make_ubuf_ptr(
                T.access_ptr(prob[sequence % 2, 0, 0], "w", ROWS + 1, BC), "bfloat16"
            )
            if not first:
                S.vsts(old_max[0], S.vld(maximum[ts, 0], "NORM"), full, "NORM_B32")
            for r in range(0, ROWS, 2):
                s00 = S.vmuls(S.vld(score[ss, r, 0], "NORM"), T.float32(scale), full)
                s01 = S.vmuls(S.vld(score[ss, r, VL], "NORM"), T.float32(scale), full)
                s10 = S.vmuls(S.vld(score[ss, r + 1, 0], "NORM"), T.float32(scale), full)
                s11 = S.vmuls(S.vld(score[ss, r + 1, VL], "NORM"), T.float32(scale), full)
                if masked:
                    neg = S.vdup(T.float32(-3.402823e38), "float32", full)
                    masked00 = S.vsel(s00, neg, S.update_mask(T.min(VL, sid * ROWS + r + 1), 32))
                    masked01 = S.vsel(
                        s01, neg, S.update_mask(T.max(0, sid * ROWS + r + 1 - VL), 32)
                    )
                    masked10 = S.vsel(s10, neg, S.update_mask(T.min(VL, sid * ROWS + r + 2), 32))
                    masked11 = S.vsel(
                        s11, neg, S.update_mask(T.max(0, sid * ROWS + r + 2 - VL), 32)
                    )
                else:
                    masked00, masked01, masked10, masked11 = s00, s01, s10, s11
                S.vsts(score[ss, r, 0], masked00, full, "NORM_B32")
                S.vsts(score[ss, r, VL], masked01, full, "NORM_B32")
                S.vsts(score[ss, r + 1, 0], masked10, full, "NORM_B32")
                S.vsts(score[ss, r + 1, VL], masked11, full, "NORM_B32")
                max0 = S.vcmax(S.vmax(masked00, masked01, full), full)
                max1 = S.vcmax(S.vmax(masked10, masked11, full), full)
                S.vsts(maximum[ts, r], max0, one, "ONEPT_B32")
                S.vsts(maximum[ts, r + 1], max1, one, "ONEPT_B32")
            S.mem_bar("VST_VLD")
            if not first:
                merged = S.vmax(S.vld(maximum[ts, 0], "NORM"), S.vld(old_max[0], "NORM"), full)
                S.vsts(maximum[ts, 0], merged, full, "NORM_B32")
                S.mem_bar("VST_VLD")
            for r in range(0, ROWS, 2):
                m0 = S.vld(maximum[ts, r], "BRC_B32")
                m1 = S.vld(maximum[ts, r + 1], "BRC_B32")
                x00, x01 = S.vld2(score[ss, r, 0], "DINTLV_B32")
                x10, x11 = S.vld2(score[ss, r + 1, 0], "DINTLV_B32")
                e00 = S.vexpdif(x00, m0, full)
                e01 = S.vexpdif(x01, m0, full)
                e10 = S.vexpdif(x10, m1, full)
                e11 = S.vexpdif(x11, m1, full)
                sum0 = S.vcadd(S.vadd(e00, e01, full), full)
                sum1 = S.vcadd(S.vadd(e10, e11, full), full)
                S.vsts(row_sum[r], sum0, one, "ONEPT_B32")
                S.vsts(row_sum[r + 1], sum1, one, "ONEPT_B32")
                even0 = S.vcvt(e00, "bfloat16", full, sat=False, part=0)
                even1 = S.vcvt(e10, "bfloat16", full, sat=False, part=0)
                odd0 = S.vcvt(e01, "bfloat16", full, sat=False, part=1)
                odd1 = S.vcvt(e11, "bfloat16", full, sat=False, part=1)
                bits0 = S.vor(  # codespell:ignore vor
                    T.reinterpret(even0, "uint16x128"), T.reinterpret(odd0, "uint16x128"), b16
                )
                bits1 = S.vor(  # codespell:ignore vor
                    T.reinterpret(even1, "uint16x128"), T.reinterpret(odd1, "uint16x128"), b16
                )
                dst = S.vsstb(
                    T.reinterpret(bits0, "bfloat16x128"),
                    dst,
                    T.int32(((ROWS + 1) << 16) | 1),
                    b16,
                    update=True,
                )
                dst = S.vsstb(
                    T.reinterpret(bits1, "bfloat16x128"),
                    dst,
                    T.int32(((ROWS + 1) << 16) | 1),
                    b16,
                    update=True,
                )
            S.mem_bar("VST_VLD")
            if first:
                S.vsts(denominator[ts, 0], S.vld(row_sum[0], "NORM"), full, "NORM_B32")
            else:
                a = S.vexpdif(S.vld(old_max[0], "NORM"), S.vld(maximum[ts, 0], "NORM"), full)
                S.vsts(alpha[ss, 0], a, full, "NORM_B32")
                updated = S.vadd(
                    S.vmul(S.vld(denominator[ts, 0], "NORM"), a, full),
                    S.vld(row_sum[0], "NORM"),
                    full,
                )
                S.vsts(denominator[ts, 0], updated, full, "NORM_B32")

    @T.macro
    def update_output(pv, accum, output, alpha, denominator, task, sequence, first, last):
        with T.SimdVF():
            full = S.pset(32, "PAT_ALL")
            for r in range(ROWS):
                for c in range(0, D0, VL):
                    current = S.vld(pv[sequence % 2, r, c], "NORM")
                    if first:
                        updated = current
                    else:
                        a = S.vld(alpha[sequence % 3, r], "BRC_B32")
                        prev = S.vld(accum[r, c], "NORM")
                        updated = S.vadd(S.vmul(a, prev, full), current, full)
                    if last:
                        d = S.vld(denominator[task % 3, r], "BRC_B32")
                        normalized = S.vdiv(updated, d, full)
                        bf16 = S.vcvt(normalized, "bfloat16", full)
                        S.vsts(output[r, c], bf16, full, "PK_B32")
                    else:
                        S.vsts(accum[r, c], updated, full, "NORM_B32")

    @T.prim_func
    def main(
        Q: T.Tensor((batch_size, q_len, num_heads, D), "bfloat16"),
        K: T.Tensor((batch_size, kv_len, num_kv_heads, D), "bfloat16"),
        V: T.Tensor((batch_size, kv_len, num_kv_heads, D0), "bfloat16"),
        O: T.Tensor((batch_size, q_len, num_heads, D0), "bfloat16"),
    ):
        with T.Kernel(num_cores) as cid:
            T.assume(q_len > 0)
            T.assume(kv_len > 0)
            T.assume(batch_size > 0)
            T.assume(num_heads > 0)
            T.assume(q_len % 128 == 0)
            T.assume(kv_len % 128 == 0)
            T.assume(kv_len >= q_len + 128)
            q_l1 = T.alloc_l1((2, BR, D), "bfloat16")
            k_l1 = T.alloc_l1((3, BC, D), "bfloat16")
            v_l1 = T.alloc_l1((3, D0, BC), "bfloat16")
            p_l1 = T.alloc_l1((2, BR, BC), "bfloat16")
            # Shared L0 rings have fixed 32 KiB slots for BOTH GEMMs.
            a_l0 = T.alloc_l0a((2, BR, BC), "bfloat16")
            b_l0 = T.alloc_l0b((2, BC, BC), "bfloat16")
            qk_l0 = T.alloc_l0c((2, BR, BC), "float32")
            pv_l0 = T.alloc_l0c((2, BR, D0), "float32")
            score = T.alloc_shared((3, ROWS, BC), "float32")
            prob = T.alloc_shared((2, ROWS + 1, BC), "bfloat16")
            T.annotate_layout({prob: make_ascend_compact_nz_layout(prob)})
            pv = T.alloc_shared((2, ROWS, D0), "float32")
            accum = T.alloc_shared((ROWS, D0), "float32")
            maximum = T.alloc_shared((3, ROWS), "float32")
            denominator = T.alloc_shared((3, ROWS), "float32")
            alpha = T.alloc_shared((3, ROWS), "float32")
            old_max = T.alloc_shared((ROWS,), "float32")
            row_sum = T.alloc_shared((ROWS,), "float32")
            output = T.alloc_shared((ROWS, D0), "bfloat16")

            tasks = T.ceildiv(TASKS - cid, num_cores)
            total = T.alloc_var("int32", init=0)
            for task in T.serial(tasks):
                total += (task * num_cores + cid) % M_TILES + PREFIX
            T.assume(total >= 0)
            T.assume(total <= T.ceildiv(TASKS, num_cores) * (kv_len // BC))

            with T.Cube():
                for e in T.serial(8):
                    T.ascend_set_flag("MTE1_MTE2", e)
                for e in T.serial(4):
                    T.ascend_set_flag("FIX_M", e)
                for e in T.serial(2):
                    T.ascend_set_flag("M_MTE1", e)
                    T.ascend_cross_core_set_flag(4, "PIPE_MTE1", 5 + e * 16)
                    T.ascend_cross_core_set_flag(4, "PIPE_MTE1", 12 + e * 16)
                tk = T.alloc_var("int32", init=0)
                kk = T.alloc_var("int32", init=0)
                tq = T.alloc_var("int32", init=0)
                kq = T.alloc_var("int32", init=0)
                tv = T.alloc_var("int32", init=0)
                kv = T.alloc_var("int32", init=0)
                l0_sequence = T.alloc_var("int32", init=0)

                # t: load K/Q; t-1: QK; t-2: load V / QK fix;
                # t-4: PV; t-5: PV fix. Drain once for ALL Q tasks.
                for step in T.serial(total + 5):
                    if step < total:
                        tile = tk * num_cores + cid
                        # Cursor invariants: the stream contains exactly the
                        # valid Q tasks and their visible KV tiles. Expose the
                        # bounds so OOB lowering need not add dynamic padding.
                        T.assume(0 <= tile < TASKS)
                        T.assume(0 <= kk < kv_len // BC)
                        batch = tile // (num_heads * M_TILES)
                        head = tile // M_TILES % num_heads
                        mt = tile % M_TILES
                        if kk == 0:
                            T.ascend_wait_flag("MTE1_MTE2", 6 + tk % 2)
                            T.copy(Q[batch, mt * BR : (mt + 1) * BR, head, :], q_l1[tk % 2, :, :])
                            T.ascend_set_flag("MTE2_MTE1", 6 + tk % 2)
                        T.ascend_wait_flag("MTE1_MTE2", 3 + step % 3)
                        T.copy(
                            K[batch, kk * BC : (kk + 1) * BC, head // group, :],
                            k_l1[step % 3, :, :],
                        )
                        T.ascend_set_flag("MTE2_MTE1", 3 + step % 3)
                        kk += 1
                        if kk == mt + PREFIX:
                            kk = 0
                            tk += 1

                    if step >= 1 and step < total + 1:
                        g = step - 1
                        tile = tq * num_cores + cid
                        T.ascend_wait_flag("FIX_M", 2 + g % 2)
                        if kq == 0:
                            T.ascend_wait_flag("MTE2_MTE1", 6 + tq % 2)
                        T.ascend_wait_flag("MTE2_MTE1", 3 + g % 3)
                        for sk in T.serial(1):
                            ls = T.floormod(l0_sequence + sk, 2)
                            T.assume(0 <= ls < 2)
                            T.ascend_wait_flag("M_MTE1", ls)
                            T.copy(q_l1[tq % 2, :, sk * 128 : (sk + 1) * 128], a_l0[ls, :, :128])
                            T.copy(k_l1[g % 3, :, sk * 128 : (sk + 1) * 128], b_l0[ls, :, :128])
                            T.ascend_set_flag("MTE1_M", ls)
                            T.ascend_wait_flag("MTE1_M", ls)
                            T.gemm(
                                a_l0[ls, :, :128],
                                b_l0[ls, :, :128],
                                qk_l0[g % 2, :, :],
                                transpose_B=True,
                                clear_accum=sk == 0,
                            )
                            T.ascend_set_flag("M_MTE1", ls)
                        l0_sequence += 1
                        T.ascend_set_flag("M_FIX", 2 + g % 2)
                        T.ascend_set_flag("MTE1_MTE2", 3 + g % 3)
                        kq += 1
                        if kq == tile % M_TILES + PREFIX:
                            T.ascend_set_flag("MTE1_MTE2", 6 + tq % 2)
                            kq = 0
                            tq += 1

                    if step >= 2 and step < total + 2:
                        g = step - 2
                        tile = tv * num_cores + cid
                        T.assume(0 <= tile < TASKS)
                        T.assume(0 <= kv < kv_len // BC)
                        batch = tile // (num_heads * M_TILES)
                        head = tile // M_TILES % num_heads
                        T.ascend_wait_flag("MTE1_MTE2", g % 3)
                        T.copy(
                            V[batch, kv * BC : (kv + 1) * BC, head // group, :],
                            v_l1[g % 3, :, :],
                            transpose=True,
                        )
                        T.ascend_set_flag("MTE2_MTE1", g % 3)
                        kv += 1
                        if kv == tile % M_TILES + PREFIX:
                            kv = 0
                            tv += 1

                    if step >= 4 and step < total + 4:
                        g = step - 4
                        ready = T.if_then_else(g % 2 == 0, 11, 13)
                        free = T.if_then_else(g % 2 == 0, 5, 12)
                        T.ascend_wait_flag("FIX_M", g % 2)
                        T.ascend_wait_flag("MTE2_MTE1", g % 3)
                        T.ascend_cross_core_wait_flag(4, "PIPE_MTE1", ready)
                        T.ascend_cross_core_wait_flag(4, "PIPE_MTE1", ready + 16)
                        ls = T.floormod(l0_sequence, 2)
                        T.assume(0 <= ls < 2)
                        T.ascend_wait_flag("M_MTE1", ls)
                        T.copy(p_l1[g % 2, :, :], a_l0[ls, :, :])
                        T.copy(v_l1[g % 3, :, :], b_l0[ls, :, :])
                        T.ascend_set_flag("MTE1_M", ls)
                        T.ascend_wait_flag("MTE1_M", ls)
                        T.gemm(
                            a_l0[ls, :, :],
                            b_l0[ls, :, :],
                            pv_l0[g % 2, :, :],
                            transpose_B=True,
                            clear_accum=True,
                        )
                        T.ascend_set_flag("M_MTE1", ls)
                        l0_sequence += 1
                        T.ascend_set_flag("M_FIX", g % 2)
                        T.ascend_cross_core_set_flag(4, "PIPE_MTE1", free)
                        T.ascend_cross_core_set_flag(4, "PIPE_MTE1", free + 16)
                        T.ascend_set_flag("MTE1_MTE2", g % 3)

                    if step >= 5:
                        g = step - 5
                        T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 3 + g % 2)
                        T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 19 + g % 2)
                        T.ascend_wait_flag("M_FIX", g % 2)
                        T.dual_copy(pv_l0[g % 2, :, :], pv[g % 2, :, :])
                        T.ascend_cross_core_set_flag(4, "PIPE_FIX", 9 + g % 2)
                        T.ascend_cross_core_set_flag(4, "PIPE_FIX", 25 + g % 2)
                        T.ascend_set_flag("FIX_M", g % 2)

                    if step >= 2 and step < total + 2:
                        g = step - 2
                        T.ascend_cross_core_wait_flag(4, "PIPE_FIX", g % 3)
                        T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 16 + g % 3)
                        T.ascend_wait_flag("M_FIX", 2 + g % 2)
                        T.dual_copy(qk_l0[g % 2, :, :], score[g % 3, :, :])
                        T.ascend_cross_core_set_flag(4, "PIPE_FIX", 6 + g % 3)
                        T.ascend_cross_core_set_flag(4, "PIPE_FIX", 22 + g % 3)
                        T.ascend_set_flag("FIX_M", 2 + g % 2)

                for e in T.serial(8):
                    T.ascend_wait_flag("MTE1_MTE2", e)
                for e in T.serial(4):
                    T.ascend_wait_flag("FIX_M", e)
                for sub in T.serial(2):
                    T.ascend_wait_flag("M_MTE1", sub)
                    for e in T.serial(5):
                        T.ascend_cross_core_wait_flag(4, "PIPE_FIX", e + sub * 16)

            with T.Vector() as sid:
                for e in T.serial(5):
                    T.ascend_cross_core_set_flag(4, "PIPE_V", e)
                for e in T.serial(3):
                    T.ascend_set_flag("MTE3_V", e)
                st = T.alloc_var("int32", init=0)
                skv = T.alloc_var("int32", init=0)
                output_task = T.alloc_var("int32", init=0)
                okv = T.alloc_var("int32", init=0)

                # Consume g=t-3 BEFORE producing t: alpha has three slots.
                # Separate Q-task cursors keep the previous denominator live.
                for step in T.serial(total + 3):
                    if step >= 3:
                        g = step - 3
                        tile = output_task * num_cores + cid
                        last = okv == tile % M_TILES + PREFIX - 1
                        T.ascend_cross_core_wait_flag(4, "PIPE_V", 9 + g % 2)
                        if last:
                            T.ascend_wait_flag("MTE3_V", 2)
                            update_output(
                                pv, accum, output, alpha, denominator, output_task, g, False, True
                            )
                        elif okv == 0:
                            update_output(
                                pv, accum, output, alpha, denominator, output_task, g, True, False
                            )
                        else:
                            update_output(
                                pv, accum, output, alpha, denominator, output_task, g, False, False
                            )
                        T.ascend_cross_core_set_flag(4, "PIPE_V", 3 + g % 2)
                        if last:
                            T.assume(0 <= tile < TASKS)
                            batch = tile // (num_heads * M_TILES)
                            head = tile // M_TILES % num_heads
                            mt = tile % M_TILES
                            T.ascend_set_flag("V_MTE3", 2)
                            T.ascend_wait_flag("V_MTE3", 2)
                            T.copy(
                                output,
                                O[
                                    batch,
                                    mt * BR + sid * ROWS : mt * BR + (sid + 1) * ROWS,
                                    head,
                                    :,
                                ],
                                l2_cache_ctrl=0,
                            )
                            T.ascend_set_flag("MTE3_V", 2)
                            okv = 0
                            output_task += 1
                        else:
                            okv += 1

                    if step < total:
                        tile = st * num_cores + cid
                        last = skv == tile % M_TILES + PREFIX - 1
                        T.ascend_wait_flag("MTE3_V", step % 2)
                        T.ascend_cross_core_wait_flag(4, "PIPE_V", 6 + step % 3)
                        if skv == 0:
                            softmax(
                                score,
                                prob,
                                maximum,
                                denominator,
                                alpha,
                                old_max,
                                row_sum,
                                st,
                                step,
                                sid,
                                True,
                                False,
                            )
                        elif last:
                            softmax(
                                score,
                                prob,
                                maximum,
                                denominator,
                                alpha,
                                old_max,
                                row_sum,
                                st,
                                step,
                                sid,
                                False,
                                True,
                            )
                        else:
                            softmax(
                                score,
                                prob,
                                maximum,
                                denominator,
                                alpha,
                                old_max,
                                row_sum,
                                st,
                                step,
                                sid,
                                False,
                                False,
                            )
                        T.ascend_set_flag("V_MTE3", step % 2)
                        T.ascend_cross_core_set_flag(4, "PIPE_V", step % 3)
                        T.ascend_cross_core_wait_flag(
                            4, "PIPE_MTE3", T.if_then_else(step % 2 == 0, 5, 12)
                        )
                        T.ascend_wait_flag("V_MTE3", step % 2)
                        T.dual_copy(prob[step % 2, :ROWS, :], p_l1[step % 2, :, :])
                        T.ascend_cross_core_set_flag(
                            4, "PIPE_MTE3", T.if_then_else(step % 2 == 0, 11, 13)
                        )
                        T.ascend_set_flag("MTE3_V", step % 2)
                        if last:
                            skv = 0
                            st += 1
                        else:
                            skv += 1

                for e in T.serial(3):
                    T.ascend_wait_flag("MTE3_V", e)
                T.ascend_cross_core_wait_flag(4, "PIPE_MTE3", 5)
                T.ascend_cross_core_wait_flag(4, "PIPE_MTE3", 12)

    return main
