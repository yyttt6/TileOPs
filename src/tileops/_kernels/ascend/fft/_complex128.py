"""Device binary64 FFT using integer significand arithmetic on Ascend."""

import functools
import hashlib
from pathlib import Path

import tilelang
import tilelang.ascend.language as T
import torch

MAX_LENGTH = 1 << 20
_HEADER = Path(__file__).with_name("_binary64.h")


def _reverse_program(batch: int, n: int, cores: int, conjugate: bool):
    size = batch * n
    width = 256
    blocks = min(cores, (size + width - 1) // width)
    bit_count = n.bit_length() - 1

    @T.prim_func
    def fft128_reverse(source: T.Tensor((size, 2), "int64"), out: T.Tensor((size, 2), "int64")):
        with T.Kernel(blocks) as block, T.SimtVF(threads=width):
            for wave in T.serial(T.ceildiv(size, blocks * width)):
                for lane in T.Parallel(width):
                    i = (wave * blocks + block) * width + lane
                    if i < size:
                        value = T.cast(i % n, "uint32")
                        a = ((value >> 1) & 0x55555555) | ((value & 0x55555555) << 1)
                        b = ((a >> 2) & 0x33333333) | ((a & 0x33333333) << 2)
                        c = ((b >> 4) & 0x0F0F0F0F) | ((b & 0x0F0F0F0F) << 4)
                        d = ((c >> 8) & 0x00FF00FF) | ((c & 0x00FF00FF) << 8)
                        e = (d >> 16) | (d << 16)
                        src = (i // n) * n + T.cast(e >> (32 - bit_count), "int32")
                        out[i, 0], out[i, 1] = source[src, 0], source[src, 1]

    return tilelang.compile(fft128_reverse, target="ascend", out_idx=[1])


def _butterfly_program(batch: int, n: int, cores: int):
    size = batch * n
    count = size // 2
    width = 256
    blocks = min(cores, (count + width - 1) // width)
    sign = 1 << 63
    stages = n.bit_length() - 1

    @T.prim_func
    def fft128_butterfly(
        data: T.Tensor((size, 2), "int64"),
        roots: T.Tensor((n // 2, 2), "int64"),
        phase: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(blocks) as block:  # noqa: SIM117 - TileLang requires separate scopes.
            with T.SimtVF(threads=width):
                for wave in T.serial(T.ceildiv(count, blocks * width)):
                    for lane in T.Parallel(width):
                        i = (wave * blocks + block) * width + lane
                        if i < count:
                            stage = phase[0]
                            half = T.shift_left(1, stage)
                            j = i & (half - 1)
                            p = (i << 1) - j
                            q = p + half
                            ar = T.reinterpret(data[p, 0], "uint64")
                            ai = T.reinterpret(data[p, 1], "uint64")
                            br = T.reinterpret(data[q, 0], "uint64")
                            bi = T.reinterpret(data[q, 1], "uint64")
                            tr = T.alloc_var("uint64")
                            ti = T.alloc_var("uint64")
                            if j == 0:
                                tr, ti = br, bi
                            elif j * 2 == half:
                                tr, ti = bi, br ^ T.uint64(sign)
                            else:
                                k = T.shift_left(j, stages - stage - 1)
                                wr = T.reinterpret(roots[k, 0], "uint64")
                                wi = T.reinterpret(roots[k, 1], "uint64")
                                rr = T.call_pure_extern(
                                    "uint64", "tileops_fft_mul64", br, wr
                                )  # lint: extern-target allow: binary64 significand arithmetic on Ascend.
                                ii = T.call_pure_extern(
                                    "uint64", "tileops_fft_mul64", bi, wi
                                )  # lint: extern-target allow: binary64 significand arithmetic on Ascend.
                                ri = T.call_pure_extern(
                                    "uint64", "tileops_fft_mul64", br, wi
                                )  # lint: extern-target allow: binary64 significand arithmetic on Ascend.
                                ir = T.call_pure_extern(
                                    "uint64", "tileops_fft_mul64", bi, wr
                                )  # lint: extern-target allow: binary64 significand arithmetic on Ascend.
                                tr = T.call_pure_extern(
                                    "uint64", "tileops_fft_add64", rr, ii ^ T.uint64(sign)
                                )  # lint: extern-target allow: binary64 significand arithmetic on Ascend.
                                ti = T.call_pure_extern(
                                    "uint64", "tileops_fft_add64", ri, ir
                                )  # lint: extern-target allow: binary64 significand arithmetic on Ascend.
                            pr = T.call_pure_extern(
                                "uint64", "tileops_fft_add64", ar, tr
                            )  # lint: extern-target allow: binary64 significand arithmetic on Ascend.
                            pi = T.call_pure_extern(
                                "uint64", "tileops_fft_add64", ai, ti
                            )  # lint: extern-target allow: binary64 significand arithmetic on Ascend.
                            qr = T.call_pure_extern(
                                "uint64", "tileops_fft_add64", ar, tr ^ T.uint64(sign)
                            )  # lint: extern-target allow: binary64 significand arithmetic on Ascend.
                            qi = T.call_pure_extern(
                                "uint64", "tileops_fft_add64", ai, ti ^ T.uint64(sign)
                            )  # lint: extern-target allow: binary64 significand arithmetic on Ascend.
                            data[p, 0], data[p, 1] = (
                                T.reinterpret(pr, "int64"),
                                T.reinterpret(pi, "int64"),
                            )
                            data[q, 0], data[q, 1] = (
                                T.reinterpret(qr, "int64"),
                                T.reinterpret(qi, "int64"),
                            )

    # Include the header's bytes in the compiler cache identity.
    digest = hashlib.sha256(_HEADER.read_bytes()).hexdigest()
    flags = [
        "-DTILEOPS_ASCEND_DEVICE",
        f"-DTILEOPS_FFT_HEADER_SHA=h{digest}",
        "-include",
        str(_HEADER),
    ]
    return tilelang.compile(
        fft128_butterfly,
        target="ascend",
        out_idx=[],
        pass_configs={"tl.device_compile_flags": flags},
    )


@functools.lru_cache(maxsize=32)
def _programs(batch: int, n: int, cores: int, device: torch.device, conjugate: bool):
    return _reverse_program(batch, n, cores, conjugate), _butterfly_program(batch, n, cores)


@functools.lru_cache(maxsize=16)
def _constants(n: int, device: torch.device):
    # Roots are shape constants. The signal and every butterfly stay on device.
    theta = (-2.0 * torch.pi / n) * torch.arange(n // 2, dtype=torch.float64, device="cpu")
    roots = torch.stack((theta.cos(), theta.sin()), dim=-1).view(torch.int64).to(device)
    phases = tuple(
        torch.tensor([s], dtype=torch.int32, device="cpu").to(device)
        for s in range(n.bit_length() - 1)
    )
    return roots, phases


def fft_complex128(x: torch.Tensor, n: int, cores: int) -> torch.Tensor:
    # Resolve the conjugate bit in the integer load, avoiding a device cast.
    conjugate = x.is_conj()
    dense = (x.conj() if conjugate else x).contiguous()
    batch = dense.numel() // n
    reverse, butterfly = _programs(batch, n, cores, dense.device, conjugate)
    roots, phases = _constants(n, dense.device)
    result = reverse(dense.view(torch.int64).reshape(-1, 2))
    for phase in phases:
        butterfly(result, roots, phase)
    return result.view(torch.complex128).reshape(dense.shape)
