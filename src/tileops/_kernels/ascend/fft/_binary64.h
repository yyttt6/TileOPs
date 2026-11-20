#ifndef TILEOPS_ASCEND_FFT_BINARY64_H
#define TILEOPS_ASCEND_FFT_BINARY64_H

#include <stdint.h>

#if defined(__CCE_AICORE__) || defined(__ASCENDC__) || defined(TILEOPS_ASCEND_DEVICE)
#define TILEOPS_FFT_INLINE __simt_callee__ __aicore__ inline
#else
#define TILEOPS_FFT_INLINE inline
#endif

namespace tileops_fft {
constexpr uint64_t sign_bit = uint64_t{1} << 63;
constexpr uint64_t hidden_bit = uint64_t{1} << 52;
constexpr uint64_t fraction_mask = hidden_bit - 1;
constexpr uint64_t infinity = uint64_t{0x7ff} << 52;
constexpr uint64_t quiet_nan = infinity | (hidden_bit >> 1);

TILEOPS_FFT_INLINE int leading_zeros(uint64_t nonzero) {
    const uint32_t upper = uint32_t(nonzero >> 32);
    return upper ? __builtin_clz(upper) : 32 + __builtin_clz(uint32_t(nonzero));
}

// Preserve a sticky bit whenever alignment discards a nonzero bit.
TILEOPS_FFT_INLINE uint64_t shift_right_jam(uint64_t value, int shift) {
    if (shift == 0) return value;
    if (shift >= 64) return value != 0;
    return (value >> shift) | ((value << (64 - shift)) != 0);
}

// The last three bits hold guard, round and sticky. A normal significand's
// leading bit is at bit 55. Round to nearest, with ties to even.
TILEOPS_FFT_INLINE uint64_t round_pack(uint64_t sign, int exponent, uint64_t sig) {
    if (exponent < 1) {
        sig = shift_right_jam(sig, 1 - exponent);
        exponent = 1;
    }
    const uint64_t tail = sig & 7;
    uint64_t mantissa = sig >> 3;
    mantissa += tail > 4 || (tail == 4 && (mantissa & 1));
    if (mantissa >= (hidden_bit << 1)) {
        mantissa >>= 1;
        ++exponent;
    }
    if (exponent >= 0x7ff) return sign | infinity;
    if (mantissa < hidden_bit) exponent = 0;
    return sign | (uint64_t(exponent) << 52) | (mantissa & fraction_mask);
}
}  // namespace tileops_fft

// Operands and results are the raw bits of IEEE binary64. Ascend vector
// cores execute only integer instructions in these two routines.
TILEOPS_FFT_INLINE uint64_t tileops_fft_add64(uint64_t a, uint64_t b) {
    using namespace tileops_fft;
    uint64_t ma = a & ~sign_bit, mb = b & ~sign_bit;
    if (ma < mb) {
        const uint64_t swap = a;
        a = b;
        b = swap;
        ma = a & ~sign_bit;
        mb = b & ~sign_bit;
    }
    if (ma >= infinity) {
        if (ma > infinity || (mb == infinity && ((a ^ b) & sign_bit))) return quiet_nan;
        return a;
    }
    const uint64_t sign = a & sign_bit;
    if (mb == 0) return ma == 0 ? ((a & b) & sign_bit) : a;
    int ea = int(ma >> 52), eb = int(mb >> 52);
    uint64_t sa = (ma & fraction_mask) | (ea == 0 ? 0 : hidden_bit);
    uint64_t sb = (mb & fraction_mask) | (eb == 0 ? 0 : hidden_bit);
    if (ea == 0) ea = 1;
    if (eb == 0) eb = 1;
    sa <<= 3;
    sb = shift_right_jam(sb << 3, ea - eb);
    if ((a ^ b) & sign_bit) {
        sa -= sb;
        if (sa == 0) return 0;
        const int distance = leading_zeros(sa) - 8;
        const int shift = distance < ea - 1 ? distance : ea - 1;
        sa <<= shift;
        ea -= shift;
    } else {
        sa += sb;
        if (sa & (hidden_bit << 4)) {
            sa = shift_right_jam(sa, 1);
            ++ea;
        }
    }
    return round_pack(sign, ea, sa);
}

TILEOPS_FFT_INLINE uint64_t tileops_fft_mul64(uint64_t a, uint64_t b) {
    using namespace tileops_fft;
    const uint64_t sign = (a ^ b) & sign_bit;
    const uint64_t ma = a & ~sign_bit, mb = b & ~sign_bit;
    if (ma > infinity || mb > infinity ||
        (ma == infinity && mb == 0) || (mb == infinity && ma == 0)) return quiet_nan;
    if (ma == infinity || mb == infinity) return sign | infinity;
    if (ma == 0 || mb == 0) return sign;

    int ea = int(ma >> 52), eb = int(mb >> 52);
    uint64_t sa = (ma & fraction_mask) | (ea == 0 ? 0 : hidden_bit);
    uint64_t sb = (mb & fraction_mask) | (eb == 0 ? 0 : hidden_bit);
    if (ea == 0) {
        const int shift = leading_zeros(sa) - 11;
        sa <<= shift;
        ea = 1 - shift;
    }
    if (eb == 0) {
        const int shift = leading_zeros(sb) - 11;
        sb <<= shift;
        eb = 1 - shift;
    }
    // 53-by-53 bit multiplication via two 32-bit limbs. A 128-bit device
    // type is unnecessary, and the cross products fit in 54 bits.
    const uint64_t a0 = uint32_t(sa), a1 = sa >> 32;
    const uint64_t b0 = uint32_t(sb), b1 = sb >> 32;
    const uint64_t p0 = a0 * b0;
    const uint64_t cross = a0 * b1 + a1 * b0 + (p0 >> 32);
    const uint64_t low = (cross << 32) | uint32_t(p0);
    const uint64_t high = a1 * b1 + (cross >> 32);
    const int carry = (high >> 41) != 0;
    const int shift = 49 + carry;
    const uint64_t sig = (high << (64 - shift)) | shift_right_jam(low, shift);
    return round_pack(sign, ea + eb - 1023 + carry, sig);
}

#undef TILEOPS_FFT_INLINE
#endif
