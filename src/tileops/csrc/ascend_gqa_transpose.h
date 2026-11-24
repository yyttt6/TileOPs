#pragma once

#include "c_api/asc_simd.h"

namespace tileops {
// Transpose packed 16x16 FP16 blocks between the two compact NZ layouts.
__aicore__ inline void gqa_transpose_v(__ubuf__ half* dst, __ubuf__ half* src) {
  for (int d = 0; d < 128; d += 16) {
    for (int k = 0; k < 64; k += 16) {
      asc_transpose(
          reinterpret_cast<__ubuf__ uint16_t*>(dst + (k / 16 * 129 + d) * 16),
          reinterpret_cast<__ubuf__ uint16_t*>(src + (d / 16 * 65 + k) * 16));
    }
  }
}
}  // namespace tileops
