#pragma once

#include <tl_templates/ascend/common.h>

namespace tileops {
template <typename A, typename B>
__aicore__ inline void conv_folded_mad(__cc__ float* acc, A a, B b, int m,
                                       int k, int n) {
  asc_mmad(acc, a, b, m, k, n, 0, 1, 0, true);
}

template <typename D>
__aicore__ inline void conv_native_fill(D dst, int blocks, uint32_t bits) {
  asc_fill_value_config config;
  config.repeat = 1;
  config.blk_num = blocks;
  config.dst_gap = 0;
  asc_fill_l1((__cbuf__ uint16_t*)dst, bits, config);
}

template <typename D, typename S>
__aicore__ inline void conv_native_load3d(D dst, S src, int h, int w, int pl,
                                          int pr, int bk, int bn, int ms,
                                          int sh, int sw, int kh, int kw,
                                          int dh, int dw, int channels) {
  set_fmatrix(uint64_t(w) | (uint64_t(h) << 16) | (uint64_t(pl) << 32) |
              (uint64_t(pr) << 40));
  set_padding(uint64_t(0));
  set_l3d_rpt((uint64_t(bn / 16) << 32) | (uint64_t(1) << 16));
  img2colv2_cbuf_to_ca((__ca__ half*)dst, (__cbuf__ half*)src, bk, bn, 0, ms,
                       sw, sh, kw, kh, dw, dh, false, false, false, false,
                       channels);
}

template <typename D>
__aicore__ inline void conv_native_store(D dst, __cc__ float* src, int n, int m,
                                         int dst_stride, int src_stride,
                                         int quant) {
  asc_set_l0c_copy_nz_para(1, 0, 0);
  asc_set_l0c_copy_channel_para(1);
  asc_copy_l0c2gm(dst, src, n, m, dst_stride, src_stride,
                  static_cast<asc_store_l2_cache_mode>(0),
                  static_cast<asc_unit_flag_mode>(0),
                  static_cast<asc_quant_mode>(quant),
                  static_cast<asc_relu_pre_mode>(0), false, false, true, false);
}
}  // namespace tileops

namespace tileops {
template <typename D, typename S>
__aicore__ inline void conv_native_input_small(D dst, S src, uint64_t stride,
                                               int positions, int channels) {
  asc_set_gm2l1_nz_para(1, 1, 1, 0);
  asc_copy_gm2l1_dn2nz((__cbuf__ half*)dst, (__gm__ half*)src, stride,
                       asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, positions,
                       channels, 0, true);
}
}  // namespace tileops

namespace tileops {
template <typename D, typename S>
__aicore__ inline void conv_bias_gm(D dst, S src, int count) {
  asc_copy_gm2l1(dst, src, 1, count / 16, asc_channel_pad_mode::NONE, 0, 0);
}
template <typename S>
__aicore__ inline void conv_bias_bt(S src, int count, int convert) {
  asc_copy_l12bt(uint64_t(0), src, convert, 1, count / 16, 0, 0);
}
template <typename A, typename B>
__aicore__ inline void conv_bias_mad(__cc__ float* acc, A a, B b, int m, int k,
                                     int n, uint64_t offset) {
  asc_mmad(acc, a, b, offset, m, k, n, asc_unit_flag_mode::DISABLE, true);
}
}  // namespace tileops

namespace tileops {
template <typename D, typename S>
__aicore__ inline void conv_resident_img(D dst, S src, int h, int w, int pl,
                                         int pr, int bk, int bn, int ms, int ks,
                                         int sh, int sw, int kh, int kw, int dh,
                                         int dw, int channels) {
  set_fmatrix(uint64_t(w) | (uint64_t(h) << 16) | (uint64_t(pl) << 32) |
              (uint64_t(pr) << 40));
  set_padding(uint64_t(0));
  set_l3d_rpt((uint64_t(bn / 16) << 32) | (uint64_t(1) << 16));
  img2colv2_cbuf_to_ca((__ca__ half*)dst, (__cbuf__ half*)src, bk, bn, ks, ms,
                       sw, sh, kw, kh, dw, dh, false, false, false, false,
                       channels);
}
}  // namespace tileops
