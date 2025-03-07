/*******************************************************************************
* Copyright 2025 FUJITSU LIMITED
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef CPU_AARCH64_JIT_INT8_KERNEL_TYPES_HPP
#define CPU_AARCH64_JIT_INT8_KERNEL_TYPES_HPP

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace matmul {

typedef enum {
    none = 0,
    per_tensor = 1,
    per_m = 2,
    per_n = 3,
    per_k = 4,
} jit_int8_broadcast_t;

struct dyn_vals_t {
    int f = 0;
    dim_t M = 0;
    dim_t K = 0;
    dim_t N = 0;
    dim_t B = 0;
    int is_s8 = 0, is_u8 = 0;
    int mtail, ktail, ntail, m_blk, k_blk, n_blk;
    int get_min_max = 0, reorder_a = 0, reorder_b = 0, cal_src = 0;
    int is_mtail = 0, is_ktail = 0;
};

struct dyn_params_t {
    const float *dyn_src;
    const int8_t *src;
    int8_t *dst;
    float *max, *min;
    int *nk, *nm, *nn;
    int *tl, *mtl, *ntl;
};

struct brg_int8_t {
    int M, K, N;
    const int m_blk = 8, n_blk = 4, k_blk = 8;
    const int ld_block = 6, rd_block = 4, bd_block = 8;
    int na, nb;
    int m_tail, n_tail, k_tail;
    int is_m_tail, is_k_tail, is_n_tail, is_zp_cal;
    int dst_dt_sz;
    bool is_s8;
    bool is_bias;
    bool with_scales;
    bool with_dst_scales;
    bool is_oc_scales;
    jit_int8_broadcast_t zp_type_a = jit_int8_broadcast_t::none;
    jit_int8_broadcast_t zp_type_b = jit_int8_broadcast_t::none;
    jit_int8_broadcast_t zp_type_c = jit_int8_broadcast_t::none;
    bool is_zp_b_int8 = false;
    bool b_reo = true;
    data_type_t zp_b_dt;
    dim_t B;
};

struct call_params_t {
    const uint8_t *src, *wei;
    float *dst;
    const float *bias, *scales, *dst_scales;
    dim_t M, K, N;
    char *buf_B_ptr_;
    int *na, *nb;
    int32_t *src_zero_point, *wei_zero_point, *dst_zero_point;
    const int8_t *wei_zero_point_buf;
    float *zp_a_ptr, *zp_b_ptr;
};

} // namespace matmul
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif