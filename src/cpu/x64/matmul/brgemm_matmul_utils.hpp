/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef CPU_X64_MATMUL_BRGEMM_MATMUL_UTILS_HPP
#define CPU_X64_MATMUL_BRGEMM_MATMUL_UTILS_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/x64/brgemm/brgemm.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace matmul {

constexpr int max_batch_ndims = DNNL_MAX_NDIMS - 2;

struct brgemm_matmul_bcast_desc_t {
    int bcast_mask; // sets bcast_dim = 1, non_bcast_dim = 0

    int first_bcast_dim {-1};
    int last_bcast_dim {-1};

    dim_t first_bcast_dim_to_last_batch_dim_prod {1};
    dim_t bcast_dims_prod {1};

    dim_t batch_dims[max_batch_ndims];
    dim_t gb_off[max_batch_ndims]; // generalized batch offset
};

struct brgemm_matmul_conf_t {
    int ndims, batch_ndims;
    dim_t M, N, K, batch, batch_without_first_dim;
    dim_t M_blk, N_blk, K_blk, M_tail, N_tail, K_tail;
    int M_chunk_size, N_chunk_size;
    dim_t LDA, LDB, LDC, LDD;
    int brgemm_batch_size;
    int wei_n_blk, wei_k_blk;
    brgemm_batch_kind_t brg_type;

    cpu_isa_t isa;

    format_tag_t src_tag, wei_tag, dst_tag, bia_tag;
    bool with_bias;
    bool with_sum;
    bool with_eltwise;
    bool with_binary;
    bool with_scales;
    bool s8s8_compensation_required;
    bool is_oscale_per_n;
    brgemm_broadcast_t src_zp_type;
    brgemm_broadcast_t wei_zp_type;
    brgemm_broadcast_t dst_zp_type;

    bool use_buffer_a;
    bool use_buffer_a_tail_only;
    bool use_buffer_b;
    bool use_buffer_c;

    brgemm_matmul_bcast_desc_t bcast_A_desc;
    brgemm_matmul_bcast_desc_t bcast_B_desc;

    data_type_t src_dt;
    data_type_t dst_dt;
    data_type_t wei_dt;
    data_type_t acc_dt;
    data_type_t bia_dt;
    int nthr;
    int nthr_k;

    // Auxiliary values for init_config() and execute()
    dim_t a_dt_sz, b_dt_sz, c_dt_sz, acc_dt_sz, bias_dt_sz;

    int M_chunks;
    int N_chunks;
    int K_chunks;
    int num_M_blocks;
    int num_N_blocks;
    dim_t M_chunk_elems;
    dim_t N_chunk_elems;
    dim_t K_chunk_elems;

    // Pre-calculated memory strides for each tensor
    dim_t A_strides[3];
    dim_t B_strides[3];
    dim_t C_strides[3];
    dim_t buffer_c_chunk_sz;
    dim_t buffer_c_per_thread_sz;

    dim_t buffer_a_chunk_sz;
    dim_t buffer_a_chunk_shift_along_m;
    dim_t buffer_a_per_thread_sz;

    dim_t buffer_b_chunk_sz;
    dim_t buffer_b_per_thread_sz;
    dim_t s8s8_comp_ithr_str;
    dim_t s8s8_comp_b_str;
    dim_t s8s8_comp_n_str;
    bool has_zero_point_a, has_zero_point_b, has_zero_point_c;
    bool post_ops_applicable;
    bool transposed_A;
    bool blocked_B;

    dim_t zp_a_comp_shift_n;
    dim_t zp_a_comp_elems_per_thr;

    dim_t zp_b_comp_result_shift_m;
    dim_t zp_b_comp_buffer_start;
    dim_t zp_b_comp_buffer_shift_m;
    dim_t zp_b_comp_elems_per_thr;

    int wsp_tile_per_thr_bytes;
    int brgemm_batch_element_per_thr_sz;
    bool is_amx;

    int required_k_granularity;
};

status_t init_brgemm_matmul_conf(cpu_isa_t isa, brgemm_matmul_conf_t &bgmmc,
        const matmul_desc_t &mmd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, primitive_attr_t &attr);

void init_scratchpad(memory_tracking::registrar_t &scratchpad,
        const brgemm_matmul_conf_t &bgmmc);

} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
