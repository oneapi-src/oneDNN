/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef OCL_RNN_UTILS_HPP
#define OCL_RNN_UTILS_HPP

#include "ocl_rnn_pd.hpp"

#define OFF6(i0,d0,i1,d1,i2,d2,i3,d3,i4,d4,i5,d5) \
    ((((((i0)*(d1)+(i1))*(d2)+(i2))*(d3)+(i3))*(d4)+(i4))*(d5)+(i5))
#define OFF5(i0,d0,i1,d1,i2,d2,i3,d3,i4,d4) \
    (((((i0)*(d1)+(i1))*(d2)+(i2))*(d3)+(i3))*(d4)+(i4))
#define OFF4(i0,d0,i1,d1,i2,d2,i3,d3) \
    ((((i0)*(d1)+(i1))*(d2)+(i2))*(d3)+(i3))
#define OFF3(i0,d0,i1,d1,i2,d2) \
    (((i0)*(d1)+(i1))*(d2)+(i2))
#define OFF2(i0,d0,i1,d1) \
    ((i0)*(d1)+(i1))

#define elemwise_sig(f)                                                 \
    void f(const exec_ctx_t &ctx, int dir, int lay, int iter, \
            int dic, int wic, int batch, \
            const memory_storage_t &workspace, \
            const memory_storage_t &bias) const

#define cell_execution_sig(f)                                                 \
    void f(const exec_ctx_t &ctx, \
            int dir, int lay, int iter, \
            int dic, int slc, int sic, int wic, int batch, int n_layer,     \
            int n_direction, int n_iter, int n_gates, int n_states,        \
            int n_bias, size_t *weights_input, int n_parts_wei_i,         \
            size_t *weights_states, int n_parts_wei_st,                   \
            const memory_storage_t &bias, const memory_storage_t &workspace, \
            const memory_storage_t &w_input, const memory_storage_t &w_state, \
            const memory_storage_t &diff_weights_layer,                   \
            const memory_storage_t &diff_weights_iter, \
            const memory_storage_t &diff_bias) const

#define grid_execution_sig(f)                                              \
    void f(const exec_ctx_t &ctx, int dic, int slc, int sic, int wic,      \
            int batch, int n_layer,                                        \
            int n_direction, int n_iter, int n_gates, int n_states,        \
            int n_bias, size_t *weights_input, int n_parts_wei_i,         \
            size_t *weights_states, int n_parts_wei_st,                   \
            const memory_storage_t &bias, const memory_storage_t &workspace, \
            const memory_storage_t &w_input, const memory_storage_t &w_state, \
            const memory_storage_t &diff_weights_layer,                   \
            const memory_storage_t &diff_weights_iter, \
            const memory_storage_t &diff_bias) const

#define gemm_sig(f)                                                          \
    void f(const exec_ctx_t &ctx, int m, int n, int k, int strideA_m, int strideA_k, \
            int strideB_n, int strideB_k, int strideC_m, int strideC_n, \
            const memory_storage_t &a, size_t off_a, \
            const memory_storage_t &b, size_t off_b, \
            const memory_storage_t &c, size_t off_c, \
            bool is_B_trans, float beta, gemm_kind_t gemm_kind) const

#define packing_sig(f)                                               \
    void f(int n_layer, int n_direction, int n_weights, int n_gates, \
            int batch, int OC_size, int IC_size, size_t *weights_,   \
            int n_parts, int *gates_per_part, const memory_storage_t &w_) const

#define free_packed_sig(f) void f(int n_layer, int n_direction, int n_parts, \
            size_t *weights_)

namespace mkldnn {
namespace impl {
namespace ocl {

namespace rnn_utils {

typedef enum execution_direction_ {
    b2t_l2r,
    b2t_r2l,
    b2t_bi_concat,
    b2t_bi_sum,
    t2b_l2r,
    t2b_r2l,
    t2b_bi_concat,
    t2b_bi_sum
} execution_direction;

bool is_training(const rnn_pd_t &pd);

size_t ws_states_size(const rnn_pd_t &pd);
size_t ws_diff_states_size(const rnn_pd_t &pd);
size_t ws_gates_size(const rnn_pd_t &pd);
size_t ws_cell_comp_size(const rnn_pd_t &pd);
size_t ws_grid_comp_size(const rnn_pd_t &pd);
size_t get_ws_size(const rnn_pd_t &pd);
size_t get_scratchpad_size(const rnn_pd_t &pd);

void set_offsets(const rnn_pd_t &pd, size_t &ws_gates_offset,
        size_t &ws_states_offset, size_t &ws_diff_states_offset,
        size_t &ws_grid_comp_offset, size_t &ws_cell_comp_offset);
}

}
}
}

#endif
