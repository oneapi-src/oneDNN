
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

#ifndef CPU_X64_RNN_BRGEMM_UTILS_RNN_HPP
#define CPU_X64_RNN_BRGEMM_UTILS_RNN_HPP

#include <memory>
#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"
#include "cpu/x64/brgemm/brgemm.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

namespace rnn_utils {
struct rnn_conf_t;
}

namespace x64 {
namespace rnn_brgemm_utils {

struct rnn_brgemm_t {
    static void init_scratchpad(const cpu::rnn_utils::rnn_conf_t &rnn,
            memory_tracking::registrar_t &scratchpad, dim_t gemm_acc_type_size,
            dim_t gemm_acc_align);
    static status_t configure_brgemm(cpu::rnn_utils::rnn_conf_t &rnn,
            alg_kind_t cell_kind, dim_t src_layer_type_size,
            dim_t scratch_type_size);
    void init_kernels(const cpu::rnn_utils::rnn_conf_t &rnn,
            data_type_t src_type, data_type_t weights_type);

    x64::brgemm_t desc_layer_b0_[3];
    x64::brgemm_t desc_iter_b0_[3];
    x64::brgemm_t desc_iter_b1_[3];
    x64::brgemm_t desc_layer_N_tail_b0_[3];
    x64::brgemm_t desc_iter_N_tail_b0_[3];
    x64::brgemm_t desc_iter_N_tail_b1_[3];

    x64::brgemm_t desc_layer_K1_tail_b1_[3];
    x64::brgemm_t desc_layer_NK1_tail_b1_[3];
    x64::brgemm_t desc_iter_K2_tail_b1_[3];
    x64::brgemm_t desc_iter_NK2_tail_b1_[3];

    x64::brgemm_t desc_proj_b0_[4];
    x64::brgemm_t desc_proj_N_tail_b0_[4];
    x64::brgemm_t desc_proj_N_tail_b1_[4];
    x64::brgemm_t desc_proj_K_tail_b1_[4];
    x64::brgemm_t desc_proj_NK_tail_b1_[4];

    std::unique_ptr<x64::brgemm_kernel_t> kernel_layer_b0_[3];
    std::unique_ptr<x64::brgemm_kernel_t> kernel_iter_b0_[3];
    std::unique_ptr<x64::brgemm_kernel_t> kernel_iter_b1_[3];
    std::unique_ptr<x64::brgemm_kernel_t> kernel_layer_N_tail_b0_[3];
    std::unique_ptr<x64::brgemm_kernel_t> kernel_iter_N_tail_b0_[3];
    std::unique_ptr<x64::brgemm_kernel_t> kernel_iter_N_tail_b1_[3];

    std::unique_ptr<x64::brgemm_kernel_t> kernel_layer_K1_tail_b1_[3];
    std::unique_ptr<x64::brgemm_kernel_t> kernel_layer_NK1_tail_b1_[3];
    std::unique_ptr<x64::brgemm_kernel_t> kernel_iter_K2_tail_b1_[3];
    std::unique_ptr<x64::brgemm_kernel_t> kernel_iter_NK2_tail_b1_[3];

    std::unique_ptr<x64::brgemm_kernel_t> kernel_proj_b0_[4];
    std::unique_ptr<x64::brgemm_kernel_t> kernel_proj_N_tail_b0_[4];
    std::unique_ptr<x64::brgemm_kernel_t> kernel_proj_N_tail_b1_[4];
    std::unique_ptr<x64::brgemm_kernel_t> kernel_proj_K_tail_b1_[4];
    std::unique_ptr<x64::brgemm_kernel_t> kernel_proj_NK_tail_b1_[4];

    char pallete_buff_[64];
    char pallete_buff_n_tail_[64];
    char pallete_buff_k1_tail_[64];
    char pallete_buff_k2_tail_[64];
    char pallete_buff_nk1_tail_[64];
    char pallete_buff_nk2_tail_[64];
    char pallete_buff_proj_[64];
    char pallete_buff_nproj_tail_[64];
    char pallete_buff_kproj_tail_[64];
    char pallete_buff_nkproj_tail_[64];
};

} // namespace rnn_brgemm_utils
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
