/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#include "jit_avx2_relu.hpp"

#include "c_types_map.hpp"
#include "type_helpers.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

void jit_avx2_relu_fwd_t::execute_forward() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<data_t*>(this->memory(0));

    const memory_desc_wrapper src_desc(conf_.src_pd());
    const memory_desc_wrapper dst_desc(conf_.dst_pd());

    const auto &jrp = kernel_->jrp;

    src += src_desc.blocking_desc().offset_padding;
    dst += dst_desc.blocking_desc().offset_padding;

    const int actual_jit_n_runs =
        jrp.block_size == 0 ? 1 : jrp.n_elems / jrp.block_size;

#   pragma omp parallel for schedule(static)
    for (int n = 0; n < actual_jit_n_runs; ++n) {
        jit_relu_call_s arg = {};
        arg.from = &src[n * jrp.block_size];
        arg.to = &dst[n * jrp.block_size];
        if (n != actual_jit_n_runs - 1) {
            arg.main_loop_iters = jrp.main_loop_iters;
            arg.process_remainder = 0;
        } else {
            arg.main_loop_iters =
                jrp.main_loop_iters + jrp.remainder_main_loop_iters;
            arg.process_remainder = 1;
        }
        (*kernel_)(&arg);
    }
}

void jit_avx2_relu_bwd_t::execute_backward() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<data_t*>(this->memory(0));

    const memory_desc_wrapper src_desc(conf_.src_pd());
    const memory_desc_wrapper diff_dst_desc(conf_.diff_dst_pd());
    const memory_desc_wrapper diff_src_desc(conf_.diff_src_pd());

    const auto &jrp = kernel_->jrp;

    src += src_desc.blocking_desc().offset_padding;
    diff_dst += diff_dst_desc.blocking_desc().offset_padding;
    diff_src += diff_src_desc.blocking_desc().offset_padding;

    const int actual_jit_n_runs =
        jrp.block_size == 0 ? 1 : jrp.n_elems / jrp.block_size;

#   pragma omp parallel for schedule(static)
    for (int n = 0; n < actual_jit_n_runs; ++n) {
        jit_relu_call_s arg = {};
        arg.for_comparison = &src[n * jrp.block_size];
        arg.from = &diff_dst[n * jrp.block_size];
        arg.to = &diff_src[n * jrp.block_size];
        if (n != actual_jit_n_runs - 1) {
            arg.main_loop_iters = jrp.main_loop_iters;
            arg.process_remainder = 0;
        } else {
            arg.main_loop_iters =
                jrp.main_loop_iters + jrp.remainder_main_loop_iters;
            arg.process_remainder = 1;
        }
        (*kernel_)(&arg);
    }
}

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
