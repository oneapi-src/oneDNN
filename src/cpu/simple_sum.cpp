/*******************************************************************************
* Copyright 2017-2025 Intel Corporation
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

#include "cpu/simple_sum.hpp"
#include "common/bfloat16.hpp"
#include "common/dnnl_thread.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <data_type_t src_data_type, data_type_t dst_data_type>
status_t simple_sum_t<src_data_type, dst_data_type>::execute(
        exec_ctx_t &ctx) const {
    auto output = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);

    const memory_desc_wrapper o_d(pd()->dst_md());
    output += o_d.blk_off(0);
    const int num_arrs = pd()->n_inputs();
    const src_data_t *input_ptrs[max_num_arrs];

    for (int a = 0; a < num_arrs; ++a) {
        const memory_desc_wrapper i_d(pd()->src_md(a));
        input_ptrs[a]
                = CTX_IN_MEM(const src_data_t *, DNNL_ARG_MULTIPLE_SRC + a)
                + i_d.blk_off(0);
    }

    const dim_t nelems = pd()->nelems_;
    const dim_t block_size = pd()->block_size_;
    const dim_t blocks_number = pd()->blocks_number_;
    const dim_t tail = pd()->tail_;

    const auto scales = pd()->scales();

    auto sum_block_xf16 = [&](dim_t start, dim_t end, int ithr) {
        const bool is_dst_xf16
                = utils::one_of(dst_data_type, data_type::bf16, data_type::f16);
        const auto xf16_params = pd()->xf16_params_;
        const auto scratchpad = ctx.get_scratchpad_grantor();
        acc_data_t *wspace = scratchpad.template get<acc_data_t>(
                memory_tracking::names::key_sum_srcs_cvt);
        acc_data_t *my_ws = &wspace[ithr * xf16_params.ws_elements_per_thread_];

        for (dim_t b = start; b < end; b += xf16_params.acc_loop_step_) {
            acc_data_t *my_acc = is_dst_xf16
                    ? &my_ws[xf16_params.ws_cvt_elements_per_thread_]
                    : (acc_data_t *)&output[b];
            dim_t current_block
                    = nstl::min(xf16_params.acc_loop_step_, end - b);
            types::cvt_to_float(my_ws, &input_ptrs[0][b], current_block);
            for (dim_t e = 0; e < current_block; e++)
                my_acc[e] = scales[0] * my_ws[e];

            for (int a = 1; a < num_arrs; a++) {
                types::cvt_to_float(my_ws, &input_ptrs[a][b], current_block);
                for (dim_t e = 0; e < current_block; e++)
                    my_acc[e] += scales[a] * my_ws[e];
            }

            if (is_dst_xf16)
                types::cvt_from_float(&output[b], my_acc, current_block);
        }
    };

    auto sum_block = [&](dim_t start, dim_t end, int ithr) {
        PRAGMA_OMP_SIMD()
        for (dim_t e = start; e < end; e++) {
            output[e] = dst_data_t(scales[0] * input_ptrs[0][e]);
        }
        for (int a = 1; a < num_arrs; a++) {
            PRAGMA_OMP_SIMD()
            for (dim_t e = start; e < end; e++) {
                output[e] += dst_data_t(scales[a] * input_ptrs[a][e]);
            }
        }
    };

    const int max_nthr = pd()->nthr_;
    parallel(max_nthr, [&](const int ithr, const int nthr) {
        dim_t start {0}, end {0};
        balance211(blocks_number, nthr, ithr, start, end);

        for (dim_t nb = start; nb < end; ++nb) {
            dim_t start_e = nb * block_size;
            dim_t end_e = start_e + block_size;
            if (src_data_type == data_type::f32)
                sum_block(start_e, end_e, ithr);
            else
                sum_block_xf16(start_e, end_e, ithr);
        }

        if (tail != 0 && ithr == nthr - 1) {
            dim_t start_e = nelems - tail;
            dim_t end_e = nelems;
            if (src_data_type == data_type::f32)
                sum_block(start_e, end_e, ithr);
            else
                sum_block_xf16(start_e, end_e, ithr);
        }
    });

    return status::success;
}

template struct simple_sum_t<data_type::f32>;
template struct simple_sum_t<data_type::bf16>;
template struct simple_sum_t<data_type::bf16, data_type::f32>;
template struct simple_sum_t<data_type::f16>;
template struct simple_sum_t<data_type::f16, data_type::f32>;
} // namespace cpu
} // namespace impl
} // namespace dnnl
