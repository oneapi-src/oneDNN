/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#include "gpu/ocl/many_inputs_sum.hpp"
namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t many_inputs_sum_t::execute(const exec_ctx_t &ctx) const {
    auto &output = CTX_OUT_STORAGE(DNNL_ARG_DST);
    const int num_arrs = pd()->n_inputs()
            - 1; // input0 is copied over to output. Accumulate the rest.
    const memory_desc_wrapper o_d(pd()->dst_md());
    const size_t nelems = o_d.nelems(true);
    compute::kernel_arg_list_t arg_list;

    int num_batches = utils::div_up(num_arrs, max_num_arrs);

    auto &input0 = CTX_IN_STORAGE(DNNL_ARG_MULTIPLE_SRC);
    const bool is_inplace = (output.data_handle() == input0.data_handle());
    if (!is_inplace) {
        auto *compute_stream
                = utils::downcast<compute::compute_stream_t *>(ctx.stream());
        CHECK(compute_stream->copy(input0, output, o_d.size(),
                compute_stream->ctx().get_deps(),
                compute_stream->ctx().get_deps()));
    }
    status_t status;
    for (int batch_iter = 0; batch_iter < num_batches; batch_iter++) {
        int kernel_num_arrs = max_num_arrs;
        if ((batch_iter == num_batches - 1) && (num_arrs % max_num_arrs)) {
            kernel_num_arrs = num_arrs % max_num_arrs;
        }
        for (int a = 0; a < max_num_arrs; ++a) {
            if (a < kernel_num_arrs) {
                auto &input = CTX_IN_STORAGE(DNNL_ARG_MULTIPLE_SRC + 1 + a
                        + batch_iter * max_num_arrs);
                arg_list.set(a, input);
            } else
                arg_list.set(a, memory_storage_t::empty_storage());
        }
        arg_list.set(many_inputs_sum_t::max_num_arrs, output);
        arg_list.set(many_inputs_sum_t::max_num_arrs + 1,
                CTX_GPU_RES_STORAGE(SCALES_));

        const size_t total_width = nelems * kernel_num_arrs;
        const size_t lws = utils::rnd_dn(256, kernel_num_arrs);

        compute::nd_range_t nd_range({utils::rnd_up(total_width, lws)}, {lws});
        if (batch_iter == num_batches - 1) {
            status = parallel_for(ctx, nd_range, kernel_, arg_list);
        } else {
            status = parallel_for(ctx, nd_range, batched_kernel_, arg_list);
        }
        if (status != dnnl_success) return status;
    }
    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
