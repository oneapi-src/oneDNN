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

#include "ocl/rnn/rnn_reorders.hpp"
#include "ocl/ocl_stream.hpp"
#include "ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace ocl {

status_t rnn_weights_reorder_t::execute(const exec_ctx_t &ctx) const {
    auto *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto &input = CTX_IN_STORAGE(DNNL_ARG_FROM);
    auto &output = CTX_OUT_STORAGE(DNNL_ARG_TO);

    const auto &jrp = ker_->jrp;
    const bool do_reorder = jrp.do_reorder;

    auto ocl_reorder = [&](const memory_storage_t &in_storage,
                               const memory_storage_t &scales_storage,
                               const memory_storage_t &out_storage) {
        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, in_storage);
        arg_list.set(1, scales_storage);
        arg_list.set(2, out_storage);

        auto nd_range = jrp.dispatch.nd_range();
        return compute_stream->parallel_for(nd_range, kernel_, arg_list);
    };

    status_t status = status::success;

    // Copy to gpu
    memory_desc_wrapper src_mdw(pd()->src_md());
    status = compute_stream->copy(
            input, do_reorder ? *temp_buf : output, src_mdw.size());

    if (status == status::success && do_reorder) {
        if (scales_buf) {
            void *tmp_ptr = nullptr;
            status = scales_buf->map_data(&tmp_ptr);
            if (status != status::success) return status;
            utils::array_copy((float *)tmp_ptr,
                    pd()->attr()->rnn_weights_qparams_.scales_,
                    jrp.scales_count);
            status = scales_buf->unmap_data(tmp_ptr);
            if (status != status::success) return status;
        }
    }

    if (status == status::success && do_reorder)
        status = ocl_reorder(*temp_buf, *scales_buf, output);

    return status;
}

} // namespace ocl
} // namespace impl
} // namespace dnnl
