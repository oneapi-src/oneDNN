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

#include "ocl/ocl_cross_engine_reorder_pd.hpp"
#include "ocl/ocl_stream.hpp"
#include "ocl/ocl_utils.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

status_t ocl_cross_engine_reorder_t::execute(const exec_ctx_t &ctx) const {
    auto &input = CTX_IN_STORAGE(MKLDNN_ARG_FROM);
    auto &output = CTX_OUT_STORAGE(MKLDNN_ARG_TO);
    const auto in_e_kind = pd()->src_engine()->kind();
    const auto out_e_kind = pd()->dst_engine()->kind();

    const auto &jrp = ker_->jrp;
    float alpha = pd()->alpha();
    float beta = pd()->beta();
    const bool do_reorder = jrp.do_reorder;

    auto &executor
            = *(utils::downcast<cl_stream_t *>(ctx.stream())->cl_executor());

    auto ocl_reorder = [&](const memory_storage_t &in_storage,
                               const memory_storage_t &out_storage) {
        kernel_.set_arg(0, in_storage);
        kernel_.set_arg(1, out_storage);
        kernel_.set_arg(2, alpha);
        kernel_.set_arg(3, beta);

        auto nd_range = cl_nd_range_t(jrp.gws_d, jrp.lws_d);
        status_t status = executor.parallel_for(nd_range, kernel_);

        return status;
    };

    status_t status = status::success;
    if (in_e_kind == engine_kind::gpu && out_e_kind == engine_kind::cpu) {
        if (do_reorder) {
            status = ocl_reorder(input, *temp_buf);
        }
        if (status == status::success) {
            // Copy to cpu
            memory_desc_wrapper dst_mdw(pd()->dst_md());
            status = executor.copy(
                    do_reorder ? *temp_buf : input, output, dst_mdw.size());
        }
    } else if (in_e_kind == engine_kind::cpu
            && out_e_kind == engine_kind::gpu) {
        // Copy to gpu
        memory_desc_wrapper src_mdw(pd()->src_md());
        status = executor.copy(
                input, do_reorder ? *temp_buf : output, src_mdw.size());
        if (status == status::success && do_reorder)
            status = ocl_reorder(*temp_buf, output);
    } else {
        status = ocl_reorder(input, output);
    }
    return status;
}

} // namespace ocl
} // namespace impl
} // namespace mkldnn
