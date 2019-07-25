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

#include "common/utils.hpp"
#include "ocl/cross_engine_reorder.hpp"
#include "ocl/ocl_stream.hpp"
#include "ocl/ocl_utils.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

status_t cross_engine_reorder_t::execute(const exec_ctx_t &ctx) const {
    auto *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto &src = CTX_IN_STORAGE(MKLDNN_ARG_FROM);
    auto &dst = CTX_OUT_STORAGE(MKLDNN_ARG_TO);
    const auto src_e_kind = pd()->src_engine()->kind();
    const auto dst_e_kind = pd()->dst_engine()->kind();

    auto exec_reorder = [&](const memory_t *src_mem, const memory_t *dst_mem) {
        exec_args_t r_args;
        r_args[MKLDNN_ARG_SRC]
                = memory_arg_t{ const_cast<memory_t *>(src_mem), true };
        r_args[MKLDNN_ARG_DST]
                = memory_arg_t{ const_cast<memory_t *>(dst_mem), false };

        exec_ctx_t r_ctx(ctx.stream(), std::move(r_args));
        return gpu_reorder_->execute(r_ctx);
    };

    if (src_e_kind == engine_kind::gpu && dst_e_kind == engine_kind::cpu) {
        if (do_reorder_) {
            status = exec_reorder(ctx.input(MKLDNN_ARG_FROM), temp_buf.get());
        }
        if (status == status::success) {
            // Copy to cpu
            memory_desc_wrapper dst_mdw(pd()->dst_md());
            status = compute_stream->copy(
                    do_reorder_ ? *temp_buf->memory_storage() : src, dst,
                    dst_mdw.size());
        }
    } else if (src_e_kind == engine_kind::cpu
            && dst_e_kind == engine_kind::gpu) {
        // Copy to gpu
        memory_desc_wrapper src_mdw(pd()->src_md());
        status = compute_stream->copy(src,
                do_reorder_ ? *temp_buf->memory_storage() : dst,
                src_mdw.size());
        if (status == status::success && do_reorder_)
            status = exec_reorder(temp_buf.get(), ctx.output(MKLDNN_ARG_TO));
    } else {
        status = exec_reorder(
                ctx.input(MKLDNN_ARG_FROM), ctx.output(MKLDNN_ARG_TO));
    }
    return status;
}

} // namespace ocl
} // namespace impl
} // namespace mkldnn
