/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
* Copyright 2020-2022 Codeplay Software Limited
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
#include "gpu/amd/miopen_lrn.hpp"
#include "gpu/amd/sycl_hip_scoped_context.hpp"
#include "gpu/amd/sycl_hip_stream.hpp"
#include "sycl/sycl_buffer_memory_storage.hpp"
#include "sycl/sycl_memory_storage_helper.hpp"
namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

status_t miopen_lrn_fwd_t::execute(const exec_ctx_t &ctx) const {

    if (memory_desc_wrapper(pd()->src_md()).has_zero_dim())
        return status::success;

    amd::sycl_hip_stream_t *hip_stream
            = utils::downcast<amd::sycl_hip_stream_t *>(ctx.stream());

    return hip_stream->interop_task([&](::sycl::handler &cgh) {
        auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
        auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);
        auto arg_wrksp = CTX_OUT_SYCL_MEMORY(DNNL_ARG_WORKSPACE);

        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(
                    hip_stream->engine());
            auto sc = hip_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = hip_stream->get_miopen_handle();

            void *src_ = arg_src.get_native_pointer(ih);
            void *dst_ = arg_dst.get_native_pointer(ih);
            void *ws_ = arg_wrksp.get_native_pointer(ih);

            std::vector<void *> args {src_, dst_, ws_};
            pd()->lrn_impl_->execute(handle, args);
        });
    });
}

status_t miopen_lrn_bwd_t::execute(const exec_ctx_t &ctx) const {
    if (memory_desc_wrapper(pd()->src_md()).has_zero_dim())
        return status::success;

    amd::sycl_hip_stream_t *hip_stream
            = utils::downcast<amd::sycl_hip_stream_t *>(ctx.stream());

    return hip_stream->interop_task([&](::sycl::handler &cgh) {
        auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
        auto arg_diff_dst = CTX_IN_SYCL_MEMORY(DNNL_ARG_DIFF_DST);
        auto arg_diff_src = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DIFF_SRC);
        auto arg_ws = CTX_IN_SYCL_MEMORY(DNNL_ARG_WORKSPACE);

        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            std::vector<void *> args;
            auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(
                    hip_stream->engine());
            auto sc = hip_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = hip_stream->get_miopen_handle();

            args.push_back(arg_src.get_native_pointer(ih));
            args.push_back(arg_ws.get_native_pointer(ih));
            args.push_back(arg_diff_src.get_native_pointer(ih));
            args.push_back(arg_diff_dst.get_native_pointer(ih));

            pd()->lrn_impl_->execute(handle, args);
        });
    });
}

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl
