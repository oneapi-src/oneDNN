/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
* Copyright 2020 Codeplay Software Limited
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

#include "gpu/amd/miopen_convolution.hpp"
#include "gpu/amd/sycl_hip_scoped_context.hpp"
#include "gpu/amd/sycl_hip_stream.hpp"
#include "gpu/amd/sycl_hip_utils.hpp"
#include "sycl/sycl_memory_storage_helper.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

status_t miopen_convolution_fwd_t::execute_convolution(
        const exec_ctx_t &ctx, bool with_bias, bool with_scratchpad) const {
    amd::sycl_hip_stream_t *hip_stream
            = utils::downcast<amd::sycl_hip_stream_t *>(ctx.stream());

    return hip_stream->interop_task([&](::sycl::handler &cgh) {
        auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
        auto arg_weights = CTX_IN_SYCL_MEMORY(DNNL_ARG_WEIGHTS);
        auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);
        auto arg_bias = CTX_IN_SYCL_MEMORY(DNNL_ARG_BIAS);
        auto arg_scratch = CTX_SCRATCH_SYCL_MEMORY(
                memory_tracking::names::key_conv_miopen_algo);
        auto arg_filter_scratch = CTX_SCRATCH_SYCL_MEMORY(
                memory_tracking::names::key_conv_miopen_filter);
        auto arg_oscale = CTX_IN_SYCL_MEMORY(DNNL_ARG_ATTR_OUTPUT_SCALES);

        impl::sycl::sycl_memory_arg_t<::sycl::access::mode::read_write>
                temp_dst;
        impl::sycl::sycl_memory_arg_t<::sycl::access::mode::read_write>
                temp_reorder;

        if (pd()->use_temp_dst()) {
            memory_storage_t *temp_dst_mem = scratch_storage.get();
            memory_storage_t *temp_reorder_mem = scratch_storage_2.get();
            temp_dst = impl::sycl::sycl_memory_arg_t<
                    ::sycl::access::mode::read_write>(temp_dst_mem, cgh);
            temp_reorder = impl::sycl::sycl_memory_arg_t<
                    ::sycl::access::mode::read_write>(temp_reorder_mem, cgh);
        }

        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(
                    hip_stream->engine());
            auto sc = hip_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = hip_stream->get_miopen_handle();

            std::vector<void *> args;
            args.push_back(arg_src.get_native_pointer(ih));
            args.push_back(arg_weights.get_native_pointer(ih));
            args.push_back(arg_dst.get_native_pointer(ih));
            args.push_back(arg_bias.get_native_pointer(ih));
            args.push_back(arg_scratch.get_native_pointer(ih));
            args.push_back(arg_filter_scratch.get_native_pointer(ih));
            args.push_back(temp_dst.get_native_pointer(ih));
            args.push_back(temp_reorder.get_native_pointer(ih));
            args.push_back(arg_oscale.get_native_pointer(ih));

            pd()->impl_->execute(handle, args);
        });
    });
}

status_t miopen_convolution_bwd_data_t::execute_convolution(
        const exec_ctx_t &ctx, bool with_bias, bool with_scratchpad) const {
    amd::sycl_hip_stream_t *hip_stream
            = utils::downcast<amd::sycl_hip_stream_t *>(ctx.stream());

    return hip_stream->interop_task([&](::sycl::handler &cgh) {
        auto arg_diff_src = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DIFF_SRC);
        auto arg_weights = CTX_IN_SYCL_MEMORY(DNNL_ARG_WEIGHTS);
        auto arg_diff_dst = CTX_IN_SYCL_MEMORY(DNNL_ARG_DIFF_DST);
        auto arg_bias = CTX_IN_SYCL_MEMORY(DNNL_ARG_BIAS);
        auto arg_scratch = CTX_SCRATCH_SYCL_MEMORY(
                memory_tracking::names::key_conv_miopen_algo);
        auto arg_filter_scratch = CTX_SCRATCH_SYCL_MEMORY(
                memory_tracking::names::key_conv_miopen_filter);

        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(
                    hip_stream->engine());
            auto sc = hip_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = hip_stream->get_miopen_handle();

            std::vector<void *> args;
            args.push_back(arg_diff_src.get_native_pointer(ih));
            args.push_back(arg_weights.get_native_pointer(ih));
            args.push_back(arg_diff_dst.get_native_pointer(ih));
            args.push_back(arg_bias.get_native_pointer(ih));
            args.push_back(arg_scratch.get_native_pointer(ih));
            args.push_back(arg_filter_scratch.get_native_pointer(ih));

            pd()->impl_->execute(handle, args);
        });
    });
}

status_t miopen_convolution_bwd_weights_t::execute_zero_dims(
        const exec_ctx_t &ctx) const {
    amd::sycl_hip_stream_t *hip_stream
            = utils::downcast<amd::sycl_hip_stream_t *>(ctx.stream());

    return hip_stream->interop_task([&](::sycl::handler &cgh) {
        auto arg_diff_weights = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DIFF_WEIGHTS);
        auto arg_diff_bias = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DIFF_BIAS);

        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(
                    hip_stream->engine());
            auto sc = hip_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = hip_stream->get_miopen_handle();

            void *weights = arg_diff_weights.get_native_pointer(ih);
            void *bias = arg_diff_bias.get_native_pointer(ih);

            pd()->impl_->execute_set_weights_bias(handle, weights, bias, 0.f);
        });
    });
}

status_t miopen_convolution_bwd_weights_t::execute_convolution(
        const exec_ctx_t &ctx, bool with_bias, bool with_scratchpad) const {
    amd::sycl_hip_stream_t *hip_stream
            = utils::downcast<amd::sycl_hip_stream_t *>(ctx.stream());

    return hip_stream->interop_task([&](::sycl::handler &cgh) {
        auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
        auto arg_diff_weights = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DIFF_WEIGHTS);
        auto arg_diff_dst = CTX_IN_SYCL_MEMORY(DNNL_ARG_DIFF_DST);
        auto arg_scratch = CTX_SCRATCH_SYCL_MEMORY(
                memory_tracking::names::key_conv_miopen_algo);
        auto arg_filter_scratch = CTX_SCRATCH_SYCL_MEMORY(
                memory_tracking::names::key_conv_miopen_filter);

        impl::sycl::sycl_memory_arg_t<::sycl::access::mode::write>
                arg_diff_bias;

        if (with_bias) {
            arg_diff_bias = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DIFF_BIAS);
        }

        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(
                    hip_stream->engine());
            auto sc = hip_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = hip_stream->get_miopen_handle();

            std::vector<void *> args;
            args.push_back(arg_src.get_native_pointer(ih));
            args.push_back(arg_diff_weights.get_native_pointer(ih));
            args.push_back(arg_diff_dst.get_native_pointer(ih));
            args.push_back(arg_diff_bias.get_native_pointer(ih));
            args.push_back(arg_scratch.get_native_pointer(ih));
            args.push_back(arg_filter_scratch.get_native_pointer(ih));

            pd()->impl_->execute(handle, args);
        });
    });
}

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl
