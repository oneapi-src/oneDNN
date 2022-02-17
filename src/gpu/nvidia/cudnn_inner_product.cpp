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

#include "gpu/nvidia/cudnn_inner_product.hpp"
#include "gpu/nvidia/cudnn_conv_inner_product.hpp"
#include "gpu/nvidia/cudnn_gemm_inner_product.hpp"
#include "gpu/nvidia/sycl_cuda_scoped_context.hpp"
#include "gpu/nvidia/sycl_cuda_stream.hpp"
#include "sycl/sycl_buffer_memory_storage.hpp"
#include "sycl_cuda_memory_storage_helper.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

status_t cudnn_inner_product_fwd_t::execute(const exec_ctx_t &ctx) const {
    if (pd()->has_zero_dim_memory()) return status::success;

    nvidia::sycl_cuda_stream_t *cuda_stream
            = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

    return cuda_stream->interop_task([&](::sycl::handler &cgh) {
        using scratch_acc_t = ::sycl::accessor<uint8_t, 1,
                ::sycl::access::mode::read_write, sycl::compat::target_device>;
        auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
        auto arg_wei = CTX_IN_SYCL_MEMORY(DNNL_ARG_WEIGHTS);
        auto arg_bias = CTX_IN_SYCL_MEMORY(DNNL_ARG_BIAS);
        auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);

        std::shared_ptr<scratch_acc_t> ip_scratch_acc;
        std::shared_ptr<scratch_acc_t> spacial_scratch_acc;
        std::shared_ptr<scratch_acc_t> scaled_bias_scratch_acc;
        if (pd()->inner_product_impl_->ip_using_scratchpad()) {
            ip_scratch_acc = std::make_shared<
                    scratch_acc_t>(CTX_SCRATCH_ACCESSOR(
                    memory_tracking::names::key_iprod_int_dat_in_acc_dt));
        }
        if (pd()->inner_product_impl_->need_to_transform_filter()) {
            spacial_scratch_acc = std::make_shared<scratch_acc_t>(
                    CTX_SCRATCH_ACCESSOR(memory_tracking::names::key_none));
        }
        if (pd()->inner_product_impl_->conv_using_scale_scratchpad()) {
            scaled_bias_scratch_acc
                    = std::make_shared<scratch_acc_t>(CTX_SCRATCH_ACCESSOR(
                            memory_tracking::names::key_conv_adjusted_scales));
        }
        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(
                    cuda_stream->engine());
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto cudnn_handle = cuda_stream->get_cudnn_handle();
            auto cublas_handle = cuda_stream->get_cublas_handle();

            std::vector<void *> args;

            args.push_back(arg_src.get_native_pointer(ih));
            args.push_back(arg_wei.get_native_pointer(ih));
            args.push_back(arg_bias.get_native_pointer(ih));
            args.push_back(arg_dst.get_native_pointer(ih));

            args.push_back((pd()->inner_product_impl_->ip_using_scratchpad()
                            ? sc.memory<void *>(ih, *ip_scratch_acc)
                            : nullptr));
            args.push_back((
                    pd()->inner_product_impl_->need_to_transform_filter()
                            ? sc.memory<void *>(ih, *spacial_scratch_acc)
                            : nullptr));
            args.push_back((
                    pd()->inner_product_impl_->conv_using_scale_scratchpad()
                            ? sc.memory<void *>(ih, *scaled_bias_scratch_acc)
                            : nullptr));

            pd()->inner_product_impl_->execute(
                    cudnn_handle, cublas_handle, args);
        });
    });
}

status_t cudnn_inner_product_bwd_data_t::execute(const exec_ctx_t &ctx) const {
    if (pd()->has_zero_dim_memory()) return status::success;
    nvidia::sycl_cuda_stream_t *cuda_stream
            = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

    return cuda_stream->interop_task([&](::sycl::handler &cgh) {
        using scratch_acc_t = ::sycl::accessor<uint8_t, 1,
                ::sycl::access::mode::read_write, sycl::compat::target_device>;
        auto arg_diff_dst = CTX_IN_SYCL_MEMORY(DNNL_ARG_DIFF_DST);
        auto arg_wei = CTX_IN_SYCL_MEMORY(DNNL_ARG_WEIGHTS);
        auto arg_diff_src = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DIFF_SRC);

        std::shared_ptr<scratch_acc_t> ip_scratch_acc;
        std::shared_ptr<scratch_acc_t> spacial_scratch_acc;
        if (pd()->inner_product_impl_->ip_using_scratchpad()) {
            ip_scratch_acc = std::make_shared<
                    scratch_acc_t>(CTX_SCRATCH_ACCESSOR(
                    memory_tracking::names::key_iprod_int_dat_in_acc_dt));
        }
        if (pd()->inner_product_impl_->need_to_transform_filter()) {
            spacial_scratch_acc = std::make_shared<scratch_acc_t>(
                    CTX_SCRATCH_ACCESSOR(memory_tracking::names::key_none));
        }
        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(
                    cuda_stream->engine());
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto cudnn_handle = cuda_stream->get_cudnn_handle();
            auto cublas_handle = cuda_stream->get_cublas_handle();

            std::vector<void *> args;

            args.push_back(arg_diff_src.get_native_pointer(ih));
            args.push_back(arg_wei.get_native_pointer(ih));
            args.push_back(arg_diff_dst.get_native_pointer(ih));
            args.push_back((pd()->inner_product_impl_->ip_using_scratchpad()
                            ? sc.memory<void *>(ih, *ip_scratch_acc)
                            : nullptr));
            args.push_back((
                    pd()->inner_product_impl_->need_to_transform_filter()
                            ? sc.memory<void *>(ih, *spacial_scratch_acc)
                            : nullptr));
            pd()->inner_product_impl_->execute(
                    cudnn_handle, cublas_handle, args);
        });
    });
}

status_t cudnn_inner_product_bwd_weights_t::execute(
        const exec_ctx_t &ctx) const {

    nvidia::sycl_cuda_stream_t *cuda_stream
            = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

    if (pd()->has_zero_dim_memory()) {
        auto wei_sz = memory_desc_wrapper(pd()->diff_weights_md(0)).size();
        size_t bias_sz = (pd()->with_bias()
                        ? memory_desc_wrapper(pd()->diff_weights_md(1)).size()
                        : 0);

        if (wei_sz != 0) {
            auto status = cuda_stream->fill(CTX_OUT_STORAGE(DNNL_ARG_DIFF_WEIGHTS), 0, wei_sz);
            /*auto status = cuda_stream->interop_task([&](::sycl::handler &cgh) {
                auto *mem_diff_wei
                        = static_cast<sycl::sycl_memory_storage_base_t *>(
                                &CTX_OUT_STORAGE(DNNL_ARG_DIFF_WEIGHTS));
                
                switch (mem_diff_wei->memory_kind()) {
                    case sycl::memory_kind::buffer: {
                        auto diff_wei_acc
                                = utils::downcast<
                                        sycl::sycl_buffer_memory_storage_t *>(
                                        mem_diff_wei)
                                          ->buffer()
                                          .get_access<
                                                  ::sycl::access::mode::write>(
                                                  cgh);
                        cgh.fill(diff_wei_acc, static_cast<uint8_t>(0));
                        break;
                    }
                    case sycl::memory_kind::usm: {
                        auto *diff_wei_ptr = utils::downcast<
                                const sycl::sycl_usm_memory_storage_t *>(
                                mem_diff_wei)
                                                     ->usm_ptr();
                        cudaMemset(static_cast<void *>(diff_wei_ptr),
                                static_cast<uint8_t>(0), wei_sz);
                        break;
                    }
                    default: assert(!"unexpected memory kind");
                }
            });*/
            if (status != status::success) return status;
        }
        if (bias_sz != 0) {
            auto status = cuda_stream->fill(CTX_OUT_STORAGE(DNNL_ARG_DIFF_BIAS), 0, bias_sz);
            /*auto status = cuda_stream->interop_task([&](::sycl::handler &cgh) {
                auto *mem_diff_bias
                        = static_cast<sycl::sycl_memory_storage_base_t *>(
                                &CTX_OUT_STORAGE(DNNL_ARG_DIFF_BIAS));
                switch (mem_diff_bias->memory_kind()) {
                    case sycl::memory_kind::buffer: {
                        auto diff_bias_acc
                                = utils::downcast<
                                        sycl::sycl_buffer_memory_storage_t *>(
                                        mem_diff_bias)
                                          ->buffer()
                                          .get_access<
                                                  ::sycl::access::mode::write>(
                                                  cgh);
                        cgh.fill(diff_bias_acc, static_cast<uint8_t>(0));
                        break;
                    }
                    case sycl::memory_kind::usm: {
                        auto *diff_bias_ptr = utils::downcast<
                                const sycl::sycl_usm_memory_storage_t *>(
                                mem_diff_bias)
                                                      ->usm_ptr();
                        cudaMemset(static_cast<void *>(diff_bias_ptr),
                                static_cast<uint8_t>(0), bias_sz);
                        break;
                    }
                    default: assert(!"unexpected memory kind");
                }
            });*/
            if (status != status::success) return status;
        }
        return status::success;
    }

    return cuda_stream->interop_task([&](::sycl::handler &cgh) {
        using scratch_acc_t = ::sycl::accessor<uint8_t, 1,
                ::sycl::access::mode::read_write, sycl::compat::target_device>;

        auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
        auto arg_diff_dst = CTX_IN_SYCL_MEMORY(DNNL_ARG_DIFF_DST);
        auto arg_diff_wei = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DIFF_WEIGHTS);
        auto arg_bias = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DIFF_BIAS);

        std::shared_ptr<scratch_acc_t> ip_scratch_acc;
        std::shared_ptr<scratch_acc_t> spacial_scratch_acc;
        if (pd()->inner_product_impl_->ip_using_scratchpad()) {
            ip_scratch_acc = std::make_shared<
                    scratch_acc_t>(CTX_SCRATCH_ACCESSOR(
                    memory_tracking::names::key_iprod_int_dat_in_acc_dt));
        }
        if (pd()->inner_product_impl_->need_to_transform_filter()) {
            spacial_scratch_acc = std::make_shared<scratch_acc_t>(
                    CTX_SCRATCH_ACCESSOR(memory_tracking::names::key_none));
        }
        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(
                    cuda_stream->engine());
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto cudnn_handle = cuda_stream->get_cudnn_handle();
            auto cublas_handle = cuda_stream->get_cublas_handle();
            std::vector<void *> args;

            args.push_back(arg_src.get_native_pointer(ih));
            args.push_back(arg_diff_dst.get_native_pointer(ih));
            args.push_back(arg_diff_wei.get_native_pointer(ih));
            args.push_back(arg_bias.get_native_pointer(ih));

            args.push_back((pd()->inner_product_impl_->ip_using_scratchpad()
                            ? sc.memory<void *>(ih, *ip_scratch_acc)
                            : nullptr));
            args.push_back((
                    pd()->inner_product_impl_->need_to_transform_filter()
                            ? sc.memory<void *>(ih, *spacial_scratch_acc)
                            : nullptr));
            pd()->inner_product_impl_->execute(
                    cudnn_handle, cublas_handle, args);
        });
    });
}

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl
