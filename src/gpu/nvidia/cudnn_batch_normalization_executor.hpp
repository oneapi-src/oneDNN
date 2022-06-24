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

#ifndef GPU_NVIDIA_CUDNN_BATCH_NORMALIZATION_EXECUTOR_HPP
#define GPU_NVIDIA_CUDNN_BATCH_NORMALIZATION_EXECUTOR_HPP

#include "common/batch_normalization_pd.hpp"
#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "gpu/nvidia/cudnn_batch_normalization_impl.hpp"
#include "gpu/nvidia/sycl_cuda_engine.hpp"
#include "gpu/nvidia/sycl_cuda_scoped_context.hpp"
#include "gpu/nvidia/sycl_cuda_stream.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"
#include "sycl/sycl_memory_storage_helper.hpp"
#include "sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct bnorm_exec_base_t {
    virtual status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            const std::shared_ptr<cudnn_batch_normalization_impl_base_t>
                    bnorm_impl) const = 0;
    virtual ~bnorm_exec_base_t() = default;

protected:
    template <::sycl::access::mode flt_m,
            ::sycl::access::mode mean_var_m = ::sycl::access::mode::write>
    void interop_task_fwd(
            std::shared_ptr<cudnn_batch_normalization_impl_base_t> bnorm_impl,
            engine_t *engine, ::sycl::handler &cgh,
            nvidia::sycl_cuda_stream_t *cuda_stream,
            impl::sycl::sycl_memory_arg_t<::sycl::access::mode::read> arg_src,
            impl::sycl::sycl_memory_arg_t<::sycl::access::mode::write> arg_dst,
            impl::sycl::sycl_memory_arg_t<flt_m> arg_scale_bias,
            impl::sycl::sycl_memory_arg_t<::sycl::access::mode::write>
                    arg_wkspace,
            bool init_ss, bool init_mv,
            impl::sycl::sycl_memory_arg_t<mean_var_m> arg_mean = {},
            impl::sycl::sycl_memory_arg_t<mean_var_m> arg_var = {}) const {

        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(engine);
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = cuda_stream->get_cudnn_handle();

            if (init_ss)
                init_scaleshift(
                        sc, ih, cuda_stream, arg_scale_bias, bnorm_impl->C());
            if (init_mv)
                init_mean_var(sc, ih, cuda_stream, arg_mean, arg_var,
                        bnorm_impl->C());

            auto *x = arg_src.get_native_pointer(ih);
            auto *y = arg_dst.get_native_pointer(ih);
            auto *mean = arg_mean.get_native_pointer(ih);
            auto *var = arg_var.get_native_pointer(ih);

            auto *scale = static_cast<uint8_t *>(
                    arg_scale_bias.get_native_pointer(ih));
            auto *bias = scale + bnorm_impl->C() * sizeof(float);
            uint8_t *y_prime = nullptr, *save_mean = nullptr,
                    *save_var = nullptr;

            if (!arg_wkspace.empty()) {
                save_mean = static_cast<uint8_t *>(
                        arg_wkspace.get_native_pointer(ih));
                save_var = save_mean + bnorm_impl->mean_var_size_bytes();
                y_prime = save_var + bnorm_impl->mean_var_size_bytes();
            }

            std::shared_ptr<bnorm_args_t> args(new bnorm_fwd_args_t(x, y, mean,
                    var, scale, bias, y_prime, save_mean, save_var));
            bnorm_impl->execute(handle, args);
        });
    }

    template <::sycl::access::mode ss_m, ::sycl::access::mode d_ss_m>
    void interop_task_bwd(
            std::shared_ptr<cudnn_batch_normalization_impl_base_t> bnorm_impl,
            engine_t *engine, ::sycl::handler &cgh,
            nvidia::sycl_cuda_stream_t *cuda_stream,
            impl::sycl::sycl_memory_arg_t<::sycl::access::mode::read> arg_src,
            impl::sycl::sycl_memory_arg_t<::sycl::access::mode::read>
                    arg_diff_dst,
            impl::sycl::sycl_memory_arg_t<::sycl::access::mode::write>
                    arg_diff_src,
            impl::sycl::sycl_memory_arg_t<ss_m> arg_scale_bias,
            impl::sycl::sycl_memory_arg_t<d_ss_m> arg_diff_scale_bias,
            impl::sycl::sycl_memory_arg_t<::sycl::access::mode::read>
                    arg_wkspace,
            impl::sycl::sycl_memory_arg_t<::sycl::access::mode::read_write>
                    arg_temp_relu,
            bool init_ss) const {

        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(engine);
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = cuda_stream->get_cudnn_handle();

            if (init_ss)
                init_scaleshift(
                        sc, ih, cuda_stream, arg_scale_bias, bnorm_impl->C());

            auto *x = arg_src.get_native_pointer(ih);
            auto *dy = arg_diff_dst.get_native_pointer(ih);
            auto *dx = arg_diff_src.get_native_pointer(ih);

            auto *scale = static_cast<uint8_t *>(
                    arg_scale_bias.get_native_pointer(ih));
            auto *bias = scale + bnorm_impl->C() * sizeof(float);

            auto *diff_scale = static_cast<uint8_t *>(
                    arg_diff_scale_bias.get_native_pointer(ih));
            auto *diff_bias = diff_scale + (bnorm_impl->C() * sizeof(float));

            auto *save_mean = static_cast<uint8_t *>(
                    arg_wkspace.get_native_pointer(ih));
            auto *save_var = save_mean + bnorm_impl->mean_var_size_bytes();
            auto *wkspace = save_var + bnorm_impl->mean_var_size_bytes();
            auto *relu_dy = arg_temp_relu.get_native_pointer(ih);

            std::shared_ptr<bnorm_args_t> args(
                    new bnorm_bwd_args_t(x, dx, dy, save_mean, save_var, scale,
                            bias, diff_scale, diff_bias, wkspace, relu_dy));
            bnorm_impl->execute(handle, args);
        });
    }

    template <typename T = float>
    void init_scaleshift(cuda_sycl_scoped_context_handler_t &sc,
            const compat::interop_handle &ih,
            nvidia::sycl_cuda_stream_t *cuda_stream, T, size_t) const {}

    template <typename T = float>
    void init_scaleshift(cuda_sycl_scoped_context_handler_t &sc,
            const compat::interop_handle &ih,
            nvidia::sycl_cuda_stream_t *cuda_stream,
            impl::sycl::sycl_memory_arg_t<::sycl::access::mode::write>
                    arg_scale_bias,
            const size_t n) const {
        T scale_val = 1, bias_val = 0;
        cuda_stream->interop_task([&](::sycl::handler &cgh) {
            T *scale_ptr
                    = static_cast<T *>(arg_scale_bias.get_native_pointer(ih));
            T *bias_ptr = scale_ptr + n;
            CUDA_EXECUTE_FUNC(cuMemsetD32Async,
                    reinterpret_cast<CUdeviceptr>(scale_ptr),
                    reinterpret_cast<int &>(scale_val), n,
                    cuda_stream->get_underlying_stream());
            CUDA_EXECUTE_FUNC(cuMemsetD32Async,
                    reinterpret_cast<CUdeviceptr>(bias_ptr),
                    reinterpret_cast<int &>(bias_val), n,
                    cuda_stream->get_underlying_stream());
            cudaDeviceSynchronize();
        });
    }

    // Handle the cases when mean and var are read-only accessors or nullptr
    template <typename T, typename U>
    void init_mean_var(cuda_sycl_scoped_context_handler_t &sc,
            const compat::interop_handle &ih,
            nvidia::sycl_cuda_stream_t *cuda_stream, T, U, const size_t) const {
    }

    template <typename T = float>
    void init_mean_var(cuda_sycl_scoped_context_handler_t &sc,
            const compat::interop_handle &ih,
            nvidia::sycl_cuda_stream_t *cuda_stream,
            impl::sycl::sycl_memory_arg_t<::sycl::access_mode::write> arg_mean,
            impl::sycl::sycl_memory_arg_t<::sycl::access_mode::write> arg_var,
            const size_t n) const {
        constexpr T mean_var_val = 0;
        cuda_stream->interop_task([&](::sycl::handler &cgh) {
            T *mean_ptr = static_cast<T *>(arg_mean.get_native_pointer(ih));
            T *var_ptr = static_cast<T *>(arg_var.get_native_pointer(ih));
            CUDA_EXECUTE_FUNC(cuMemsetD32Async,
                    reinterpret_cast<CUdeviceptr>(mean_ptr), mean_var_val, n,
                    cuda_stream->get_underlying_stream());
            CUDA_EXECUTE_FUNC(cuMemsetD32Async,
                    reinterpret_cast<CUdeviceptr>(var_ptr), mean_var_val, n,
                    cuda_stream->get_underlying_stream());
            cudaDeviceSynchronize();
        });
    }
};

struct bnorm_exec_fwd_inf_t : public bnorm_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            std::shared_ptr<cudnn_batch_normalization_impl_base_t> bnorm_impl)
            const override {
        nvidia::sycl_cuda_stream_t *cuda_stream
                = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());
        auto n_channels = bnorm_impl->C();
        ::sycl::buffer<uint8_t> scale_bias_buff(n_channels * 2 * sizeof(float));

        return cuda_stream->interop_task([&](::sycl::handler &cgh) {
            auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
            auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);
            auto arg_scale_bias = impl::sycl::sycl_memory_arg_t<
                    ::sycl::access::mode::write>(scale_bias_buff, cgh);
            auto arg_wkspace = bnorm_impl->is_training()
                    ? CTX_OUT_SYCL_MEMORY(DNNL_ARG_WORKSPACE)
                    : impl::sycl::sycl_memory_arg_t<
                            ::sycl::access::mode::write>();

            bool init_ss = true, init_mean_var = false;

            interop_task_fwd(bnorm_impl, engine, cgh, cuda_stream, arg_src,
                    arg_dst, arg_scale_bias, arg_wkspace, init_ss,
                    init_mean_var);
        });
    }
};

struct bnorm_exec_fwd_inf_ss_t : public bnorm_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            std::shared_ptr<cudnn_batch_normalization_impl_base_t> bnorm_impl)
            const override {
        nvidia::sycl_cuda_stream_t *cuda_stream
                = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

        return cuda_stream->interop_task([&](::sycl::handler &cgh) {
            auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
            auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);
            auto arg_scale_bias = CTX_IN_SYCL_MEMORY(DNNL_ARG_SCALE_SHIFT);
            auto arg_wkspace = bnorm_impl->is_training()
                    ? CTX_OUT_SYCL_MEMORY(DNNL_ARG_WORKSPACE)
                    : impl::sycl::sycl_memory_arg_t<
                            ::sycl::access::mode::write>();

            bool init_ss = false, init_mean_var = false;

            interop_task_fwd(bnorm_impl, engine, cgh, cuda_stream, arg_src,
                    arg_dst, arg_scale_bias, arg_wkspace, init_ss,
                    init_mean_var);
        });
    }
};

struct bnorm_exec_fwd_inf_stats_t : public bnorm_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            std::shared_ptr<cudnn_batch_normalization_impl_base_t> bnorm_impl)
            const override {
        nvidia::sycl_cuda_stream_t *cuda_stream
                = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());
        auto n_channels = bnorm_impl->C();
        ::sycl::buffer<uint8_t> scale_bias_buff(n_channels * 2 * sizeof(float));

        return cuda_stream->interop_task([&](::sycl::handler &cgh) {
            auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
            auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);
            auto arg_mean = CTX_IN_SYCL_MEMORY(DNNL_ARG_MEAN);
            auto arg_var = CTX_IN_SYCL_MEMORY(DNNL_ARG_VARIANCE);
            auto arg_scale_bias = impl::sycl::sycl_memory_arg_t<
                    ::sycl::access::mode::write>(scale_bias_buff, cgh);
            auto arg_wkspace = bnorm_impl->is_training()
                    ? CTX_OUT_SYCL_MEMORY(DNNL_ARG_WORKSPACE)
                    : impl::sycl::sycl_memory_arg_t<
                            ::sycl::access::mode::write>();

            bool init_ss = true, init_mean_var = false;

            interop_task_fwd(bnorm_impl, engine, cgh, cuda_stream, arg_src,
                    arg_dst, arg_scale_bias, arg_wkspace, init_ss,
                    init_mean_var, arg_mean, arg_var);
        });
    }
};

struct bnorm_exec_fwd_inf_ss_stats_t : public bnorm_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            std::shared_ptr<cudnn_batch_normalization_impl_base_t> bnorm_impl)
            const override {
        nvidia::sycl_cuda_stream_t *cuda_stream
                = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

        return cuda_stream->interop_task([&](::sycl::handler &cgh) {
            auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
            auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);
            auto arg_mean = CTX_IN_SYCL_MEMORY(DNNL_ARG_MEAN);
            auto arg_var = CTX_IN_SYCL_MEMORY(DNNL_ARG_VARIANCE);
            auto arg_scale_bias = CTX_IN_SYCL_MEMORY(DNNL_ARG_SCALE_SHIFT);
            auto arg_wkspace = bnorm_impl->is_training()
                    ? CTX_OUT_SYCL_MEMORY(DNNL_ARG_WORKSPACE)
                    : impl::sycl::sycl_memory_arg_t<
                            ::sycl::access::mode::write>();

            bool init_ss = false, init_mean_var = false;

            interop_task_fwd(bnorm_impl, engine, cgh, cuda_stream, arg_src,
                    arg_dst, arg_scale_bias, arg_wkspace, init_ss,
                    init_mean_var, arg_mean, arg_var);
        });
    }
};

struct bnorm_exec_fwd_t : public bnorm_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            std::shared_ptr<cudnn_batch_normalization_impl_base_t> bnorm_impl)
            const override {
        nvidia::sycl_cuda_stream_t *cuda_stream
                = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());
        auto n_channels = bnorm_impl->C();
        ::sycl::buffer<uint8_t> scale_bias_buff(n_channels * 2 * sizeof(float));

        return cuda_stream->interop_task([&](::sycl::handler &cgh) {
            auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
            auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);
            auto arg_mean = CTX_OUT_SYCL_MEMORY(DNNL_ARG_MEAN);
            auto arg_var = CTX_OUT_SYCL_MEMORY(DNNL_ARG_VARIANCE);
            auto arg_scale_bias = impl::sycl::sycl_memory_arg_t<
                    ::sycl::access::mode::write>(scale_bias_buff, cgh);
            auto arg_wkspace = bnorm_impl->is_training()
                    ? CTX_OUT_SYCL_MEMORY(DNNL_ARG_WORKSPACE)
                    : impl::sycl::sycl_memory_arg_t<
                            ::sycl::access::mode::write>();

            bool init_ss = true, init_mean_var = true;

            interop_task_fwd(bnorm_impl, engine, cgh, cuda_stream, arg_src,
                    arg_dst, arg_scale_bias, arg_wkspace, init_ss,
                    init_mean_var, arg_mean, arg_var);
        });
    }
};

struct bnorm_exec_fwd_ss_t : public bnorm_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            std::shared_ptr<cudnn_batch_normalization_impl_base_t> bnorm_impl)
            const override {
        nvidia::sycl_cuda_stream_t *cuda_stream
                = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

        return cuda_stream->interop_task([&](::sycl::handler &cgh) {
            auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
            auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);
            auto arg_mean = CTX_OUT_SYCL_MEMORY(DNNL_ARG_MEAN);
            auto arg_var = CTX_OUT_SYCL_MEMORY(DNNL_ARG_VARIANCE);
            auto arg_scale_bias = CTX_IN_SYCL_MEMORY(DNNL_ARG_SCALE_SHIFT);
            auto arg_wkspace = bnorm_impl->is_training()
                    ? CTX_OUT_SYCL_MEMORY(DNNL_ARG_WORKSPACE)
                    : impl::sycl::sycl_memory_arg_t<
                            ::sycl::access::mode::write>();

            bool init_ss = false, init_mean_var = true;

            interop_task_fwd(bnorm_impl, engine, cgh, cuda_stream, arg_src,
                    arg_dst, arg_scale_bias, arg_wkspace, init_ss,
                    init_mean_var, arg_mean, arg_var);
        });
    }
};

struct bnorm_exec_bwd_t : public bnorm_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            std::shared_ptr<cudnn_batch_normalization_impl_base_t> bnorm_impl)
            const override {
        nvidia::sycl_cuda_stream_t *cuda_stream
                = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());
        auto n_channels = bnorm_impl->C();
        ::sycl::buffer<uint8_t> scale_bias_buff(n_channels * 2 * sizeof(float));
        ::sycl::buffer<uint8_t> diff_scale_bias_buff(
                n_channels * 2 * sizeof(float));

        return cuda_stream->interop_task([&](::sycl::handler &cgh) {
            auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
            auto arg_diff_dst = CTX_IN_SYCL_MEMORY(DNNL_ARG_DIFF_DST);
            auto arg_diff_src = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DIFF_SRC);
            auto arg_wkspace = CTX_IN_SYCL_MEMORY(DNNL_ARG_WORKSPACE);
            auto arg_scale_bias = impl::sycl::sycl_memory_arg_t<
                    ::sycl::access::mode::write>(scale_bias_buff, cgh);
            auto arg_diff_scale_bias = impl::sycl::sycl_memory_arg_t<
                    ::sycl::access::mode::write>(diff_scale_bias_buff, cgh);

            auto arg_temp_relu
                    = CTX_SCRATCH_SYCL_MEMORY(memory_tracking::names::key_none);

            bool init_ss = true;

            interop_task_bwd(bnorm_impl, engine, cgh, cuda_stream, arg_src,
                    arg_diff_dst, arg_diff_src, arg_scale_bias,
                    arg_diff_scale_bias, arg_wkspace, arg_temp_relu, init_ss);
        });
    }
};

struct bnorm_exec_bwd_dw_ss_t : public bnorm_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            std::shared_ptr<cudnn_batch_normalization_impl_base_t> bnorm_impl)
            const override {
        nvidia::sycl_cuda_stream_t *cuda_stream
                = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

        return cuda_stream->interop_task([&](::sycl::handler &cgh) {
            auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
            auto arg_diff_dst = CTX_IN_SYCL_MEMORY(DNNL_ARG_DIFF_DST);
            auto arg_diff_src = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DIFF_SRC);
            auto arg_scale_bias = CTX_IN_SYCL_MEMORY(DNNL_ARG_SCALE_SHIFT);
            auto arg_diff_scale_bias
                    = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DIFF_SCALE_SHIFT);
            auto arg_wkspace = CTX_IN_SYCL_MEMORY(DNNL_ARG_WORKSPACE);
            bool init_ss = false;

            auto arg_temp_relu
                    = CTX_SCRATCH_SYCL_MEMORY(memory_tracking::names::key_none);

            interop_task_bwd(bnorm_impl, engine, cgh, cuda_stream, arg_src,
                    arg_diff_dst, arg_diff_src, arg_scale_bias,
                    arg_diff_scale_bias, arg_wkspace, arg_temp_relu, init_ss);
        });
    }
};

struct bnorm_exec_bwd_d_ss_t : public bnorm_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            std::shared_ptr<cudnn_batch_normalization_impl_base_t> bnorm_impl)
            const override {
        nvidia::sycl_cuda_stream_t *cuda_stream
                = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());
        auto n_channels = bnorm_impl->C();
        ::sycl::buffer<uint8_t> diff_scale_bias_buff(
                n_channels * 2 * sizeof(float));

        return cuda_stream->interop_task([&](::sycl::handler &cgh) {
            auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
            auto arg_diff_dst = CTX_IN_SYCL_MEMORY(DNNL_ARG_DIFF_DST);
            auto arg_diff_src = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DIFF_SRC);
            auto arg_scale_bias = CTX_IN_SYCL_MEMORY(DNNL_ARG_SCALE_SHIFT);
            auto arg_diff_scale_bias = impl::sycl::sycl_memory_arg_t<
                    ::sycl::access::mode::write>(diff_scale_bias_buff, cgh);
            auto arg_wkspace = CTX_IN_SYCL_MEMORY(DNNL_ARG_WORKSPACE);
            auto arg_temp_relu
                    = CTX_SCRATCH_SYCL_MEMORY(memory_tracking::names::key_none);

            bool init_ss = false;

            interop_task_bwd(bnorm_impl, engine, cgh, cuda_stream, arg_src,
                    arg_diff_dst, arg_diff_src, arg_scale_bias,
                    arg_diff_scale_bias, arg_wkspace, arg_temp_relu, init_ss);
        });
    }
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
