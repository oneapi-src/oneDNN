/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include <CL/cl.h>

#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "ocl/gemm_inner_product.hpp"
#include "ocl/gemm_x8s8s32x_inner_product.hpp"
#include "ocl/jit_gen9_common_convolution.hpp"
#include "ocl/jit_gen9_gemm.hpp"
#include "ocl/jit_gen9_gemm_x8x8s32.hpp"
#include "ocl/ocl_kernel_list.hpp"
#include "ocl/ocl_memory_storage.hpp"
#include "ocl/ocl_stream.hpp"
#include "ocl/ocl_utils.hpp"
#include "ocl/ref_batch_normalization.hpp"
#include "ocl/ref_binary.hpp"
#include "ocl/ref_convolution.hpp"
#include "ocl/ref_deconvolution.hpp"
#include "ocl/ref_eltwise.hpp"
#include "ocl/ref_inner_product.hpp"
#include "ocl/ref_layer_normalization.hpp"
#include "ocl/ref_lrn.hpp"
#include "ocl/ref_matmul.hpp"
#include "ocl/ref_pooling.hpp"
#include "ocl/ref_resampling.hpp"
#include "ocl/ref_shuffle.hpp"
#include "ocl/ref_softmax.hpp"
#include "ocl/rnn/ref_rnn.hpp"

#include "ocl/ocl_engine.hpp"

namespace dnnl {
namespace impl {
namespace ocl {

status_t ocl_gpu_engine_t::init() {
    CHECK(compute_engine_t::init());

    cl_int err = CL_SUCCESS;
    if (is_user_context_) {
        err = clRetainContext(context_);
        if (err != CL_SUCCESS) context_ = nullptr;
    } else {
        context_
                = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    }

    OCL_CHECK(err);

    status_t status
            = ocl_utils::check_device(engine_kind::gpu, device_, context_);
    if (status != status::success) return status;

    stream_t *service_stream_ptr;
    status = create_stream(&service_stream_ptr, stream_flags::default_flags);
    if (status != status::success) return status;
    service_stream_.reset(service_stream_ptr);
    return status::success;
}

status_t ocl_gpu_engine_t::create_memory_storage(
        memory_storage_t **storage, unsigned flags, size_t size, void *handle) {
    auto _storage = new ocl_memory_storage_t(this);
    if (_storage == nullptr) return status::out_of_memory;
    status_t status = _storage->init(flags, size, handle);
    if (status != status::success) {
        delete _storage;
        return status;
    }
    *storage = _storage;
    return status::success;
}

status_t ocl_gpu_engine_t::create_stream(stream_t **stream, unsigned flags) {
    return ocl_stream_t::create_stream(stream, this, flags);
}

status_t ocl_gpu_engine_t::create_stream(
        stream_t **stream, cl_command_queue queue) {
    return ocl_stream_t::create_stream(stream, this, queue);
}

status_t ocl_gpu_engine_t::create_kernels(
        std::vector<compute::kernel_t> *kernels,
        const std::vector<const char *> &kernel_names,
        const compute::kernel_ctx_t &kernel_ctx) const {
    const std::string &options = kernel_ctx.options();

    std::vector<const char *> code_strings;
    code_strings.reserve(kernel_names.size());
    for (auto *kernel_name : kernel_names) {
        const char *code = get_ocl_kernel_source(kernel_name);
        code_strings.push_back(code);
    }

    *kernels = std::vector<compute::kernel_t>(kernel_names.size());
    for (size_t i = 0; i < kernel_names.size(); ++i) {
        if (!kernel_names[i] || (*kernels)[i]) continue;

        const char *code = code_strings[i];

        cl_int err;
        cl_program program
                = clCreateProgramWithSource(context(), 1, &code, nullptr, &err);
        OCL_CHECK(err);

        cl_device_id dev = device();
        err = clBuildProgram(
                program, 1, &dev, options.c_str(), nullptr, nullptr);
#ifndef NDEBUG
        if (err != CL_SUCCESS) {
            size_t log_length = 0;
            err = clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0,
                    nullptr, &log_length);
            assert(err == CL_SUCCESS);

            std::vector<char> log_buf(log_length);
            err = clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
                    log_length, log_buf.data(), 0);
            assert(err == CL_SUCCESS);
            printf("Error during the build of OpenCL program.\nBuild "
                   "log:\n%s\n",
                    log_buf.data());
            OCL_CHECK(err);
        }
#endif
        for (size_t j = i; j < kernel_names.size(); ++j) {
            if (code_strings[j] == code_strings[i]) {
                cl_kernel ocl_kernel
                        = clCreateKernel(program, kernel_names[j], &err);
                OCL_CHECK(err);
                (*kernels)[j]
                        = compute::kernel_t(new ocl_gpu_kernel_t(ocl_kernel));
            }
        }

        OCL_CHECK(clReleaseProgram(program));
    }
    return status::success;
    ;
}

using pd_create_f = dnnl::impl::engine_t::primitive_desc_create_f;

namespace {
using namespace dnnl::impl::data_type;

#define INSTANCE(...) &primitive_desc_t::create<__VA_ARGS__::pd_t>
static const pd_create_f ocl_impl_list[] = {
        /*eltwise*/
        INSTANCE(ref_eltwise_fwd_t),
        INSTANCE(ref_eltwise_bwd_t),
        /*deconv*/
        INSTANCE(ref_deconvolution_fwd_t),
        INSTANCE(ref_deconvolution_bwd_data_t),
        INSTANCE(ref_deconvolution_bwd_weights_t),
        /*conv*/
        INSTANCE(jit_gen9_common_convolution_fwd_t),
        INSTANCE(jit_gen9_common_convolution_bwd_data_t),
        INSTANCE(jit_gen9_common_convolution_bwd_weights_t),
        INSTANCE(ref_convolution_fwd_t),
        INSTANCE(ref_convolution_bwd_data_t),
        INSTANCE(ref_convolution_bwd_weights_t),
        /*bnorm*/
        INSTANCE(ref_batch_normalization_fwd_t),
        INSTANCE(ref_batch_normalization_bwd_t),
        /*pool*/
        INSTANCE(ref_pooling_fwd_t),
        INSTANCE(ref_pooling_bwd_t),
        /* lrn */
        INSTANCE(ref_lrn_fwd_t),
        INSTANCE(ref_lrn_bwd_t),
        /*inner_product*/
        INSTANCE(gemm_x8s8s32x_inner_product_fwd_t),
        INSTANCE(gemm_inner_product_fwd_t),
        INSTANCE(gemm_inner_product_bwd_data_t),
        INSTANCE(gemm_inner_product_bwd_weights_t),

        INSTANCE(ref_inner_product_fwd_t),
        INSTANCE(ref_inner_product_bwd_data_t),
        INSTANCE(ref_inner_product_bwd_weights_t),
        /*softmax*/
        INSTANCE(ref_softmax_fwd_t),
        INSTANCE(ref_softmax_bwd_t),
        /* gemm */
        INSTANCE(jit_gen9_gemm_x8x8s32_t<s8, s8, s32>),
        INSTANCE(jit_gen9_gemm_x8x8s32_t<s8, u8, s32>),
        INSTANCE(jit_gen9_gemm_x8x8s32_t<u8, s8, s32>),
        INSTANCE(jit_gen9_gemm_x8x8s32_t<u8, u8, s32>),
        INSTANCE(jit_gen9_gemm_t<bf16, bf16, bf16, f32>),
        INSTANCE(jit_gen9_gemm_t<bf16, bf16, f32>),
        INSTANCE(jit_gen9_gemm_t<f16, f16, f32>),
        INSTANCE(jit_gen9_gemm_t<f16>),
        INSTANCE(jit_gen9_gemm_t<f32>),
        /*rnn*/
        INSTANCE(ref_rnn_fwd_u8s8_t),
        INSTANCE(ref_rnn_fwd_f16_t),
        INSTANCE(ref_rnn_fwd_f32_t),
        INSTANCE(ref_rnn_bwd_f32_t),
        INSTANCE(ref_rnn_fwd_bf16_t),
        INSTANCE(ref_rnn_bwd_bf16_t),
        /* shuffle */
        INSTANCE(ref_shuffle_t),
        /*layer normalization */
        INSTANCE(ref_layer_normalization_fwd_t),
        INSTANCE(ref_layer_normalization_bwd_t),
        /* binary */
        INSTANCE(ref_binary_t),
        /* matmul */
        INSTANCE(ref_matmul_t),
        /*resampling*/
        INSTANCE(ref_resampling_fwd_t),
        INSTANCE(ref_resampling_bwd_t),
        nullptr,
};

#undef INSTANCE
} // namespace

const pd_create_f *ocl_gpu_engine_impl_list_t::get_implementation_list() {
    return ocl_impl_list;
}

} // namespace ocl
} // namespace impl
} // namespace dnnl
