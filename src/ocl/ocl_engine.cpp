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

#include <CL/cl.h>

#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"
#include "ocl/gemm_inner_product.hpp"
#include "ocl/jit_gen9_common_convolution.hpp"
#include "ocl/jit_gen9_gemm.hpp"
#include "ocl/ocl_memory_storage.hpp"
#include "ocl/ocl_stream.hpp"
#include "ocl/ocl_utils.hpp"
#include "ocl/ref_batch_normalization.hpp"
#include "ocl/ref_convolution.hpp"
#include "ocl/ref_eltwise.hpp"
#include "ocl/ref_inner_product.hpp"
#include "ocl/ref_lrn.hpp"
#include "ocl/ref_pooling.hpp"
#include "ocl/ref_rnn.hpp"
#include "ocl/ref_softmax.hpp"
#include "ocl/ref_deconvolution.hpp"
#include "ocl/ref_shuffle.hpp"
#include "ocl/ocl_engine.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

status_t ocl_engine_t::init() {
    CHECK(cl_engine_t::init());

    cl_int err = CL_SUCCESS;
    if (is_user_context_) {
        err = clRetainContext(context_);
        if (err != CL_SUCCESS)
            context_ = nullptr;
    } else {
        context_
                = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    }

    OCL_CHECK(err);

    status_t status
            = ocl_utils::check_device(engine_kind::gpu, device_, context_);
    if (status != status::success)
        return status;

    stream_t *service_stream_ptr;
    status = create_stream(&service_stream_ptr, stream_flags::default_flags);
    if (status != status::success)
        return status;
    service_stream_.reset(service_stream_ptr);
    return status::success;
}

status_t ocl_engine_t::create_memory_storage(
        memory_storage_t **storage, unsigned flags, size_t size, void *handle) {
    return safe_ptr_assign<memory_storage_t>(
            *storage, new ocl_memory_storage_t(this, flags, size, handle));
}

status_t ocl_engine_t::create_stream(stream_t **stream, unsigned flags) {
    return ocl_stream_t::create_stream(stream, this, flags);
}

status_t ocl_engine_t::create_stream(
        stream_t **stream, cl_command_queue queue) {
    return ocl_stream_t::create_stream(stream, this, queue);
}

using pd_create_f = mkldnn::impl::engine_t::primitive_desc_create_f;

namespace {
using namespace mkldnn::impl::data_type;

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
    INSTANCE(jit_gen9_common_convolution_fwd_t<u8, s8, u8, s32>),
    INSTANCE(jit_gen9_common_convolution_fwd_t<f16>),
    INSTANCE(jit_gen9_common_convolution_fwd_t<f32>),
    INSTANCE(jit_gen9_common_convolution_bwd_data_t<f16, f16, f16, f16>),
    INSTANCE(jit_gen9_common_convolution_bwd_data_t<f32, f32, f32, f32>),
    INSTANCE(jit_gen9_common_convolution_bwd_weights_t<f32, f32, f32, f32>),
    INSTANCE(ref_convolution_fwd_t),
    INSTANCE(ref_convolution_bwd_data_t),
    INSTANCE(ref_convolution_bwd_weights_t),
    /*bnorm*/
    INSTANCE(ref_batch_normalization_fwd_t<f16>),
    INSTANCE(ref_batch_normalization_fwd_t<f32>),
    INSTANCE(ref_batch_normalization_bwd_t<f32>),
    /*pool*/
    INSTANCE(ref_pooling_fwd_t<u8, s32>),
    INSTANCE(ref_pooling_fwd_t<s8, s32>),
    INSTANCE(ref_pooling_fwd_t<f16>),
    INSTANCE(ref_pooling_fwd_t<f32>),
    INSTANCE(ref_pooling_bwd_t<f32>),
    /* lrn */
    INSTANCE(ref_lrn_fwd_t<f16>),
    INSTANCE(ref_lrn_fwd_t<f32>),
    INSTANCE(ref_lrn_bwd_t<f32>),
    /*inner_product*/
    INSTANCE(gemm_inner_product_fwd_t<f16>),
    INSTANCE(gemm_inner_product_fwd_t<f32>),
    INSTANCE(gemm_inner_product_bwd_data_t<f32>),
    INSTANCE(gemm_inner_product_bwd_weights_t<f32>),
    INSTANCE(ref_inner_product_fwd_t<f16>),
    INSTANCE(ref_inner_product_fwd_t<f32>),
    INSTANCE(ref_inner_product_bwd_data_t<f32, f32, f32, f32>),
    INSTANCE(ref_inner_product_bwd_weights_t<f32>),
    /*softmax*/
    INSTANCE(ref_softmax_fwd_t<f16>),
    INSTANCE(ref_softmax_fwd_t<f32>),
    INSTANCE(ref_softmax_bwd_t<f32>),
    /* gemm */
    INSTANCE(jit_gen9_gemm_t<f16>),
    INSTANCE(jit_gen9_gemm_t<f32>),
    /*rnn*/
    INSTANCE(ref_rnn_fwd_f16_t),
    INSTANCE(ref_rnn_fwd_f32_t),
    INSTANCE(ref_rnn_bwd_f32_t),
    /* shuffle */
    INSTANCE(ref_shuffle_t<4>),
    INSTANCE(ref_shuffle_t<2>),
    INSTANCE(ref_shuffle_t<1>),
    nullptr,
};

#undef INSTANCE
} // namespace

const pd_create_f *ocl_engine_t::get_implementation_list() const {
    return ocl_impl_list;
}

} // namespace ocl
} // namespace impl
} // namespace mkldnn
