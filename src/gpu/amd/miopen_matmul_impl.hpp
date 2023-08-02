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

#ifndef GPU_AMD_MIOPEN_MATMUL_IMPL_HPP
#define GPU_AMD_MIOPEN_MATMUL_IMPL_HPP

#include "gpu/amd/sycl_hip_engine.hpp"
#include "gpu/amd/sycl_hip_utils.hpp"
#include "miopen/miopen.h"
#include "rocblas/rocblas.h"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct miopen_matmul_impl_t {
    bool with_eltwise(int position, const matmul_pd_t *pd) const {
        return pd->attr()->post_ops_.contain(primitive_kind::eltwise, position);
    }

    float eltwise_alpha(const matmul_pd_t *pd) const {
        int eltwise_idx_ = pd->attr()->post_ops_.find(primitive_kind::eltwise);
        return with_eltwise(0, pd) || with_eltwise(1, pd)
                ? pd->attr()->post_ops_.entry_[eltwise_idx_].eltwise.alpha
                : 1.0f;
    }

    alg_kind_t eltwise_algo(const matmul_pd_t *pd) const {
        int eltwise_idx_ = pd->attr()->post_ops_.find(primitive_kind::eltwise);
        return with_eltwise(0, pd) || with_eltwise(1, pd)
                ? pd->attr()->post_ops_.entry_[eltwise_idx_].eltwise.alg
                : dnnl_alg_kind_undef;
    }

    bool with_sum(const matmul_pd_t *pd) const {
        return pd->attr()->post_ops_.contain(primitive_kind::sum, 0)
                || pd->attr()->post_ops_.contain(primitive_kind::sum, 1);
    }

    // Returns scaling factor for post-ops=sum operation
    float sum_scale(const matmul_pd_t *pd) const {
        int sum_idx_ = pd->attr()->post_ops_.find(primitive_kind::sum);
        return pd->attr()->post_ops_.entry_[sum_idx_].sum.scale;
    }

    // creates operation descriptor based on the element-wise operation specified
    status_t create_and_set_op_descriptor(const matmul_pd_t *pd) {
        CHECK(MIOPEN_EXECUTE_FUNC_S(
                miopenCreateActivationDescriptor, &act_desc_));

        miopenActivationMode_t mode;
        switch (eltwise_algo(pd)) {
            case alg_kind::eltwise_relu:
                mode = miopenActivationMode_t::miopenActivationLEAKYRELU;
                break;
            case alg_kind::eltwise_tanh:
                mode = miopenActivationMode_t::miopenActivationTANH;
                break;
            case alg_kind::eltwise_elu:
                mode = miopenActivationMode_t::miopenActivationELU;
                break;
            case alg_kind::eltwise_logistic:
                mode = miopenActivationMode_t::miopenActivationLOGISTIC;
                break;
            default: return status::unimplemented;
        }

        //parameters for SetActivationDescriptor
        float activAlpha;
        float activBeta;
        float activGamma;
        double ceiling = eltwise_alpha(pd);

        if (mode == miopenActivationMode_t::miopenActivationTANH)
            activAlpha = activBeta = 1;
        else if (mode == miopenActivationMode_t::miopenActivationELU)
            activAlpha = ceiling;
        else if (mode == miopenActivationMode_t::miopenActivationLEAKYRELU)
            activAlpha = ceiling;

        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenSetActivationDescriptor, act_desc_,
                mode, activAlpha, activBeta, activGamma));

        return status::success;
    }

    status_t init(matmul_pd_t *pd) {
        CHECK(get_rocblas_data_type(pd->src_md()->data_type, src_type_));
        CHECK(get_rocblas_data_type(
                pd->weights_md()->data_type, weights_type_));
        isbatched_ = pd->batched();

        memory_desc_wrapper src_d = memory_desc_wrapper(pd->src_md());
        memory_desc_wrapper weights_d = memory_desc_wrapper(pd->weights_md());
        memory_desc_wrapper dst_d = memory_desc_wrapper(pd->dst_md());

        with_bias_ = pd->with_bias();

        CHECK(get_rocblas_data_type(pd->dst_md()->data_type, dst_type_));

        if (with_eltwise(0, pd) || with_eltwise(1, pd)) {
            with_eltwise_ = true;
            CHECK(create_and_set_op_descriptor(pd));
        }

        // Set parameter when post-op sum is specified
        if (with_sum(pd)) {
            post_op_sum_ = sum_scale(pd);
            sum_scale_s32 = post_op_sum_;
            sum_scale_f32 = post_op_sum_;
        }

        has_runtime_params_ = src_d.has_runtime_dims_or_strides()
                || dst_d.has_runtime_dims_or_strides()
                || weights_d.has_runtime_dims_or_strides();

        if (!has_runtime_params_) {
            // Initialise all gemm parameters if there are no runtime parameters
            init_parameters(src_d, weights_d, dst_d,
                    memory_desc_wrapper(pd->weights_md(1)));
        }

        //setting compute type based on destination type
        acc_type_ = (dst_type_ == rocblas_datatype_bf16_r
                            || dst_type_ == rocblas_datatype_f16_r)
                ? rocblas_datatype_f32_r
                : dst_type_;

        return status::success;
    }

    bool isbatched() { return isbatched_; }
    bool with_bias() { return with_bias_; }
    bool with_scratchpad() { return with_scratchpad_; }
    bool has_runtime_params() { return has_runtime_params_; }

    dnnl_data_type_t get_scratchpad_type() { return scratchpad_type_; }

    void convert_dims_matmul(
            const dnnl_dim_t *dims, int *new_dims, int n_dims) {
        new_dims[0] = 1;
        for (size_t i = 0; i < n_dims; i++) {
            new_dims[i + 1] = static_cast<int>(dims[i]);
        }
        for (size_t i = n_dims; i < 4; i++) {
            new_dims[i + 1] = 1;
        }
    }

    int get_ld(const memory_desc_wrapper desc, rocblas_operation trans) {
        const int ndims = desc.ndims();
        const auto *strides = &desc.blocking_desc().strides[ndims - 2];
        const int ld
                = strides[trans == rocblas_operation::rocblas_operation_none
                                ? 0
                                : 1];
        return ld;
    }

    int get_batch_stride(const memory_desc_wrapper desc) {
        auto dims = desc.dims();
        auto strides = desc.blocking_desc().strides;
        return dims[0] == 1 ? 0 : strides[0];
    }

    status_t init_gemm_parameters(const memory_desc_wrapper src_d,
            const memory_desc_wrapper weights_d,
            const memory_desc_wrapper dst_d) {
        weightBroadcastNeeded = false;
        srcBroadcastNeeded = false;

        if (isbatched_) {
            const auto src_batch = src_d.dims()[0];
            const auto wei_batch = weights_d.dims()[0];

            if (src_batch > wei_batch)
                weightBroadcastNeeded = true;
            else if (wei_batch > src_batch)
                srcBroadcastNeeded = true;

            batch_count_ = dst_d.dims()[0];
        }

        const dim_t M = dst_d.dims()[isbatched_ + 1];
        const dim_t N = dst_d.dims()[isbatched_ + 0];
        const dim_t K = src_d.dims()[isbatched_ + 1];

        M_ = (int)M;
        N_ = (int)N;
        K_ = (int)K;

        const auto dst_strides_org = dst_d.blocking_desc().strides;
        const auto &src_strides = &src_d.blocking_desc().strides[isbatched_];
        const auto &weights_strides
                = &weights_d.blocking_desc().strides[isbatched_];

        // A matrix is the weights
        transA_ = weights_strides[1] == 1
                        && weights_d.dims()[isbatched_ + 0] > 1
                ? rocblas_operation::rocblas_operation_none
                : rocblas_operation::rocblas_operation_transpose;

        // B matrix is the src
        transB_ = src_strides[1] == 1 && src_d.dims()[isbatched_ + 0] > 1
                ? rocblas_operation::rocblas_operation_none
                : rocblas_operation::rocblas_operation_transpose;

        // C matrix is the dst
        transC_ = dst_strides_org[dst_d.ndims() - 1] != 1
                ? rocblas_operation::rocblas_operation_transpose
                : rocblas_operation::rocblas_operation_none;

        lda_ = get_ld(weights_d, transA_);
        ldb_ = get_ld(src_d, transB_);
        ldc_ = get_ld(dst_d, transC_);

        if (isbatched_) {
            stride_a_ = get_batch_stride(weights_d);
            stride_b_ = get_batch_stride(src_d);
            stride_c_ = get_batch_stride(dst_d);
        }

        return status::success;
    }

    status_t init_parameters(const memory_desc_wrapper src_d,
            const memory_desc_wrapper weights_d,
            const memory_desc_wrapper dst_d, const memory_desc_wrapper bias_d) {
        // Matmul supports runtime paramters for dimensions and scales.
        // We need to initialize them in the execute function.
        CHECK(init_gemm_parameters(src_d, weights_d, dst_d));

        if (with_bias_ || with_eltwise_) {
            // Initialise MIOpen descriptors
            miopenDataType_t data_types[NUM_IO];
            int ndims = dst_d.ndims() < 4 ? 4 : dst_d.ndims();
            int dims[NUM_IO][DNNL_MAX_NDIMS];
            int strides[NUM_IO][DNNL_MAX_NDIMS];

            convert_dims_matmul(dst_d.dims(), dims[dst], dst_d.ndims());
            CHECK(convert_data_type(dst_d.md_, &data_types[dst], false));
            convert_dims_matmul(
                    dst_d.blocking_desc().strides, strides[dst], dst_d.ndims());
            CHECK(create_and_set_tensor_descriptor(&tensor_descs_[dst],
                    data_types[dst], ndims, dims[dst], strides[dst]));

            if (with_bias_) {
                // Create bias and destination tensor descriptors
                convert_dims_matmul(bias_d.dims(), dims[bias], bias_d.ndims());
                convert_dims_matmul(bias_d.blocking_desc().strides,
                        strides[bias], bias_d.ndims());
                CHECK(convert_data_type(bias_d.md_, &data_types[bias], false));
                CHECK(create_and_set_tensor_descriptor(&tensor_descs_[bias],
                        data_types[bias], ndims, dims[bias], strides[bias]));
            }
        }
        return status::success;
    }

    void execute(rocblas_handle rocblas_handle, miopenHandle_t miopen_handle,
            void *a, void *b, void *c, void *bias, void *scratch) {

        const void *alpha = get_gemm_alpha();
        const void *beta = get_gemm_beta();
        scratch = c;
        temp_mem_desc_ = tensor_descs_[io::dst];
        int solution_index = 0;
        uint32_t flags = 0;

        auto flip_op = [](rocblas_operation op) {
            return (op == rocblas_operation::rocblas_operation_transpose)
                    ? rocblas_operation::rocblas_operation_none
                    : rocblas_operation::rocblas_operation_transpose;
        };

        if (isbatched_) {
            if (weightBroadcastNeeded)
                stride_a_ = 0;
            else if (srcBroadcastNeeded)
                stride_b_ = 0;

            if (transC_ == rocblas_operation::rocblas_operation_transpose) {
                ROCBLAS_EXECUTE_FUNC(rocblas_gemm_strided_batched_ex,
                        rocblas_handle, flip_op(transB_), flip_op(transA_), N_,
                        M_, K_, alpha, b, src_type_, ldb_, stride_b_, a,
                        weights_type_, lda_, stride_a_, beta, scratch,
                        dst_type_, ldc_, stride_c_, scratch, dst_type_, ldc_,
                        stride_c_, batch_count_, acc_type_, gemm_algo_,
                        solution_index, (uint32_t)flags);
            } else {
                ROCBLAS_EXECUTE_FUNC(rocblas_gemm_strided_batched_ex,
                        rocblas_handle, transA_, transB_, M_, N_, K_, alpha, a,
                        weights_type_, lda_, stride_a_, b, src_type_, ldb_,
                        stride_b_, beta, scratch, dst_type_, ldc_, stride_c_,
                        scratch, dst_type_, ldc_, stride_c_, batch_count_,
                        acc_type_, gemm_algo_, solution_index, (uint32_t)flags);
            }
        } else {
            if (transC_ == rocblas_operation::rocblas_operation_transpose) {
                ROCBLAS_EXECUTE_FUNC(rocblas_gemm_ex, rocblas_handle,
                        flip_op(transB_), flip_op(transA_), N_, M_, K_, alpha,
                        b, src_type_, ldb_, a, weights_type_, lda_, beta,
                        scratch, dst_type_, ldc_, scratch, dst_type_, ldc_,
                        acc_type_, gemm_algo_, solution_index, (uint32_t)flags);
            } else {
                ROCBLAS_EXECUTE_FUNC(rocblas_gemm_ex, rocblas_handle, transA_,
                        transB_, M_, N_, K_, alpha, a, weights_type_, lda_, b,
                        src_type_, ldb_, beta, scratch, dst_type_, ldc_,
                        scratch, dst_type_, ldc_, acc_type_, gemm_algo_,
                        solution_index, (uint32_t)flags);
            }
        }
        if (with_bias_) {
            // When bias is specified call miopenOpTensor()
            float bias_beta = 0;
            float alpha = 1;
            float scales = 1;

            MIOPEN_EXECUTE_FUNC(miopenOpTensor, miopen_handle,
                    miopenTensorOpAdd, &alpha, temp_mem_desc_, scratch, &scales,
                    tensor_descs_[io::bias], bias, &bias_beta, temp_mem_desc_,
                    scratch);
        }
        if (with_eltwise_) {
            // Perform elementwise operation if specified
            float alpha = 1;
            float beta = 0;

            MIOPEN_EXECUTE_FUNC(miopenActivationForward, miopen_handle,
                    act_desc_, &alpha, temp_mem_desc_, scratch, &beta,
                    temp_mem_desc_, scratch);
        }
    }

    ~miopen_matmul_impl_t() { cleanup(); }

    void cleanup() {
        if (act_desc_) {
            MIOPEN_EXECUTE_FUNC_V(miopenDestroyActivationDescriptor, act_desc_);
            act_desc_ = nullptr;
        }

        for (size_t i = 0; i < NUM_IO; i++) {
            if (tensor_descs_[i]) {
                MIOPEN_EXECUTE_FUNC_V(
                        miopenDestroyTensorDescriptor, tensor_descs_[i]);
                tensor_descs_[i] = nullptr;
            }
        }
    }

private:
    status_t get_rocblas_data_type(
            dnnl_data_type_t data_type, rocblas_datatype &blas_dt) {
        switch (data_type) {
            case dnnl_data_type_t::dnnl_f32:
                blas_dt = rocblas_datatype_f32_r;
                return status::success;
            case dnnl_data_type_t::dnnl_f16:
                blas_dt = rocblas_datatype_f16_r;
                return status::success;
            case dnnl_data_type_t::dnnl_s8:
                blas_dt = rocblas_datatype_i8_r;
                return status::success;
            case dnnl_data_type_t::dnnl_s32:
                blas_dt = rocblas_datatype_i32_r;
                return status::success;
            case dnnl_data_type_t::dnnl_bf16:
                blas_dt = rocblas_datatype_bf16_r;
                return status::success;
            default: return status::unimplemented;
        }
        return status::unimplemented;
    }

    const void *get_gemm_alpha() const {
        switch (acc_type_) {
            case rocblas_datatype::rocblas_datatype_i32_r:
                return reinterpret_cast<const void *>(&alpha_s32);
            case rocblas_datatype::rocblas_datatype_f32_r:
                return reinterpret_cast<const void *>(&alpha_f32);
            default: assert(!"unknown acc type"); return nullptr;
        }
    }

    const void *get_gemm_beta() const {
        switch (acc_type_) {
            case rocblas_datatype::rocblas_datatype_i32_r:
                return reinterpret_cast<const void *>(&sum_scale_s32);
            case rocblas_datatype::rocblas_datatype_f32_r:
                return reinterpret_cast<const void *>(&sum_scale_f32);
            default: assert(!"unknown acc type"); return nullptr;
        }
    }

    rocblas_operation transA_;
    rocblas_operation transB_;
    rocblas_operation transC_;
    int M_, N_, K_;
    int lda_, ldb_, ldc_;
    long long int stride_a_, stride_b_, stride_c_;
    bool weightBroadcastNeeded = false, srcBroadcastNeeded = false;
    int alpha_s32 = 1, sum_scale_s32 = 0;
    float alpha_f32 = 1.0f, sum_scale_f32 = 0.0f;
    bool isbatched_ = false, with_bias_ = false;
    bool with_eltwise_ = false;
    bool with_scratchpad_ = false, has_runtime_params_ = false;
    dnnl_data_type_t scratchpad_type_;
    rocblas_datatype src_type_, weights_type_, dst_type_;
    rocblas_datatype acc_type_;
    rocblas_gemm_algo gemm_algo_
            = rocblas_gemm_algo::rocblas_gemm_algo_standard;
    int batch_count_;
    enum io { bias = 0, dst, NUM_IO };
    miopenTensorDescriptor_t tensor_descs_[NUM_IO] = {},
                             temp_mem_desc_ = nullptr;
    miopenActivationDescriptor_t act_desc_ = nullptr;
    float post_op_sum_;
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
