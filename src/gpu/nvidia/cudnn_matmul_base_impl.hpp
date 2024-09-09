/*******************************************************************************
* Copyright 2024 Intel Corporation
* Copyright 2024 Codeplay Software Limited
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

#ifndef GPU_NVIDIA_CUDNN_MATMUL_BASE_IMPL_HPP
#define GPU_NVIDIA_CUDNN_MATMUL_BASE_IMPL_HPP

#include "cublasLt.h"
#include "cudnn.h"

#include "gpu/nvidia/engine.hpp"
#include "gpu/nvidia/stream.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cudnn_matmul_base_impl_t {
    virtual status_t init_gemm_parameters(const memory_desc_wrapper src_d,
            const memory_desc_wrapper weights_d,
            const memory_desc_wrapper dst_d)
            = 0;
    virtual void init_scratchpad(matmul_pd_t *pd) = 0;
    virtual void cleanup() = 0;

    bool isbatched() { return isbatched_; }
    bool with_bias() { return with_separate_bias_; }
    bool bias_dt_mismatch() { return bias_dt_mismatch_; }
    bool has_runtime_params() { return has_runtime_params_; }

    bool with_eltwise(int position, const matmul_pd_t *pd) {
        return pd->attr()->post_ops_.contain(primitive_kind::eltwise, position);
    }

    bool with_sum(const matmul_pd_t *pd) {
        return pd->attr()->post_ops_.contain(primitive_kind::sum, 0)
                || pd->attr()->post_ops_.contain(primitive_kind::sum, 1);
    }

    // Returns scaling factor for post-ops=sum operation
    float sum_scale(const matmul_pd_t *pd) {
        int sum_idx_ = pd->attr()->post_ops_.find(primitive_kind::sum);
        return pd->attr()->post_ops_.entry_[sum_idx_].sum.scale;
    }

    float eltwise_alpha(const matmul_pd_t *pd) {
        int eltwise_idx_ = pd->attr()->post_ops_.find(primitive_kind::eltwise);
        return with_eltwise(0, pd) || with_eltwise(1, pd)
                ? pd->attr()->post_ops_.entry_[eltwise_idx_].eltwise.alpha
                : 1.0f;
    }

    float eltwise_beta(const matmul_pd_t *pd) {
        int eltwise_idx_ = pd->attr()->post_ops_.find(primitive_kind::eltwise);
        return with_eltwise(0, pd) || with_eltwise(1, pd)
                ? pd->attr()->post_ops_.entry_[eltwise_idx_].eltwise.beta
                : 0.0f;
    }

    alg_kind_t eltwise_algo(const matmul_pd_t *pd) {
        int eltwise_idx_ = pd->attr()->post_ops_.find(primitive_kind::eltwise);
        return with_eltwise(0, pd) || with_eltwise(1, pd)
                ? pd->attr()->post_ops_.entry_[eltwise_idx_].eltwise.alg
                : dnnl_alg_kind_undef;
    }

    int get_ld(const memory_desc_wrapper desc, cublasOperation_t trans) {
        const int ndims = desc.ndims();
        const auto *strides = &desc.blocking_desc().strides[ndims - 2];
        const int ld = strides[trans == cublasOperation_t::CUBLAS_OP_N ? 0 : 1];
        return ld;
    }
    // creates operation descriptor based on the elemen-wise operation specified
    status_t create_and_set_op_descriptor(
            const matmul_pd_t *pd, cudnnActivationDescriptor_t &act_desc) {
        CHECK(CUDNN_EXECUTE_FUNC_S(cudnnCreateActivationDescriptor, &act_desc));

        cudnnActivationMode_t mode;

        switch (eltwise_algo(pd)) {
            case alg_kind::eltwise_relu:
                mode = cudnnActivationMode_t::CUDNN_ACTIVATION_RELU;
                break;
            case alg_kind::eltwise_tanh:
                mode = cudnnActivationMode_t::CUDNN_ACTIVATION_TANH;
                break;
            case alg_kind::eltwise_elu:
                mode = cudnnActivationMode_t::CUDNN_ACTIVATION_ELU;
                break;
            case alg_kind::eltwise_logistic:
                mode = cudnnActivationMode_t::CUDNN_ACTIVATION_SIGMOID;
                break;
            default: return status::unimplemented;
        }

        // NaNs by default are propagated in oneDNN, although the forward
        // convolution routine does not support this.
        auto propagate_nan = cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN;

        // For ReLU, a ceiling of 0 means no limit.
        double ceiling = eltwise_alpha(pd);

        CHECK(CUDNN_EXECUTE_FUNC_S(cudnnSetActivationDescriptor, act_desc, mode,
                propagate_nan, ceiling));

        return status::success;
    }

    void convert_dims_matmul(
            const dnnl_dim_t *dims, int *new_dims, int n_dims) {
        // Moving the dimensions because cudnnAddTensor doesn't work when
        // bia_mask=1
        if (n_dims == 3) { return convert_dims(dims, new_dims, n_dims); }
        new_dims[0] = 1;
        for (size_t i = 0; i < n_dims; i++) {
            new_dims[i + 1] = static_cast<int>(dims[i]);
        }
        for (size_t i = n_dims; i < 4; i++) {
            new_dims[i + 1] = 1;
        }
    }

    int get_batch_stride(const memory_desc_wrapper desc) {
        auto dims = desc.dims();
        auto strides = desc.blocking_desc().strides;
        return dims[0] == 1 ? 0 : strides[0];
    }

    virtual ~cudnn_matmul_base_impl_t() = default;

    size_t bias_scratch_size() { return reorder_scratch_size_; }

protected:
    int lda_, ldb_, ldc_;
    long long int stride_a_, stride_b_, stride_c_;
    bool isbatched_ = false, with_separate_bias_ = false,
         bias_dt_mismatch_ = false, with_dst_scale_ = false;
    bool reorder_required_ = false, with_separate_eltwise_ = false;
    bool has_runtime_params_ = false;
    cudaDataType_t src_type_, weights_type_, dst_type_;
    cudaDataType_t acc_type_ = cudaDataType_t::CUDA_R_32F;
    int batch_count_;
    enum io { bias = 0, dst, NUM_IO };
    cudnnTensorDescriptor_t tensor_descs_[NUM_IO] = {},
                            temp_mem_desc_ = nullptr;
    cudnnActivationDescriptor_t act_desc_ = nullptr;
    float post_op_sum_ = 0;
    size_t algo_scratch_size_ = 0;
    size_t reorder_scratch_size_ = 0;

    status_t handle_post_ops(cudnnHandle_t cudnn_handle, void *dst, void *bias,
            void *reorder_scratch, float host_dst_scale) {
        if (with_separate_bias_) {
            // When bias is specified call cudnnAddTensor()
            float bias_beta = 1;
            auto scale = (with_separate_eltwise_ ? 1 : 1.0f / host_dst_scale);
            CUDNN_EXECUTE_FUNC(cudnnAddTensor, cudnn_handle, &scale,
                    tensor_descs_[io::bias], bias, &bias_beta, temp_mem_desc_,
                    reorder_scratch);
        }
        if (with_separate_eltwise_) {
            // Perform elementwise operation if specified
            float alpha = 1.0f / host_dst_scale;
            float beta = 0;
            CUDNN_EXECUTE_FUNC(cudnnActivationForward, cudnn_handle, act_desc_,
                    &alpha, temp_mem_desc_, reorder_scratch, &beta,
                    temp_mem_desc_, reorder_scratch);
        }
        if (reorder_required_) {
            // Reorder from scratchpad to destination if required
            float reorder_alpha = 1;
            CUDNN_EXECUTE_FUNC(cudnnTransformTensor, cudnn_handle,
                    &reorder_alpha, temp_mem_desc_, reorder_scratch,
                    &post_op_sum_, tensor_descs_[io::dst], dst);
        }

        return status::success;
    }
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
