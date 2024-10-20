/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#ifndef GPU_NVIDIA_CUDNN_MATMUL_IMPL_HPP
#define GPU_NVIDIA_CUDNN_MATMUL_IMPL_HPP

#include "cudnn.h"

#include "gpu/nvidia/cudnn_matmul_base_impl.hpp"
#include "gpu/nvidia/engine.hpp"
#include "gpu/nvidia/sycl_cuda_scoped_context.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cublas_params : cublas_base_params {

    status_t init(const memory_desc_t *src_md, const memory_desc_t *weights_md,
            const memory_desc_t *dst_md, const memory_desc_t *bias_md,
            const primitive_attr_t *attr, bool batched, bool with_bias) {

        CHECK(get_cublas_data_type(src_md->data_type, src_type_));

        CHECK(get_cublas_data_type(weights_md->data_type, weights_type_));

        isbatched_ = batched;

        memory_desc_wrapper src_d = memory_desc_wrapper(src_md);
        memory_desc_wrapper weights_d = memory_desc_wrapper(weights_md);
        memory_desc_wrapper dst_d = memory_desc_wrapper(dst_md);

        if (!(src_d.is_plain() && weights_d.is_plain() && dst_d.is_plain())) {
            return status::unimplemented;
        }

        with_dst_scale_ = !attr->scales_.get(DNNL_ARG_DST).has_default_values();
        with_separate_bias_ = with_bias;
        if ((with_separate_bias_)
                && (bias_md->data_type != dst_md->data_type)) {
            // When datatype of bias is different from the dst,
            // we need to reorder the output.
            bias_dt_mismatch_ = true;
            reorder_required_ = true;
            CHECK(get_cublas_data_type(bias_md->data_type, dst_type_));
        } else {
            CHECK(get_cublas_data_type(dst_md->data_type, dst_type_));
        }

        // cuBLAS only supports s8s8f32 configuration.
        // Hence, one final reorder is required if the cfg = s8s8s8
        if (dst_type_ == cudaDataType_t::CUDA_R_8I) {
            reorder_required_ = true;
            dst_type_ = cudaDataType_t::CUDA_R_32F;
        }

        if (with_eltwise(0, attr) || with_eltwise(1, attr)) {
            with_separate_eltwise_ = true;
            CHECK(create_and_set_op_descriptor(attr, act_desc_));
        }

        // Set parameter when post-op sum is specified
        if (with_sum(attr)) { post_op_sum_ = sum_scale(attr); }

        has_runtime_params_ = src_d.has_runtime_dims_or_strides()
                || dst_d.has_runtime_dims_or_strides()
                || weights_d.has_runtime_dims_or_strides();

        if (!has_runtime_params_) {
            // Initialise all gemm parameters if there are no runtime parameters
            set_params(src_d, weights_d, dst_d, memory_desc_wrapper(bias_md));
        }

        return status::success;
    }

    status_t init_from_params(const std::shared_ptr<cublas_params> &other) {
        if (!other) { return status::invalid_arguments; }
        src_type_ = other->src_type_;
        weights_type_ = other->weights_type_;
        isbatched_ = other->isbatched_;
        with_dst_scale_ = other->with_dst_scale_;
        with_separate_bias_ = other->with_separate_bias_;
        bias_dt_mismatch_ = other->bias_dt_mismatch_;
        reorder_required_ = other->reorder_required_;
        dst_type_ = other->dst_type_;
        with_separate_eltwise_ = other->with_separate_eltwise_;
        has_runtime_params_ = other->has_runtime_params_;
        return status::success;
    }

    status_t set_gemm_params(const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d) {

        if (isbatched_) batch_count_ = dst_d.dims()[0];
        const dim_t M = dst_d.dims()[isbatched_ + 1];
        const dim_t N = dst_d.dims()[isbatched_ + 0];
        const dim_t K = src_d.dims()[isbatched_ + 1];

        M_ = (int)M;
        N_ = (int)N;
        K_ = (int)K;

        const auto &dst_strides = &dst_d.blocking_desc().strides[isbatched_];
        const auto &src_strides = &src_d.blocking_desc().strides[isbatched_];
        const auto &weights_strides
                = &weights_d.blocking_desc().strides[isbatched_];

        // A matrix is the weights
        transA_ = weights_strides[1] == 1
                        && weights_d.dims()[isbatched_ + 0] > 1
                ? cublasOperation_t::CUBLAS_OP_N
                : cublasOperation_t::CUBLAS_OP_T;
        // B matrix is the src
        transB_ = src_strides[1] == 1 && src_d.dims()[isbatched_ + 0] > 1
                ? cublasOperation_t::CUBLAS_OP_N
                : cublasOperation_t::CUBLAS_OP_T;
        // C matrix is the dst
        transC_ = dst_strides[1] == 1 && dst_d.dims()[isbatched_ + 0] > 1
                ? cublasOperation_t::CUBLAS_OP_N
                : cublasOperation_t::CUBLAS_OP_T;

        lda_ = get_ld(weights_d, transA_);
        ldb_ = get_ld(src_d, transB_);
        ldc_ = get_ld(dst_d, transC_);

        if (isbatched_) {
            // These parameters are required for cublasGemmStridedBatchedEx()
            stride_a_ = get_batch_stride(weights_d);
            stride_b_ = get_batch_stride(src_d);
            stride_c_ = get_batch_stride(dst_d);

            // Enable broadcast semantics.
            if (src_d.dims()[0] > weights_d.dims()[0])
                stride_a_ = 0;
            else if (src_d.dims()[0] < weights_d.dims()[0])
                stride_b_ = 0;
        }

        return status::success;
    }

    status_t set_params(const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d,
            const memory_desc_wrapper &bias_d) {
        // Matmul supports runtime paramters for dimensions and scales.
        // We need to initialize them in the execute function.
        CHECK(set_gemm_params(src_d, weights_d, dst_d));

        if (with_separate_bias_ || reorder_required_ || with_separate_eltwise_
                || with_dst_scale_) {
            // Initialise cuDNN descriptors
            cudnnDataType_t data_types[NUM_IO];
            int ndims = dst_d.ndims() < 4 ? 4 : dst_d.ndims();
            int dims[NUM_IO][DNNL_MAX_NDIMS];
            int strides[NUM_IO][DNNL_MAX_NDIMS];

            convert_dims_matmul(dst_d.dims(), dims[dst], dst_d.ndims());
            CHECK(convert_data_type(dst_d.md_, &data_types[dst], false));
            convert_dims_matmul(
                    dst_d.blocking_desc().strides, strides[dst], dst_d.ndims());
            CHECK(create_and_set_tensor_descriptor(&tensor_descs_[dst],
                    data_types[dst], ndims, dims[dst], strides[dst]));

            if (reorder_required_ && !bias_dt_mismatch_) {
                // If reorder is required, we need to create a scratchpad memory
                // to store the intermediate result
                CHECK(create_and_set_tensor_descriptor(&temp_mem_desc_,
                        cudnnDataType_t::CUDNN_DATA_FLOAT, ndims, dims[dst],
                        strides[dst]));
            }

            if (with_separate_bias_) {
                // Create bias and destination tensor descriptors
                convert_dims_matmul(bias_d.dims(), dims[bias], bias_d.ndims());
                convert_dims_matmul(bias_d.blocking_desc().strides,
                        strides[bias], bias_d.ndims());
                CHECK(convert_data_type(bias_d.md_, &data_types[bias], false));
                CHECK(create_and_set_tensor_descriptor(&tensor_descs_[bias],
                        data_types[bias], ndims, dims[bias], strides[bias]));
                if (bias_dt_mismatch_) {
                    CHECK(create_and_set_tensor_descriptor(&temp_mem_desc_,
                            data_types[bias], ndims, dims[dst], strides[dst]));
                }
            }
        }

        const auto dst_nelems = dst_d.nelems(true);
        reorder_scratch_size_ = dst_nelems * sizeof(float);

        return status::success;
    }

    size_t scratchpad_size(const memory_desc_t *dst_md) const {
        const auto dst_nelems = memory_desc_wrapper(dst_md).nelems(true);
        return dst_nelems * sizeof(float);
    }

    void init_scratchpad(const memory_desc_t *dst_md,
            memory_tracking::registrar_t scratchpad) {
        auto reorder_scratch_size = scratchpad_size(dst_md);
        if (reorder_scratch_size > 0) {
            scratchpad.book(memory_tracking::names::key_matmul_dst_in_acc_dt,
                    reorder_scratch_size, 1);
        }
    }

    void convert_dims_matmul(
            const dnnl_dim_t *dims, int *new_dims, int n_dims) {
        // Moving the dimensions because cudnnAddTensor doesn't work when
        // bia_mask=1
        if (n_dims == 3) { return convert_dims(dims, new_dims, n_dims); }
        new_dims[0] = 1;
        for (int i = 0; i < n_dims; i++) {
            new_dims[i + 1] = static_cast<int>(dims[i]);
        }
        for (int i = n_dims; i < 4; i++) {
            new_dims[i + 1] = 1;
        }
    }

    int get_ld(const memory_desc_wrapper desc, cublasOperation_t trans) {
        const int ndims = desc.ndims();
        const auto *strides = &desc.blocking_desc().strides[ndims - 2];
        const int ld = strides[trans == cublasOperation_t::CUBLAS_OP_N ? 0 : 1];
        return ld;
    }

    // creates operation descriptor based on the elemen-wise operation specified
    status_t create_and_set_op_descriptor(const primitive_attr_t *attr,
            cudnnActivationDescriptor_t &act_desc) {
        CHECK(CUDNN_EXECUTE_FUNC_S(cudnnCreateActivationDescriptor, &act_desc));

        cudnnActivationMode_t mode;

        switch (eltwise_algo(attr)) {
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
        double ceiling = eltwise_alpha(attr);

        CHECK(CUDNN_EXECUTE_FUNC_S(cudnnSetActivationDescriptor, act_desc, mode,
                propagate_nan, ceiling));

        return status::success;
    }

    float eltwise_alpha(const primitive_attr_t *attr) {
        int eltwise_idx_ = attr->post_ops_.find(primitive_kind::eltwise);
        return with_eltwise(0, attr) || with_eltwise(1, attr)
                ? attr->post_ops_.entry_[eltwise_idx_].eltwise.alpha
                : 1.0f;
    }

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

    void cleanup() const {
        if (act_desc_) {
            CUDNN_EXECUTE_FUNC_V(cudnnDestroyActivationDescriptor, act_desc_);
        }
        if ((reorder_required_ && !bias_dt_mismatch_)
                || ((with_separate_bias_ && bias_dt_mismatch_)
                        && temp_mem_desc_)) {
            CUDNN_EXECUTE_FUNC_V(cudnnDestroyTensorDescriptor, temp_mem_desc_);
        }
        for (size_t i = 0; i < NUM_IO; i++) {
            if (tensor_descs_[i]) {
                CUDNN_EXECUTE_FUNC_V(
                        cudnnDestroyTensorDescriptor, tensor_descs_[i]);
            }
        }
    }

    int lda_, ldb_, ldc_;

    int64_t stride_a_, stride_b_, stride_c_;

    enum io { bias = 0, dst, NUM_IO };
    cudnnTensorDescriptor_t tensor_descs_[NUM_IO] = {},
                            temp_mem_desc_ = nullptr;
    cudnnActivationDescriptor_t act_desc_ = nullptr;

    cublasOperation_t transA_;
    cublasOperation_t transB_;
    cublasOperation_t transC_;
    cublasGemmAlgo_t gemm_algo_
            = cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP;
};

struct cudnn_matmul_impl_t {

    void set_non_runtime_params(
            const std::shared_ptr<cublas_params> &matmul_params) {
        matmul_params_ = matmul_params;
    }

    void execute(cublasHandle_t cublas_handle, cudnnHandle_t cudnn_handle,
            const std::shared_ptr<cublas_params> &matmul_params, void *a,
            void *b, void *c, void *bias, void *reorder_scratch,
            void *src_scale, void *wei_scale, void *dst_scale) {

        // use cached params unless using runtime dimensions
        std::shared_ptr<cublas_params> params
                = matmul_params->has_runtime_params_ ? matmul_params
                                                     : matmul_params_;

        float gemm_beta = 0;
        if (!params->bias_dt_mismatch_ && !params->reorder_required_) {
            // Case where no reorder is required, scratchpad points to dst (c)
            reorder_scratch = c;
            params->temp_mem_desc_
                    = params->tensor_descs_[cublas_params::io::dst];
            gemm_beta = params->post_op_sum_;
        }
        auto flip_op = [](cublasOperation_t op) {
            return (op == cublasOperation_t::CUBLAS_OP_T)
                    ? cublasOperation_t::CUBLAS_OP_N
                    : cublasOperation_t::CUBLAS_OP_T;
        };

        float scale = 1.0f;
        float host_dst_scale = 1.0f;
        if (src_scale) {
            float host_src_scale = 1.0f;
            CUDA_EXECUTE_FUNC(cuMemcpy, (CUdeviceptr)&host_src_scale,
                    (CUdeviceptr)src_scale, sizeof(float));
            scale *= host_src_scale;
        }
        if (wei_scale) {
            float host_wei_scale = 1.0f;
            CUDA_EXECUTE_FUNC(cuMemcpy, (CUdeviceptr)&host_wei_scale,
                    (CUdeviceptr)wei_scale, sizeof(float));
            scale *= host_wei_scale;
        }
        if (dst_scale) {
            CUDA_EXECUTE_FUNC(cuMemcpy, (CUdeviceptr)&host_dst_scale,
                    (CUdeviceptr)dst_scale, sizeof(float));
            // For eltwise post-ops, apply the dst scale afterward
            if (!params->with_separate_eltwise_) scale /= host_dst_scale;
        }

        auto M = params->M_;
        auto N = params->N_;
        auto K = params->K_;

        auto lda = params->lda_;
        auto ldb = params->ldb_;
        auto ldc = params->ldc_;

        auto src_type = params->src_type_;
        auto weights_type = params->weights_type_;
        auto dst_type = params->dst_type_;

        auto stride_a = params->stride_a_;
        auto stride_b = params->stride_b_;
        auto stride_c = params->stride_c_;

        auto batch_count = params->batch_count_;
        auto acc_type = params->acc_type_;
        auto gemm_algo = params->gemm_algo_;

        auto transA = params->transA_;
        auto transB = params->transB_;
        auto transC = params->transC_;

        if (params->isbatched_) {
            // Calls cublasGemmStridedBatchedEx()
            if (transC == cublasOperation_t::CUBLAS_OP_T) {
                CUBLAS_EXECUTE_FUNC(cublasGemmStridedBatchedEx, cublas_handle,
                        flip_op(transB), flip_op(transA), N, M, K, &scale, b,
                        src_type, ldb, stride_b, a, weights_type, lda, stride_a,
                        &gemm_beta, reorder_scratch, dst_type, ldc, stride_c,
                        batch_count, acc_type, gemm_algo);

            } else {
                CUBLAS_EXECUTE_FUNC(cublasGemmStridedBatchedEx, cublas_handle,
                        transA, transB, M, N, K, &scale, a, weights_type, lda,
                        stride_a, b, src_type, ldb, stride_b, &gemm_beta,
                        reorder_scratch, dst_type, ldc, stride_c, batch_count,
                        acc_type, gemm_algo);
            }
        } else {
            // Calls cublasGemmEx()
            if (transC == cublasOperation_t::CUBLAS_OP_T) {
                CUBLAS_EXECUTE_FUNC(cublasGemmEx, cublas_handle,
                        flip_op(transB), flip_op(transA), N, M, K, &scale, b,
                        src_type, ldb, a, weights_type, lda, &gemm_beta,
                        reorder_scratch, dst_type, ldc, acc_type, gemm_algo);
            } else {
                CUBLAS_EXECUTE_FUNC(cublasGemmEx, cublas_handle, transA, transB,
                        M, N, K, &scale, a, weights_type, lda, b, src_type, ldb,
                        &gemm_beta, reorder_scratch, dst_type, ldc, acc_type,
                        gemm_algo);
            }
        }
        params->handle_post_ops(
                cudnn_handle, c, bias, reorder_scratch, host_dst_scale);
    }

    ~cudnn_matmul_impl_t() {
        if (matmul_params_) { matmul_params_->cleanup(); }
    }

private:
    std::shared_ptr<cublas_params> matmul_params_;
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
