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

#ifndef GPU_NVIDIA_CUDNN_MATMUL_IMPL_HPP
#define GPU_NVIDIA_CUDNN_MATMUL_IMPL_HPP

#include "cudnn.h"

#include "gpu/nvidia/sycl_cuda_engine.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cudnn_matmul_impl_t {

    bool with_eltwise(int position, const matmul_pd_t *pd) const {
        return pd->attr()->post_ops_.contain(primitive_kind::eltwise, position);
    }

    float eltwise_alpha(const matmul_pd_t *pd) const {
        int eltwise_idx_ = pd->attr()->post_ops_.find(primitive_kind::eltwise);
        return with_eltwise(0, pd) || with_eltwise(1, pd)
                ? pd->attr()->post_ops_.entry_[eltwise_idx_].eltwise.alpha
                : 1.0f;
    }

    float eltwise_beta(const matmul_pd_t *pd) const {
        int eltwise_idx_ = pd->attr()->post_ops_.find(primitive_kind::eltwise);
        return with_eltwise(0, pd) || with_eltwise(1, pd)
                ? pd->attr()->post_ops_.entry_[eltwise_idx_].eltwise.beta
                : 0.0f;
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

    // creates operation descriptor based on the elemen-wise operation specified
    status_t create_and_set_op_descriptor(const matmul_pd_t *pd) {
        CHECK(CUDNN_EXECUTE_FUNC_S(
                cudnnCreateActivationDescriptor, &act_desc_));

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

        CHECK(CUDNN_EXECUTE_FUNC_S(cudnnSetActivationDescriptor, act_desc_,
                mode, propagate_nan, ceiling));

        return status::success;
    }

    status_t init(matmul_pd_t *pd) {
        CHECK(get_cublas_data_type(pd->src_md()->data_type, src_type_));
        CHECK(get_cublas_data_type(pd->weights_md()->data_type, weights_type_));

        isbatched_ = pd->batched();

        memory_desc_wrapper src_d = memory_desc_wrapper(pd->src_md());
        memory_desc_wrapper weights_d = memory_desc_wrapper(pd->weights_md());
        memory_desc_wrapper dst_d = memory_desc_wrapper(pd->dst_md());

        with_dst_scale_
                = !pd->attr()->scales_.get(DNNL_ARG_DST).has_default_values();
        with_bias_ = pd->with_bias();
        if ((with_bias_)
                && (pd->weights_md(1)->data_type != pd->dst_md()->data_type)) {
            // When datatype of bias is different from the dst,
            // we need to reorder the output.
            bias_dt_mismatch_ = true;
            reorder_required_ = true;
            CHECK(get_cublas_data_type(
                    pd->weights_md(1)->data_type, dst_type_));
        } else {
            CHECK(get_cublas_data_type(pd->dst_md()->data_type, dst_type_));
        }

        // cuBLAS only supports s8s8f32 configuration.
        // Hence, one final reorder is required if the cfg = s8s8s8
        if (dst_type_ == cudaDataType_t::CUDA_R_8I) {
            reorder_required_ = true;
            dst_type_ = cudaDataType_t::CUDA_R_32F;
        }

        if (with_eltwise(0, pd) || with_eltwise(1, pd)) {
            with_eltwise_ = true;
            CHECK(create_and_set_op_descriptor(pd));
        }

        // Set parameter when post-op sum is specified
        if (with_sum(pd)) { post_op_sum_ = sum_scale(pd); }

        has_runtime_params_ = src_d.has_runtime_dims_or_strides()
                || dst_d.has_runtime_dims_or_strides()
                || weights_d.has_runtime_dims_or_strides();

        if (!has_runtime_params_) {
            // Initialise all gemm parameters if there are no runtime parameters
            init_parameters(src_d, weights_d, dst_d,
                    memory_desc_wrapper(pd->weights_md(1)));
        }

        return status::success;
    }

    bool isbatched() { return isbatched_; }
    bool with_bias() { return with_bias_; }
    bool has_runtime_params() { return has_runtime_params_; }

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

    int get_ld(const memory_desc_wrapper desc, cublasOperation_t trans) {
        const int ndims = desc.ndims();
        const auto *strides = &desc.blocking_desc().strides[ndims - 2];
        const int ld = strides[trans == cublasOperation_t::CUBLAS_OP_N ? 0 : 1];
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

    status_t init_parameters(const memory_desc_wrapper src_d,
            const memory_desc_wrapper weights_d,
            const memory_desc_wrapper dst_d, const memory_desc_wrapper bias_d) {
        // Matmul supports runtime paramters for dimensions and scales.
        // We need to initialize them in the execute function.
        CHECK(init_gemm_parameters(src_d, weights_d, dst_d));

        if (with_bias_ || reorder_required_ || with_eltwise_
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

            if (with_bias_) {
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
        return status::success;
    }

    void execute(cublasHandle_t cublas_handle, cudnnHandle_t cudnn_handle,
            void *a, void *b, void *c, void *bias, void *scratch,
            void *src_scale, void *wei_scale, void *dst_scale) {
        float gemm_beta = 0;
        if (!bias_dt_mismatch_ && !reorder_required_) {
            // Case where no reorder is required, scratchpad points to dst (c)
            scratch = c;
            temp_mem_desc_ = tensor_descs_[io::dst];
            gemm_beta = post_op_sum_;
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
            if (!with_eltwise_) scale /= host_dst_scale;
        }

        if (isbatched_) {
            // Calls cublasGemmStridedBatchedEx()
            if (transC_ == cublasOperation_t::CUBLAS_OP_T) {
                CUBLAS_EXECUTE_FUNC(cublasGemmStridedBatchedEx, cublas_handle,
                        flip_op(transB_), flip_op(transA_), N_, M_, K_, &scale,
                        b, src_type_, ldb_, stride_b_, a, weights_type_, lda_,
                        stride_a_, &gemm_beta, scratch, dst_type_, ldc_,
                        stride_c_, batch_count_, acc_type_, gemm_algo_);

            } else {
                CUBLAS_EXECUTE_FUNC(cublasGemmStridedBatchedEx, cublas_handle,
                        transA_, transB_, M_, N_, K_, &scale, a, weights_type_,
                        lda_, stride_a_, b, src_type_, ldb_, stride_b_,
                        &gemm_beta, scratch, dst_type_, ldc_, stride_c_,
                        batch_count_, acc_type_, gemm_algo_);
            }
        } else {
            // Calls cublasGemmEx()
            if (transC_ == cublasOperation_t::CUBLAS_OP_T) {
                CUBLAS_EXECUTE_FUNC(cublasGemmEx, cublas_handle,
                        flip_op(transB_), flip_op(transA_), N_, M_, K_, &scale,
                        b, src_type_, ldb_, a, weights_type_, lda_, &gemm_beta,
                        scratch, dst_type_, ldc_, acc_type_, gemm_algo_);
            } else {
                CUBLAS_EXECUTE_FUNC(cublasGemmEx, cublas_handle, transA_,
                        transB_, M_, N_, K_, &scale, a, weights_type_, lda_, b,
                        src_type_, ldb_, &gemm_beta, scratch, dst_type_, ldc_,
                        acc_type_, gemm_algo_);
            }
        }
        if (with_bias_) {
            // When bias is specified call cudnnAddTensor()
            float bias_beta = 1;
            scale = (with_eltwise_ ? 1 : 1.0f / host_dst_scale);
            CUDNN_EXECUTE_FUNC(cudnnAddTensor, cudnn_handle, &scale,
                    tensor_descs_[io::bias], bias, &bias_beta, temp_mem_desc_,
                    scratch);
        }
        if (with_eltwise_) {
            // Perform elementwise operation if specified
            float alpha = 1.0f / host_dst_scale;
            float beta = 0;
            CUDNN_EXECUTE_FUNC(cudnnActivationForward, cudnn_handle, act_desc_,
                    &alpha, temp_mem_desc_, scratch, &beta, temp_mem_desc_,
                    scratch);
        }
        if (reorder_required_) {
            // Reorder from scratchpad to destination if required
            float reorder_alpha = 1;
            CUDNN_EXECUTE_FUNC(cudnnTransformTensor, cudnn_handle,
                    &reorder_alpha, temp_mem_desc_, scratch, &post_op_sum_,
                    tensor_descs_[io::dst], c);
        }
    }

    ~cudnn_matmul_impl_t() { cleanup(); }

    void cleanup() {
        if (act_desc_) {
            CUDNN_EXECUTE_FUNC_V(cudnnDestroyActivationDescriptor, act_desc_);
            act_desc_ = nullptr;
        }
        if ((reorder_required_ && !bias_dt_mismatch_)
                || ((with_bias_ && bias_dt_mismatch_) && temp_mem_desc_)) {
            CUDNN_EXECUTE_FUNC_V(cudnnDestroyTensorDescriptor, temp_mem_desc_);
            temp_mem_desc_ = nullptr;
        }
        for (size_t i = 0; i < NUM_IO; i++) {
            if (tensor_descs_[i]) {
                CUDNN_EXECUTE_FUNC_V(
                        cudnnDestroyTensorDescriptor, tensor_descs_[i]);
                tensor_descs_[i] = nullptr;
            }
        }
    }

private:
    status_t get_cublas_data_type(
            dnnl_data_type_t data_type, cudaDataType_t &blas_dt) {
        switch (data_type) {
            case dnnl_data_type_t::dnnl_f32:
                blas_dt = CUDA_R_32F;
                return status::success;
            case dnnl_data_type_t::dnnl_f16:
                blas_dt = CUDA_R_16F;
                return status::success;
            case dnnl_data_type_t::dnnl_bf16:
                blas_dt = CUDA_R_16BF;
                return status::success;
            case dnnl_data_type_t::dnnl_s8:
                blas_dt = CUDA_R_8I;
                return status::success;
            default: return status::unimplemented;
        }
        return status::unimplemented;
    }
    cublasOperation_t transA_;
    cublasOperation_t transB_;
    cublasOperation_t transC_;
    int M_, N_, K_;
    int lda_, ldb_, ldc_;
    long long int stride_a_, stride_b_, stride_c_;
    bool isbatched_ = false, with_bias_ = false, bias_dt_mismatch_ = false,
         with_dst_scale_ = false;
    bool reorder_required_ = false, with_eltwise_ = false;
    bool has_runtime_params_ = false;
    cudaDataType_t src_type_, weights_type_, dst_type_;
    cudaDataType_t acc_type_ = cudaDataType_t::CUDA_R_32F;
    cublasGemmAlgo_t gemm_algo_
            = cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    int batch_count_;
    enum io { bias = 0, dst, NUM_IO };
    cudnnTensorDescriptor_t tensor_descs_[NUM_IO] = {},
                            temp_mem_desc_ = nullptr;
    cudnnActivationDescriptor_t act_desc_ = nullptr;
    float post_op_sum_;
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
