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

#ifndef GPU_NVIDIA_CUDNN_MATMUL_LT_IMPL_HPP
#define GPU_NVIDIA_CUDNN_MATMUL_LT_IMPL_HPP

#include <cublasLt.h>
#include "cudnn.h"
#include <cublas_v2.h>

#include "common/float16.hpp"
#include "gpu/nvidia/cudnn_matmul_base_impl.hpp"
#include "gpu/nvidia/cudnn_matmul_executor.hpp"
#include "gpu/nvidia/engine.hpp"
#include "gpu/nvidia/stream.hpp"
#include "gpu/nvidia/sycl_cuda_scoped_context.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

template <typename T,
        typename = typename std::enable_if<std::is_integral_v<T>>::type>
T ceildiv(T n, T d) {
    return (n + d - 1) / d;
}

struct cudnn_matmul_lt_impl_t : cudnn_matmul_base_impl_t {

    status_t init(matmul_pd_t *pd, impl::engine_t *engine) {

        CHECK(get_cublas_data_type(pd->src_md()->data_type, src_type_));
        CHECK(get_cublas_data_type(pd->weights_md()->data_type, weights_type_));
        CHECK(get_cublas_data_type(pd->dst_md()->data_type, dst_type_));

        auto src_d = memory_desc_wrapper(pd->src_md());
        auto weights_d = memory_desc_wrapper(pd->weights_md());
        auto dst_d = memory_desc_wrapper(pd->dst_md());
        auto bias_d = memory_desc_wrapper(pd->weights_md(1));

        isbatched_ = pd->batched() && dst_d.dims()[0];

        has_runtime_params_ = src_d.has_runtime_dims_or_strides()
                || dst_d.has_runtime_dims_or_strides()
                || weights_d.has_runtime_dims_or_strides();

        if (!pd->attr()->scales_.get(DNNL_ARG_SRC).has_default_values()) {
            auto src_scale = pd->attr()->scales_.get(DNNL_ARG_SRC);
            if (src_scale.mask_ != 0) { multi_src_scale_ = true; }
        }

        if (!pd->attr()->scales_.get(DNNL_ARG_WEIGHTS).has_default_values()) {
            auto wei_scale = pd->attr()->scales_.get(DNNL_ARG_WEIGHTS);
            if (wei_scale.mask_ != 0) { multi_wei_scale_ = true; }
        }

        with_dst_scale_
                = !pd->attr()->scales_.get(DNNL_ARG_DST).has_default_values();
        if (with_dst_scale_) {
            auto dst_scale = pd->attr()->scales_.get(DNNL_ARG_DST);
            if (dst_scale.mask_ != 0) { multi_dst_scale_ = true; }
        }

        // Initialise flags and variables for the imma case (E.g. imma_case_ flag).
        check_imma_case(src_d, weights_d, dst_d);

        with_bias_ = pd->with_bias();

        bool dst_row_major = !is_md_col_major(dst_d);

        // Check if bias can be used in the epilogue
        if (with_bias_) {
            if (has_runtime_params_) {
                return status::unimplemented;
            } else {
                bias_dt_mismatch_ = (pd->weights_md(1)->data_type
                        != pd->dst_md()->data_type);
                if (imma_case_) {
                    with_separate_bias_ = true;
                    if (dst_d.data_type() == dnnl_s8) {
                        reorder_required_ = true;
                    } else {
                        reorder_required_ = false;
                    }
                } else {
                    if (bias_dt_mismatch_ || dst_row_major) {
                        with_separate_bias_ = true;
                        reorder_required_ = false;
                    }
                    if (!with_separate_bias_ && !dst_row_major) {
                        // bias epilogue not supported for dst dim = 1
                        if ((bias_d.dims()[1 + isbatched_] != M_
                                    || bias_d.dims()[0 + isbatched_] != 1)
                                || M_ == 1 || N_ == 1 || has_runtime_params_) {
                            with_separate_bias_ = true;
                            reorder_required_ = false;
                        }
                    }
                }
            }
        }
        with_bias_epilogue_ = with_bias_ && !with_separate_bias_;

        // Check if activation can be used in epilogue
        if (with_eltwise(0, pd) || with_eltwise(1, pd)) {
            if (dst_d.has_runtime_dims_or_strides()) {
                return status::unimplemented;
            } else {
                with_relu_ = eltwise_algo(pd) == alg_kind::eltwise_relu;
                if (!with_relu_ || dst_row_major || with_separate_bias_) {
                    with_separate_eltwise_ = true;
                }
            }
        }
        with_relu_epilogue_ = with_relu_ && !with_separate_eltwise_;

        // Separate activation is not supported and separate bias in non-imma case not supported.
        if ((with_separate_bias_ && !imma_case_) || with_separate_eltwise_) {
            return status::unimplemented;
        }

        // CublasLt is only used for the IMMA case and when the bias and relu are used in the epilogue
        if (!imma_case_ && !with_relu_epilogue_ && !with_bias_epilogue_) {
            return status::unimplemented;
        }

        // Imma case only supports default epilogue
        if (imma_case_ && with_relu_epilogue_) { return status::unimplemented; }

        // we use separate bias to support imma case
        if (imma_case_ && with_bias_epilogue_) {
            with_separate_bias_ = true;
            with_bias_epilogue_ = false;
        }

        // if dst case is single value but we have post ops
        if (with_dst_scale_
                && (with_bias_epilogue_ || with_separate_bias_
                        || with_relu_epilogue_)) {
            multi_dst_scale_ = true;
        }

        const bool supports_ampere_layout = has_imma_ampere_layout_support(
                utils::downcast<nvidia::engine_t *>(engine)->device());

        imma_ampere_case_ = imma_case_ && supports_ampere_layout;
        imma_plain_case_ = imma_case_ && !imma_ampere_case_;

        // Set parameter when post-op sum is specified
        if (with_sum(pd)) { post_op_sum_ = sum_scale(pd); }

        // Initialise scaling parameters
        alpha_beta_size_bytes_ = dst_d.data_type_size();
        if (dst_d.data_type() == dnnl_s8) {
            alpha_beta_size_bytes_ = sizeof(float);
        }
        alpha_ = std::malloc(alpha_beta_size_bytes_);
        beta_ = std::malloc(alpha_beta_size_bytes_);

        // Initialise all gemm parameters
        if (!has_runtime_params_) {
            CHECK(init_parameters(src_d, weights_d, dst_d, bias_d, engine));
            init_scratchpad(pd);
        }

        return status::success;
    }

    void check_imma_case(const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d) {
        if (src_d.data_type() == dnnl_s8 && weights_d.data_type() == dnnl_s8
                && (dst_d.data_type() == dnnl_s32
                        || dst_d.data_type() == dnnl_s8)) {
            // weights blocked in Ab32a
            w_blocked_ = is_md_col32(weights_d);
            bool weights_supported = weights_d.has_runtime_dims_or_strides()
                    || w_blocked_ || weights_d.is_plain();

            // src not blocked
            src_blocked_ = src_d.is_cublaslt_blocked_desc();
            bool src_supported = src_d.has_runtime_dims_or_strides()
                    || src_blocked_ || src_d.is_plain();

            // dst blocked in Ab32a
            dst_blocked_ = is_md_col32(dst_d);
            bool dst_supported = dst_d.has_runtime_dims_or_strides()
                    || dst_blocked_ || dst_d.is_plain();

            imma_case_ = weights_supported && src_supported && dst_supported;
        }
    }

    status_t init_parameters(const memory_desc_wrapper src_d,
            const memory_desc_wrapper weights_d,
            const memory_desc_wrapper dst_d, const memory_desc_wrapper bias_d,
            impl::engine_t *engine) {
        batch_count_ = isbatched_ ? dst_d.dims()[0] : 1;
        M_ = static_cast<uint64_t>(dst_d.dims()[isbatched_ + 1]);
        N_ = static_cast<uint64_t>(dst_d.dims()[isbatched_ + 0]);
        K_ = static_cast<uint64_t>(src_d.dims()[isbatched_ + 1]);

        if (is_imma_case()) {
            w_blocked_ = is_md_col32(weights_d);
            dst_blocked_ = is_md_col32(dst_d);
            src_blocked_ = src_d.is_cublaslt_blocked_desc();

            CUBLAS_EXECUTE_FUNC(cublasLtMatrixTransformDescCreate, &trans_desc_,
                    CUDA_R_32I);

            if (!imma_ampere_case_) {
                const bool all_plain_layout = src_d.is_plain()
                        && weights_d.is_plain() && dst_d.is_plain();
                // The plain imma configuration requires TN->N
                const bool is_tnn = is_md_col_major(src_d)
                        && !is_md_col_major(weights_d)
                        && is_md_col_major(dst_d);
                // The plain imma configuration is only supported if dimensions are multiples of 4
                const bool are_dims_ok
                        = M_ % 4 == 0 && K_ % 4 == 0 && N_ % 4 == 0;
                if (!(is_tnn && all_plain_layout && are_dims_ok)) {
                    return status::unimplemented;
                }
            } else {
                CHECK(init_imma_ampere_sizes(src_d, weights_d, dst_d));
            }
        }

        // Matmul supports runtime paramters for dimensions and scales.
        // We need to initialize them in the execute function.
        CHECK(init_gemm_parameters(src_d, weights_d, dst_d));

        auto &sycl_engine = *utils::downcast<nvidia::engine_t *>(engine);

        impl::stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto cuda_stream = utils::downcast<nvidia::stream_t *>(service_stream);
        auto cublas_handle = cuda_stream->get_cublas_handle();
        auto lt_handle = (cublasLtHandle_t)cublas_handle;
        CHECK(init_scratchpad_size(lt_handle, src_d, weights_d, dst_d));

        if (with_separate_bias_) {
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

            // Create bias and destination tensor descriptors
            convert_dims_matmul(bias_d.dims(), dims[bias], bias_d.ndims());
            convert_dims_matmul(bias_d.blocking_desc().strides, strides[bias],
                    bias_d.ndims());
            CHECK(convert_data_type(bias_d.md_, &data_types[bias], false));
            CHECK(create_and_set_tensor_descriptor(&tensor_descs_[bias],
                    data_types[bias], ndims, dims[bias], strides[bias]));

            CHECK(create_and_set_tensor_descriptor(&tensor_descs_[dst],
                    data_types[dst], ndims, dims[dst], strides[dst]));
        }
        return status::success;
    }

    status_t init_gemm_parameters(const memory_desc_wrapper src_d,
            const memory_desc_wrapper weights_d,
            const memory_desc_wrapper dst_d) override {
        // C matrix is the dst
        trans_c_ = !is_md_col_major(dst_d);
        // A matrix is the weights
        trans_a_ = !is_md_col_major(weights_d);
        // B matrix is the src
        trans_b_ = !is_md_col_major(src_d);

        if (imma_ampere_case_) {
            // IMMA kernels support only NT->N config
            //trans_b_ = false;
            if (w_blocked_) { trans_a_ = false; }
            if (dst_blocked_) { trans_c_ = false; }
        }
        auto dst_dt = dst_d.data_type();
        if (imma_case_ && reorder_required_) { dst_dt = dnnl_s32; }

        if (dst_dt == dnnl_s8 || dst_dt == dnnl_bf16) {
            CHECK(get_cublas_data_type(dnnl_f32, acc_type_));
        } else {
            CHECK(get_cublas_data_type(dst_dt, acc_type_));
        }
        CHECK(get_cublas_data_type(src_d.data_type(), src_type_));
        CHECK(get_cublas_data_type(weights_d.data_type(), weights_type_));
        CHECK(get_cublas_data_type(dst_dt, dst_type_));

        if (dst_dt == dnnl_f16 && src_d.data_type() == dnnl_f16
                && weights_d.data_type() == dnnl_f16) {
            compute_type_ = CUBLAS_COMPUTE_16F;
        } else if (src_d.data_type() == dnnl_s8
                && weights_d.data_type() == dnnl_s8
                && (dst_dt == dnnl_s32 || dst_dt == dnnl_s8)) {
            compute_type_ = CUBLAS_COMPUTE_32I;
        }

        CUBLAS_EXECUTE_FUNC(cublasLtMatmulDescCreate, &operation_desc_,
                compute_type_, acc_type_);

        if (batch_count_ != 1) {
            stride_a_ = get_batch_stride(weights_d);
            stride_b_ = src_d.is_cublaslt_blocked_desc()
                    ? (K_ * N_)
                    : get_batch_stride(src_d);
            stride_c_ = get_batch_stride(dst_d);

            // Enable broadcast semantics.
            if (src_d.dims()[0] > weights_d.dims()[0]) {
                stride_a_ = 0;
                stride_a_blocked_ = 0;
            } else if (src_d.dims()[0] < weights_d.dims()[0]) {
                stride_b_ = 0;
                stride_b_blocked_ = 0;
            }
        }

        if (!imma_ampere_case_) {
            create_non_blocked_layouts();
        } else {
            create_blocked_layouts();
        }

        return status::success;
    }

    status_t init_imma_ampere_sizes(const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d) {
        a_blocked_ld_ = c_blocked_ld_ = M_ * 32;
        b_blocked_ld_ = ceildiv(N_, static_cast<uint64_t>(32)) * 32 * 32;
        stride_b_blocked_
                = ceildiv(K_, static_cast<uint64_t>(32)) * b_blocked_ld_;
        source_size_ = batch_count_ * stride_b_blocked_ * src_d.data_type_size()
                * 32;
        stride_a_blocked_
                = ceildiv(K_, static_cast<uint64_t>(32)) * a_blocked_ld_;
        if (!w_blocked_) {
            weight_size_ = batch_count_ * stride_a_blocked_
                    * weights_d.data_type_size() * 32;
        }
        stride_c_blocked_
                = ceildiv(N_, static_cast<uint64_t>(32)) * c_blocked_ld_;
        if (!dst_blocked_) {
            dest_size_ = batch_count_ * stride_c_blocked_
                    * dst_d.data_type_size() * 32;
        }
        return status::success;
    }

    // Initialization for scratchpad memory
    status_t init_scratchpad_size(cublasLtHandle_t lt_handle,
            const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d) {

        reorder_scratch_size_ = 0;

        CUBLAS_EXECUTE_FUNC(cublasLtMatmulPreferenceCreate, &preference_);

        // 1GB limit
        uint64_t workspace_size = getenv_int_user(
                "CUBLASLT_MAX_MATMUL_WORKSPACE_SIZE", 1073741824);

        CUBLAS_EXECUTE_FUNC(cublasLtMatmulPreferenceSetAttribute, preference_,
                CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size,
                sizeof(workspace_size));
        cublasLtReductionScheme_t reduction_scheme
                = CUBLASLT_REDUCTION_SCHEME_MASK;
        CUBLAS_EXECUTE_FUNC(cublasLtMatmulPreferenceSetAttribute, preference_,
                CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reduction_scheme,
                sizeof(reduction_scheme));

        int num_results = 0;
        if (imma_ampere_case_) {
            CUBLAS_EXECUTE_FUNC(cublasLtMatmulAlgoGetHeuristic, lt_handle,
                    operation_desc_, blocked_a_layout_, blocked_b_layout_,
                    blocked_c_layout_, blocked_c_layout_, preference_,
                    1 /* Num requested algos*/, &heuristic_results_,
                    &num_results);
        } else {
            CUBLAS_EXECUTE_FUNC(cublasLtMatmulAlgoGetHeuristic, lt_handle,
                    operation_desc_, a_layout_, b_layout_, c_layout_, c_layout_,
                    preference_, 1 /* Num requested algos*/,
                    &heuristic_results_, &num_results);
        }

        if (num_results == 0) { return status_t::dnnl_runtime_error; }
        gemm_algo_ = heuristic_results_.algo;
        algo_scratch_size_ = heuristic_results_.workspaceSize;

        const auto dst_nelems = dst_d.nelems(true);
        reorder_scratch_size_ = dst_nelems * sizeof(float);

        if (multi_src_scale_ || acc_type_ == CUDA_R_32I) {
            src_scale_size_ = src_d.size();
        }
        if (multi_wei_scale_ || acc_type_ == CUDA_R_32I) {
            wei_scale_size_ = weights_d.size();
        }

        return status_t::dnnl_success;
    }

    void init_scratchpad(matmul_pd_t *pd) override {
        auto scratchpad = pd->scratchpad_registry().registrar();
        if (reorder_scratch_size_ > 0) {
            scratchpad.book(memory_tracking::names::key_matmul_dst_in_acc_dt,
                    reorder_scratch_size_, 1, 256);
        }
        if (algo_scratch_size_ > 0) {
            scratchpad.book(memory_tracking::names::key_matmul_lt_algo_scratch,
                    algo_scratch_size_, 1, 256);
        }
        if (weight_size_ > 0) {
            scratchpad.book(memory_tracking::names::key_gemm_blocked_a,
                    weight_size_, 1, 256);
        }
        if (source_size_ > 0) {
            scratchpad.book(memory_tracking::names::key_gemm_blocked_b,
                    source_size_, 1, 256);
        }
        if (dest_size_ > 0) {
            scratchpad.book(memory_tracking::names::key_matmul_lt_block_c,
                    dest_size_, 1, 256);
        }
        if (src_scale_size_ > 0) {
            scratchpad.book(memory_tracking::names::key_matmul_lt_src_scale,
                    src_scale_size_, 1, 256);
        }
        if (wei_scale_size_ > 0) {
            scratchpad.book(memory_tracking::names::key_matmul_lt_wei_scale,
                    wei_scale_size_, 1, 256);
        }
    }

    void execute(cublasHandle_t cublas_handle, cudnnHandle_t cudnn_handle,
            void *a, void *b, void *c, void *bias, void *algo_scratch,
            void *reorder_scratch, void *block_a_scratch, void *block_b_scratch,
            void *block_c_scratch, void *scaled_src, void *scaled_wt,
            void *src_scale, void *wei_scale, void *dst_scale) {
        // read from binary output instead of input if multi scale or s32 scaling
        if ((src_scale && acc_type_ == CUDA_R_32I) || multi_wei_scale_) {
            b = scaled_src;
        }
        if ((wei_scale && acc_type_ == CUDA_R_32I) || multi_wei_scale_) {
            a = scaled_wt;
        }

        cudaStream_t streamId;
        auto lt_handle = (cublasLtHandle_t)(cublas_handle);
        CUBLAS_EXECUTE_FUNC(cublasGetStream, cublas_handle, &streamId);

        if (imma_ampere_case_) {
            if (!src_blocked_) {
                transform_matrix(lt_handle, b_layout_, b, blocked_b_layout_,
                        block_b_scratch, !trans_b_, streamId);
                b = block_b_scratch;
            }
            if (!w_blocked_) {
                transform_matrix(lt_handle, a_layout_, a, blocked_a_layout_,
                        block_a_scratch, trans_a_, streamId);
                a = block_a_scratch;
            }
        }

        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
        if (with_bias_epilogue_) {
            if (with_relu_epilogue_) {
                epilogue = CUBLASLT_EPILOGUE_RELU_BIAS;
            } else {
                epilogue = CUBLASLT_EPILOGUE_BIAS;
            }
            CUBLAS_EXECUTE_FUNC(cublasLtMatmulDescSetAttribute, operation_desc_,
                    CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
        } else if (with_relu_epilogue_ && !with_bias_epilogue_) {
            epilogue = CUBLASLT_EPILOGUE_RELU;
        }
        CUBLAS_EXECUTE_FUNC(cublasLtMatmulDescSetAttribute, operation_desc_,
                CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));

        float scale = 1.0f;
        float host_dst_scale = 1.0f;
        if (src_scale && !multi_src_scale_ && acc_type_ != CUDA_R_32I) {
            float host_src_scale = 1.0f;
            CUDA_EXECUTE_FUNC(cuMemcpy, (CUdeviceptr)&host_src_scale,
                    (CUdeviceptr)src_scale, sizeof(float));
            scale *= host_src_scale;
        }
        if (wei_scale && !multi_wei_scale_ && acc_type_ != CUDA_R_32I) {
            float host_wei_scale = 1.0f;
            CUDA_EXECUTE_FUNC(cuMemcpy, (CUdeviceptr)&host_wei_scale,
                    (CUdeviceptr)wei_scale, sizeof(float));
            scale *= host_wei_scale;
        }
        if (dst_scale && !multi_dst_scale_ && acc_type_ != CUDA_R_32I) {
            CUDA_EXECUTE_FUNC(cuMemcpy, (CUdeviceptr)&host_dst_scale,
                    (CUdeviceptr)dst_scale, sizeof(float));
            // only applied here if no post ops used
            scale /= host_dst_scale;
        }

        if (acc_type_ == CUDA_R_16F) {
            dnnl::impl::float16_t half_scale = scale;
            dnnl::impl::float16_t half_gemm_beta = post_op_sum_;
            *static_cast<float16_t *>(alpha_) = half_scale;
            *static_cast<float16_t *>(beta_) = half_gemm_beta;
        } else {
            *static_cast<float *>(alpha_) = scale;
            *static_cast<float *>(beta_) = post_op_sum_;
        }

        if (imma_ampere_case_) {
            if (!dst_blocked_) {
                std::memset(beta_, 0, alpha_beta_size_bytes_);
            }
            c = reorder_required_ ? reorder_scratch : c;
            void *tmp_c = dst_blocked_ ? c : block_c_scratch;
            CUBLAS_EXECUTE_FUNC(cublasLtMatmul, lt_handle, operation_desc_,
                    alpha_, a, blocked_a_layout_, b, blocked_b_layout_, beta_,
                    tmp_c, blocked_c_layout_, tmp_c, blocked_c_layout_,
                    &gemm_algo_, algo_scratch, heuristic_results_.workspaceSize,
                    streamId);

            if (!dst_blocked_) {
                transform_matrix(lt_handle, blocked_c_layout_, block_c_scratch,
                        c_layout_, c, trans_c_, streamId, post_op_sum_);
            }
        } else {
            CUBLAS_EXECUTE_FUNC(cublasLtMatmul, lt_handle, operation_desc_,
                    alpha_, a, a_layout_, b, b_layout_, beta_, c, c_layout_, c,
                    c_layout_, &gemm_algo_, algo_scratch,
                    heuristic_results_.workspaceSize, streamId);
        }
    }

    ~cudnn_matmul_lt_impl_t() { cleanup(); }

    void rt_cleanup() {
        if (a_layout_) {
            CUBLAS_EXECUTE_FUNC(cublasLtMatrixLayoutDestroy, a_layout_);
            a_layout_ = nullptr;
        }
        if (b_layout_) {
            CUBLAS_EXECUTE_FUNC(cublasLtMatrixLayoutDestroy, b_layout_);
            b_layout_ = nullptr;
        }
        if (c_layout_) {
            CUBLAS_EXECUTE_FUNC(cublasLtMatrixLayoutDestroy, c_layout_);
            c_layout_ = nullptr;
        }

        if (operation_desc_) {
            CUBLAS_EXECUTE_FUNC(cublasLtMatmulDescDestroy, operation_desc_);
            operation_desc_ = nullptr;
        }

        if (imma_ampere_case_) {

            if (trans_desc_) {
                CUBLAS_EXECUTE_FUNC(
                        cublasLtMatrixTransformDescDestroy, trans_desc_);
                trans_desc_ = nullptr;
            }
            if (blocked_a_layout_) {
                CUBLAS_EXECUTE_FUNC(
                        cublasLtMatrixLayoutDestroy, blocked_a_layout_);
                blocked_a_layout_ = nullptr;
            }
            if (blocked_b_layout_) {
                CUBLAS_EXECUTE_FUNC(
                        cublasLtMatrixLayoutDestroy, blocked_b_layout_);
                blocked_b_layout_ = nullptr;
            }
            if (blocked_c_layout_) {
                CUBLAS_EXECUTE_FUNC(
                        cublasLtMatrixLayoutDestroy, blocked_c_layout_);
                blocked_c_layout_ = nullptr;
            }
        }

        for (size_t i = 0; i < NUM_IO; i++) {
            if (tensor_descs_[i]) {
                CUDNN_EXECUTE_FUNC_V(
                        cudnnDestroyTensorDescriptor, tensor_descs_[i]);
                tensor_descs_[i] = nullptr;
            }
        }
    }

    void cleanup() override {
        std::free(alpha_);
        alpha_ = nullptr;
        std::free(beta_);
        beta_ = nullptr;

        if (preference_) {
            CUBLAS_EXECUTE_FUNC(cublasLtMatmulPreferenceDestroy, preference_);
            preference_ = nullptr;
        }

        rt_cleanup();

        for (size_t i = 0; i < NUM_IO; i++) {
            if (tensor_descs_[i]) {
                CUDNN_EXECUTE_FUNC_V(
                        cudnnDestroyTensorDescriptor, tensor_descs_[i]);
                tensor_descs_[i] = nullptr;
            }
        }
    }

    bool is_imma_case() { return imma_case_; }

    size_t algo_scratch_size() const { return algo_scratch_size_; }
    size_t block_a_scratch_size() const { return source_size_; }
    size_t block_b_scratch_size() const { return weight_size_; }
    size_t block_c_scratch_size() const { return dest_size_; }
    size_t src_scale_size() const { return src_scale_size_; }
    size_t wei_scale_size() const { return wei_scale_size_; }

    bool multi_src_scale() const { return multi_src_scale_; }
    bool multi_wei_scale() const { return multi_wei_scale_; }
    bool multi_dst_scale() const { return multi_dst_scale_; }
    cudaDataType_t scale_type() const { return acc_type_; }

private:
    cublasLtMatmulDesc_t operation_desc_;

    cublasLtMatrixLayout_t a_layout_;
    cublasLtMatrixLayout_t b_layout_;
    cublasLtMatrixLayout_t c_layout_;
    cublasLtMatrixLayout_t blocked_a_layout_;
    cublasLtMatrixLayout_t blocked_b_layout_;
    cublasLtMatrixLayout_t blocked_c_layout_;

    bool multi_src_scale_ = false;
    bool multi_wei_scale_ = false;
    bool multi_dst_scale_ = false;

    size_t src_scale_size_ = 0;
    size_t wei_scale_size_ = 0;

    bool with_bias_ = false;
    bool with_bias_epilogue_ = false;
    bool with_relu_;
    bool with_relu_epilogue_ = false;
    bool imma_case_ = false;
    bool imma_ampere_case_ = false;
    bool imma_plain_case_ = false;
    bool w_blocked_ = false;
    bool src_blocked_ = false;
    bool dst_blocked_ = false;
    cublasLtMatrixTransformDesc_t trans_desc_;
    uint64_t source_size_ = 0;
    uint64_t weight_size_ = 0;
    uint64_t dest_size_ = 0;

    uint64_t M_, N_, K_;

    int64_t stride_a_, stride_b_, stride_c_, stride_a_blocked_,
            stride_b_blocked_, stride_c_blocked_, a_blocked_ld_, b_blocked_ld_,
            c_blocked_ld_;

    bool trans_a_ = false, trans_b_ = false, trans_c_ = false;

    cublasComputeType_t compute_type_ = CUBLAS_COMPUTE_32F;

    cudaDataType_t src_type_, weights_type_, dst_type_;
    size_t alpha_beta_size_bytes_ = 0;
    void *alpha_ = nullptr;
    void *beta_ = nullptr;

    cublasLtMatmulAlgo_t gemm_algo_;

    cublasLtMatmulPreference_t preference_;

    cublasLtMatmulHeuristicResult_t heuristic_results_;

    status_t create_matrix_layout(cublasLtMatrixLayout_t &layout,
            cublasLtOrder_t order, cublasOperation_t trans, uint64_t row,
            uint64_t col, uint64_t ld, const cudaDataType_t data_type,
            cublasLtMatmulDescAttributes_t trans_attr, uint64_t stride) {
        CUBLAS_EXECUTE_FUNC(cublasLtMatmulDescSetAttribute, operation_desc_,
                trans_attr, &trans, sizeof(trans));

        CUBLAS_EXECUTE_FUNC(
                cublasLtMatrixLayoutCreate, &layout, data_type, row, col, ld);

        CUBLAS_EXECUTE_FUNC(cublasLtMatrixLayoutSetAttribute, layout,
                CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));

        if (batch_count_ != 1) {
            CUBLAS_EXECUTE_FUNC(cublasLtMatrixLayoutSetAttribute, layout,
                    CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count_,
                    sizeof(batch_count_));
            CUBLAS_EXECUTE_FUNC(cublasLtMatrixLayoutSetAttribute, layout,
                    CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride,
                    sizeof(stride));
        }
        return status_t::dnnl_success;
    }

    void transform_matrix(cublasLtHandle_t handle,
            cublasLtMatrixLayout_t in_layout, void *in,
            cublasLtMatrixLayout_t out_layout, void *out, bool transpose,
            cudaStream_t stream, int beta = 0) {
        int alpha = 1;

        cublasOperation_t transform_trans
                = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
        CUBLAS_EXECUTE_FUNC(cublasLtMatrixTransformDescSetAttribute,
                trans_desc_, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA,
                &transform_trans, sizeof(transform_trans));
        CUBLAS_EXECUTE_FUNC(cublasLtMatrixTransform, handle, trans_desc_,
                &alpha, in, in_layout, &beta, out, out_layout, out, out_layout,
                stream);
    }

    void create_non_blocked_layouts() {
        auto trans_op = cublasOperation_t::CUBLAS_OP_N;
        auto order = CUBLASLT_ORDER_COL;

        auto row = M_;
        auto col = K_;
        maybe_swap(row, col, trans_op, order, trans_a_);
        create_matrix_layout(a_layout_, CUBLASLT_ORDER_COL, trans_op, row, col,
                row, weights_type_, CUBLASLT_MATMUL_DESC_TRANSA, stride_a_);

        row = K_;
        col = N_;
        trans_op = cublasOperation_t::CUBLAS_OP_N;
        maybe_swap(row, col, trans_op, order, trans_b_);
        create_matrix_layout(b_layout_, CUBLASLT_ORDER_COL, trans_op, row, col,
                row, src_type_, CUBLASLT_MATMUL_DESC_TRANSB, stride_b_);

        row = M_;
        col = N_;
        order = CUBLASLT_ORDER_COL;
        maybe_swap(row, col, trans_op, order, trans_c_);
        create_matrix_layout(c_layout_, order, cublasOperation_t::CUBLAS_OP_N,
                row, col, row, dst_type_, CUBLASLT_MATMUL_DESC_TRANSC,
                stride_c_);
    }

    void create_blocked_layouts() {
        create_matrix_layout(blocked_a_layout_, CUBLASLT_ORDER_COL32,
                cublasOperation_t::CUBLAS_OP_N, M_, K_, a_blocked_ld_,
                weights_type_, CUBLASLT_MATMUL_DESC_TRANSA, stride_a_blocked_);

        create_matrix_layout(blocked_b_layout_, CUBLASLT_ORDER_COL32_2R_4R4,
                cublasOperation_t::CUBLAS_OP_N, N_, K_, b_blocked_ld_,
                src_type_, CUBLASLT_MATMUL_DESC_TRANSB, stride_b_blocked_);

        create_matrix_layout(blocked_c_layout_, CUBLASLT_ORDER_COL32,
                cublasOperation_t::CUBLAS_OP_N, M_, N_, c_blocked_ld_,
                dst_type_, CUBLASLT_MATMUL_DESC_TRANSC, stride_c_blocked_);

        uint64_t row, col;
        if (!w_blocked_) {
            row = M_;
            col = K_;
            if (trans_a_) { std::swap(row, col); }
            create_matrix_layout(a_layout_, CUBLASLT_ORDER_COL,
                    cublasOperation_t::CUBLAS_OP_N, row, col, row,
                    weights_type_, CUBLASLT_MATMUL_DESC_TRANSA, stride_a_);
        }

        if (!src_blocked_) {
            row = K_;
            col = N_;
            if (trans_b_) { std::swap(row, col); }
            create_matrix_layout(b_layout_, CUBLASLT_ORDER_COL,
                    cublasOperation_t::CUBLAS_OP_N, row, col, row, src_type_,
                    CUBLASLT_MATMUL_DESC_TRANSB, stride_b_);
        }

        if (!dst_blocked_) {
            row = M_;
            col = N_;
            if (trans_c_) { std::swap(row, col); }
            create_matrix_layout(c_layout_, CUBLASLT_ORDER_COL,
                    cublasOperation_t::CUBLAS_OP_N, row, col, row, dst_type_,
                    CUBLASLT_MATMUL_DESC_TRANSC, stride_c_);
        }

        // Constraint for Turing/Ampere kernels matmul config needs to be
        // A^N B^T
        cublasOperation_t b_trans_t = cublasOperation_t::CUBLAS_OP_T;
        CUBLAS_EXECUTE_FUNC(cublasLtMatmulDescSetAttribute, operation_desc_,
                CUBLASLT_MATMUL_DESC_TRANSB, &b_trans_t, sizeof(b_trans_t));
    }

    bool is_md_col_major(const memory_desc_wrapper &md) {
        if (md.is_blocking_desc()) {
            const auto &md_strides = &md.blocking_desc().strides[isbatched_];
            return (md_strides[1] == 1 && md.dims()[isbatched_ + 0] > 1);
        }
        return false;
    }

    void maybe_swap(uint64_t &row, uint64_t &col, cublasOperation_t &op,
            cublasLtOrder_t order, bool transpose) {
        if (transpose) {
            std::swap(row, col);
            op = cublasOperation_t::CUBLAS_OP_T;
            order = CUBLASLT_ORDER_ROW;
        }
    };
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
