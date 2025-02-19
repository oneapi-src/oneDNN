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

struct cublas_lt_params : cublas_base_params {

    status_t init(impl::engine_t *engine, const memory_desc_t *src_md,
            const memory_desc_t *weights_md, const memory_desc_t *dst_md,
            const memory_desc_t *bias_md, const primitive_attr_t *attr,
            bool batched, bool with_bias) {
        CHECK(get_cublas_data_type(src_md->data_type, src_type_));
        CHECK(get_cublas_data_type(weights_md->data_type, weights_type_));
        CHECK(get_cublas_data_type(dst_md->data_type, dst_type_));

        auto src_d = memory_desc_wrapper(*src_md);
        auto weights_d = memory_desc_wrapper(*weights_md);
        auto dst_d = memory_desc_wrapper(*dst_md);

        isbatched_ = batched && dst_d.dims()[0];

        has_runtime_params_ = src_d.has_runtime_dims_or_strides()
                || dst_d.has_runtime_dims_or_strides()
                || weights_d.has_runtime_dims_or_strides();

        with_dst_scale_ = !attr->scales_.get(DNNL_ARG_DST).has_default_values();
        if (with_dst_scale_) {
            auto dst_scale = attr->scales_.get(DNNL_ARG_DST);
            if (dst_scale.mask_ != 0) { multi_dst_scale_ = true; }
        }

        // Initialise flags and variables for the imma case (E.g. imma_case_ flag).
        check_imma_case(src_d, weights_d, dst_d);

        with_bias_ = with_bias;

        bool dst_row_major = !is_md_col_major(dst_d);

        // Check if bias can be used in the epilogue
        if (with_bias_) {
            if (has_runtime_params_) {
                return status::unimplemented;
            } else {
                bias_dt_mismatch_ = (bias_md->data_type != dst_md->data_type);
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
                        memory_desc_wrapper bias_d
                                = memory_desc_wrapper(*bias_md);
                        if ((bias_d.dims()[1 + isbatched_]
                                            != static_cast<dim_t>(M_)
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
        if (with_eltwise(0, attr) || with_eltwise(1, attr)) {
            if (dst_d.has_runtime_dims_or_strides()) {
                return status::unimplemented;
            } else {
                with_relu_ = eltwise_algo(attr) == alg_kind::eltwise_relu;
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
        if (with_sum(attr)) { post_op_sum_ = sum_scale(attr); }

        // Initialise scaling parameters
        alpha_beta_size_bytes_ = dst_d.data_type_size();
        if (dst_d.data_type() == dnnl_s8) {
            alpha_beta_size_bytes_ = sizeof(float);
        }
        alpha_ = std::malloc(alpha_beta_size_bytes_);
        beta_ = std::malloc(alpha_beta_size_bytes_);

        // Initialise all gemm parameters
        if (!has_runtime_params_) {
            CHECK(set_params(src_d, weights_d, dst_d, engine));
        }

        return status::success;
    }

    status_t init_from_params(const std::shared_ptr<cublas_lt_params> &other) {
        if (!other) { return status::invalid_arguments; }
        src_type_ = other->src_type_;
        weights_type_ = other->weights_type_;
        dst_type_ = other->dst_type_;
        isbatched_ = other->isbatched_;
        has_runtime_params_ = other->has_runtime_params_;
        with_dst_scale_ = other->with_dst_scale_;
        multi_dst_scale_ = other->multi_dst_scale_;
        with_bias_ = other->with_bias_;
        bias_dt_mismatch_ = other->bias_dt_mismatch_;
        with_separate_bias_ = other->with_separate_bias_;
        reorder_required_ = other->reorder_required_;
        with_bias_epilogue_ = other->with_bias_epilogue_;
        with_relu_epilogue_ = other->with_relu_epilogue_;
        imma_ampere_case_ = other->imma_ampere_case_;
        imma_plain_case_ = other->imma_plain_case_;
        alpha_beta_size_bytes_ = other->alpha_beta_size_bytes_;
        alpha_ = std::malloc(alpha_beta_size_bytes_);
        beta_ = std::malloc(alpha_beta_size_bytes_);
        return status::success;
    }

    status_t set_params(const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d, impl::engine_t *engine) {
        batch_count_ = isbatched_ ? dst_d.dims()[0] : 1;
        M_ = static_cast<uint64_t>(dst_d.dims()[isbatched_ + 1]);
        N_ = static_cast<uint64_t>(dst_d.dims()[isbatched_ + 0]);
        K_ = static_cast<uint64_t>(src_d.dims()[isbatched_ + 1]);

        if (imma_case_) {
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
        CHECK(set_gemm_params(src_d, weights_d, dst_d));

        auto &sycl_engine = *utils::downcast<nvidia::engine_t *>(engine);

        impl::stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto cuda_stream = utils::downcast<nvidia::stream_t *>(service_stream);
        auto cublas_handle = cuda_stream->get_cublas_handle();
        auto lt_handle = (cublasLtHandle_t)cublas_handle;
        CHECK(init_scratchpad_size(lt_handle, src_d, weights_d, dst_d));

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

    status_t set_gemm_params(const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d) {
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

        return status_t::dnnl_success;
    }

    void init_scratchpad(memory_tracking::registrar_t scratchpad) {
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
    }

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

    size_t bias_scratch_size() { return reorder_scratch_size_; }
    bool has_runtime_params() { return has_runtime_params_; }

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

    void rt_cleanup() const {
        if (a_layout_) {
            CUBLAS_EXECUTE_FUNC(cublasLtMatrixLayoutDestroy, a_layout_);
        }
        if (b_layout_) {
            CUBLAS_EXECUTE_FUNC(cublasLtMatrixLayoutDestroy, b_layout_);
        }
        if (c_layout_) {
            CUBLAS_EXECUTE_FUNC(cublasLtMatrixLayoutDestroy, c_layout_);
        }

        if (operation_desc_) {
            CUBLAS_EXECUTE_FUNC(cublasLtMatmulDescDestroy, operation_desc_);
        }

        if (imma_ampere_case_) {
            if (trans_desc_) {
                CUBLAS_EXECUTE_FUNC(
                        cublasLtMatrixTransformDescDestroy, trans_desc_);
            }
            if (blocked_a_layout_) {
                CUBLAS_EXECUTE_FUNC(
                        cublasLtMatrixLayoutDestroy, blocked_a_layout_);
            }
            if (blocked_b_layout_) {
                CUBLAS_EXECUTE_FUNC(
                        cublasLtMatrixLayoutDestroy, blocked_b_layout_);
            }
            if (blocked_c_layout_) {
                CUBLAS_EXECUTE_FUNC(
                        cublasLtMatrixLayoutDestroy, blocked_c_layout_);
            }
        }
    }

    void cleanup() const {
        std::free(alpha_);
        std::free(beta_);

        if (preference_) {
            CUBLAS_EXECUTE_FUNC(cublasLtMatmulPreferenceDestroy, preference_);
        }
        rt_cleanup();
    }

    cublasLtMatmulDesc_t operation_desc_;

    cublasLtMatrixLayout_t a_layout_;
    cublasLtMatrixLayout_t b_layout_;
    cublasLtMatrixLayout_t c_layout_;
    cublasLtMatrixLayout_t blocked_a_layout_;
    cublasLtMatrixLayout_t blocked_b_layout_;
    cublasLtMatrixLayout_t blocked_c_layout_;

    bool multi_dst_scale_ = false;

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

    int64_t stride_a_, stride_b_, stride_c_, stride_a_blocked_,
            stride_b_blocked_, stride_c_blocked_, a_blocked_ld_, b_blocked_ld_,
            c_blocked_ld_;

    bool trans_a_ = false, trans_b_ = false, trans_c_ = false;

    cublasComputeType_t compute_type_ = CUBLAS_COMPUTE_32F;

    size_t alpha_beta_size_bytes_ = 0;
    void *alpha_ = nullptr;
    void *beta_ = nullptr;

    size_t algo_scratch_size_ = 0;

    cublasLtMatmulAlgo_t gemm_algo_;

    cublasLtMatmulPreference_t preference_;

    cublasLtMatmulHeuristicResult_t heuristic_results_;
};

struct cudnn_matmul_lt_impl_t {

    void set_non_runtime_params(
            const std::shared_ptr<cublas_lt_params> &matmul_params) {
        matmul_params_ = matmul_params;
    }

    void execute(cublasHandle_t cublas_handle,
            const std::shared_ptr<cublas_lt_params> matmul_params, void *a,
            void *b, void *c, void *bias, void *algo_scratch,
            void *reorder_scratch, void *block_a_scratch, void *block_b_scratch,
            void *block_c_scratch, void * /* src_scale */,
            void * /* wei_scale */, void *dst_scale) {

        // use cached params unless using runtime dimensions
        std::shared_ptr<cublas_lt_params> params
                = matmul_params->has_runtime_params_ ? matmul_params
                                                     : matmul_params_;

        auto acc_type = params->acc_type_;

        cudaStream_t streamId;
        auto lt_handle = (cublasLtHandle_t)(cublas_handle);
        CUBLAS_EXECUTE_FUNC(cublasGetStream, cublas_handle, &streamId);

        auto b_layout = params->b_layout_;
        auto blocked_b_layout = params->blocked_b_layout_;
        auto a_layout = params->a_layout_;
        auto blocked_a_layout = params->blocked_a_layout_;

        auto imma_ampere_case = params->imma_ampere_case_;

        if (imma_ampere_case) {
            if (!params->src_blocked_) {
                transform_matrix(lt_handle, params, b_layout, b,
                        blocked_b_layout, block_b_scratch, !params->trans_b_,
                        streamId);
                b = block_b_scratch;
            }
            if (!params->w_blocked_) {
                transform_matrix(lt_handle, params, a_layout, a,
                        blocked_a_layout, block_a_scratch, params->trans_a_,
                        streamId);
                a = block_a_scratch;
            }
        }

        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
        auto with_bias_epilogue = params->with_bias_epilogue_;
        auto with_relu_epilogue = params->with_relu_epilogue_;

        auto operation_desc = params->operation_desc_;

        if (with_bias_epilogue) {
            if (with_relu_epilogue) {
                epilogue = CUBLASLT_EPILOGUE_RELU_BIAS;
            } else {
                epilogue = CUBLASLT_EPILOGUE_BIAS;
            }
            CUBLAS_EXECUTE_FUNC(cublasLtMatmulDescSetAttribute, operation_desc,
                    CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
        } else if (with_relu_epilogue && !with_bias_epilogue) {
            epilogue = CUBLASLT_EPILOGUE_RELU;
        }
        CUBLAS_EXECUTE_FUNC(cublasLtMatmulDescSetAttribute, operation_desc,
                CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));

        float scale = 1.0f;
        float host_dst_scale = 1.0f;
        if (dst_scale && !params->multi_dst_scale_ && acc_type != CUDA_R_32I) {
            CUDA_EXECUTE_FUNC(cuMemcpy, (CUdeviceptr)&host_dst_scale,
                    (CUdeviceptr)dst_scale, sizeof(float));
            // only applied here if no post ops used
            scale /= host_dst_scale;
        }

        auto alpha = params->alpha_;
        auto beta = params->beta_;
        auto post_op_sum = params->post_op_sum_;

        if (acc_type == CUDA_R_16F) {
            dnnl::impl::float16_t half_scale = scale;
            dnnl::impl::float16_t half_gemm_beta = post_op_sum;
            *static_cast<float16_t *>(alpha) = half_scale;
            *static_cast<float16_t *>(beta) = half_gemm_beta;
        } else {
            *static_cast<float *>(alpha) = scale;
            *static_cast<float *>(beta) = post_op_sum;
        }

        auto dst_blocked = params->dst_blocked_;
        auto c_layout = params->c_layout_;
        auto gemm_algo = params->gemm_algo_;
        auto heuristic_results = params->heuristic_results_;

        if (imma_ampere_case) {
            if (!dst_blocked) {
                std::memset(beta, 0, params->alpha_beta_size_bytes_);
            }
            auto blocked_c_layout = params->blocked_c_layout_;
            c = params->reorder_required_ ? reorder_scratch : c;
            void *tmp_c = dst_blocked ? c : block_c_scratch;
            CUBLAS_EXECUTE_FUNC(cublasLtMatmul, lt_handle, operation_desc,
                    alpha, a, blocked_a_layout, b, blocked_b_layout, beta,
                    tmp_c, blocked_c_layout, tmp_c, blocked_c_layout,
                    &gemm_algo, algo_scratch, heuristic_results.workspaceSize,
                    streamId);

            if (!dst_blocked) {
                transform_matrix(lt_handle, params, blocked_c_layout,
                        block_c_scratch, c_layout, c, params->trans_c_,
                        streamId, post_op_sum);
            }
        } else {
            CUBLAS_EXECUTE_FUNC(cublasLtMatmul, lt_handle, operation_desc,
                    alpha, a, a_layout, b, b_layout, beta, c, c_layout, c,
                    c_layout, &gemm_algo, algo_scratch,
                    heuristic_results.workspaceSize, streamId);
        }
    }

    ~cudnn_matmul_lt_impl_t() {
        if (matmul_params_) { matmul_params_->cleanup(); }
    }

private:
    void transform_matrix(cublasLtHandle_t handle,
            const std::shared_ptr<cublas_lt_params> &params,
            cublasLtMatrixLayout_t in_layout, void *in,
            cublasLtMatrixLayout_t out_layout, void *out, bool transpose,
            cudaStream_t stream, int beta = 0) {
        int alpha = 1;
        cublasLtMatrixTransformDesc_t trans_desc = params->trans_desc_;
        cublasOperation_t transform_trans
                = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
        CUBLAS_EXECUTE_FUNC(cublasLtMatrixTransformDescSetAttribute, trans_desc,
                CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &transform_trans,
                sizeof(transform_trans));
        CUBLAS_EXECUTE_FUNC(cublasLtMatrixTransform, handle, trans_desc, &alpha,
                in, in_layout, &beta, out, out_layout, out, out_layout, stream);
    }

    std::shared_ptr<cublas_lt_params> matmul_params_;
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
