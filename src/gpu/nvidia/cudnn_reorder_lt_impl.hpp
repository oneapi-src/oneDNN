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

#ifndef GPU_NVIDIA_CUDNN_REORDER_LT_IMPL_HPP
#define GPU_NVIDIA_CUDNN_REORDER_LT_IMPL_HPP

#include <cublasLt.h>
#include "common/type_helpers.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cublaslt_reorder_t {
public:
    bool trans;
    status_t init(reorder_pd_t *pd) {
        // If any of the dimensions are 0 we should not continue with creating
        // cudnn descriptors
        memory_desc_wrapper src_wrap(pd->src_md());
        memory_desc_wrapper dst_wrap(pd->dst_md());

        if (src_wrap.size() == 0) { return status::success; }
        // Validity checks
        if (pd->dst_md()->ndims != pd->src_md()->ndims) {
            return status::unimplemented;
        }

        const bool is_batched = src_wrap.ndims() > 2 && dst_wrap.dims()[0];
        batch_count_ = is_batched ? dst_wrap.dims()[0] : 1;

        CUBLAS_EXECUTE_FUNC(
                cublasLtMatrixTransformDescCreate, &trans_desc_, CUDA_R_32I);

        beta_ = pd->beta();

        CHECK(get_cublas_data_type(pd->src_md()->data_type, src_data_type_));
        CHECK(get_cublas_data_type(pd->dst_md()->data_type, dst_data_type_));
        // take into account conversion from/to float
        if (src_data_type_ == cudaDataType_t::CUDA_R_32F) {
            src_data_type_ = cudaDataType_t::CUDA_R_8I;
        }
        if (dst_data_type_ == cudaDataType_t::CUDA_R_32F) {
            dst_data_type_ = cudaDataType_t::CUDA_R_8I;
        }

        ampere_src_ = src_wrap.is_cublaslt_blocked_desc();

        if (ampere_src_) {
            convert_dims(pd->dst_md()->padded_dims, dims_, pd->dst_md()->ndims);
        } else {
            convert_dims(pd->src_md()->padded_dims, dims_, pd->src_md()->ndims);
        }

        trans = false;
        row_ = dims_[is_batched + 0];
        col_ = dims_[is_batched + 1];
        int plain_ld = 0;
        if (!ampere_src_) {
            if (src_wrap.matches_one_of_tag(format_tag::ab, format_tag::abc)
                    != format_tag::undef) {
                non_blocked_order_ = CUBLASLT_ORDER_COL;
                plain_ld = row_;
                trans = true;
            } else {
                non_blocked_order_ = CUBLASLT_ORDER_ROW;
                plain_ld = col_;
            }
        } else {
            if (dst_wrap.matches_one_of_tag(format_tag::ab, format_tag::acb)
                    != format_tag::undef) {
                non_blocked_order_ = CUBLASLT_ORDER_COL;
                plain_ld = row_;
                trans = true;
            } else {
                non_blocked_order_ = CUBLASLT_ORDER_ROW;
                plain_ld = col_;
            }
        }

        uint64_t blocked_ld
                = ceildiv(row_, static_cast<uint64_t>(32)) * 32 * 32;
        auto stride_b_blocked_
                = ceildiv(col_, static_cast<uint64_t>(32)) * blocked_ld;
        if (ampere_src_) {
            create_matrix_layout(src_layout_, col32_2r_4r4, col_, row_,
                    blocked_ld, src_data_type_, stride_b_blocked_);
            create_matrix_layout(dst_layout_, non_blocked_order_, row_, col_,
                    plain_ld, dst_data_type_, row_ * col_);

        } else {
            create_matrix_layout(src_layout_, non_blocked_order_, col_, row_,
                    col_, src_data_type_, row_ * col_);
            create_matrix_layout(dst_layout_, col32_2r_4r4, row_, col_,
                    blocked_ld, dst_data_type_, stride_b_blocked_);
        }

        return status::success;
    }

    void execute(cublasHandle_t cublas_handle, void *src, void *dst,
            void *src_scale, void *dst_scale) {

        cudaStream_t streamId;
        auto lt_handle = (cublasLtHandle_t)(cublas_handle);
        CUBLAS_EXECUTE_FUNC(cublasGetStream, cublas_handle, &streamId);
        int alpha = 1;
        if (src_scale) {
            float host_src_scale = 1.0f;
            CUDA_EXECUTE_FUNC(cuMemcpyAsync, (CUdeviceptr)&host_src_scale,
                    (CUdeviceptr)src_scale, sizeof(float), streamId);
            alpha *= host_src_scale;
        }
        int beta = beta_;
        if (dst_scale) {
            float host_dst_scale = 1.0f;
            CUDA_EXECUTE_FUNC(cuMemcpyAsync, (CUdeviceptr)&host_dst_scale,
                    (CUdeviceptr)dst_scale, sizeof(float), streamId);
            alpha /= host_dst_scale;
            beta /= host_dst_scale;
        }
        cublasOperation_t transform_trans = trans ? CUBLAS_OP_T : CUBLAS_OP_N;
        CUBLAS_EXECUTE_FUNC(cublasLtMatrixTransformDescSetAttribute,
                trans_desc_, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA,
                &transform_trans, sizeof(transform_trans));
        CUBLAS_EXECUTE_FUNC(cublasLtMatrixTransform, lt_handle, trans_desc_,
                &alpha, src, src_layout_, &beta, dst, dst_layout_, dst,
                dst_layout_, streamId);
    }

    ~cublaslt_reorder_t() { cleanup(); }

    void cleanup() {
        if (src_layout_) {
            CUBLAS_EXECUTE_FUNC(cublasLtMatrixLayoutDestroy, src_layout_);
            src_layout_ = nullptr;
        }
        if (dst_layout_) {
            CUBLAS_EXECUTE_FUNC(cublasLtMatrixLayoutDestroy, dst_layout_);
            dst_layout_ = nullptr;
        }
        if (trans_desc_) {
            CUBLAS_EXECUTE_FUNC(
                    cublasLtMatrixTransformDescDestroy, trans_desc_);
            trans_desc_ = nullptr;
        }
    }

protected:
    cudaDataType_t src_data_type_;
    cudaDataType_t dst_data_type_;
    int dims_[DNNL_MAX_NDIMS];
    float beta_ = 0.0f;

    cublasLtMatrixTransformDesc_t trans_desc_;
    cublasLtMatrixLayout_t src_layout_;
    cublasLtMatrixLayout_t dst_layout_;

    int batch_count_;
    uint64_t row_, col_;

    cublasLtOrder_t col32_2r_4r4 = CUBLASLT_ORDER_COL32_2R_4R4;
    cublasLtOrder_t non_blocked_order_ = CUBLASLT_ORDER_COL;

    bool ampere_src_ = false;

    status_t create_matrix_layout(cublasLtMatrixLayout_t &layout,
            cublasLtOrder_t order, uint64_t row, uint64_t col, uint64_t ld,
            const cudaDataType_t data_type, uint64_t stride) {

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
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
