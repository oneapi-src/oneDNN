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

struct cublas_base_params {

    bool with_eltwise(int position, const primitive_attr_t *attr) {
        return attr->post_ops_.contain(primitive_kind::eltwise, position);
    }

    bool with_sum(const primitive_attr_t *attr) {
        return attr->post_ops_.contain(primitive_kind::sum, 0)
                || attr->post_ops_.contain(primitive_kind::sum, 1);
    }

    // Returns scaling factor for post-ops=sum operation
    float sum_scale(const primitive_attr_t *attr) {
        int sum_idx_ = attr->post_ops_.find(primitive_kind::sum);
        return attr->post_ops_.entry_[sum_idx_].sum.scale;
    }

    alg_kind_t eltwise_algo(const primitive_attr_t *attr) {
        int eltwise_idx_ = attr->post_ops_.find(primitive_kind::eltwise);
        return with_eltwise(0, attr) || with_eltwise(1, attr)
                ? attr->post_ops_.entry_[eltwise_idx_].eltwise.alg
                : dnnl_alg_kind_undef;
    }

    int get_batch_stride(const memory_desc_wrapper desc) {
        auto dims = desc.dims();
        auto strides = desc.blocking_desc().strides;
        return dims[0] == 1 ? 0 : strides[0];
    }

    uint64_t M_, N_, K_;

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
    size_t reorder_scratch_size_ = 0;

    cublasOperation_t transA_;
    cublasOperation_t transB_;
    cublasOperation_t transC_;
    cublasGemmAlgo_t gemm_algo_
            = cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP;
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
