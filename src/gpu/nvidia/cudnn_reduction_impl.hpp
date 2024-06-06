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

#ifndef GPU_NVIDIA_CUDNN_REDUCTION_IMPL_HPP
#define GPU_NVIDIA_CUDNN_REDUCTION_IMPL_HPP

#include "cudnn.h"

#include "gpu/nvidia/engine.hpp"
#include "gpu/nvidia/stream.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cudnn_reduction_impl_base_t {
    enum io { src_0 = 0, dst_0, NUM_IO };
    cudnnDataType_t data_types[NUM_IO];
    int ndims;
    int dims[NUM_IO][DNNL_MAX_NDIMS];
    cudnnTensorDescriptor_t tensor_descs[NUM_IO] = {};
    cudnnReduceTensorDescriptor_t reduce_desc_;
    cudnnReduceTensorOp_t alg_kind;
    cudnnNanPropagation_t Nanprop = cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN;
    cudnnReduceTensorIndices_t Indices
            = cudnnReduceTensorIndices_t::CUDNN_REDUCE_TENSOR_NO_INDICES;
    cudnnIndicesType_t IndicesType = cudnnIndicesType_t::CUDNN_32BIT_INDICES;

    size_t workSpaceSize = 0;
    float beta = 0.0f;
    float alpha = 1.0f;

    virtual ~cudnn_reduction_impl_base_t() {
        if (reduce_desc_) {
            CUDNN_EXECUTE_FUNC_V(
                    cudnnDestroyReduceTensorDescriptor, reduce_desc_);
        }
        for (size_t i = 0; i < NUM_IO; i++) {
            if (tensor_descs[i]) {
                CUDNN_EXECUTE_FUNC_V(
                        cudnnDestroyTensorDescriptor, tensor_descs[i]);
            }
        }
    }

    virtual status_t init(reduction_pd_t *pd) = 0;

    virtual void create_and_set_workspace(
            reduction_pd_t *pd, impl::engine_t *engine)
            = 0;

    void execute(cudnnHandle_t handle, void *a, void *c, void *scratch) {
        CUDNN_EXECUTE_FUNC(cudnnReduceTensor, handle, reduce_desc_, nullptr, 0,
                scratch, workSpaceSize, &alpha, tensor_descs[src_0], a, &beta,
                tensor_descs[dst_0], c);
    }

    status_t create_and_set_reduction_descriptor(const reduction_pd_t *pd) {
        CHECK(CUDNN_EXECUTE_FUNC_S(
                cudnnCreateReduceTensorDescriptor, &reduce_desc_));

        CHECK(CUDNN_EXECUTE_FUNC_S(cudnnSetReduceTensorDescriptor, reduce_desc_,
                alg_kind, CUDNN_DATA_FLOAT, Nanprop, Indices, IndicesType));
        return status::success;
    }

    status_t convert_alg_kind(
            alg_kind_t alg_kind, cudnnReduceTensorOp_t *cudnn_alg_kind) const {

        switch (alg_kind) {
            case alg_kind::reduction_max:
                *cudnn_alg_kind
                        = cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_MAX;
                break;
            case alg_kind::reduction_min:
                *cudnn_alg_kind
                        = cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_MIN;
                break;
            case alg_kind::reduction_sum:
                *cudnn_alg_kind
                        = cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_ADD;
                break;
            case alg_kind::reduction_mul:
                *cudnn_alg_kind
                        = cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_MUL;
                break;
            case alg_kind::reduction_mean:
                *cudnn_alg_kind
                        = cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_AVG;
                break;
            default: return status::unimplemented;
        }
        return status::success;
    }
};

struct cudnn_reduction_impl_t : public cudnn_reduction_impl_base_t {
    int strides[NUM_IO][DNNL_MAX_NDIMS];

    status_t init(reduction_pd_t *pd) override {
        if (has_zero_dims(pd->src_md()->dims, pd->src_md()->ndims)) {
            return status::success;
        }

        if (pd->src_md()->ndims > 5) { return status::invalid_arguments; }
        ndims = pd->src_md()->ndims < 4 ? 4 : pd->src_md()->ndims;

        convert_dims(
                pd->src_md()->padded_dims, dims[src_0], pd->src_md()->ndims);
        convert_dims(
                pd->dst_md()->padded_dims, dims[dst_0], pd->src_md()->ndims);

        convert_dims(pd->src_md()->format_desc.blocking.strides, strides[src_0],
                pd->src_md()->ndims);
        convert_dims(pd->dst_md()->format_desc.blocking.strides, strides[dst_0],
                pd->src_md()->ndims);

        alg_kind_t alg = pd->desc()->alg_kind;

        auto alg_ok = convert_alg_kind(alg, &alg_kind);
        if (alg_ok != status::success) { return status::unimplemented; }

        CHECK(convert_data_type(pd->src_md(), &data_types[src_0]));
        CHECK(convert_data_type(pd->dst_md(), &data_types[dst_0]));

        CHECK(create_and_set_tensor_descriptor(&tensor_descs[src_0],
                data_types[src_0], ndims, dims[src_0], strides[src_0]));
        CHECK(create_and_set_tensor_descriptor(&tensor_descs[dst_0],
                data_types[dst_0], ndims, dims[dst_0], strides[dst_0]));
        CHECK(create_and_set_reduction_descriptor(pd));
        return status::success;
    }

    void create_and_set_workspace(
            reduction_pd_t *pd, impl::engine_t *engine) override {
        auto sycl_engine = utils::downcast<nvidia::engine_t *>(engine);

        impl::stream_t *service_stream;
        sycl_engine->get_service_stream(service_stream);

        auto cuda_stream = utils::downcast<nvidia::stream_t *>(service_stream);
        auto cuda_handle = cuda_stream->get_cudnn_handle();

        CUDNN_EXECUTE_FUNC_S(cudnnGetReductionWorkspaceSize, cuda_handle,
                reduce_desc_, tensor_descs[src_0], tensor_descs[dst_0],
                &workSpaceSize);

        pd->scratchpad_registry().registrar().book(
                memory_tracking::names::key_none, workSpaceSize, size_t(1));
    }
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
