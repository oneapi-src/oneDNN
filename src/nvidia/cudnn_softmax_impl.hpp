/***************************************************************************
 *  Copyright 2020 Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a copy of the License has been included in this
 *  repository.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 **************************************************************************/

#ifndef CUDNN_SOFTMAX_IMPL_HPP
#define CUDNN_SOFTMAX_IMPL_HPP

#include "cudnn.h"

#include "nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace cuda {

struct cudnn_softmax_impl_base_t {
    cudnnDataType_t data_type;
    int ndims;
    cudnnSoftmaxAlgorithm_t alg_kind;
    // cuDNN only supports softmax on channel dimension
    cudnnSoftmaxMode_t mode = cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_CHANNEL;
    // DNNL softmax primitive doesn't support any post-ops or attributes,
    // hence we can set alpha = 1 and beta = 0 for all cases
    float alpha = 1.0f;
    float beta = 0.0f;

    virtual ~cudnn_softmax_impl_base_t() {}

    virtual status_t init(const softmax_pd_t *pd) = 0;

    virtual void execute(cudnnHandle_t handle, void **x, int size) = 0;

    // Mapping between dnnl algorithm and cuDNN softmax algorithm
    status_t convert_alg_kind(
            bool is_log_softmax, cudnnSoftmaxAlgorithm_t *cuda_alg_kind) {
        if (is_log_softmax) {
            *cuda_alg_kind = cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_LOG;
        } else {
            *cuda_alg_kind = cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_ACCURATE;
        }
        return status::success;
    }
};

struct cudnn_softmax_fwd_impl_t : public cudnn_softmax_impl_base_t {
    int dims[DNNL_MAX_NDIMS];
    cudnnTensorDescriptor_t tensor_desc;
    int strides[DNNL_MAX_NDIMS];

    virtual status_t init(const softmax_pd_t *pd) override {
        // If any of the dimensions are 0 we should not continue with creating cudnn descriptors
        if (has_zero_dims(pd->src_md(0)->dims, pd->ndims())) {
            return status::success;
        }

        if (pd->ndims() > CUDNN_DIM_MAX) { return status::invalid_arguments; }
        ndims = pd->ndims() < 4 ? 4 : pd->ndims();

        convert_dims(pd->src_md()->padded_dims, dims, pd->ndims());

        convert_dims(pd->src_md()->format_desc.blocking.strides, strides,
                pd->ndims());

        convert_alg_kind(pd->is_logsoftmax(), &alg_kind);

        assert(pd->src_md()->data_type == pd->dst_md()->data_type);

        CHECK(convert_data_type(pd->src_md(), &data_type));

        CHECK(create_and_set_tensor_descriptor(
                &tensor_desc, data_type, ndims, dims, strides));
        return status::success;
    }

    virtual void execute(cudnnHandle_t handle, void **x, int size) override {
        // Confirm that 2 arguments were passed, src and dst
        assert(size == 2);
        CUDNN_EXECUTE_FUNC(cudnnSoftmaxForward, handle, alg_kind, mode, &alpha,
                tensor_desc, x[0], &beta, tensor_desc, x[1]);
    }

    ~cudnn_softmax_fwd_impl_t() {
        CUDNN_EXECUTE_FUNC_V(cudnnDestroyTensorDescriptor, tensor_desc);
    }
};

struct cudnn_softmax_bwd_impl_t : public cudnn_softmax_impl_base_t {
    int dims[DNNL_MAX_NDIMS];
    int dims_dst[DNNL_MAX_NDIMS];
    cudnnTensorDescriptor_t tensor_dst_desc;
    cudnnTensorDescriptor_t tensor_diff_desc;
    int strides[DNNL_MAX_NDIMS];

    virtual status_t init(const softmax_pd_t *pd) override {
        // If any of the dimensions are 0 we should not continue with creating cudnn descriptors
        if (memory_desc_wrapper(pd->desc()->diff_desc).has_zero_dim())
            return status::success;

        if (pd->ndims() > CUDNN_DIM_MAX) { return status::invalid_arguments; }
        ndims = pd->ndims() < 4 ? 4 : pd->ndims();

        // Initialise data from descriptors
        convert_dims(pd->dst_md()->padded_dims, dims_dst, pd->ndims());
        convert_dims(pd->diff_src_md()->padded_dims, dims, pd->ndims());

        convert_dims(pd->dst_md()->format_desc.blocking.strides, strides,
                pd->ndims());

        convert_alg_kind(pd->is_logsoftmax(), &alg_kind);

        assert(pd->diff_dst_md()->data_type == pd->dst_md()->data_type);
        assert(pd->diff_dst_md()->data_type == pd->diff_src_md()->data_type);

        CHECK(convert_data_type(pd->dst_md(), &data_type));

        CHECK(create_and_set_tensor_descriptor(
                &tensor_dst_desc, data_type, ndims, dims_dst, strides));
        CHECK(create_and_set_tensor_descriptor(
                &tensor_diff_desc, data_type, ndims, dims, strides));
        return status::success;
    }

    virtual void execute(cudnnHandle_t handle, void **x, int size) override {
        // Assert that 3 arguments were passed src, diff_dst and diff_src
        assert(size == 3);
        CUDNN_EXECUTE_FUNC(cudnnSoftmaxBackward, handle, alg_kind, mode, &alpha,
                tensor_dst_desc, x[0], tensor_diff_desc, x[1], &beta,
                tensor_diff_desc, x[2]);
    }

    ~cudnn_softmax_bwd_impl_t() {
        CUDNN_EXECUTE_FUNC_V(cudnnDestroyTensorDescriptor, tensor_dst_desc);
        CUDNN_EXECUTE_FUNC_V(cudnnDestroyTensorDescriptor, tensor_diff_desc);
    }
};

} // namespace cuda
} // namespace impl
} // namespace dnnl

#endif //CUDNN_SOFTMAX_IMPL_HPP
