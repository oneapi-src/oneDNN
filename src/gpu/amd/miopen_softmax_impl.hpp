/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
* Copyright 2020-2022 Codeplay Software Limited 
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

#ifndef GPU_AMD_MIOPEN_SOFTMAX_IMPL_HPP
#define GPU_AMD_MIOPEN_SOFTMAX_IMPL_HPP

#include "gpu/amd/sycl_hip_utils.hpp"
#include "miopen/miopen.h"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct miopen_softmax_impl_base_t {
    enum io { src = 0, dst, d_src, d_dst, NUM_IO };
    int strides[NUM_IO][DNNL_MAX_NDIMS];
    miopenDataType_t data_type;
    int ndims;
    miopenSoftmaxAlgorithm_t alg_kind;

    miopenSoftmaxMode_t mode = miopenSoftmaxMode_t::MIOPEN_SOFTMAX_MODE_CHANNEL;

    // oneDNN softmax primitive doesn't support any post-ops or attributes,
    // hence we can set alpha = 1 and beta = 0 for all cases
    float alpha = 1.0f;
    float beta = 0.0f;

    virtual ~miopen_softmax_impl_base_t() {}

    virtual status_t init(const softmax_pd_t *pd) = 0;

    virtual void execute(miopenHandle_t handle, void **x, int size) const = 0;

    // Mapping between dnnl algorithm and MIOpen softmax algorithm
    status_t convert_alg_kind(bool is_log_softmax,
            miopenSoftmaxAlgorithm_t *miopen_alg_kind) const {
        if (is_log_softmax) {
            *miopen_alg_kind = miopenSoftmaxAlgorithm_t::MIOPEN_SOFTMAX_LOG;
        } else {
            *miopen_alg_kind
                    = miopenSoftmaxAlgorithm_t::MIOPEN_SOFTMAX_ACCURATE;
        }

        return status::success;
    }
};

struct miopen_softmax_fwd_impl_t : public miopen_softmax_impl_base_t {
    int dims[NUM_IO][DNNL_MAX_NDIMS];
    miopenTensorDescriptor_t tensor_desc;

    status_t init(const softmax_pd_t *pd) override {

        if (pd->has_zero_dim_memory()) return status::success;

        if (pd->ndims() > MIOPEN_DIM_MAX) { return status::invalid_arguments; }
        ndims = pd->ndims() < 4 ? 4 : pd->ndims();

        convert_dims(pd->src_md()->padded_dims, dims[src], pd->ndims());
        convert_dims(pd->src_md()->format_desc.blocking.strides, strides[src],
                pd->ndims());
        convert_dims(pd->dst_md()->format_desc.blocking.strides, strides[dst],
                pd->ndims());

        convert_alg_kind(pd->is_logsoftmax(), &alg_kind);

        assert(pd->src_md()->data_type == pd->dst_md()->data_type);

        CHECK(convert_data_type(pd->src_md(), &data_type));

        CHECK(create_and_set_tensor_descriptor(
                &tensor_desc, data_type, 4, dims[src], strides[src]));

        return status::success;
    }

    void execute(miopenHandle_t handle, void **x, int size) const override {
        // Confirm that 2 arguments were passed, src and dst
        assert(size == 2);

        MIOPEN_EXECUTE_FUNC(miopenSoftmaxForward_V2, handle, &alpha,
                tensor_desc, x[0], &beta, tensor_desc, x[1], alg_kind, mode);
    }

    ~miopen_softmax_fwd_impl_t() {
        MIOPEN_EXECUTE_FUNC_V(miopenDestroyTensorDescriptor, tensor_desc);
    }
};

struct miopen_softmax_bwd_impl_t : public miopen_softmax_impl_base_t {
    int dims[NUM_IO][DNNL_MAX_NDIMS];
    miopenTensorDescriptor_t tensor_dst_desc;
    miopenTensorDescriptor_t tensor_diff_desc;

    status_t init(const softmax_pd_t *pd) override {

        if (pd->has_zero_dim_memory()) return status::success;
        if (pd->ndims() > MIOPEN_DIM_MAX) { return status::invalid_arguments; }
        ndims = pd->ndims() < 4 ? 4 : pd->ndims();

        convert_dims(pd->dst_md()->padded_dims, dims[dst], pd->ndims());
        convert_dims(pd->diff_src_md()->padded_dims, dims[d_src], pd->ndims());

        convert_alg_kind(pd->is_logsoftmax(), &alg_kind);

        assert(pd->diff_dst_md()->data_type == pd->dst_md()->data_type);
        assert(pd->diff_dst_md()->data_type == pd->diff_src_md()->data_type);
        CHECK(convert_data_type(pd->dst_md(), &data_type));

        convert_dims(pd->dst_md()->format_desc.blocking.strides, strides[dst],
                pd->ndims());
        convert_dims(pd->diff_src_md()->format_desc.blocking.strides,
                strides[d_src], pd->ndims());
        convert_dims(pd->diff_dst_md()->format_desc.blocking.strides,
                strides[d_dst], pd->ndims());

        CHECK(create_and_set_tensor_descriptor(
                &tensor_dst_desc, data_type, 4, dims[dst], strides[dst]));
        CHECK(create_and_set_tensor_descriptor(
                &tensor_diff_desc, data_type, 4, dims[d_src], strides[d_src]));

        return status::success;
    }

    void execute(miopenHandle_t handle, void **x, int size) const override {
        // Assert that 3 arguments were passed src, diff_dst and diff_src
        assert(size == 3);

        MIOPEN_EXECUTE_FUNC(miopenSoftmaxBackward_V2, handle, &alpha,
                tensor_dst_desc, x[0], tensor_diff_desc, x[1], &beta,
                tensor_diff_desc, x[2], alg_kind, mode);
    }

    ~miopen_softmax_bwd_impl_t() {
        MIOPEN_EXECUTE_FUNC_V(miopenDestroyTensorDescriptor, tensor_dst_desc);
        MIOPEN_EXECUTE_FUNC_V(miopenDestroyTensorDescriptor, tensor_diff_desc);
    }
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
