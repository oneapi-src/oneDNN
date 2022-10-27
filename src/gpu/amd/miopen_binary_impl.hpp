/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#ifndef GPU_AMD_MIOPEN_BINARY_IMPL_HPP
#define GPU_AMD_MIOPEN_BINARY_IMPL_HPP
#include "gpu/amd/sycl_hip_utils.hpp"
#include <miopen/miopen.h>

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct miopen_binary_impl_base_t {
    enum io { src_0 = 0, src_1, dst_0, NUM_IO };
    miopenDataType_t data_types[NUM_IO];
    int ndims;
    int dims[NUM_IO][DNNL_MAX_NDIMS];
    miopenTensorDescriptor_t tensor_descs[NUM_IO] = {};
    miopenTensorOp_t alg_kind;
    float beta = 0.0f;
    const float alpha = 1.0f;

    virtual ~miopen_binary_impl_base_t() {
        for (size_t i = 0; i < NUM_IO; i++) {
            if (tensor_descs[i]) {
                MIOPEN_EXECUTE_FUNC_V(
                        miopenDestroyTensorDescriptor, tensor_descs[i]);
            }
        }
    }

    virtual status_t init(const binary_pd_t *pd) = 0;

    void execute(miopenHandle_t handle, void *a, void *b, void *c,
            const void *s0, const void *s1) const {
        MIOPEN_EXECUTE_FUNC(miopenOpTensor, handle, alg_kind, s0 ? s0 : &alpha,
                tensor_descs[src_0], a, s1 ? s1 : &alpha, tensor_descs[src_1],
                b, &beta, tensor_descs[dst_0], c);
    }

    status_t convert_alg_kind(
            alg_kind_t alg_kind, miopenTensorOp_t *miopen_alg_kind) const {

        switch (alg_kind) {
            case alg_kind::binary_add:
                *miopen_alg_kind = miopenTensorOp_t::miopenTensorOpAdd;
                break;
            case alg_kind::binary_mul:
                *miopen_alg_kind = miopenTensorOp_t::miopenTensorOpMul;
                break;
            case alg_kind::binary_min:
                *miopen_alg_kind = miopenTensorOp_t::miopenTensorOpMin;
                break;
            case alg_kind::binary_max:
                *miopen_alg_kind = miopenTensorOp_t::miopenTensorOpMax;
                break;
            default: return status::unimplemented;
        }
        return status::success;
    }
};

struct miopen_binary_impl_t : public miopen_binary_impl_base_t {
    int strides[NUM_IO][DNNL_MAX_NDIMS];

    status_t init(const binary_pd_t *pd) override {

        if (has_zero_dims(pd->src_md(0)->dims, pd->ndims())) {
            return status::success;
        }
        if (pd->ndims() > MIOPEN_DIM_MAX) { return status::invalid_arguments; }
        ndims = pd->ndims() < 4 ? 4 : pd->ndims();
        convert_dims(pd->src_md(0)->padded_dims, dims[src_0], pd->ndims());
        convert_dims(pd->src_md(1)->padded_dims, dims[src_1], pd->ndims());
        convert_dims(pd->dst_md()->padded_dims, dims[dst_0], pd->ndims());

        convert_dims(pd->src_md(0)->format_desc.blocking.strides,
                strides[src_0], pd->ndims());
        convert_dims(pd->src_md(1)->format_desc.blocking.strides,
                strides[src_1], pd->ndims());
        convert_dims(pd->dst_md()->format_desc.blocking.strides, strides[dst_0],
                pd->ndims());
        alg_kind_t alg = pd->desc()->alg_kind;
        auto alg_ok = convert_alg_kind(alg, &alg_kind);
        if (alg_ok != status::success) { return status::unimplemented; }

        CHECK(convert_data_type(pd->src_md(0), &data_types[src_0]));
        CHECK(convert_data_type(pd->src_md(1), &data_types[src_1]));
        CHECK(convert_data_type(pd->dst_md(), &data_types[dst_0]));

        CHECK(create_and_set_tensor_descriptor(&tensor_descs[src_0],
                data_types[src_0], ndims, dims[src_0], strides[src_0]));
        CHECK(create_and_set_tensor_descriptor(&tensor_descs[src_1],
                data_types[src_1], ndims, dims[src_1], strides[src_1]));
        CHECK(create_and_set_tensor_descriptor(&tensor_descs[dst_0],
                data_types[dst_0], ndims, dims[dst_0], strides[dst_0]));

        return status::success;
    }
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
