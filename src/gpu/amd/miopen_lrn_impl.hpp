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

#ifndef GPU_AMD_MIOPEN_LRN_IMPL_HPP
#define GPU_AMD_MIOPEN_LRN_IMPL_HPP

#include "gpu/amd/sycl_hip_utils.hpp"
#include <hip/hip_runtime.h>

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct miopen_lrn_impl_base_t {

    virtual ~miopen_lrn_impl_base_t() {
        if (lrn_desc) {
            MIOPEN_EXECUTE_FUNC_V(miopenDestroyLRNDescriptor, lrn_desc);
        }
        for (size_t i = 0; i < NUM_IO; i++) {
            if (tensor_descs[i]) {
                MIOPEN_EXECUTE_FUNC_V(
                        miopenDestroyTensorDescriptor, tensor_descs[i]);
            }
        }
    }

    virtual status_t init(lrn_pd_t *pd) = 0;
    virtual void execute(
            miopenHandle_t handle, const std::vector<void *> &args) const = 0;

    size_t get_workspace_size() const { return workspace_size; }

protected:
    enum io { src_idx = 0, dst_idx, d_src_idx, d_dst_idx, NUM_IO };
    miopenDataType_t data_types[NUM_IO];
    int ndims;
    int dst_size;
    int dims[NUM_IO][DNNL_MAX_NDIMS];
    int strides[NUM_IO][DNNL_MAX_NDIMS];
    float alpha = 1.0f;
    float beta = 0.0f;
    bool is_training;
    double lrn_alpha;
    double lrn_beta;
    double lrn_K;
    unsigned int lrn_N;
    size_t workspace_size;
    miopenLRNMode_t lrn_mode;
    miopenLRNDescriptor_t lrn_desc = nullptr;
    miopenTensorDescriptor_t tensor_descs[NUM_IO] = {};

    virtual status_t init_common(lrn_pd_t *pd) {
        ndims = std::max(4, pd->ndims());
        if (ndims > 4) { return status::invalid_arguments; }

        const auto lrn_desc = pd->desc();
        const auto dst_wrap = memory_desc_wrapper(pd->dst_md());

        dst_size = dst_wrap.nelems();
        is_training = pd->desc()->prop_kind == prop_kind::forward_training;

        lrn_K = lrn_desc->lrn_k;
        lrn_N = lrn_desc->local_size;
        lrn_alpha = lrn_desc->lrn_alpha;
        lrn_beta = lrn_desc->lrn_beta;

        // Initialise lrn algorithm
        CHECK(convert_alg_kind(pd->desc()->alg_kind, &lrn_mode));

        convert_dims(pd->src_md()->padded_dims, dims[src_idx], pd->ndims());
        convert_dims(pd->src_md()->format_desc.blocking.strides,
                strides[src_idx], pd->ndims());

        // Set datatype
        CHECK(convert_data_type(pd->src_md(), &data_types[src_idx]));

        // Initialise tensor descriptor
        CHECK(create_and_set_tensor_descriptor(&tensor_descs[src_idx],
                data_types[src_idx], ndims, dims[src_idx], strides[src_idx]));
        CHECK(create_and_set_lrn_descriptor());

        return status::success;
    }

    virtual status_t create_and_set_lrn_descriptor() {
        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenCreateLRNDescriptor, &lrn_desc));
        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenSetLRNDescriptor, lrn_desc, lrn_mode,
                lrn_N, lrn_alpha, lrn_beta, lrn_K));
        return status::success;
    }

    status_t convert_alg_kind(
            alg_kind_t alg_kind, miopenLRNMode_t *miopen_alg_kind) {
        if (alg_kind == alg_kind::lrn_across_channels) {
            *miopen_alg_kind = miopenLRNMode_t::miopenLRNCrossChannel;
        } else if (alg_kind == alg_kind::lrn_within_channel) {
            *miopen_alg_kind = miopenLRNMode_t::miopenLRNWithinChannel;
        } else {
            return status::unimplemented;
        }
        return status::success;
    }
};

struct miopen_lrn_fwd_impl_t : public miopen_lrn_impl_base_t {
    bool do_backward = false;

    virtual status_t init(lrn_pd_t *pd) override {
        CHECK(init_common(pd));

        convert_dims(pd->dst_md()->padded_dims, dims[dst_idx], pd->ndims());
        convert_dims(pd->dst_md()->format_desc.blocking.strides,
                strides[dst_idx], pd->ndims());

        CHECK(convert_data_type(pd->dst_md(), &data_types[dst_idx]));
        CHECK(create_and_set_tensor_descriptor(&tensor_descs[dst_idx],
                data_types[dst_idx], ndims, dims[dst_idx], strides[dst_idx]));

        if (is_training) { do_backward = true; }
        MIOPEN_EXECUTE_FUNC(miopenLRNGetWorkSpaceSize, tensor_descs[dst_idx],
                &workspace_size);

        return status::success;
    }

    void execute(miopenHandle_t handle,
            const std::vector<void *> &args) const override {

        void *ws_miopen = args[2];
        void *ws_dst_values = (uint8_t *)args[2] + get_workspace_size();

        MIOPEN_EXECUTE_FUNC(miopenLRNForward, handle, lrn_desc, &alpha,
                tensor_descs[src_idx], args[0], &beta, tensor_descs[dst_idx],
                args[1], do_backward, ws_miopen);

        if (is_training) {
            float alpha1 = 1.f;
            float alpha2 = 0.f;
            float beta = 0.f;

            MIOPEN_EXECUTE_FUNC(miopenOpTensor, handle, miopenTensorOpAdd,
                    &alpha1, tensor_descs[dst_idx], args[1], &alpha2,
                    tensor_descs[dst_idx], args[1], &beta,
                    tensor_descs[dst_idx], ws_dst_values);
        }
    }
};

struct miopen_lrn_bwd_impl_t : public miopen_lrn_impl_base_t {

    status_t init(lrn_pd_t *pd) override {
        CHECK(init_common(pd));

        // Set dimensions
        convert_dims(
                pd->diff_dst_md()->padded_dims, dims[dst_idx], pd->ndims());
        convert_dims(
                pd->diff_src_md()->padded_dims, dims[d_src_idx], pd->ndims());
        convert_dims(
                pd->diff_dst_md()->padded_dims, dims[d_dst_idx], pd->ndims());

        // Set strides
        convert_dims(pd->diff_dst_md()->format_desc.blocking.strides,
                strides[dst_idx], pd->ndims());
        convert_dims(pd->diff_src_md()->format_desc.blocking.strides,
                strides[d_src_idx], pd->ndims());
        convert_dims(pd->diff_dst_md()->format_desc.blocking.strides,
                strides[d_dst_idx], pd->ndims());

        // Set datatypes
        CHECK(convert_data_type(pd->diff_dst_md(), &data_types[dst_idx]));
        CHECK(convert_data_type(pd->diff_src_md(), &data_types[d_src_idx]));
        CHECK(convert_data_type(pd->diff_dst_md(), &data_types[d_dst_idx]));

        // Initialise tensor descriptors
        CHECK(create_and_set_tensor_descriptor(&tensor_descs[dst_idx],
                data_types[dst_idx], ndims, dims[dst_idx], strides[dst_idx]));
        CHECK(create_and_set_tensor_descriptor(&tensor_descs[d_src_idx],
                data_types[d_src_idx], ndims, dims[d_src_idx],
                strides[d_src_idx]));
        CHECK(create_and_set_tensor_descriptor(&tensor_descs[d_dst_idx],
                data_types[d_dst_idx], ndims, dims[d_dst_idx],
                strides[d_dst_idx]));

        MIOPEN_EXECUTE_FUNC(miopenLRNGetWorkSpaceSize, tensor_descs[dst_idx],
                &workspace_size);
        return status::success;
    }

    void execute(miopenHandle_t handle,
            const std::vector<void *> &args) const override {
        void *ws_miopen = args[1];
        void *ws_dst_values = (uint8_t *)args[1] + get_workspace_size();

        MIOPEN_EXECUTE_FUNC_V(miopenLRNBackward, handle, lrn_desc, &alpha,
                tensor_descs[dst_idx], ws_dst_values, tensor_descs[d_dst_idx],
                args[d_dst_idx], tensor_descs[src_idx], args[src_idx], &beta,
                tensor_descs[d_src_idx], args[d_src_idx], ws_miopen);
    }
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
