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

#ifndef GPU_AMD_MIOPEN_POOLING_IMPL_HPP
#define GPU_AMD_MIOPEN_POOLING_IMPL_HPP

#include "gpu/amd/sycl_hip_utils.hpp"
#include <miopen/miopen.h>

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct miopen_pooling_impl_base_t {
    virtual status_t init(pooling_pd_t *pd) = 0;

    virtual ~miopen_pooling_impl_base_t() {
        for (size_t i = 0; i < NUM_IO; ++i) {
            if (tensor_descs_[i]) {
                MIOPEN_EXECUTE_FUNC_V(
                        miopenDestroyTensorDescriptor, tensor_descs_[i]);
            }
        }

        if (pool_desc_) {
            MIOPEN_EXECUTE_FUNC_V(miopenDestroyPoolingDescriptor, pool_desc_);
        }
    }

    virtual void execute(
            miopenHandle_t handle, void *x, void *y, void *ws) const = 0;

    size_t get_ws_size_miopen() const { return ws_size_miopen_; }

protected:
    status_t init_common(pooling_pd_t *pd) {
        ndims_ = std::max(4, pd->ndims());
        kernel_ndims_ = ndims_ - 2;

        // Only 1D, 2D and 3D pooling is supported by MIOpen
        if (kernel_ndims_ > 3) { return status::unimplemented; }

        is_training_ = pd->desc()->prop_kind == prop_kind::forward_training;
        bool is_fwd = pd->is_fwd();
        auto src_md = is_fwd ? pd->src_md() : pd->diff_src_md();
        auto dst_md = is_fwd ? pd->dst_md() : pd->diff_dst_md();

        if (has_zero_dims(src_md->dims, pd->ndims())
                || has_zero_dims(dst_md->dims, pd->ndims())) {
            return status::success;
        }

        if (is_training_) {
            auto src_wrap = memory_desc_wrapper(src_md);
            x_size_bytes_ = src_wrap.size();
        }

        convert_dims(src_md->padded_dims, dims_[src], pd->ndims());
        convert_dims(dst_md->padded_dims, dims_[dst], pd->ndims());

        convert_dims(src_md->format_desc.blocking.strides, strides_[src],
                pd->ndims());
        convert_dims(dst_md->format_desc.blocking.strides, strides_[dst],
                pd->ndims());

        convert_dims(pd->desc()->kernel, kernel_dims_, kernel_ndims_);

        // If 1D pooling
        if (pd->ndims() == 3) {
            // Convert to [n, c, 1, w] since the current format is
            // [n, c, w, 1]
            dims_[src][3] = dims_[src][2];
            dims_[src][2] = 1;

            dims_[dst][3] = dims_[dst][2];
            dims_[dst][2] = 1;

            // Set kernel dimensions to [1, kw]
            kernel_dims_[1] = kernel_dims_[0];
            kernel_dims_[0] = 1;
        }

        if (ndims_ == 4) {
            kernel_padding_[0] = static_cast<int>(pd->padT());
            kernel_padding_[1] = static_cast<int>(pd->padL());

            kernel_strides_[0] = static_cast<int>(pd->KSH());
            kernel_strides_[1] = static_cast<int>(pd->KSW());
        } else {
            kernel_padding_[0] = static_cast<int>(pd->padFront());
            kernel_padding_[1] = static_cast<int>(pd->padT());
            kernel_padding_[2] = static_cast<int>(pd->padL());

            kernel_strides_[0] = static_cast<int>(pd->KSD());
            kernel_strides_[1] = static_cast<int>(pd->KSH());
            kernel_strides_[2] = static_cast<int>(pd->KSW());
        }

        CHECK(convert_data_type(src_md, &data_types_[src]));
        CHECK(convert_data_type(dst_md, &data_types_[dst]));

        CHECK(convert_alg_kind(pd->desc()->alg_kind, &pool_mode_));

        CHECK(create_and_set_tensor_descriptor(&tensor_descs_[src],
                data_types_[src], ndims_, dims_[src], strides_[src]));
        CHECK(create_and_set_tensor_descriptor(&tensor_descs_[dst],
                data_types_[dst], ndims_, dims_[dst], strides_[dst]));
        CHECK(create_and_set_pooling_descriptor(pd));

        return status::success;
    }

    status_t create_and_set_pooling_descriptor(const pooling_pd_t *pd) {
        CHECK(MIOPEN_EXECUTE_FUNC_S(
                miopenCreatePoolingDescriptor, &pool_desc_));
        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenSetNdPoolingDescriptor, pool_desc_,
                pool_mode_, kernel_ndims_, kernel_dims_, kernel_padding_,
                kernel_strides_));
        miopenSetPoolingIndexType(pool_desc_, index_type);
        miopenSetPoolingWorkSpaceIndexMode(pool_desc_, workspace_mode);

        return status::success;
    }

    status_t convert_alg_kind(
            alg_kind_t alg_kind, miopenPoolingMode_t *miopen_alg_kind) const {
        switch (alg_kind) {
            case alg_kind::pooling_max:
                *miopen_alg_kind = miopenPoolingMax;
                break;
            case alg_kind::pooling_avg_include_padding:
                *miopen_alg_kind = miopenPoolingAverageInclusive;
                break;
            case alg_kind::pooling_avg_exclude_padding:
                *miopen_alg_kind = miopenPoolingAverage;
                break;
            default: return status::unimplemented;
        }

        return status::success;
    }

    enum io { src = 0, dst, NUM_IO };
    miopenDataType_t data_types_[NUM_IO];
    miopenTensorDescriptor_t tensor_descs_[NUM_IO] = {};
    miopenPoolingDescriptor_t pool_desc_;

    miopenPoolingMode_t pool_mode_ = miopenPoolingMode_t::miopenPoolingMax;
    miopenIndexType_t index_type = miopenIndexType_t::miopenIndexUint32;
    miopenPoolingWorkspaceIndexMode_t workspace_mode
            = miopenPoolingWorkspaceIndexMode_t::
                    miopenPoolingWorkspaceIndexImage;

    int dims_[NUM_IO][DNNL_MAX_NDIMS];
    int strides_[NUM_IO][DNNL_MAX_NDIMS];
    int kernel_dims_[DNNL_MAX_NDIMS];
    int kernel_padding_[DNNL_MAX_NDIMS];
    int kernel_strides_[DNNL_MAX_NDIMS];
    const float alpha_ = 1.f, beta_ = 0.f;
    int ndims_, kernel_ndims_;
    bool is_training_ = false;

    size_t ws_size_miopen_ = 0;
    size_t x_size_bytes_ = 0;
};

struct miopen_pooling_fwd_impl_t : public miopen_pooling_impl_base_t {
    bool do_backward = false;

    status_t init(pooling_pd_t *pd) override {
        CHECK(init_common(pd));
        if (is_training_) {
            do_backward = true;
            MIOPEN_EXECUTE_FUNC(miopenPoolingGetWorkSpaceSizeV2, pool_desc_,
                    tensor_descs_[dst], &ws_size_miopen_);
        }

        return status::success;
    }

    void execute(
            miopenHandle_t handle, void *x, void *y, void *ws) const override {
        void *ws_miopen = is_training_ ? ws : nullptr;

        MIOPEN_EXECUTE_FUNC(miopenPoolingForward, handle, pool_desc_, &alpha_,
                tensor_descs_[src], x, &beta_, tensor_descs_[dst], y,
                do_backward, ws_miopen, ws_size_miopen_);

        if (is_training_) {
            void *ws_x = (uint8_t *)ws_miopen + ws_size_miopen_;
            void *ws_y = (uint8_t *)ws_x + x_size_bytes_;

            // Copy x and y into workspace so that they can be used
            // in the backward pass
            int alpha2 = 0;
            miopenTensorOp_t tensorOp = miopenTensorOpAdd;
            miopenOpTensor(handle, tensorOp, &alpha_, tensor_descs_[src], ws_x,
                    &alpha2, tensor_descs_[src], x, &beta_, tensor_descs_[src],
                    ws_x);

            miopenOpTensor(handle, tensorOp, &alpha_, tensor_descs_[dst], ws_y,
                    &alpha2, tensor_descs_[dst], y, &beta_, tensor_descs_[dst],
                    ws_y);
        }
    }
};

struct miopen_pooling_bwd_impl_t : public miopen_pooling_impl_base_t {
    status_t init(pooling_pd_t *pd) override {
        CHECK(init_common(pd));
        MIOPEN_EXECUTE_FUNC(miopenPoolingGetWorkSpaceSizeV2, pool_desc_,
                tensor_descs_[dst], &ws_size_miopen_);

        return status::success;
    }

    void execute(miopenHandle_t handle, void *dx, void *dy,
            void *ws) const override {
        void *ws_miopen = ws;
        void *ws_x = (uint8_t *)ws + ws_size_miopen_;
        void *ws_y = (uint8_t *)ws_x + x_size_bytes_;

        MIOPEN_EXECUTE_FUNC(miopenPoolingBackward, handle, pool_desc_, &alpha_,
                tensor_descs_[dst], ws_y, tensor_descs_[dst], dy,
                tensor_descs_[src], ws_x, &beta_, tensor_descs_[src], dx,
                ws_miopen);
    }
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
