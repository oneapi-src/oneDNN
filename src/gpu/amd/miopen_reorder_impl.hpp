/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#ifndef GPU_AMD_MIOPEN_REORDER_IMPL_HPP
#define GPU_AMD_MIOPEN_REORDER_IMPL_HPP

#include "common/type_helpers.hpp"
#include "gpu/amd/sycl_hip_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct miopen_reorder_generic_t {
public:
    virtual status_t init(const reorder_pd_t *pd) = 0;

    virtual void execute(miopenHandle_t handle, void *src, void *dst,
            void *src_scale, void *dst_scale) const = 0;

    virtual ~miopen_reorder_generic_t() {
        MIOPEN_EXECUTE_FUNC_V(miopenDestroyTensorDescriptor, src_desc_);
        MIOPEN_EXECUTE_FUNC_V(miopenDestroyTensorDescriptor, dst_desc_);
    }

    int dst_offset_in_bytes() { return dst_offset_in_bytes_; }
    int src_offset_in_bytes() { return src_offset_in_bytes_; }

protected:
    miopenDataType_t src_data_type_;
    miopenDataType_t dst_data_type_;
    int ndims_;
    int dims_[DNNL_MAX_NDIMS];
    miopenTensorDescriptor_t src_desc_;
    miopenTensorDescriptor_t dst_desc_;
    float alpha_, beta_;
    int dst_offset_in_bytes_ = 0;
    int src_offset_in_bytes_ = 0;
    int nelems;

    data_type_t src_dt;
    data_type_t dst_dt;
};

// This structure is used when the memory format does not include blocking
struct miopen_reorder_stride_t : public miopen_reorder_generic_t {
public:
    status_t init(const reorder_pd_t *pd) override {
        // If any of the dimensions are 0 we should not continue with creating
        // MIOpen descriptors
        memory_desc_wrapper src_mdw(pd->src_md());
        if (src_mdw.size() == 0) { return status::success; }

        if (src_mdw.has_runtime_dims_or_strides()) {
            return status::unimplemented;
        }

        nelems = src_mdw.nelems();

        // Validity checks
        assert(pd->dst_md()->ndims == pd->src_md()->ndims);
        dst_offset_in_bytes_ = pd->dst_md()->offset0
                * types::data_type_size(pd->dst_md()->data_type);
        src_offset_in_bytes_ = pd->src_md()->offset0
                * types::data_type_size(pd->src_md()->data_type);
        beta_ = pd->beta();

        CHECK(convert_data_type(pd->src_md(), &src_data_type_));
        CHECK(convert_data_type(pd->dst_md(), &dst_data_type_));

        convert_dims(pd->dst_md()->dims, dims_, pd->dst_md()->ndims);
        convert_dims(pd->src_md()->format_desc.blocking.strides, src_strides_,
                pd->src_md()->ndims);
        convert_dims(pd->dst_md()->format_desc.blocking.strides, dst_strides_,
                pd->dst_md()->ndims);

        ndims_ = pd->dst_md()->ndims > 4 ? pd->dst_md()->ndims : 4;

        bool vectorized = has_different_block_size(pd->src_md(), pd->dst_md());
        if (!vectorized) {
            adjust_dim_for_dnn(dims_, pd->dst_md()->ndims, pd->src_md());
            adjust_stride_for_dnn(
                    src_strides_, pd->dst_md()->ndims, pd->src_md());
            adjust_stride_for_dnn(
                    dst_strides_, pd->dst_md()->ndims, pd->dst_md());
            ndims_ = pd->dst_md()->ndims >= 4 ? pd->dst_md()->ndims
                            + pd->dst_md()->format_desc.blocking.inner_nblks
                                              : 4;
        }
        convert_data_type(pd->src_md(), &src_data_type_, vectorized);
        convert_data_type(pd->dst_md(), &dst_data_type_, vectorized);

        src_dt = pd->src_md()->data_type;
        dst_dt = pd->dst_md()->data_type;

        // Create and set source tensor descriptor
        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenCreateTensorDescriptor, &src_desc_));
        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenSetTensorDescriptor, src_desc_,
                src_data_type_, ndims_, dims_, src_strides_));

        // Create and set destination tensor descriptor
        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenCreateTensorDescriptor, &dst_desc_));
        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenSetTensorDescriptor, dst_desc_,
                dst_data_type_, ndims_, dims_, dst_strides_));

        return status::success;
    }

    void execute(miopenHandle_t handle, void *src, void *dst, void *src_scale,
            void *dst_scale) const override {
        float alpha = 1.0f;
        if (src_scale) {
            float host_src_scale = 1.0f;
            HIP_EXECUTE_FUNC(hipMemcpy, (HIPdeviceptr)&host_src_scale,
                    (HIPdeviceptr)src_scale, sizeof(float), hipMemcpyDefault);
            alpha *= host_src_scale;
        }
        float beta = beta_;
        if (dst_scale) {
            float host_dst_scale = 1.0f;
            HIP_EXECUTE_FUNC(hipMemcpy, (HIPdeviceptr)&host_dst_scale,
                    (HIPdeviceptr)dst_scale, sizeof(float), hipMemcpyDefault);
            alpha /= host_dst_scale;
            beta /= host_dst_scale;
        }

        MIOPEN_EXECUTE_FUNC(miopenTransformTensor, handle, &alpha, src_desc_,
                src, &beta, dst_desc_, dst);
    }

private:
    int src_strides_[DNNL_MAX_NDIMS];
    int dst_strides_[DNNL_MAX_NDIMS];

    using miopen_reorder_generic_t::miopen_reorder_generic_t;
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_AMD_MIOPEN_REORDER_IMPL_HPP
