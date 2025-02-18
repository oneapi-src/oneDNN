/*******************************************************************************
* Copyright 2025 Intel Corporation
* Copyright 2025 Codeplay Software Limited
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

#ifndef GPU_GENERIC_SYCL_REF_GROUP_NORMALIZATION_HPP
#define GPU_GENERIC_SYCL_REF_GROUP_NORMALIZATION_HPP

#include "common/group_normalization_pd.hpp"
#include "common/primitive_attr.hpp"
#include "common/primitive_exec_types.hpp"
#include "gpu/generic/sycl/engine.hpp"
#include "gpu/generic/sycl/group_normalization_kernel.hpp"
#include "gpu/generic/sycl/sycl_gpu_primitive.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"
#include "gpu/generic/sycl/sycl_utils.hpp"

namespace dnnl::impl::gpu::generic::sycl {
struct ref_group_normalization_fwd_t : public gpu::generic::sycl::primitive_t {
    using gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public group_normalization_fwd_pd_t {
        using group_normalization_fwd_pd_t::group_normalization_fwd_pd_t;
        DECLARE_COMMON_PD_T(
                "ref:sycl:group_normalization", ref_group_normalization_fwd_t);

        status_t init(impl::engine_t *engine);

        ::sycl::nd_range<2> launch_range;
        sycl_group_norm_conf_t conf_;
    };

    kernel_t kernel_;
    status_t init(impl::engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const {
        return dynamic_cast<pd_t *>(primitive_t::pd().get());
    }
};

struct ref_group_normalization_bwd_t : public gpu::generic::sycl::primitive_t {
    using gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public group_normalization_bwd_pd_t {
        using group_normalization_bwd_pd_t ::group_normalization_bwd_pd_t;
        DECLARE_COMMON_PD_T("ref:sycl:group_normalization_bwd",
                ref_group_normalization_bwd_t);

        status_t init(impl::engine_t *engine);

        ::sycl::nd_range<1> launch_range;
        sycl_gnorm_bwd_conf_t conf_;
    };

    kernel_t kernel_;
    status_t init(impl::engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

    const pd_t *pd() const {
        return dynamic_cast<pd_t *>(primitive_t::pd().get());
    }
};

} // namespace dnnl::impl::gpu::generic::sycl

#endif
