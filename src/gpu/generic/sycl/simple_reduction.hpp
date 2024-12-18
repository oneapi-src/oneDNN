/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef GPU_GENERIC_SYCL_SIMPLE_REDUCTION_HPP
#define GPU_GENERIC_SYCL_SIMPLE_REDUCTION_HPP

#include "common/primitive_desc_iterator.hpp"
#include "common/reorder.hpp"
#include "common/reorder_pd.hpp"
#include "gpu/generic/sycl/sycl_gpu_primitive.hpp"
#include "gpu/generic/sycl/sycl_io_helper.hpp"
#include "gpu/generic/sycl/sycl_post_ops.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"
#include "gpu/generic/sycl/sycl_utils.hpp"
#include "gpu/gpu_reduction_pd.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

struct simple_reduction_t : public gpu::generic::sycl::primitive_t {
    using gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public gpu_reduction_pd_t {
        using gpu_reduction_pd_t::gpu_reduction_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", simple_reduction_t);

        status_t init(impl::engine_t *engine) {
            using sm = primitive_attr_t::skip_mask_t;

            memory_desc_wrapper src_wrap(src_md());
            memory_desc_wrapper dst_wrap(dst_md());

            bool ok = set_default_params() == status::success
                    && attr()->has_default_values(sm::post_ops)
                    && sycl_post_ops_t::post_ops_ok(attr())
                    && attr_.set_default_formats(dst_md()) == status::success
                    && src_wrap.is_plain() && dst_wrap.is_plain()
                    && src_wrap.ndims() == dst_wrap.ndims()
                    && md_dims_in_range(src_md()) && md_dims_in_range(dst_md());
            if (!ok) return status::unimplemented;

            return init_conf();
        }

        sycl_simple_reduction_conf_t conf_;
        dim_t dst_nelems_;

    private:
        status_t init_conf();
    };

    status_t init(impl::engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    kernel_t kernel_;
    std::shared_ptr<impl::primitive_t> reorder_p_;
};

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
