/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#include "gpu/generic/sycl/ref_reorder.hpp"
#include "gpu/generic/sycl/reorder_kernels.hpp"
#include "gpu/generic/sycl/sycl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

status_t ref_reorder_t::pd_t::init_conf() {
    conf_ = sycl_reorder_conf_t();

    conf_.src_md = xpu::sycl::md_t(src_md(0));
    conf_.dst_md = xpu::sycl::md_t(dst_md());

    conf_.wk_size = memory_desc_wrapper(src_md(0)).nelems();

    conf_.do_scale_src = !attr()->scales_.has_default_values(DNNL_ARG_SRC_0);
    conf_.scale_src_mask = attr()->scales_.get_mask(DNNL_ARG_SRC_0);
    conf_.do_scale_dst = !attr()->scales_.has_default_values(DNNL_ARG_DST);
    conf_.scale_dst_mask = attr()->scales_.get_mask(DNNL_ARG_DST);
    conf_.post_ops = sycl_post_ops_t(attr(), dst_md());

    return status::success;
}

status_t ref_reorder_t::init(impl::engine_t *engine) {
    const auto kid = ::sycl::get_kernel_id<reorder_kernel_t>();
    CHECK(create_kernel(engine, kid, &kernel_));
    return status::success;
}

status_t ref_reorder_t::execute(const exec_ctx_t &ctx) const {
    parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        reorder_kernel_t reorder_kernel(pd()->conf_, cgh, ctx);

        cgh.parallel_for(get_range(ctx, pd()->conf_.wk_size), reorder_kernel);
    });

    return status::success;
}

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl
