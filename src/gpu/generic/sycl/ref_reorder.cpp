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

    conf_.src_md = xpu::sycl::md_t(arg_md(DNNL_ARG_FROM));
    conf_.dst_md = xpu::sycl::md_t(arg_md(DNNL_ARG_TO));

    //get padded number of elements from source and destination
    auto dst_nelems = memory_desc_wrapper(arg_md(DNNL_ARG_TO)).nelems(true);

    // To cover cases when src has more padding than destination, in that case
    // simply setting the range as dst_nelems does not cover all the source elements
    conf_.num_elements = dst_nelems;

    conf_.do_scale_src = !attr()->scales_.has_default_values(DNNL_ARG_FROM);
    conf_.scale_src_mask = attr()->scales_.get_mask(DNNL_ARG_FROM);
    conf_.do_scale_dst = !attr()->scales_.has_default_values(DNNL_ARG_TO);
    conf_.scale_dst_mask = attr()->scales_.get_mask(DNNL_ARG_TO);
    conf_.apply_src_zp
            = !attr()->zero_points_.has_default_values(DNNL_ARG_FROM);
    conf_.src_zp_mask = attr()->zero_points_.get_mask(DNNL_ARG_FROM);
    conf_.apply_dst_zp = !attr()->zero_points_.has_default_values(DNNL_ARG_TO);
    conf_.dst_zp_mask = attr()->zero_points_.get_mask(DNNL_ARG_TO);
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

        cgh.parallel_for(
                get_range(ctx, pd()->conf_.num_elements), reorder_kernel);
    });

    return status::success;
}

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl
