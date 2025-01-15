/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#include "gpu/generic/sycl/ref_softmax.hpp"
#include "gpu/generic/sycl/softmax_kernels.hpp"
#include "gpu/generic/sycl/sycl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

status_t ref_sycl_softmax_fwd_t::pd_t::init_conf() {
    conf_ = sycl_softmax_conf_t();
    conf_.src_md = xpu::sycl::md_t(src_md());
    conf_.dst_md = xpu::sycl::md_t(dst_md());
    conf_.alg_kind = desc()->alg_kind;
    conf_.axis = axis();
    conf_.axis_size = axis_size(true);
    conf_.inner_size = inner_size();
    conf_.outer_size = outer_size();
    conf_.channels = axis_size();
    conf_.wk_size = inner_size() * outer_size();

    conf_.do_scale_src = !attr()->scales_.has_default_values(DNNL_ARG_SRC);
    conf_.do_scale_dst = !attr()->scales_.has_default_values(DNNL_ARG_DST);

    conf_.post_ops = sycl_post_ops_t(attr(), dst_md());

    return status::success;
}

status_t ref_sycl_softmax_fwd_t::init(impl::engine_t *engine) {
    const auto kid = ::sycl::get_kernel_id<softmax_fwd_kernel_vec_t>();
    return create_kernel(engine, kid, &kernel_);
}

status_t ref_sycl_softmax_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
    if (pd()->has_zero_dim_memory()) return status::success;

    return parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        softmax_fwd_kernel_vec_t softmax_fwd_kernel_(pd()->conf_, cgh, ctx);

        cgh.parallel_for(
                get_range(ctx, pd()->conf_.wk_size), softmax_fwd_kernel_);
    });
}

status_t ref_sycl_softmax_bwd_t::pd_t::init_conf() {
    conf_ = sycl_softmax_conf_t();
    conf_.dst_md = xpu::sycl::md_t(dst_md());
    conf_.diff_dst_md = xpu::sycl::md_t(diff_dst_md());
    conf_.diff_src_md = xpu::sycl::md_t(diff_src_md());
    conf_.alg_kind = desc()->alg_kind;
    conf_.axis = axis();
    conf_.axis_size = axis_size(true);
    conf_.inner_size = inner_size();
    conf_.outer_size = outer_size();
    conf_.channels = axis_size();
    conf_.wk_size = inner_size() * outer_size();

    return status::success;
}

status_t ref_sycl_softmax_bwd_t::init(impl::engine_t *engine) {
    const auto kid = ::sycl::get_kernel_id<softmax_bwd_kernel_vec_t>();
    return create_kernel(engine, kid, &kernel_);
}

status_t ref_sycl_softmax_bwd_t::execute_backward(const exec_ctx_t &ctx) const {
    if (pd()->has_zero_dim_memory()) return status::success;

    return parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        softmax_bwd_kernel_vec_t softmax_bwd_kernel(pd()->conf_, cgh, ctx);

        cgh.parallel_for(
                get_range(ctx, pd()->conf_.wk_size), softmax_bwd_kernel);
    });
}

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl
