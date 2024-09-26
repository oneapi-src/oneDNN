/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#include "gpu/generic/sycl/ref_lrn.hpp"
#include "gpu/generic/sycl/lrn_kernels.hpp"
#include "gpu/generic/sycl/sycl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

status_t ref_sycl_lrn_fwd_t::pd_t::init_conf() {
    conf_ = sycl_lrn_conf_t();
    conf_.src_md = xpu::sycl::md_t(src_md());
    conf_.dst_md = xpu::sycl::md_t(dst_md());
    conf_.alg_kind = desc()->alg_kind;

    const memory_desc_wrapper data_d(src_md());
    conf_.mb = MB();
    conf_.c = C();
    conf_.d = D();
    conf_.h = H();
    conf_.w = W();
    conf_.stride_mb = data_d.blocking_desc().strides[0];
    conf_.ndims = data_d.ndims();
    conf_.wk_size = data_d.nelems();
    conf_.alpha = static_cast<float>(desc()->lrn_alpha);
    conf_.beta = static_cast<float>(desc()->lrn_beta);
    conf_.k = static_cast<float>(desc()->lrn_k);
    conf_.size = desc()->local_size;

    if (desc()->alg_kind == alg_kind::lrn_across_channels) {
        conf_.compute_n_summands = conf_.size;
    } else { // within_channel
        dim_t n_summands = 1;
        for (auto d = conf_.ndims - 2; d > 0; --d)
            n_summands *= conf_.size;
        conf_.compute_n_summands = n_summands;
    }
    return status::success;
}

status_t ref_sycl_lrn_fwd_t::init(impl::engine_t *engine) {
    const auto kid = ::sycl::get_kernel_id<lrn_fwd_kernel_vec_t>();
    return create_kernel(engine, kid, &kernel_);
}

status_t ref_sycl_lrn_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
    using namespace alg_kind;
    using namespace format_tag;

    return parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        const format_tag_t tag = pd()->dat_tag_;

        lrn_fwd_kernel_vec_t lrn_fwd_kernel_(pd()->conf_, cgh, ctx, tag);

        cgh.parallel_for(get_range(ctx, pd()->conf_.wk_size), lrn_fwd_kernel_);
    });
}

status_t ref_sycl_lrn_bwd_t::pd_t::init_conf() {
    conf_ = sycl_lrn_conf_t();
    conf_.src_md = xpu::sycl::md_t(src_md());
    conf_.diff_dst_md = xpu::sycl::md_t(diff_dst_md());
    conf_.diff_src_md = xpu::sycl::md_t(diff_src_md());
    conf_.alg_kind = desc()->alg_kind;

    const memory_desc_wrapper data_d(src_md());
    conf_.mb = MB();
    conf_.c = C();
    conf_.d = D();
    conf_.h = H();
    conf_.w = W();
    conf_.stride_mb = data_d.blocking_desc().strides[0];
    conf_.ndims = data_d.ndims();
    conf_.wk_size = data_d.nelems();
    conf_.alpha = static_cast<float>(desc()->lrn_alpha);
    conf_.beta = static_cast<float>(desc()->lrn_beta);
    conf_.k = static_cast<float>(desc()->lrn_k);
    conf_.size = desc()->local_size;
    if (desc()->alg_kind == alg_kind::lrn_across_channels) {
        conf_.compute_n_summands = conf_.size;
    } else { // within_channel
        dim_t n_summands = 1;
        for (auto d = conf_.ndims - 2; d > 0; --d)
            n_summands *= conf_.size;
        conf_.compute_n_summands = n_summands;
    }

    return status::success;
}

status_t ref_sycl_lrn_bwd_t::init(impl::engine_t *engine) {
    const auto kid = ::sycl::get_kernel_id<lrn_bwd_kernel_vec_t>();
    return create_kernel(engine, kid, &kernel_);
}

status_t ref_sycl_lrn_bwd_t::execute_backward(const exec_ctx_t &ctx) const {
    return parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        const format_tag_t tag = pd()->dat_tag_;
        lrn_bwd_kernel_vec_t lrn_bwd_kernel_(pd()->conf_, cgh, ctx, tag);

        cgh.parallel_for(get_range(ctx, pd()->conf_.wk_size), lrn_bwd_kernel_);
    });
}

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl
