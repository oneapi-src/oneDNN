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

#include "gpu/generic/sycl/ref_eltwise.hpp"
#include "gpu/generic/sycl/eltwise_kernels.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

status_t ref_sycl_eltwise_fwd_t::pd_t::init_conf() {
    conf_ = sycl_eltwise_conf_t();
    conf_.src_md = xpu::sycl::md_t(src_md());
    conf_.dst_md = xpu::sycl::md_t(dst_md());
    conf_.wk_size = memory_desc_wrapper(src_md()).nelems();
    conf_.alg_kind = desc()->alg_kind;
    conf_.alpha = desc()->alpha;
    conf_.beta = desc()->beta;
    conf_.block_size = 16;
    conf_.wg_size = 32;
    conf_.mb = MB();
    conf_.c = C();
    conf_.d = D();
    conf_.h = H();
    conf_.w = W();

    if (attr()->post_ops_.len() > sycl_post_ops_t::max_post_ops) {
        return status::unimplemented;
    }
    conf_.post_po_len = attr()->post_ops_.len();
    conf_.post_ops = sycl_post_ops_t(attr(), dst_md()->data_type);

    const int block_size = conf_.block_size;
    const int wg_size = conf_.wg_size;
    const int t_work = conf_.wk_size;
    int wg_work = wg_size * block_size;
    int wg_cnt = (t_work + wg_work - 1) / wg_work;
    wg_thr = wg_cnt * wg_size;
    wg_thr = wg_thr < 1 ? 1 : wg_thr;

    return status::success;
}

status_t ref_sycl_eltwise_fwd_t::init(impl::engine_t *engine) {
    const auto kid = ::sycl::get_kernel_id<eltwise_fwd_kernel_vec_t>();
    return create_kernel(engine, kid, &kernel_);
}

status_t ref_sycl_eltwise_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
    return parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        eltwise_fwd_kernel_vec_t eltwise_fwd_kernel_(pd()->conf_, cgh, ctx);

        cgh.parallel_for(::sycl::nd_range<1>(pd()->wg_thr, pd()->conf_.wg_size),
                eltwise_fwd_kernel_);
    });
}

status_t ref_sycl_eltwise_bwd_t::pd_t::init_conf() {
    conf_ = sycl_eltwise_conf_t();
    conf_.src_md = xpu::sycl::md_t(data_md(0));
    conf_.diff_src_md = xpu::sycl::md_t(diff_src_md());
    conf_.diff_dst_md = xpu::sycl::md_t(diff_dst_md());
    conf_.block_size = 16;
    conf_.wg_size = 32;
    conf_.wk_size = memory_desc_wrapper(data_md(0)).nelems();
    conf_.alg_kind = desc()->alg_kind;
    conf_.alpha = desc()->alpha;
    conf_.beta = desc()->beta;
    conf_.mb = MB();
    conf_.c = C();
    conf_.d = D();
    conf_.h = H();
    conf_.w = W();

    const int block_size = conf_.block_size;
    const int wg_size = conf_.wg_size;
    const int t_work = conf_.wk_size;
    int wg_work = wg_size * block_size;
    int wg_cnt = (t_work + wg_work - 1) / wg_work;
    wg_thr = wg_cnt * wg_size;
    wg_thr = wg_thr < 1 ? 1 : wg_thr;

    return status::success;
}

status_t ref_sycl_eltwise_bwd_t::init(impl::engine_t *engine) {
    const auto kid = ::sycl::get_kernel_id<eltwise_bwd_kernel_vec_t>();
    return create_kernel(engine, kid, &kernel_);
}

status_t ref_sycl_eltwise_bwd_t::execute_backward(const exec_ctx_t &ctx) const {
    return parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        eltwise_bwd_kernel_vec_t eltwise_bwd_kernel_(
                pd()->conf_, cgh, ctx, pd()->use_dst());

        cgh.parallel_for(::sycl::nd_range<1>(pd()->wg_thr, pd()->conf_.wg_size),
                eltwise_bwd_kernel_);
    });
}

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl
