/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "gpu/sycl/ref_lrn.hpp"
#include "gpu/sycl/lrn_kernels.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

using namespace impl::sycl;
status_t ref_sycl_lrn_fwd_t::pd_t::init_conf() {
    conf_ = sycl_lrn_conf_t();
    conf_.src_md = sycl_md_t(src_md());
    conf_.dst_md = sycl_md_t(dst_md());
    conf_.alg_kind = desc()->alg_kind;
    conf_.block_size = 16;
    conf_.wg_size = 32;

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

status_t ref_sycl_lrn_fwd_t::init(engine_t *engine) {
    const auto kid = ::sycl::get_kernel_id<lrn_fwd_kernel_vec_t>();
    return create_kernel(engine, kid, &kernel_);
}

status_t ref_sycl_lrn_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
    using namespace alg_kind;
    using namespace format_tag;

    return parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        auto src_mem_arg = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC);
        auto dst_mem_arg = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DST);

        const auto block_size = pd()->conf_.block_size;
        const auto wg_size = pd()->conf_.wg_size;
        const auto t_work = pd()->conf_.wk_size;
        const auto wg_work = wg_size * block_size;
        const auto wg_cnt = (t_work + wg_work - 1) / wg_work;
        auto n_thr = wg_cnt * wg_size;
        n_thr = n_thr < 1 ? 1 : n_thr;
        const format_tag_t tag = pd()->dat_tag_;

        lrn_fwd_kernel_vec_t lrn_fwd_kernel_(
                pd()->conf_, src_mem_arg, dst_mem_arg, tag);

        cgh.parallel_for(::sycl::nd_range<1>(n_thr, wg_size), lrn_fwd_kernel_);
    });
}

status_t ref_sycl_lrn_bwd_t::pd_t::init_conf() {
    conf_ = sycl_lrn_conf_t();
    conf_.src_md = sycl_md_t(src_md());
    conf_.diff_dst_md = sycl_md_t(diff_dst_md());
    conf_.diff_src_md = sycl_md_t(diff_src_md());
    conf_.alg_kind = desc()->alg_kind;
    conf_.block_size = 16;
    conf_.wg_size = 32;

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

status_t ref_sycl_lrn_bwd_t::init(engine_t *engine) {
    const auto kid = ::sycl::get_kernel_id<lrn_bwd_kernel_vec_t>();
    return create_kernel(engine, kid, &kernel_);
}

status_t ref_sycl_lrn_bwd_t::execute_backward(const exec_ctx_t &ctx) const {
    return parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        auto diff_src_arg = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC);
        auto diff_dst_mem_arg = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_DST);
        auto diff_src_mem_arg = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_SRC);

        const format_tag_t tag = pd()->dat_tag_;

        lrn_bwd_kernel_vec_t lrn_bwd_kernel_(pd()->conf_, diff_src_arg,
                diff_dst_mem_arg, diff_src_mem_arg, tag);

        const int block_size = pd()->conf_.block_size;
        const int wg_size = pd()->conf_.wg_size;
        const int t_work = pd()->conf_.wk_size;
        int wg_work = wg_size * block_size;
        int wg_cnt = (t_work + wg_work - 1) / wg_work;
        int wg_thr = wg_cnt * wg_size;
        wg_thr = wg_thr < 1 ? 1 : wg_thr;
        cgh.parallel_for(::sycl::nd_range<1>(wg_thr, wg_size), lrn_bwd_kernel_);
    });
}

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl
