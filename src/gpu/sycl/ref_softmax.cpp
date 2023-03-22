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

#include "gpu/sycl/ref_softmax.hpp"
#include "gpu/sycl/softmax_kernels.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

using namespace impl::sycl;
status_t ref_sycl_softmax_fwd_t::pd_t::init_conf() {
    conf_ = sycl_softmax_conf_t();
    conf_.src_md = sycl_md_t(src_md());
    conf_.dst_md = sycl_md_t(dst_md());
    conf_.alg_kind = desc()->alg_kind;
    conf_.block_size = 16;
    conf_.wg_size = 32;
    conf_.axis = axis();
    conf_.axis_size = axis_size(true);
    conf_.inner_size = inner_size();
    conf_.outer_size = outer_size();
    conf_.channels = axis_size();
    conf_.wk_size = inner_size() * outer_size();

    return status::success;
}

status_t ref_sycl_softmax_fwd_t::init(engine_t *engine) {
    const auto kid = ::sycl::get_kernel_id<softmax_fwd_kernel_vec_t>();
    return create_kernel(engine, kid, &kernel_);
}

status_t ref_sycl_softmax_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
    return parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        auto src_mem_arg = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC);
        auto dst_mem_arg = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DST);

        float scale_val = 1.00;
        if (!pd()->attr()->scales_.defined()) {
            const auto scales_src
                    = ctx.memory_mdw(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
            auto src_scales = CTX_IN_SYCL_KERNEL_MEMORY(
                    DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
            if (src_scales.get_pointer() != NULL)
                scale_val = load_float_value(
                        scales_src.data_type(), src_scales.get_pointer(), 0);

            const auto scales_dst
                    = ctx.memory_mdw(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);
            auto dst_scales = CTX_IN_SYCL_KERNEL_MEMORY(
                    DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);
            if (dst_scales.get_pointer() != NULL)
                scale_val /= load_float_value(
                        scales_dst.data_type(), dst_scales.get_pointer(), 0);
        }

        const auto block_size = pd()->conf_.block_size;
        const auto wg_size = pd()->conf_.wg_size;
        const auto t_work = pd()->conf_.wk_size;
        const auto wg_work = wg_size * block_size;
        const auto wg_cnt = (t_work + wg_work - 1) / wg_work;
        auto n_thr = wg_cnt * wg_size;
        n_thr = n_thr < 1 ? 1 : n_thr;

        softmax_fwd_kernel_vec_t softmax_fwd_kernel_(
                pd()->conf_, src_mem_arg, dst_mem_arg, scale_val);

        cgh.parallel_for(
                ::sycl::nd_range<1>(n_thr, wg_size), softmax_fwd_kernel_);
    });
}

status_t ref_sycl_softmax_bwd_t::pd_t::init_conf() {
    conf_ = sycl_softmax_conf_t();
    conf_.dst_md = sycl_md_t(dst_md());
    conf_.diff_dst_md = sycl_md_t(diff_dst_md());
    conf_.diff_src_md = sycl_md_t(diff_src_md());
    conf_.alg_kind = desc()->alg_kind;
    conf_.block_size = 16;
    conf_.wg_size = 32;
    conf_.axis = axis();
    conf_.axis_size = axis_size(true);
    conf_.inner_size = inner_size();
    conf_.outer_size = outer_size();
    conf_.channels = axis_size();
    conf_.wk_size = inner_size() * outer_size();

    return status::success;
}

status_t ref_sycl_softmax_bwd_t::init(engine_t *engine) {
    const auto kid = ::sycl::get_kernel_id<softmax_bwd_kernel_vec_t>();
    return create_kernel(engine, kid, &kernel_);
}

status_t ref_sycl_softmax_bwd_t::execute_backward(const exec_ctx_t &ctx) const {
    return parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        auto dst = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_DST);
        auto diff_dst = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_DST);
        auto diff_src = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_SRC);

        softmax_bwd_kernel_vec_t softmax_bwd_kernel(
                pd()->conf_, dst, diff_dst, diff_src);

        const auto block_size = pd()->conf_.block_size;
        const auto wg_size = pd()->conf_.wg_size;
        const auto t_work = pd()->conf_.wk_size;
        const auto wg_work = wg_size * block_size;
        const auto wg_cnt = (t_work + wg_work - 1) / wg_work;
        auto n_thr = wg_cnt * wg_size;
        n_thr = n_thr < 1 ? 1 : n_thr;

        cgh.parallel_for(
                ::sycl::nd_range<1>(n_thr, wg_size), softmax_bwd_kernel);
    });
}

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl
