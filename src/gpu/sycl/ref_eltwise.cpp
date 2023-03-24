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

#include "gpu/sycl/ref_eltwise.hpp"
#include "gpu/sycl/eltwise_kernels.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

using namespace impl::sycl;
status_t ref_sycl_eltwise_fwd_t::pd_t::init_conf() {
    conf_ = sycl_eltwise_conf_t();
    conf_.src_md = sycl_md_t(src_md());
    conf_.dst_md = sycl_md_t(dst_md());
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

    conf_.post_po_len = attr()->post_ops_.len();
    conf_.post_ops = sycl_post_ops_t(attr());

    for (auto i = 0; i < conf_.post_po_len; ++i)
        conf_.binary_src_arr[i] = sycl_md_t(
                arg_md(DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1));

    const int block_size = conf_.block_size;
    const int wg_size = conf_.wg_size;
    const int t_work = conf_.wk_size;
    int wg_work = wg_size * block_size;
    int wg_cnt = (t_work + wg_work - 1) / wg_work;
    wg_thr = wg_cnt * wg_size;
    wg_thr = wg_thr < 1 ? 1 : wg_thr;

    return status::success;
}

status_t ref_sycl_eltwise_fwd_t::init(engine_t *engine) {
    const auto kid = ::sycl::get_kernel_id<eltwise_fwd_kernel_vec_t>();
    return create_kernel(engine, kid, &kernel_);
}

status_t ref_sycl_eltwise_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
    return parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        auto src_mem_arg = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC);
        auto dst_mem_arg = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DST);

        auto src_mem_po_1 = CTX_IN_SYCL_KERNEL_MEMORY(
                (DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1));
        auto src_mem_po_2 = CTX_IN_SYCL_KERNEL_MEMORY(
                (DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1));
        auto src_mem_po_3 = CTX_IN_SYCL_KERNEL_MEMORY(
                (DNNL_ARG_ATTR_MULTIPLE_POST_OP(2) | DNNL_ARG_SRC_1));
        auto src_mem_po_4 = CTX_IN_SYCL_KERNEL_MEMORY(
                (DNNL_ARG_ATTR_MULTIPLE_POST_OP(3) | DNNL_ARG_SRC_1));
        auto src_mem_po_5 = CTX_IN_SYCL_KERNEL_MEMORY(
                (DNNL_ARG_ATTR_MULTIPLE_POST_OP(4) | DNNL_ARG_SRC_1));

        eltwise_fwd_kernel_vec_t eltwise_fwd_kernel_(pd()->conf_, src_mem_arg,
                dst_mem_arg, src_mem_po_1, src_mem_po_2, src_mem_po_3,
                src_mem_po_4, src_mem_po_5);

        cgh.parallel_for(::sycl::nd_range<1>(pd()->wg_thr, pd()->conf_.wg_size),
                eltwise_fwd_kernel_);
    });
}

status_t ref_sycl_eltwise_bwd_t::pd_t::init_conf() {
    conf_ = sycl_eltwise_conf_t();
    conf_.src_md = sycl_md_t(data_md(0));
    conf_.diff_src_md = sycl_md_t(diff_src_md());
    conf_.diff_dst_md = sycl_md_t(diff_dst_md());
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

status_t ref_sycl_eltwise_bwd_t::init(engine_t *engine) {
    const auto kid = ::sycl::get_kernel_id<eltwise_bwd_kernel_vec_t>();
    return create_kernel(engine, kid, &kernel_);
}

status_t ref_sycl_eltwise_bwd_t::execute_backward(const exec_ctx_t &ctx) const {
    return parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        auto src_mem_arg = pd()->use_dst()
                ? CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_DST)
                : CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC);

        auto diff_dst_mem_arg = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_DST);
        auto diff_src_mem_arg = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_SRC);

        eltwise_bwd_kernel_vec_t eltwise_bwd_kernel_(
                pd()->conf_, diff_dst_mem_arg, src_mem_arg, diff_src_mem_arg);

        cgh.parallel_for(::sycl::nd_range<1>(pd()->wg_thr, pd()->conf_.wg_size),
                eltwise_bwd_kernel_);
    });
}

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl
