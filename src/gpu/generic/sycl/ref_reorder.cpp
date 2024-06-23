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

#include "gpu/generic/sycl/ref_reorder.hpp"
#include "gpu/generic/sycl/reorder_kernels.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

status_t ref_reorder_t::pd_t::init_conf() {
    conf_ = sycl_reorder_conf_t();

    conf_.src_md = xpu::sycl::md_t(src_md(0));
    conf_.dst_md = xpu::sycl::md_t(dst_md());

    // XXX: should probably be tuned.
    conf_.block_size = 16;
    conf_.wg_size = 32;

    conf_.wk_size = memory_desc_wrapper(src_md(0)).nelems();

    conf_.do_scale_src
            = !attr()->scales_.get(DNNL_ARG_SRC_0).has_default_values();
    conf_.scale_src_mask = attr()->scales_.get(DNNL_ARG_SRC_0).mask_;
    conf_.do_scale_dst
            = !attr()->scales_.get(DNNL_ARG_DST).has_default_values();
    conf_.scale_dst_mask = attr()->scales_.get(DNNL_ARG_DST).mask_;
    conf_.post_ops = sycl_post_ops_t(attr());

    return status::success;
}

status_t ref_reorder_t::init(impl::engine_t *engine) {
    const auto kid = ::sycl::get_kernel_id<reorder_kernel_t>();
    CHECK(create_kernel(engine, kid, &kernel_));
    return status::success;
}

status_t ref_reorder_t::execute(const exec_ctx_t &ctx) const {
    parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        auto src_mem_arg = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC_0);
        auto src_scale_mem_arg = CTX_IN_SYCL_KERNEL_MEMORY(
                DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0);
        auto dst_mem_arg = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DST);
        auto dst_scale_mem_arg = CTX_IN_SYCL_KERNEL_MEMORY(
                DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);

        auto scales_src_dt = (pd()->conf_.do_scale_src)
                ? ctx.memory_mdw(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0)
                          .data_type()
                : data_type_t::dnnl_f32;
        auto scales_dst_dt = (pd()->conf_.do_scale_dst)
                ? ctx.memory_mdw(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST)
                          .data_type()
                : data_type_t::dnnl_f32;

        reorder_kernel_t reorder_kernel(pd()->conf_, src_mem_arg, dst_mem_arg,
                src_scale_mem_arg, dst_scale_mem_arg, scales_src_dt,
                scales_dst_dt);

        const int block_size = pd()->conf_.block_size;
        const int wg_size = pd()->conf_.wg_size;

        const int t_work = pd()->conf_.wk_size;
        const int wg_work = wg_size * block_size;
        const int wg_cnt = utils::div_up(t_work, wg_work);

        cgh.parallel_for(
                ::sycl::nd_range<1>(wg_cnt * wg_size, wg_size), reorder_kernel);
    });

    return status::success;
}

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl
