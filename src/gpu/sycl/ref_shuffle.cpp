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

#include "gpu/sycl/ref_shuffle.hpp"
#include "gpu/sycl/shuffle_kernels.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

using namespace impl::sycl;
using namespace impl::format_tag;
using namespace impl::utils;

status_t ref_shuffle_t::pd_t::init_conf() {
    conf_ = sycl_shuffle_conf_t();
    const memory_desc_wrapper data_d(src_md(0));

    conf_.axis = axis();
    conf_.axis_size = axis_size();
    conf_.ndims = ndims();
    auto dim = desc()->dst_desc.dims;
    conf_.ndims_d = desc()->dst_desc.ndims;
    conf_.outer_size = utils::array_product(dim, conf_.axis);
    conf_.inner_size = utils::array_product(
            dim + conf_.axis + 1, conf_.ndims_d - conf_.axis - 1);
    conf_.group_size = group_size();
    conf_.wg_size = 32;
    conf_.MB = MB();
    conf_.C = C();

    const bool has_spatial = utils::one_of(data_d.ndims(), 3, 4, 5);
    if (has_spatial) {
        conf_.H = H();
        conf_.W = W();
        conf_.D = D();
        conf_.HW = conf_.H * conf_.W;
        conf_.SP = conf_.D * conf_.HW;
    }
    conf_.stat_md = sycl_md_t(src_md(0));
    conf_.work_amount = memory_desc_wrapper(src_md()).nelems();
    conf_.src_md = sycl_md_t(src_md(0));
    conf_.dst_md = sycl_md_t(dst_md(0));

    if (ndims() == 5) {
        const auto tag
                = memory_desc_matches_one_of_tag(*data_d.md_, ncdhw, ndhwc);
        if (!tag) return status::unimplemented;
        conf_.tag = tag;
    } else if (ndims() == 4) {
        const auto tag
                = memory_desc_matches_one_of_tag(*data_d.md_, nchw, nhwc);
        if (!tag) return status::unimplemented;
        conf_.tag = tag;
    } else
        conf_.tag = any;

    conf_.stride_m = data_d.blocking_desc().strides[0];
    conf_.block_size = data_d.blocking_desc().strides[conf_.ndims - 1];

    conf_.transpose_row
            = is_fwd() ? conf_.group_size : conf_.axis_size / conf_.group_size;
    conf_.transpose_col
            = is_fwd() ? conf_.axis_size / conf_.group_size : conf_.group_size;

    int d2 = (conf_.block_size > conf_.C) ? 1 : conf_.block_size;

    const int t_work = (conf_.MB) * ((conf_.C / d2)) * conf_.SP;
    const int wg_work = conf_.wg_size * conf_.block_size;
    const int wg_cnt = (t_work + wg_work - 1) / wg_work;
    conf_.nthr = wg_cnt * conf_.wg_size;

    if (memory_desc_wrapper(src_md()).nelems() % conf_.block_size != 0)
        return status::unimplemented;

    return status::success;
}

status_t ref_shuffle_t::init(engine_t *engine) {
    if (pd()->conf_.axis == 1 && one_of(pd()->conf_.tag, nchw, ncdhw)) {
        const auto kid = ::sycl::get_kernel_id<shuffle_kernel_vec1_t>();
        CHECK(create_kernel(engine, kid, &kernel_));
    } else if (pd()->conf_.axis == 1 && one_of(pd()->conf_.tag, nhwc, ndhwc)) {
        const auto kid = ::sycl::get_kernel_id<shuffle_kernel_vec2_t>();
        CHECK(create_kernel(engine, kid, &kernel_));
    } else {
        const auto kid = ::sycl::get_kernel_id<shuffle_kernel_vec3_t>();
        CHECK(create_kernel(engine, kid, &kernel_));
    }
    return status::success;
}

status_t ref_shuffle_t::execute(const exec_ctx_t &ctx) const {
    return parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        auto src_mem_arg = pd()->is_fwd()
                ? CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC)
                : CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_DST);

        auto dst_mem_arg = pd()->is_fwd()
                ? CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DST)
                : CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_SRC);
        int axis = pd()->axis();

        shuffle_kernel_vec1_t shuffle_kernel1(
                pd()->conf_, src_mem_arg, dst_mem_arg);

        shuffle_kernel_vec2_t shuffle_kernel2(
                pd()->conf_, src_mem_arg, dst_mem_arg);

        shuffle_kernel_vec3_t shuffle_kernel3(
                pd()->conf_, src_mem_arg, dst_mem_arg);

        const int block_size = pd()->conf_.block_size;
        const int wg_size = pd()->conf_.wg_size;

        int d2 = (block_size > pd()->conf_.C) ? 1 : block_size;

        const int t_work
                = (pd()->conf_.MB) * ((pd()->conf_.C / d2)) * pd()->conf_.SP;

        std::cout << std::endl;
        const int wg_work = wg_size * block_size;
        const int wg_cnt = (t_work + wg_work - 1) / wg_work;
        const int n_thr = wg_cnt * wg_size;

        if (axis == 1 && one_of(pd()->conf_.tag, nchw, ncdhw)) {
            cgh.parallel_for(
                    ::sycl::nd_range<1>(n_thr, wg_size), shuffle_kernel1);
        } else if (axis == 1 && one_of(pd()->conf_.tag, nhwc, ndhwc)) {
            cgh.parallel_for(
                    ::sycl::nd_range<1>(n_thr, wg_size), shuffle_kernel2);
        } else {
            cgh.parallel_for(
                    ::sycl::nd_range<1>(n_thr, wg_size), shuffle_kernel3);
        }
    });
}

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl
