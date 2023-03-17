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

#include "gpu/sycl/ref_prelu.hpp"
#include "common/utils.hpp"
#include "gpu/sycl/prelu_kernels.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

using namespace impl::sycl;

status_t ref_prelu_fwd_t::pd_t::init_conf() {
    if (has_zero_dim_memory()) return status::success;

    conf_ = sycl_prelu_conf_t();

    const memory_desc_wrapper data_d(src_md(0));
    const memory_desc_wrapper weights_d(weights_md(0));
    conf_.data_md = sycl_md_t(src_md(0));
    conf_.weights_md = sycl_md_t(weights_md(0));
    conf_.dst_md = sycl_md_t(dst_md(0));
    conf_.ndims = ndims();
    conf_.mask = utils::get_dims_mask(data_d.dims(), weights_d.dims(), ndims());

    conf_.block_size = 16;
    conf_.wg_size = 32;
    conf_.work_amount = memory_desc_wrapper(src_md(0)).nelems();
    conf_.work_amount_wei = memory_desc_wrapper(weights_md(0)).nelems();
    int work_per_wg = conf_.wg_size * conf_.block_size;
    int n_wgs = (conf_.work_amount + work_per_wg - 1) / work_per_wg;
    conf_.n_thr = n_wgs * conf_.wg_size;

    return status::success;
}

status_t ref_prelu_fwd_t::init(engine_t *engine) {
    const auto kid = ::sycl::get_kernel_id<prelu_fwd_kernel_vec_t>();
    return create_kernel(engine, kid, &kernel_);
}

status_t ref_prelu_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
    if (pd()->has_zero_dim_memory()) return status::success;

    return parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        auto data = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC);
        auto weights = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_WEIGHTS);
        auto dst = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DST);
        auto nelems_A = memory_desc_wrapper(pd()->src_md(0)).nelems();
        int tot_work = nelems_A;
        prelu_fwd_kernel_vec_t prelu_fwd_kernel(
                pd()->conf_, data, weights, dst);
        const int block_size = pd()->conf_.block_size;
        const int wg_size = pd()->conf_.wg_size;
        int work_per_wg = wg_size * block_size;
        int n_wgs = (tot_work + work_per_wg - 1) / work_per_wg;
        int n_thr = n_wgs * wg_size;
        cgh.parallel_for(::sycl::nd_range<1>(n_thr, wg_size), prelu_fwd_kernel);
    });
}

status_t ref_prelu_bwd_t::pd_t::init_conf() {
    if (has_zero_dim_memory()) return status::success;
    conf_ = sycl_prelu_conf_t();
    conf_.data_md = sycl_md_t(src_md(0));
    conf_.weights_md = sycl_md_t(weights_md(0));
    conf_.diff_data_md = sycl_md_t(diff_src_md(0));
    conf_.diff_weights_md = sycl_md_t(diff_weights_md(0));
    conf_.diff_dst_md = sycl_md_t(diff_dst_md(0));
    conf_.ndims = ndims();

    const memory_desc_wrapper weights_d(weights_md(0));
    const memory_desc_wrapper data_d(src_md(0));
    conf_.bcast_type = dnnl::impl::get_rhs_arg_broadcasting_strategy(
            *weights_d.md_, data_d);
    conf_.mask = utils::get_dims_mask(data_d.dims(), weights_d.dims(), ndims());
    conf_.block_size = 16;
    conf_.wg_size = 32;
    conf_.work_amount_src = memory_desc_wrapper(src_md(0)).nelems();
    conf_.work_amount = memory_desc_wrapper(weights_md(0)).nelems();
    conf_.work_load = conf_.work_amount_src / conf_.work_amount;
    int work_per_wg = conf_.wg_size * conf_.block_size;
    int n_wgs = (conf_.work_amount_src + work_per_wg - 1) / work_per_wg;
    conf_.n_thr = n_wgs * conf_.wg_size;

    return status::success;
}

status_t ref_prelu_bwd_t::init(engine_t *engine) {
    const auto kid = ::sycl::get_kernel_id<prelu_bwd_kernel_vec_t>();
    return create_kernel(engine, kid, &kernel_);
}

status_t ref_prelu_bwd_t::execute_backward(const exec_ctx_t &ctx) const {
    if (pd()->has_zero_dim_memory()) return status::success;

    return parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        auto data = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC);
        auto weights = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_WEIGHTS);
        auto diff_data = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_SRC);
        auto diff_weights = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_WEIGHTS);
        auto diff_dst = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_DST);
        auto scratchpad = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_SCRATCHPAD);
        auto nelems_A = memory_desc_wrapper(pd()->src_md(0)).nelems();
        int tot_work = nelems_A;

        prelu_bwd_kernel_vec_t prelu_bwd_kernel(pd()->conf_, data, diff_data,
                weights, diff_weights, diff_dst, scratchpad);
        const int block_size = pd()->conf_.block_size;
        const int wg_size = pd()->conf_.wg_size;
        int work_per_wg = wg_size * block_size;
        int n_wgs = (tot_work + work_per_wg - 1) / work_per_wg;
        int n_thr = n_wgs * wg_size;
        cgh.parallel_for(::sycl::nd_range<1>(n_thr, wg_size), prelu_bwd_kernel);
    });
}

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl
