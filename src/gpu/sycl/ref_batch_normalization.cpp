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

#include "common/c_types_map.hpp"
#include "common/dnnl_traits.hpp"
#include "xpu/sycl/types.hpp"

#include "gpu/sycl/batch_normalizations_kernels.hpp"
#include "gpu/sycl/ref_batch_normalization.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {
using namespace impl::sycl;

status_t ref_batch_normalization_fwd_t::pd_t::init_conf() {
    conf_ = sycl_batch_normalization_conf_t();

    conf_.ndims = ndims();
    conf_.flags = desc()->flags;
    conf_.wk_size = memory_desc_wrapper(src_md(0)).nelems();
    // Here and below only set the memory descriptors in `conf_` if they are used
    // by the current configuration. Setting them uncoditionally triggers the
    // assertion in xpu::sycl::md_t's constructor in Debug mode.
    if (fuse_norm_add_relu()) { conf_.src1_md = xpu::sycl::md_t(dst_md(3)); }
    conf_.block_size = 16;
    conf_.wg_size = 32;
    conf_.dir = !is_fwd();
    conf_.use_scale = use_scale();
    conf_.use_shift = use_shift();
    conf_.data_md = xpu::sycl::md_t(src_md(0));
    if (use_scale() || use_shift()) {
        conf_.data_scaleshift_md = xpu::sycl::md_t(weights_md(0));
    }
    if (is_training() || use_global_stats()) {
        conf_.stat_md = stats_is_src() ? xpu::sycl::md_t(src_md(1))
                                       : xpu::sycl::md_t(dst_md(1));
        conf_.var_md = stats_is_src() ? xpu::sycl::md_t(src_md(2))
                                      : xpu::sycl::md_t(dst_md(2));
    }
    conf_.dst_md = xpu::sycl::md_t(dst_md(0));
    if (is_training()) { conf_.ws_dt = workspace_md(0)->data_type; }
    int work_per_wg = conf_.wg_size * conf_.block_size;
    int n_wgs = (C() + work_per_wg - 1) / work_per_wg;
    conf_.n_thr = n_wgs * conf_.wg_size;
    conf_.N = MB();
    conf_.C = C();
    conf_.D = D();
    conf_.H = H();
    conf_.W = W();
    conf_.batch_norm_epsilon = desc()->batch_norm_epsilon;
    conf_.save_stats = is_training();
    conf_.calculate_stats = !stats_is_src();
    conf_.fuse_norm_relu = fuse_norm_relu();
    conf_.fuse_norm_add_relu = fuse_norm_add_relu();
    conf_.zero_dims = has_zero_dim_memory();
    conf_.is_training = is_training();
    conf_.with_relu = with_relu_post_op(is_training());
    if (conf_.fuse_norm_add_relu || conf_.fuse_norm_relu || conf_.with_relu) {
        conf_.alpha = alpha();
    }

    return status::success;
}

status_t ref_batch_normalization_fwd_t::init(impl::engine_t *engine) {
    if (pd()->stats_is_src()) {
        const auto kid
                = ::sycl::get_kernel_id<batch_normalization_fwd_kernel_vec_t>();
        CHECK(create_kernel(engine, kid, &kernel_));
    } else {
        const auto kid = ::sycl::get_kernel_id<
                batch_normalization_fwd_kernel_vec_t1>();
        CHECK(create_kernel(engine, kid, &kernel_));
    }
    return status::success;
}

status_t ref_batch_normalization_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {

    if (pd()->stats_is_src())
        return parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
            auto data = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC);
            auto scale = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SCALE); //added
            auto src1 = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC_1);
            auto shift = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SHIFT);
            auto mean = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_MEAN);
            auto var = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_VARIANCE);
            auto dst = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DST);
            auto ws = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_WORKSPACE);
            batch_normalization_fwd_kernel_vec_t batch_normalization_fwd_kernel(
                    pd()->conf_, data, scale, shift, mean, var, dst, ws, src1);
            const int block_size = pd()->conf_.block_size;
            const int wg_size = pd()->conf_.wg_size;
            int work_per_wg = wg_size * block_size;
            int n_wgs = (pd()->C() + work_per_wg - 1) / work_per_wg;
            int n_thr = n_wgs * wg_size;
            cgh.parallel_for(::sycl::nd_range<1>(n_thr, wg_size),
                    batch_normalization_fwd_kernel);
        });
    else
        return parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
            auto data = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC);
            auto scale = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SCALE); //added
            auto src1 = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC_1);
            auto shift = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SHIFT);
            auto mean = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_MEAN);
            auto var = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_VARIANCE);
            auto dst = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DST);
            auto ws = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_WORKSPACE);
            batch_normalization_fwd_kernel_vec_t1
                    batch_normalization_fwd_kernel1(pd()->conf_, data, scale,
                            shift, dst, mean, var, ws, src1);
            const int block_size = pd()->conf_.block_size;
            const int wg_size = pd()->conf_.wg_size;
            int work_per_wg = wg_size * block_size;
            int n_wgs = (pd()->C() + work_per_wg - 1) / work_per_wg;
            int n_thr = n_wgs * wg_size;
            cgh.parallel_for(::sycl::nd_range<1>(n_thr, wg_size),
                    batch_normalization_fwd_kernel1);
        });
}

status_t ref_batch_normalization_bwd_t::pd_t::init_conf() {
    conf_ = sycl_batch_normalization_conf_t();
    conf_.ndims = ndims();
    conf_.flags = desc()->flags;
    conf_.block_size = (16);
    conf_.wg_size = (32);
    conf_.prop_kind = desc_.prop_kind;
    conf_.use_scale = use_scale();
    conf_.use_shift = use_shift();
    conf_.data_md = xpu::sycl::md_t(src_md(0));
    conf_.diff_data_md = xpu::sycl::md_t(diff_src_md(0));
    if (fuse_norm_add_relu()) {
        conf_.diff_src1_dt = diff_dst_md(1)->data_type;
    }
    if (use_scale() || use_shift()) {
        conf_.data_scaleshift_md = xpu::sycl::md_t(weights_md(0));
        conf_.diff_data_scaleshift_md = xpu::sycl::md_t(diff_weights_md(0));
    }
    conf_.stat_md = xpu::sycl::md_t(stat_md());
    conf_.var_md = xpu::sycl::md_t(src_md(2));
    conf_.diff_dst_md = xpu::sycl::md_t(diff_dst_md(0));
    if (fuse_norm_add_relu() || fuse_norm_relu()) {
        conf_.ws_dt = workspace_md(0)->data_type;
    }
    int work_per_wg = conf_.wg_size * conf_.block_size;
    int n_wgs = (C() + work_per_wg - 1) / work_per_wg;
    conf_.n_thr = n_wgs * conf_.wg_size;
    conf_.zero_dims = has_zero_dim_memory();
    conf_.N = MB();
    conf_.C = C();
    conf_.D = D();
    conf_.H = H();
    conf_.W = W();

    conf_.batch_norm_epsilon = desc()->batch_norm_epsilon;
    conf_.fuse_norm_relu = fuse_norm_relu();
    conf_.fuse_norm_add_relu = fuse_norm_add_relu();
    conf_.calculate_diff_stats = !use_global_stats();
    if (fuse_norm_add_relu()) { conf_.alpha = alpha(); }

    return status::success;
}

status_t ref_batch_normalization_bwd_t::init(impl::engine_t *engine) {
    const auto kid
            = ::sycl::get_kernel_id<batch_normalization_bwd_kernel_vec_t>();
    return create_kernel(engine, kid, &kernel_);
}

status_t ref_batch_normalization_bwd_t::execute_backward(
        const exec_ctx_t &ctx) const {

    return parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        auto data = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC);
        auto diff_data = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_SRC);
        auto diff_src1 = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_SRC_1);
        auto scale = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SCALE);
        auto diff_scale
                = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_SCALE); //added
        auto diff_shift = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_SHIFT);
        auto mean = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_MEAN);
        auto var = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_VARIANCE);
        auto dst = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_DST);
        auto diff_dst = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_DST);
        auto ws = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_WORKSPACE);
        const int block_size = pd()->conf_.block_size;
        const int wg_size = pd()->conf_.wg_size;
        int work_per_wg = wg_size * block_size;
        int n_wgs = (pd()->C() + work_per_wg - 1) / work_per_wg;
        int n_thr = n_wgs * wg_size;
        batch_normalization_bwd_kernel_vec_t batch_normalization_bwd_kernel(
                pd()->conf_, data, diff_data, scale, diff_scale, diff_shift,
                mean, var, diff_dst, dst, ws, diff_src1);

        cgh.parallel_for(::sycl::nd_range<1>(n_thr, wg_size),
                batch_normalization_bwd_kernel);
    });
}

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl
