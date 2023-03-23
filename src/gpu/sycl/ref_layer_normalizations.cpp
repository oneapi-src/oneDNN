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
#include "gpu/sycl/ref_layer_normalizations.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_traits.hpp"
#include "gpu/sycl/layer_normalizations_kernels.hpp"
#include "gpu/sycl/sycl_types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

using namespace impl::sycl;

status_t ref_layer_normalization_fwd_t::pd_t::init_conf() {
    conf_ = sycl_layer_normalization_conf_t();

    conf_.var_md = stats_are_src() ? sycl_md_t(src_md(2))
            : is_training()        ? sycl_md_t(dst_md(2))
                                   : sycl_md_t {};
    conf_.ndims = ndims();
    conf_.flags = desc()->flags;
    conf_.wk_size = memory_desc_wrapper(src_md(0)).nelems();
    conf_.stat_d = sycl_md_t(stat_md());
    conf_.block_size = 16;
    conf_.wg_size = 32;

    conf_.rt_scaling = !attr()->scales_.defined();
    conf_.src_def = attr()->scales_.get(DNNL_ARG_SRC).has_default_values();
    conf_.dst_def = attr()->scales_.get(DNNL_ARG_DST).has_default_values();

    conf_.use_scale = use_scale();
    conf_.use_shift = use_shift();
    conf_.use_ss = conf_.use_scale || conf_.use_shift;
    conf_.data_md = sycl_md_t(src_md(0));
    conf_.data_scaleshift_md = sycl_md_t(weights_md(0));

    conf_.stat_md = stats_are_src() ? sycl_md_t(src_md(1))
            : is_training()         ? sycl_md_t(dst_md(2))
                                    : sycl_md_t {};
    conf_.dst_md = sycl_md_t(dst_md(0));
    conf_.shift_off = conf_.use_ss && !has_zero_dim_memory()
            ? conf_.data_scaleshift_md.off(1, 0)
            : 0;
    conf_.N = across_axis();
    conf_.C = norm_axis();
    conf_.layer_norm_epsilon = desc()->layer_norm_epsilon;
    conf_.save_stats = is_training();
    conf_.calculate_stats = !stats_are_src();
    conf_.zero_dims = has_zero_dim_memory();

    return status::success;
}

status_t ref_layer_normalization_fwd_t::init(engine_t *engine) {
    if (pd()->stats_are_src()) {
        const auto kid
                = ::sycl::get_kernel_id<layer_normalization_fwd_kernel_vec_t>();
        CHECK(create_kernel(engine, kid, &kernel_));
    } else {
        const auto kid = ::sycl::get_kernel_id<
                layer_normalization_fwd_kernel_vec1_t>();
        CHECK(create_kernel(engine, kid, &kernel_));
    }
    return status::success;
}

status_t ref_layer_normalization_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    if (pd()->stats_are_src()) {
        return parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
            auto data = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC);
            auto scale = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SCALE);
            auto shift = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SHIFT);
            auto mean = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_MEAN);
            auto src_scales = CTX_IN_SYCL_KERNEL_MEMORY(
                    DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
            auto dst_scales = CTX_IN_SYCL_KERNEL_MEMORY(
                    DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);
            auto var = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_VARIANCE);
            auto dst = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DST);
            layer_normalization_fwd_kernel_vec_t layer_normalization_fwd_kernel(
                    pd()->conf_, data, scale, shift, mean, var, dst, src_scales,
                    dst_scales);
            const int block_size = pd()->conf_.block_size;
            const int wg_size = pd()->conf_.wg_size;

            int work_per_wg = wg_size * block_size;
            int n_wgs = (pd()->across_axis() + work_per_wg - 1) / work_per_wg;
            int n_thr = n_wgs * wg_size;
            cgh.parallel_for(::sycl::nd_range<1>(n_thr, wg_size),
                    layer_normalization_fwd_kernel);
        });
    } else {
        return parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
            auto data = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC);
            auto scale = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SCALE);
            auto shift = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SHIFT);
            auto mean = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_MEAN);
            auto src_scales = CTX_IN_SYCL_KERNEL_MEMORY(
                    DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
            auto dst_scales = CTX_IN_SYCL_KERNEL_MEMORY(
                    DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);
            auto var = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_VARIANCE);
            auto dst = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DST);
            layer_normalization_fwd_kernel_vec1_t
                    layer_normalization_fwd_kernel1(pd()->conf_, data, scale,
                            shift, dst, mean, var, src_scales, dst_scales);

            const int block_size = pd()->conf_.block_size;
            const int wg_size = pd()->conf_.wg_size;

            int work_per_wg = wg_size * block_size;
            int n_wgs = (pd()->across_axis() + work_per_wg - 1) / work_per_wg;
            int n_thr = n_wgs * wg_size;
            cgh.parallel_for(::sycl::nd_range<1>(n_thr, wg_size),
                    layer_normalization_fwd_kernel1);
        });
    }

    return status::success;
}

status_t ref_layer_normalization_bwd_t::pd_t::init_conf() {
    conf_ = sycl_layer_normalization_conf_t();

    conf_.var_md = sycl_md_t(src_md(2));
    conf_.ndims = ndims();
    conf_.flags = desc()->flags;
    conf_.block_size = (16);
    conf_.wg_size = (32);
    conf_.prop_kind = desc_.prop_kind;
    conf_.use_scale = use_scale();
    conf_.use_shift = use_shift();
    conf_.use_ss = conf_.use_scale || conf_.use_shift;
    conf_.data_md = sycl_md_t(src_md(0));
    conf_.diff_data_md = sycl_md_t(diff_src_md(0));
    conf_.data_scaleshift_md = sycl_md_t(weights_md(0));
    conf_.diff_data_scaleshift_md
            = conf_.use_ss ? sycl_md_t(diff_weights_md(0)) : sycl_md_t {};
    conf_.stat_md = sycl_md_t(src_md(1));
    conf_.diff_dst_md = sycl_md_t(diff_dst_md(0));
    conf_.stat_d = sycl_md_t(stat_md());
    conf_.zero_dims = has_zero_dim_memory();
    auto nelems_A = memory_desc_wrapper(src_md(0)).nelems();
    conf_.diff_shift_off = conf_.use_ss && !conf_.zero_dims
            ? conf_.diff_data_scaleshift_md.off(1, 0)
            : 0;
    int work_per_wg = conf_.wg_size * conf_.block_size;
    int n_wgs = (nelems_A + work_per_wg - 1) / work_per_wg;
    conf_.n_thr = n_wgs * conf_.wg_size;
    conf_.zero_dims = has_zero_dim_memory();
    conf_.N = across_axis();
    conf_.C = norm_axis();
    conf_.layer_norm_epsilon = desc()->layer_norm_epsilon;
    conf_.save_stats = is_training();
    conf_.calculate_diff_stats = !use_global_stats();

    return status::success;
}

status_t ref_layer_normalization_bwd_t::init(engine_t *engine) {
    if (pd()->use_scale() || pd()->use_shift()) {
        const auto kid
                = ::sycl::get_kernel_id<layer_normalization_bwd_kernel_vec_t>();
        CHECK(create_kernel(engine, kid, &kernel_));
    }
    const auto kid2
            = ::sycl::get_kernel_id<layer_normalization_bwd_kernel_vec2_t>();
    return create_kernel(engine, kid2, &kernel2_);
}

status_t ref_layer_normalization_bwd_t::execute_backward(
        const exec_ctx_t &ctx) const {
    if (pd()->conf_.use_scale || pd()->conf_.use_shift) {
        auto status = parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
            auto data = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC);
            auto diff_data = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_SRC);
            auto scale = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SCALE);
            auto diff_scale = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_SCALE);
            auto diff_shift = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_SHIFT);
            auto mean = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_MEAN);
            auto var = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_VARIANCE);
            auto diff_dst = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_DST);
            auto nelems_A = memory_desc_wrapper(pd()->src_md(0)).nelems();
            const int block_size = pd()->conf_.block_size;
            const int wg_size = pd()->conf_.wg_size;
            int work_per_wg = wg_size * block_size;
            int n_wgs = (nelems_A + work_per_wg - 1) / work_per_wg;

            int n_thr = n_wgs * wg_size;
            layer_normalization_bwd_kernel_vec_t layer_normalization_bwd_kernel(
                    pd()->conf_, data, diff_data, scale, diff_scale, diff_shift,
                    mean, var, diff_dst);

            cgh.parallel_for(::sycl::nd_range<1>(n_thr, wg_size),
                    layer_normalization_bwd_kernel);
        });
        CHECK(status);
    }

    return parallel_for(ctx, kernel2_, [&](::sycl::handler &cgh) {
        auto data = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC);
        auto diff_data = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_SRC);
        auto scale = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SCALE);
        auto diff_scale = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_SCALE);
        auto diff_shift = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_SHIFT);
        auto mean = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_MEAN);
        auto var = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_VARIANCE);
        auto diff_dst = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_DST);
        auto nelems_A = memory_desc_wrapper(pd()->src_md(0)).nelems();
        const int block_size = pd()->conf_.block_size;
        const int wg_size = pd()->conf_.wg_size;
        int work_per_wg = wg_size * block_size;
        int n_wgs = (nelems_A + work_per_wg - 1) / work_per_wg;

        int n_thr = n_wgs * wg_size;
        layer_normalization_bwd_kernel_vec2_t layer_normalization_bwd_kernel2(
                pd()->conf_, data, diff_data, scale, diff_scale, diff_shift,
                mean, var, diff_dst);

        cgh.parallel_for(::sycl::nd_range<1>(n_thr, wg_size),
                layer_normalization_bwd_kernel2);
    });
}

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl
