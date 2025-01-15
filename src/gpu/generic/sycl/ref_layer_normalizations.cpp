/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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
#include "gpu/generic/sycl/ref_layer_normalizations.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_traits.hpp"
#include "gpu/generic/sycl/layer_normalizations_kernels.hpp"
#include "gpu/generic/sycl/sycl_utils.hpp"
#include "xpu/sycl/types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

status_t ref_layer_normalization_fwd_t::pd_t::init_conf() {
    conf_ = sycl_layer_normalization_conf_t();

    if (stats_are_src()) {
        conf_.var_dt = src_md(2)->data_type;
    } else if (is_training()) {
        conf_.var_dt = dst_md(2)->data_type;
    }

    conf_.ndims = ndims();
    conf_.flags = desc()->flags;
    conf_.wk_size = memory_desc_wrapper(src_md(0)).nelems();
    conf_.block_size = 16;

    conf_.rt_scaling = !attr()->scales_.has_default_values();
    conf_.src_def = attr()->scales_.has_default_values(DNNL_ARG_SRC);
    conf_.dst_def = attr()->scales_.has_default_values(DNNL_ARG_DST);

    conf_.scales_src_dt = attr()->scales_.get_data_type(DNNL_ARG_SRC);
    conf_.scales_dst_dt = attr()->scales_.get_data_type(DNNL_ARG_DST);

    conf_.use_scale = use_scale();
    conf_.use_shift = use_shift();
    conf_.use_ss = conf_.use_scale || conf_.use_shift;
    conf_.data_md = xpu::sycl::md_t(src_md(0));
    if (conf_.use_ss) {
        conf_.data_scaleshift_md = xpu::sycl::md_t(weights_md(0));
    }

    conf_.stat_md = stats_are_src() ? xpu::sycl::md_t(src_md(1))
            : is_training()         ? xpu::sycl::md_t(dst_md(2))
                                    : xpu::sycl::md_t {};
    conf_.dst_md = xpu::sycl::md_t(dst_md(0));
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

status_t ref_layer_normalization_fwd_t::init(impl::engine_t *engine) {
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
    if (memory_desc_wrapper(pd()->dst_md()).size() == 0) return status::success;

    if (pd()->stats_are_src()) {
        return parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
            layer_normalization_fwd_kernel_vec_t layer_normalization_fwd_kernel(
                    pd()->conf_, cgh, ctx);

            cgh.parallel_for(get_range(ctx, pd()->conf_.wk_size),
                    layer_normalization_fwd_kernel);
        });
    } else {
        return parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
            layer_normalization_fwd_kernel_vec1_t
                    layer_normalization_fwd_kernel1(pd()->conf_, cgh, ctx);

            cgh.parallel_for(get_range(ctx, pd()->conf_.wk_size),
                    layer_normalization_fwd_kernel1);
        });
    }

    return status::success;
}

status_t ref_layer_normalization_bwd_t::pd_t::init_conf() {
    conf_ = sycl_layer_normalization_conf_t();

    conf_.var_dt = src_md(2)->data_type;
    conf_.ndims = ndims();
    conf_.flags = desc()->flags;
    conf_.block_size = (16);
    conf_.wg_size = (32);
    conf_.prop_kind = desc_.prop_kind;
    conf_.use_scale = use_scale();
    conf_.use_shift = use_shift();
    conf_.use_ss = conf_.use_scale || conf_.use_shift;
    conf_.data_md = xpu::sycl::md_t(src_md(0));
    conf_.diff_data_md = xpu::sycl::md_t(diff_src_md(0));
    if (conf_.use_ss) {
        conf_.data_scaleshift_md = xpu::sycl::md_t(weights_md(0));
        conf_.diff_data_scaleshift_md = xpu::sycl::md_t(diff_weights_md(0));
    }
    conf_.stat_md = xpu::sycl::md_t(src_md(1));
    conf_.diff_dst_md = xpu::sycl::md_t(diff_dst_md(0));
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

status_t ref_layer_normalization_bwd_t::init(impl::engine_t *engine) {
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
    if (memory_desc_wrapper(pd()->diff_src_md()).size() == 0)
        return status::success;

    if (pd()->conf_.use_scale || pd()->conf_.use_shift) {
        auto status = parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
            auto nelems_A = memory_desc_wrapper(pd()->src_md(0)).nelems();
            const int block_size = pd()->conf_.block_size;
            const int wg_size = pd()->conf_.wg_size;
            int work_per_wg = wg_size * block_size;
            int n_wgs = (nelems_A + work_per_wg - 1) / work_per_wg;

            int n_thr = n_wgs * wg_size;
            layer_normalization_bwd_kernel_vec_t layer_normalization_bwd_kernel(
                    pd()->conf_, cgh, ctx);

            cgh.parallel_for(::sycl::nd_range<1>(n_thr, wg_size),
                    layer_normalization_bwd_kernel);
        });
        CHECK(status);
    }

    return parallel_for(ctx, kernel2_, [&](::sycl::handler &cgh) {
        auto nelems_A = memory_desc_wrapper(pd()->src_md(0)).nelems();
        const int block_size = pd()->conf_.block_size;
        const int wg_size = pd()->conf_.wg_size;
        int work_per_wg = wg_size * block_size;
        int n_wgs = (nelems_A + work_per_wg - 1) / work_per_wg;

        int n_thr = n_wgs * wg_size;
        layer_normalization_bwd_kernel_vec2_t layer_normalization_bwd_kernel2(
                pd()->conf_, cgh, ctx);

        cgh.parallel_for(::sycl::nd_range<1>(n_thr, wg_size),
                layer_normalization_bwd_kernel2);
    });
}

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl
