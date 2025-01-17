/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#include "gpu/generic/sycl/ref_convolution.hpp"
#include "gpu/generic/sycl/convolution_kernels.hpp"
#include "gpu/generic/sycl/sycl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

status_t ref_convolution_fwd_t::pd_t::init_conf() {
    conf_ = sycl_convolution_fwd_conf_t();

    conf_.data_md = xpu::sycl::md_t(src_md());
    conf_.weights_md = xpu::sycl::md_t(weights_md(0));
    if (with_bias()) {
        conf_.bias_dt = weights_md(1)->data_type;
        conf_.has_bias = true;
    }
    conf_.dst_md = xpu::sycl::md_t(dst_md());
    conf_.ndims = ndims();

    conf_.wk_size = memory_desc_wrapper(dst_md()).nelems();

    conf_.do_scale_data = !attr()->scales_.has_default_values(DNNL_ARG_SRC_0);
    conf_.do_scale_weights
            = !attr()->scales_.has_default_values(DNNL_ARG_WEIGHTS);
    conf_.do_scale_dst = !attr()->scales_.has_default_values(DNNL_ARG_DST);
    conf_.single_weight_scale = attr()->scales_.get_mask(DNNL_ARG_WEIGHTS) == 0;

    conf_.use_data_zeropoints
            = !attr()->zero_points_.has_default_values(DNNL_ARG_SRC_0);
    conf_.use_dst_zeropoints
            = !attr()->zero_points_.has_default_values(DNNL_ARG_DST);
    conf_.single_data_zeropoint = attr()->zero_points_.common(DNNL_ARG_SRC_0);
    conf_.single_dst_zeropoint
            = attr()->zero_points_.get_mask(DNNL_ARG_DST) == 0;

    conf_.post_ops = sycl_post_ops_t(attr(), dst_md());

    conf_.padding[0] = static_cast<int>(desc()->padding[0][0]);
    conf_.padding[1] = static_cast<int>(desc()->padding[0][1]);
    conf_.padding[2] = static_cast<int>(desc()->padding[0][2]);

    conf_.strides[0] = static_cast<int>(desc()->strides[0]);
    conf_.strides[1] = static_cast<int>(desc()->strides[1]);
    conf_.strides[2] = static_cast<int>(desc()->strides[2]);

    conf_.dilation[0] = static_cast<int>(desc()->dilates[0]);
    conf_.dilation[1] = static_cast<int>(desc()->dilates[1]);
    conf_.dilation[2] = static_cast<int>(desc()->dilates[2]);
    return status::success;
}

status_t ref_convolution_fwd_t::init(impl::engine_t *engine) {
    const auto kid = ::sycl::get_kernel_id<convolution_kernel_fwd_t>();
    CHECK(create_kernel(engine, kid, &kernel_));
    return status::success;
}

status_t ref_convolution_fwd_t::execute(const exec_ctx_t &ctx) const {
    if (memory_desc_wrapper(pd()->dst_md()).size() == 0) return status::success;

    parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        convolution_kernel_fwd_t convolution_kernel(pd()->conf_, cgh, ctx);

        cgh.parallel_for(
                get_range(ctx, pd()->conf_.wk_size), convolution_kernel);
    });

    return status::success;
}

status_t ref_convolution_bwd_data_t::pd_t::init_conf() {
    conf_ = sycl_convolution_bwd_data_conf_t();

    conf_.diff_data_md = xpu::sycl::md_t(diff_src_md());
    conf_.weights_md = xpu::sycl::md_t(weights_md(0));
    if (with_bias()) {
        conf_.bias_dt = weights_md(1)->data_type;
        conf_.has_bias = true;
    }
    conf_.diff_dst_md = xpu::sycl::md_t(diff_dst_md());
    conf_.ndims = ndims();

    conf_.wk_size = memory_desc_wrapper(diff_src_md()).nelems();

    conf_.do_scale_data = !attr()->scales_.has_default_values(DNNL_ARG_SRC_0);
    conf_.do_scale_weights
            = !attr()->scales_.has_default_values(DNNL_ARG_WEIGHTS);
    conf_.do_scale_dst = !attr()->scales_.has_default_values(DNNL_ARG_DST);
    conf_.single_weight_scale = attr()->scales_.get_mask(DNNL_ARG_WEIGHTS) == 0;

    conf_.use_data_zeropoints
            = !attr()->zero_points_.has_default_values(DNNL_ARG_SRC_0);
    conf_.use_dst_zeropoints
            = !attr()->zero_points_.has_default_values(DNNL_ARG_DST);
    conf_.single_data_zeropoint = attr()->zero_points_.common(DNNL_ARG_SRC_0);
    conf_.single_dst_zeropoint
            = attr()->zero_points_.get_mask(DNNL_ARG_DST) == 0;

    conf_.post_ops = sycl_post_ops_t(attr(), diff_src_md());

    conf_.padding[0] = static_cast<int>(desc()->padding[0][0]);
    conf_.padding[1] = static_cast<int>(desc()->padding[0][1]);
    conf_.padding[2] = static_cast<int>(desc()->padding[0][2]);

    conf_.strides[0] = static_cast<int>(desc()->strides[0]);
    conf_.strides[1] = static_cast<int>(desc()->strides[1]);
    conf_.strides[2] = static_cast<int>(desc()->strides[2]);

    conf_.dilation[0] = static_cast<int>(desc()->dilates[0]);
    conf_.dilation[1] = static_cast<int>(desc()->dilates[1]);
    conf_.dilation[2] = static_cast<int>(desc()->dilates[2]);
    return status::success;
}

status_t ref_convolution_bwd_data_t::init(impl::engine_t *engine) {
    const auto kid = ::sycl::get_kernel_id<convolution_kernel_bwd_data_t>();
    CHECK(create_kernel(engine, kid, &kernel_));
    return status::success;
}

status_t ref_convolution_bwd_data_t::execute(const exec_ctx_t &ctx) const {
    if (memory_desc_wrapper(pd()->diff_src_md()).size() == 0)
        return status::success;

    parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        convolution_kernel_bwd_data_t convolution_kernel(pd()->conf_, cgh, ctx);

        cgh.parallel_for(
                get_range(ctx, pd()->conf_.wk_size), convolution_kernel);
    });

    return status::success;
}

status_t ref_convolution_bwd_weights_t::pd_t::init_conf() {
    conf_ = sycl_convolution_bwd_weights_conf_t();

    conf_.data_md = xpu::sycl::md_t(src_md());
    conf_.diff_weights_md = xpu::sycl::md_t(diff_weights_md(0));
    if (with_bias()) {
        conf_.bias_dt = diff_weights_md(1)->data_type;
        conf_.has_bias = true;
    }
    conf_.diff_dst_md = xpu::sycl::md_t(diff_dst_md());
    conf_.ndims = ndims();

    conf_.wk_size = memory_desc_wrapper(diff_weights_md()).nelems();

    conf_.do_scale_data = !attr()->scales_.has_default_values(DNNL_ARG_SRC_0);
    conf_.do_scale_weights
            = !attr()->scales_.has_default_values(DNNL_ARG_WEIGHTS);
    conf_.do_scale_dst = !attr()->scales_.has_default_values(DNNL_ARG_DST);
    conf_.single_weight_scale = attr()->scales_.get_mask(DNNL_ARG_WEIGHTS) == 0;

    conf_.use_data_zeropoints
            = !attr()->zero_points_.has_default_values(DNNL_ARG_SRC_0);
    conf_.use_dst_zeropoints
            = !attr()->zero_points_.has_default_values(DNNL_ARG_DST);
    conf_.single_data_zeropoint = attr()->zero_points_.common(DNNL_ARG_SRC_0);
    conf_.single_dst_zeropoint
            = attr()->zero_points_.get_mask(DNNL_ARG_DST) == 0;

    conf_.post_ops = sycl_post_ops_t(attr(), dst_md());

    conf_.padding[0] = static_cast<int>(desc()->padding[0][0]);
    conf_.padding[1] = static_cast<int>(desc()->padding[0][1]);
    conf_.padding[2] = static_cast<int>(desc()->padding[0][2]);

    conf_.strides[0] = static_cast<int>(desc()->strides[0]);
    conf_.strides[1] = static_cast<int>(desc()->strides[1]);
    conf_.strides[2] = static_cast<int>(desc()->strides[2]);

    conf_.dilation[0] = static_cast<int>(desc()->dilates[0]);
    conf_.dilation[1] = static_cast<int>(desc()->dilates[1]);
    conf_.dilation[2] = static_cast<int>(desc()->dilates[2]);
    return status::success;
}

status_t ref_convolution_bwd_weights_t::init(impl::engine_t *engine) {
    const auto kid = ::sycl::get_kernel_id<convolution_kernel_bwd_weights_t>();
    CHECK(create_kernel(engine, kid, &kernel_));
    return status::success;
}

status_t ref_convolution_bwd_weights_t::execute(const exec_ctx_t &ctx) const {

    parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        convolution_kernel_bwd_weights_t convolution_kernel(
                pd()->conf_, cgh, ctx, DNNL_ARG_SRC, DNNL_ARG_DIFF_DST);

        cgh.parallel_for(
                get_range(ctx, pd()->conf_.wk_size), convolution_kernel);
    });

    return status::success;
}

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl
