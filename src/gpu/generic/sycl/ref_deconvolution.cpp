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

#include "gpu/generic/sycl/ref_deconvolution.hpp"
#include "gpu/generic/sycl/convolution_kernels.hpp"
#include "gpu/generic/sycl/sycl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

status_t ref_deconvolution_bwd_weights_t::pd_t::init_conf() {
    conf_ = sycl_convolution_bwd_weights_conf_t();

    conf_.diff_dst_md = xpu::sycl::md_t(src_md());
    if (with_bias()) {
        conf_.bias_dt = diff_weights_md(1)->data_type;
        conf_.has_bias = true;
    }
    conf_.data_md = xpu::sycl::md_t(diff_dst_md());
    conf_.ndims = ndims();

    memory_desc_t diff_weights_md_copy = *diff_weights_md(0);

    //IC and OC are the other way around compared to convolution
    bool no_groups = diff_weights_md(0)->ndims == diff_dst_md()->ndims;
    auto &strides = diff_weights_md_copy.format_desc.blocking.strides;

    auto recalc_strides_swap_dims = [&](int dim0, int dim1) {
        int bigger_stride_idx = strides[dim0] > strides[dim1] ? dim0 : dim1;
        int smaller_stride_idx = strides[dim0] > strides[dim1] ? dim1 : dim0;
        for (int i = 0; i < diff_weights_md(0)->ndims; i++) {
            if (strides[smaller_stride_idx] < strides[i]
                    && strides[i] < strides[bigger_stride_idx]) {
                strides[i] /= diff_weights_md_copy.dims[bigger_stride_idx];
                strides[i] *= diff_weights_md_copy.dims[smaller_stride_idx];
            }
        }
    };

    if (no_groups) {
        std::swap(strides[0], strides[1]);
        recalc_strides_swap_dims(0, 1);
        std::swap(diff_weights_md_copy.dims[0], diff_weights_md_copy.dims[1]);
    } else {
        std::swap(diff_weights_md_copy.dims[1], diff_weights_md_copy.dims[2]);
        recalc_strides_swap_dims(1, 2);
        std::swap(strides[1], strides[2]);
    }

    conf_.diff_weights_md = xpu::sycl::md_t(&diff_weights_md_copy);

    conf_.wk_size = memory_desc_wrapper(diff_weights_md()).nelems();

    conf_.padding[0] = static_cast<int>(desc()->padding[0][0]);
    conf_.padding[1] = static_cast<int>(desc()->padding[0][1]);
    conf_.padding[2] = static_cast<int>(desc()->padding[0][2]);

    conf_.strides[0] = static_cast<int>(desc()->strides[0]);
    conf_.strides[1] = static_cast<int>(desc()->strides[1]);
    conf_.strides[2] = static_cast<int>(desc()->strides[2]);

    conf_.dilation[0] = static_cast<int>(desc()->dilates[0]);
    conf_.dilation[1] = static_cast<int>(desc()->dilates[1]);
    conf_.dilation[2] = static_cast<int>(desc()->dilates[2]);
    conf_.is_deconvolution = true;
    return status::success;
}

status_t ref_deconvolution_bwd_weights_t::init(impl::engine_t *engine) {
    const auto kid = ::sycl::get_kernel_id<convolution_kernel_bwd_weights_t>();
    CHECK(create_kernel(engine, kid, &kernel_));
    return status::success;
}

status_t ref_deconvolution_bwd_weights_t::execute(const exec_ctx_t &ctx) const {

    parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        convolution_kernel_bwd_weights_t convolution_kernel(
                pd()->conf_, cgh, ctx, DNNL_ARG_DIFF_DST, DNNL_ARG_SRC);

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
