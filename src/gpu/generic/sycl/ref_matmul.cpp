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

#include "gpu/generic/sycl/ref_matmul.hpp"
#include "gpu/generic/sycl/matmul_kernels.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

status_t ref_matmul_t::pd_t::init_conf() {
    conf_ = sycl_matmul_conf_t();

    int matmul_dim_1 = ndims() - 2;
    int matmul_dim_2 = ndims() - 1;

    memory_desc_t data_md_copy = *src_md();
    auto &data_strides = data_md_copy.format_desc.blocking.strides;
    if (data_strides[matmul_dim_1] < data_strides[matmul_dim_2]) {
        std::swap(data_strides[matmul_dim_1], data_strides[matmul_dim_2]);
        std::swap(data_md_copy.dims[matmul_dim_1],
                data_md_copy.dims[matmul_dim_2]);
        conf_.transpose_data = true;
    }
    conf_.data_md = xpu::sycl::md_t(&data_md_copy);

    memory_desc_t weights_md_copy = *weights_md();
    auto &weights_strides = weights_md_copy.format_desc.blocking.strides;
    if (weights_strides[matmul_dim_1] < weights_strides[matmul_dim_2]) {
        std::swap(weights_strides[matmul_dim_1], weights_strides[matmul_dim_2]);
        std::swap(weights_md_copy.dims[matmul_dim_1],
                weights_md_copy.dims[matmul_dim_2]);
        conf_.transpose_weights = true;
    }
    conf_.weights_md = xpu::sycl::md_t(&weights_md_copy);

    memory_desc_t dst_md_copy = *dst_md();
    auto &dst_strides = dst_md_copy.format_desc.blocking.strides;
    if (dst_strides[matmul_dim_1] < dst_strides[matmul_dim_2]) {
        std::swap(dst_strides[matmul_dim_1], dst_strides[matmul_dim_2]);
        std::swap(
                dst_md_copy.dims[matmul_dim_1], dst_md_copy.dims[matmul_dim_2]);
        conf_.transpose_dst = true;
    }
    conf_.dst_md = xpu::sycl::md_t(&dst_md_copy);

    if (with_bias()) {
        memory_desc_t bias_md_copy = *weights_md(1);
        auto &bias_strides = bias_md_copy.format_desc.blocking.strides;
        if (bias_strides[matmul_dim_1] < bias_strides[matmul_dim_2]) {
            std::swap(bias_strides[matmul_dim_1], bias_strides[matmul_dim_2]);
            std::swap(bias_md_copy.dims[matmul_dim_1],
                    bias_md_copy.dims[matmul_dim_2]);
            conf_.transpose_bias = true;
        }
        conf_.bias_md = xpu::sycl::md_t(&bias_md_copy);
    }

    dims_t dst_blocks;
    for (int i = 0; i < matmul_kernel_fwd_t::max_supported_ndims; i++) {
        if (i < conf_.dst_md.ndims()) {
            dst_blocks[i] = conf_.dst_md.dims()[i];
        } else {
            dst_blocks[i] = 1;
        }
    }
    dst_blocks[matmul_dim_1] = math::div_up(
            dst_blocks[matmul_dim_1], matmul_kernel_fwd_t::register_block_N);
    dst_blocks[matmul_dim_2] = math::div_up(
            dst_blocks[matmul_dim_2], matmul_kernel_fwd_t::register_block_M);
    int n_blocks = 1;
    for (int i = 0; i < matmul_kernel_fwd_t::max_supported_ndims; i++) {
        n_blocks *= dst_blocks[i];
    }
    conf_.wk_size = n_blocks;

    int high_two_bits = 3 << (ndims() - 2);
    // last two dimensions of data and weights are never broadcast
    conf_.data_mask
            = utils::get_dims_mask(dst_md()->dims, src_md()->dims, ndims())
            | high_two_bits;
    conf_.weights_mask
            = utils::get_dims_mask(dst_md()->dims, weights_md(0)->dims, ndims())
            | high_two_bits;
    conf_.bias_mask = utils::get_dims_mask(
            dst_md()->dims, weights_md(1)->dims, ndims());

    conf_.do_scale_data
            = !attr()->scales_.get(DNNL_ARG_SRC_0).has_default_values();
    conf_.do_scale_weights
            = !attr()->scales_.get(DNNL_ARG_WEIGHTS).has_default_values();
    conf_.do_scale_dst
            = !attr()->scales_.get(DNNL_ARG_DST).has_default_values();
    conf_.single_weights_scale
            = attr()->scales_.get(DNNL_ARG_WEIGHTS).mask_ == 0;

    conf_.use_data_zeropoints
            = !attr()->zero_points_.has_default_values(DNNL_ARG_SRC_0);
    conf_.use_weights_zeropoints
            = !attr()->zero_points_.has_default_values(DNNL_ARG_WEIGHTS_0);
    conf_.use_dst_zeropoints
            = !attr()->zero_points_.has_default_values(DNNL_ARG_DST);

    conf_.use_dropout = !attr()->dropout_.has_default_values();

    conf_.post_ops = sycl_post_ops_t(attr());

    for (auto i = 0; i < conf_.post_ops.get_post_op(); ++i) {
        const auto &e = attr()->post_ops_.entry_[i];
        if (e.is_binary() || e.is_prelu()) {
            conf_.binary_src_arr[i] = xpu::sycl::md_t(
                    arg_md(DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1));
        }
    }
    return status::success;
}

status_t ref_matmul_t::init(impl::engine_t *engine) {
    const auto kid = ::sycl::get_kernel_id<matmul_kernel_fwd_t>();
    CHECK(create_kernel(engine, kid, &kernel_));
    return status::success;
}

status_t ref_matmul_t::execute(const exec_ctx_t &ctx) const {

    parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        matmul_kernel_fwd_t matmul_kernel(pd()->conf_, cgh, ctx);

        const int block_size = 32;
        const int wg_size = 32;

        const int t_work = pd()->conf_.wk_size;
        const int wg_work = wg_size * block_size;
        const int wg_cnt = utils::div_up(t_work, wg_work);

        cgh.parallel_for(
                ::sycl::nd_range<1>(wg_cnt * wg_size, wg_size), matmul_kernel);
    });

    return status::success;
}

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl
