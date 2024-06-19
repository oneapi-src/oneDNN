/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#include "gpu/intel/ocl/ref_matmul.hpp"

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "gpu/intel/compute/utils.hpp"
namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

status_t ref_matmul_t::execute_ref(const exec_ctx_t &ctx) const {
    const auto &a = CTX_IN_STORAGE(DNNL_ARG_SRC);
    const auto &b = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    const auto &bias = CTX_IN_STORAGE(DNNL_ARG_BIAS);

    auto &c = CTX_OUT_STORAGE(DNNL_ARG_DST);

    auto &src_scales = CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    auto &wei_scales = CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
    auto &dst_scales = CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);
    const auto &a0 = CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC);
    const auto &b0
            = CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS);
    const auto &c0 = CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST);

    const auto a_d = ctx.memory_mdw(DNNL_ARG_SRC, pd()->src_md());
    const auto b_d = ctx.memory_mdw(DNNL_ARG_WEIGHTS, pd()->weights_md());
    const auto c_d = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());
    const auto bia_d = ctx.memory_mdw(DNNL_ARG_BIAS, pd()->weights_md(1));

    // All tensors must have the same order.
    // If order > 2D, all dimensions above 2 will be combined into a single
    // batch dimension. For this reason block formats are not supported.

    const int last = c_d.ndims() - 1;

    dnnl_dims_t bia_stride {0};
    if (bia_d.data_type() != data_type::undef) {
        const auto &bia_strides = bia_d.blocking_desc().strides;
        for (int i = 0; i < bia_d.ndims(); i++) {
            if (bia_d.dims()[last - i] > 1) {
                bia_stride[i] = bia_strides[last - i];
            } else {
                bia_stride[i] = 0;
            }
        }
    }

    dnnl_dims_t a_stride {0};
    dnnl_dims_t b_stride {0};
    dnnl_dims_t c_stride {0};
    const auto &a_strides = a_d.blocking_desc().strides;
    const auto &b_strides = b_d.blocking_desc().strides;
    const auto &c_strides = c_d.blocking_desc().strides;
    for (int i = 0; i < c_d.ndims(); i++) {
        if (a_d.dims()[last - i] > 1) { a_stride[i] = a_strides[last - i]; }
        if (b_d.dims()[last - i] > 1) { b_stride[i] = b_strides[last - i]; }
        if (c_d.dims()[last - i] > 1) { c_stride[i] = c_strides[last - i]; }
    }

    const dim_t D3 = c_d.ndims() > 5 ? c_d.dims()[last - 5] : 1;
    const dim_t D2 = c_d.ndims() > 4 ? c_d.dims()[last - 4] : 1;
    const dim_t D1 = c_d.ndims() > 3 ? c_d.dims()[last - 3] : 1;
    const dim_t D0 = c_d.ndims() > 2 ? c_d.dims()[last - 2] : 1;
    const dim_t M = c_d.dims()[last - 1];
    const dim_t N = c_d.dims()[last];
    const dim_t K = a_d.dims()[last];

    const auto &attr_scales = pd()->attr()->scales_;
    const int wei_scale_mask = attr_scales.get(DNNL_ARG_WEIGHTS).mask_;
    const bool wei_scale_per_n = wei_scale_mask & pd()->wei_qmask_N();
    const bool wei_scale_per_k = wei_scale_mask & pd()->wei_qmask_K();
    const auto wei_scale_group_ndim = attr_scales.get(DNNL_ARG_WEIGHTS).ndims_;
    const auto wei_scale_group_k = wei_scale_group_ndim > 0
            ? attr_scales.get(DNNL_ARG_WEIGHTS).group_dims_[0]
            : (wei_scale_per_k ? 1 : K);
    const dim_t wei_scale_stride_n = wei_scale_per_n ? 1 : 0;
    const dim_t wei_scale_stride_k
            = wei_scale_group_k < K ? wei_scale_per_n ? N : 1 : 0;
    const auto wei_scale_ngroups_k = K / wei_scale_group_k;

    const int src_scale_mask = attr_scales.get(DNNL_ARG_SRC).mask_;
    const bool src_scale_per_k = src_scale_mask & pd()->src_qmask_K();
    const auto src_scale_group_ndim = attr_scales.get(DNNL_ARG_SRC).ndims_;
    const auto src_scale_group_k = src_scale_group_ndim > 0
            ? attr_scales.get(DNNL_ARG_SRC).group_dims_[1]
            : (src_scale_per_k ? 1 : K);
    const dim_t src_scale_stride_k = src_scale_group_k < K ? 1 : 0;
    const auto src_scale_ngroups_k = K / src_scale_group_k;

    const auto &attr_zps = pd()->attr()->zero_points_;
    int wei_zp_mask = 0;
    attr_zps.get(DNNL_ARG_WEIGHTS, &wei_zp_mask);
    const bool wei_zp_per_n = wei_zp_mask & pd()->wei_qmask_N();
    const bool wei_zp_per_k = wei_zp_mask & pd()->wei_qmask_K();
    const auto wei_zp_group_ndims = attr_zps.get_groups_ndims(DNNL_ARG_WEIGHTS);
    const auto wei_zp_group_k = wei_zp_group_ndims > 0
            ? attr_zps.get_groups(DNNL_ARG_WEIGHTS)[0]
            : (wei_zp_per_k ? 1 : K);
    const dim_t wei_zp_stride_n = wei_zp_per_n ? 1 : 0;
    const dim_t wei_zp_stride_k = wei_zp_group_k < K ? wei_zp_per_n ? N : 1 : 0;
    const auto wei_zp_ngroups_k = K / wei_zp_group_k;

    // For compute kernel, the minimal group is picked.
    const auto scale_ngroups_k
            = std::max(src_scale_ngroups_k, wei_scale_ngroups_k);
    const auto ngroups_k = std::max(wei_zp_ngroups_k, scale_ngroups_k);
    const auto group_K = K / ngroups_k;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, a);
    arg_list.set(1, b);
    arg_list.set(2, c);
    arg_list.set(3, bias);
    arg_list.set(4, a0);
    arg_list.set(5, b0);
    arg_list.set(6, wei_zp_stride_n);
    arg_list.set(7, wei_zp_stride_k);
    arg_list.set(8, wei_zp_group_k);
    arg_list.set(9, c0);
    arg_list.set(10, src_scales);
    arg_list.set(11, src_scale_stride_k);
    arg_list.set(12, src_scale_group_k);
    arg_list.set(13, wei_scales);
    arg_list.set(14, wei_scale_stride_n);
    arg_list.set(15, wei_scale_stride_k);
    arg_list.set(16, wei_scale_group_k);
    arg_list.set(17, dst_scales);
    arg_list.set(18, group_K);
    arg_list.set(19, K);
    arg_list.set(20, N);
    arg_list.set(21, M);
    arg_list.set(22, D0);
    arg_list.set(23, D1);
    arg_list.set(24, D2);
    arg_list.set(25, bia_stride[5]);
    arg_list.set(26, bia_stride[4]);
    arg_list.set(27, bia_stride[3]);
    arg_list.set(28, bia_stride[2]);
    arg_list.set(29, bia_stride[1]);
    arg_list.set(30, bia_stride[0]);
    arg_list.set(31, a_stride[5]);
    arg_list.set(32, a_stride[4]);
    arg_list.set(33, a_stride[3]);
    arg_list.set(34, a_stride[2]);
    arg_list.set(35, a_stride[1]);
    arg_list.set(36, a_stride[0]);
    arg_list.set(37, b_stride[5]);
    arg_list.set(38, b_stride[4]);
    arg_list.set(39, b_stride[3]);
    arg_list.set(40, b_stride[2]);
    arg_list.set(41, b_stride[1]);
    arg_list.set(42, b_stride[0]);
    arg_list.set(43, c_stride[5]);
    arg_list.set(44, c_stride[4]);
    arg_list.set(45, c_stride[3]);
    arg_list.set(46, c_stride[2]);
    arg_list.set(47, c_stride[1]);
    arg_list.set(48, c_stride[0]);

    append_post_ops_to_arg_list(ctx, arg_list, 49, pd()->attr()->post_ops_);

    compute::range_t gws = {1, (size_t)N, (size_t)(D0 * D1 * D2 * D3)};
    auto nd_range = compute::nd_range_t(gws);

    status_t status = parallel_for(ctx, nd_range, kernel_, arg_list);

    ctx.zero_pad_output(DNNL_ARG_DST);
    return status;
}

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
