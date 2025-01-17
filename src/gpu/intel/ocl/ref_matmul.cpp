/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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
    const int wei_scale_mask = attr_scales.get_mask(DNNL_ARG_WEIGHTS);
    const bool wei_scale_per_k = wei_scale_mask & pd()->wei_qmask_K();
    const auto wei_scale_group_k
            = !attr_scales.get(DNNL_ARG_WEIGHTS).has_default_groups()
            ? attr_scales.get_group(DNNL_ARG_WEIGHTS, 0)
            : (wei_scale_per_k ? 1 : K);
    const auto wei_scale_group_n = attr_scales.get_group(DNNL_ARG_WEIGHTS, 1);
    const auto wei_scale_ngroups_k = K / wei_scale_group_k;
    // Identify wei_scales dimensions as user may not pass them.
    dims_t wei_scale_dims {};
    dims_t wei_scale_strides {};
    utils::copy_dims_with_mask(
            wei_scale_dims, b_d.dims(), b_d.ndims(), wei_scale_mask);
    wei_scale_dims[b_d.ndims() - 1] /= wei_scale_group_n;
    wei_scale_dims[b_d.ndims() - 2] /= wei_scale_group_k;

    dim_t last_scale_dim = 0;
    dim_t last_scale_stride = 0;
    for (int d = b_d.ndims() - 1; d >= 0; d--) {
        if (wei_scale_dims[d] == 0) continue;
        wei_scale_strides[d] = last_scale_stride == 0
                ? 1
                : last_scale_dim * last_scale_stride;
        last_scale_stride = wei_scale_strides[d];
        last_scale_dim = wei_scale_dims[d];
    }

    const dim_t wei_scale_stride_n = wei_scale_strides[b_d.ndims() - 1];
    const dim_t wei_scale_stride_k = wei_scale_strides[b_d.ndims() - 2];
    const dim_t wei_scale_stride_b0
            = b_d.ndims() > 2 ? wei_scale_strides[b_d.ndims() - 3] : 0;
    const dim_t wei_scale_stride_b1
            = b_d.ndims() > 3 ? wei_scale_strides[b_d.ndims() - 4] : 0;

    const int src_scale_mask = attr_scales.get_mask(DNNL_ARG_SRC);
    const bool src_scale_per_k = src_scale_mask & pd()->src_qmask_K();
    const auto src_scale_group_k
            = !attr_scales.get(DNNL_ARG_SRC).has_default_groups()
            ? attr_scales.get_group(DNNL_ARG_SRC, 1)
            : (src_scale_per_k ? 1 : K);
    const auto src_scale_ngroups_k = K / src_scale_group_k;
    // Identify src_scales dimensions as user may not pass them.
    dims_t src_scale_dims {};
    dims_t src_scale_strides {};
    utils::copy_dims_with_mask(
            src_scale_dims, a_d.dims(), a_d.ndims(), src_scale_mask);
    src_scale_dims[a_d.ndims() - 1] /= src_scale_group_k;

    last_scale_dim = 0;
    last_scale_stride = 0;
    for (int d = a_d.ndims() - 1; d >= 0; d--) {
        if (src_scale_dims[d] == 0) continue;
        src_scale_strides[d] = last_scale_stride == 0
                ? 1
                : last_scale_dim * last_scale_stride;
        last_scale_stride = src_scale_strides[d];
        last_scale_dim = src_scale_dims[d];
    }

    const dim_t src_scale_stride_k = src_scale_strides[a_d.ndims() - 1];
    const dim_t src_scale_stride_m = src_scale_strides[a_d.ndims() - 2];
    const dim_t src_scale_stride_b0
            = a_d.ndims() > 2 ? src_scale_strides[a_d.ndims() - 3] : 0;
    const dim_t src_scale_stride_b1
            = a_d.ndims() > 3 ? src_scale_strides[a_d.ndims() - 4] : 0;

    const auto &attr_zps = pd()->attr()->zero_points_;
    int wei_zp_mask = attr_zps.get_mask(DNNL_ARG_WEIGHTS);
    const auto wei_zp_group_k = attr_zps.get_group(DNNL_ARG_WEIGHTS, 0);
    const auto wei_zp_group_n = attr_zps.get_group(DNNL_ARG_WEIGHTS, 1);
    const auto wei_zp_ngroups_k = K / wei_zp_group_k;
    // Identify wei_zp dimensions as user may not pass them.
    dims_t wei_zp_dims {};
    dims_t wei_zp_strides {};
    utils::copy_dims_with_mask(
            wei_zp_dims, b_d.dims(), b_d.ndims(), wei_zp_mask);
    wei_zp_dims[b_d.ndims() - 1] /= wei_zp_group_n;
    wei_zp_dims[b_d.ndims() - 2] /= wei_zp_group_k;

    dim_t last_zp_dim = 0;
    dim_t last_zp_stride = 0;
    for (int d = b_d.ndims() - 1; d >= 0; d--) {
        if (wei_zp_dims[d] == 0) continue;
        wei_zp_strides[d]
                = last_zp_stride == 0 ? 1 : last_zp_dim * last_zp_stride;
        last_zp_stride = wei_zp_strides[d];
        last_zp_dim = wei_zp_dims[d];
    }
    const dim_t wei_zp_stride_n = wei_zp_strides[b_d.ndims() - 1];
    const dim_t wei_zp_stride_k = wei_zp_strides[b_d.ndims() - 2];
    const dim_t wei_zp_stride_b0
            = b_d.ndims() > 2 ? wei_zp_strides[b_d.ndims() - 3] : 0;
    const dim_t wei_zp_stride_b1
            = b_d.ndims() > 3 ? wei_zp_strides[b_d.ndims() - 4] : 0;

    int src_zp_mask = attr_zps.get_mask(DNNL_ARG_SRC);
    const auto src_zp_group_k = attr_zps.get_group(DNNL_ARG_SRC, 1);
    const auto src_zp_ngroups_k = K / src_zp_group_k;
    // Identify src_zp dimensions as user may not pass them.
    dims_t src_zp_dims {};
    dims_t src_zp_strides {};
    utils::copy_dims_with_mask(
            src_zp_dims, a_d.dims(), a_d.ndims(), src_zp_mask);
    src_zp_dims[a_d.ndims() - 1] /= src_zp_group_k;

    last_zp_dim = 0;
    last_zp_stride = 0;
    for (int d = a_d.ndims() - 1; d >= 0; d--) {
        if (src_zp_dims[d] == 0) continue;
        src_zp_strides[d]
                = last_zp_stride == 0 ? 1 : last_zp_dim * last_zp_stride;
        last_zp_stride = src_zp_strides[d];
        last_zp_dim = src_zp_dims[d];
    }
    const dim_t src_zp_stride_k = src_zp_strides[a_d.ndims() - 1];
    const dim_t src_zp_stride_m = src_zp_strides[a_d.ndims() - 2];

    // For compute kernel, the minimal group is picked.
    const auto scale_ngroups_k
            = std::max(src_scale_ngroups_k, wei_scale_ngroups_k);
    const auto zp_ngroups_k = std::max(src_zp_ngroups_k, wei_zp_ngroups_k);
    const auto ngroups_k = std::max(zp_ngroups_k, scale_ngroups_k);
    const auto group_K = K / ngroups_k;

    const bool subbyte_pack
            = pd()->subbyte_pack_; //(c_d.data_type() == data_type::f4_e2m1);
    const dim_t nelems = c_d.nelems();
    auto tmp = ctx.get_scratchpad_grantor().get_memory_storage(
            memory_tracking::names::key_matmul_pack_space);

    compute::kernel_arg_list_t arg_list;
    int arg_idx = 0;
    arg_list.set(arg_idx++, a);
    arg_list.set(arg_idx++, b);
    arg_list.set(arg_idx++, subbyte_pack ? *tmp : c);
    arg_list.set(arg_idx++, bias);
    arg_list.set(arg_idx++, a0);
    arg_list.set(arg_idx++, src_zp_stride_k);
    arg_list.set(arg_idx++, src_zp_stride_m);
    arg_list.set(arg_idx++, src_zp_group_k);
    arg_list.set(arg_idx++, b0);
    arg_list.set(arg_idx++, wei_zp_stride_n);
    arg_list.set(arg_idx++, wei_zp_stride_k);
    arg_list.set(arg_idx++, wei_zp_stride_b0);
    arg_list.set(arg_idx++, wei_zp_stride_b1);
    arg_list.set(arg_idx++, wei_zp_group_n);
    arg_list.set(arg_idx++, wei_zp_group_k);
    arg_list.set(arg_idx++, c0);
    arg_list.set(arg_idx++, src_scales);
    arg_list.set(arg_idx++, src_scale_stride_k);
    arg_list.set(arg_idx++, src_scale_stride_m);
    arg_list.set(arg_idx++, src_scale_stride_b0);
    arg_list.set(arg_idx++, src_scale_stride_b1);
    arg_list.set(arg_idx++, src_scale_group_k);
    arg_list.set(arg_idx++, wei_scales);
    arg_list.set(arg_idx++, wei_scale_stride_n);
    arg_list.set(arg_idx++, wei_scale_stride_k);
    arg_list.set(arg_idx++, wei_scale_stride_b0);
    arg_list.set(arg_idx++, wei_scale_stride_b1);
    arg_list.set(arg_idx++, wei_scale_group_n);
    arg_list.set(arg_idx++, wei_scale_group_k);
    arg_list.set(arg_idx++, dst_scales);
    arg_list.set(arg_idx++, group_K);
    arg_list.set(arg_idx++, K);
    arg_list.set(arg_idx++, N);
    arg_list.set(arg_idx++, M);
    arg_list.set(arg_idx++, D0);
    arg_list.set(arg_idx++, D1);
    arg_list.set(arg_idx++, D2);
    arg_list.set(arg_idx++, bia_stride[5]);
    arg_list.set(arg_idx++, bia_stride[4]);
    arg_list.set(arg_idx++, bia_stride[3]);
    arg_list.set(arg_idx++, bia_stride[2]);
    arg_list.set(arg_idx++, bia_stride[1]);
    arg_list.set(arg_idx++, bia_stride[0]);
    arg_list.set(arg_idx++, a_stride[5]);
    arg_list.set(arg_idx++, a_stride[4]);
    arg_list.set(arg_idx++, a_stride[3]);
    arg_list.set(arg_idx++, a_stride[2]);
    arg_list.set(arg_idx++, a_stride[1]);
    arg_list.set(arg_idx++, a_stride[0]);
    arg_list.set(arg_idx++, b_stride[5]);
    arg_list.set(arg_idx++, b_stride[4]);
    arg_list.set(arg_idx++, b_stride[3]);
    arg_list.set(arg_idx++, b_stride[2]);
    arg_list.set(arg_idx++, b_stride[1]);
    arg_list.set(arg_idx++, b_stride[0]);
    arg_list.set(arg_idx++, c_stride[5]);
    arg_list.set(arg_idx++, c_stride[4]);
    arg_list.set(arg_idx++, c_stride[3]);
    arg_list.set(arg_idx++, c_stride[2]);
    arg_list.set(arg_idx++, c_stride[1]);
    arg_list.set(arg_idx++, c_stride[0]);

    const bool dropout = !pd()->attr()->dropout_.has_default_values();
    if (dropout) {
        arg_list.set(arg_idx++, CTX_OUT_STORAGE(DNNL_ARG_ATTR_DROPOUT_MASK));
        arg_list.set(arg_idx++, CTX_IN_STORAGE(DNNL_ARG_ATTR_DROPOUT_SEED));
        arg_list.set(
                arg_idx++, CTX_IN_STORAGE(DNNL_ARG_ATTR_DROPOUT_PROBABILITY));
    }

    const bool sround = !pd()->attr()->rounding_mode_.has_default_values();
    if (sround) {
        arg_list.set(arg_idx++, CTX_IN_STORAGE(DNNL_ARG_ATTR_ROUNDING_SEED));
    }
    append_post_ops_to_arg_list(
            ctx, arg_list, arg_idx, pd()->attr()->post_ops_);

    compute::range_t gws = {1, (size_t)N, (size_t)(D0 * D1 * D2 * D3)};
    auto nd_range = compute::nd_range_t(gws);

    status_t status = parallel_for(ctx, nd_range, kernels_[0], arg_list);

    ctx.zero_pad_output(DNNL_ARG_DST);

    if (!subbyte_pack) return status;
    compute::kernel_arg_list_t repack_arg_list;
    repack_arg_list.set(0, *tmp);
    repack_arg_list.set(1, c);
    repack_arg_list.set(2, into<dim_t>(nelems));
    repack_arg_list.set(3, 4);
    compute::range_t repack_gws((nelems * 4 + 7) / 8);
    compute::nd_range_t repack_nd_range(repack_gws);
    return large_parallel_for(
            ctx, repack_nd_range, kernels_[1], repack_arg_list, 4);
}

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
