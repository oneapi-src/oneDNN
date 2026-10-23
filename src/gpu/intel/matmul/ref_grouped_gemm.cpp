/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#include "gpu/intel/matmul/ref_grouped_gemm.hpp"

#if DNNL_EXPERIMENTAL_GROUPED_MEMORY

#include <algorithm>

#include "common/c_types_map.hpp"
#include "gpu/intel/compute/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace matmul {

// Two grouped matmul patterns are supported:
// 2D grouped src (variable M) x 3D dense wei -> 2D grouped dst (variable M)
// 2D grouped src (variable K) x 2D grouped wei (variable M) -> dense 3D dst
status_t ref_grouped_t::execute(const exec_ctx_t &ctx) const {
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper wei_d(pd()->weights_md(0));
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const bool is_2dby2d = pd()->is_2dby2d();

    const auto &src_grouped = src_d.sparse_desc().grouped_desc;
    const dim_t group_count = pd()->group_count_;
    const dim_t N = dst_d.dims()[dst_d.ndims() - 1];
    const dim_t total_tokens = src_d.dims()[0];

    // Note, that below are constants for specific patterns
    const dim_t K_fixed
            = is_2dby2d ? 0 : wei_d.dims()[wei_d.ndims() - 2]; // 2Dx3D
    const dim_t M_fixed = is_2dby2d ? src_d.dims()[0] : 0; // 2Dx2D

    // Strides for accessing elements within inner GEMM
    //
    // 2Dx3D
    // row-major src [total_M, K]: stride_m = K, stride_k = 1
    // dense wei [G, K, N]:        stride_k = N, stride_n = 1
    // dst [total_M, N] row-major: stride_m = N, stride_n = 1
    //
    // 2Dx2D
    // col-major src [M, total_K]: stride_m = 1, stride_k = M
    // row-major wei [total_K, N]: stride_k = N, stride_n = 1
    // dst [G, M, N] row-major:    stride_m = N, stride_n = 1
    const auto src_strides = src_d.strides();
    const auto wei_strides = wei_d.strides();
    const auto dst_strides = dst_d.strides();
    const dim_t src_stride_m = src_strides[0];
    const dim_t src_stride_k = src_strides[1];
    const dim_t wei_stride_k = wei_strides[wei_d.ndims() - 2];
    const dim_t wei_stride_n = wei_strides[wei_d.ndims() - 1];
    const dim_t dst_stride_m = dst_strides[dst_d.ndims() - 2];
    const dim_t dst_stride_n = dst_strides[dst_d.ndims() - 1];

    // Strides to access different groups, i.e. base_g_ptr = g_start * g_stride
    //
    // 2Dx3D
    // row-major src [total_M, K]: stride = K
    // dense wei [G, K, N]:        stride = K * N
    // dst [total_M, N] row-major: stride = N
    //
    // 2Dx2D
    // col-major src [M, total_K]: stride = M
    // row-major wei [total_K, N]: stride = N
    // dst [G, M, N] row-major:    stride = M * N
    const dim_t src_group_stride = src_strides[src_grouped.variable_dim_idx];
    const dim_t wei_group_stride = wei_strides[0];
    const dim_t dst_group_stride = dst_strides[0];

    const auto &src_data = CTX_IN_STORAGE(DNNL_ARG_SRC, 0);
    const auto &src_offsets = CTX_IN_STORAGE(DNNL_ARG_SRC, 1);
    const auto &wei_data = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS, 0);
    auto &dst_data = CTX_OUT_STORAGE(DNNL_ARG_DST, 0);

    const bool with_bias = pd()->with_bias();
    const auto &attr_scales = pd()->attr()->scales_;
    const bool with_src_scales = !attr_scales.has_default_values(DNNL_ARG_SRC);
    const auto src_scale_group_k = attr_scales.get_group(DNNL_ARG_SRC, -1);
    const dim_t src_scale_ngroups_k
            = src_scale_group_k > 1 ? K_fixed / src_scale_group_k : 1;

    const bool with_wei_scales
            = !attr_scales.has_default_values(DNNL_ARG_WEIGHTS);
    const auto wei_scale_group_k = attr_scales.get_group(DNNL_ARG_WEIGHTS, -2);
    const dim_t wei_scale_ngroups_k
            = wei_scale_group_k > 1 ? K_fixed / wei_scale_group_k : 1;
    const dim_t wei_scale_group_n = attr_scales.get_group(DNNL_ARG_WEIGHTS, -1);

    const auto &attr_zps = pd()->attr()->zero_points_;
    const bool with_src_zp = !attr_zps.has_default_values(DNNL_ARG_SRC);
    const auto src_zp_group_k = attr_zps.get_group(DNNL_ARG_SRC, -1);
    const dim_t src_zp_ngroups_k
            = src_zp_group_k > 1 ? K_fixed / src_zp_group_k : 1;

    const bool with_wei_zp = !attr_zps.has_default_values(DNNL_ARG_WEIGHTS);
    // Wei zp is dense [G, K/gK, N/gN], dims not in the mask have size 1
    const int wei_zp_mask = attr_zps.get_mask(DNNL_ARG_WEIGHTS);
    const auto wei_zp_group_k = attr_zps.get_group(DNNL_ARG_WEIGHTS, -2);
    const dim_t wei_zp_ngroups_k
            = (wei_zp_mask & (1 << 1)) ? K_fixed / wei_zp_group_k : 1;
    const dim_t wei_zp_group_n = attr_zps.get_group(DNNL_ARG_WEIGHTS, -1);
    const dim_t wei_zp_ngroups_n
            = (wei_zp_mask & (1 << 2)) ? N / wei_zp_group_n : 1;
    const dim_t wei_zp_stride_g = (wei_zp_mask & (1 << 0))
            ? wei_zp_ngroups_k * wei_zp_ngroups_n
            : 0;
    const dim_t wei_zp_stride_k
            = (wei_zp_mask & (1 << 1)) ? wei_zp_ngroups_n : 0;
    const dim_t wei_zp_stride_n = (wei_zp_mask & (1 << 2)) ? 1 : 0;

    // Finest K-group granularity across src/wei scales and ZPs
    const dim_t n_k_groups = is_2dby2d
            ? 1
            : std::max({src_scale_ngroups_k, wei_scale_ngroups_k,
                      src_zp_ngroups_k, wei_zp_ngroups_k});

    compute::kernel_arg_list_t arg_list;
    int arg_idx = 0;
    arg_list.set(arg_idx++, src_data);
    arg_list.set(arg_idx++, src_offsets);
    arg_list.set(arg_idx++, wei_data);
    // Tensor sharing src split: wei (2Dx2D) or dst (2Dx3D)
    if (is_2dby2d) {
        arg_list.set(arg_idx++, CTX_IN_STORAGE(DNNL_ARG_WEIGHTS, 1));
    } else {
        arg_list.set(arg_idx++, CTX_OUT_STORAGE(DNNL_ARG_DST, 1));
    }
    arg_list.set(arg_idx++, dst_data);
    arg_list.set(arg_idx++, (int)group_count);
    arg_list.set(arg_idx++, (int)is_2dby2d);
    arg_list.set(arg_idx++, M_fixed);
    arg_list.set(arg_idx++, K_fixed);
    arg_list.set(arg_idx++, N);
    arg_list.set(arg_idx++, src_stride_m);
    arg_list.set(arg_idx++, src_stride_k);
    arg_list.set(arg_idx++, wei_stride_k);
    arg_list.set(arg_idx++, wei_stride_n);
    arg_list.set(arg_idx++, dst_stride_m);
    arg_list.set(arg_idx++, dst_stride_n);
    arg_list.set(arg_idx++, src_group_stride);
    arg_list.set(arg_idx++, wei_group_stride);
    arg_list.set(arg_idx++, dst_group_stride);
    arg_list.set(arg_idx++, n_k_groups);
    if (with_bias) arg_list.set(arg_idx++, CTX_IN_STORAGE(DNNL_ARG_BIAS));
    if (with_src_scales) {
        arg_list.set(
                arg_idx++, CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC));
        arg_list.set(arg_idx++, src_scale_ngroups_k);
    }
    if (with_wei_scales) {
        arg_list.set(arg_idx++,
                CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS));
        arg_list.set(arg_idx++, wei_scale_ngroups_k);
        arg_list.set(arg_idx++, wei_scale_group_n);
    }
    if (with_src_zp) {
        arg_list.set(arg_idx++,
                CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC));
        arg_list.set(arg_idx++, src_zp_ngroups_k);
    }
    if (with_wei_zp) {
        arg_list.set(arg_idx++,
                CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS));
        arg_list.set(arg_idx++, wei_zp_ngroups_k);
        arg_list.set(arg_idx++, wei_zp_group_n);
        arg_list.set(arg_idx++, wei_zp_stride_g);
        arg_list.set(arg_idx++, wei_zp_stride_k);
        arg_list.set(arg_idx++, wei_zp_stride_n);
    }

    // Post-ops apply to the 2Dx3D pattern only (grouped dst)
    arg_idx = append_post_ops_to_arg_list(
            ctx, arg_list, arg_idx, pd()->generic_po_, pd()->group_po_dst_md_);

    // Simple 3D dispatch for ref impl clarity
    compute::range_t gws
            = {(size_t)group_count, (size_t)total_tokens, (size_t)N};

    return parallel_for(ctx, compute::nd_range_t(gws), kernel_, arg_list);
}

} // namespace matmul
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // DNNL_EXPERIMENTAL_GROUPED_MEMORY
