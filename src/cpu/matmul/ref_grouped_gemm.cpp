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

#include "cpu/matmul/ref_grouped_gemm.hpp"

#if DNNL_EXPERIMENTAL_GROUPED_MEMORY

#include <algorithm>

#include "common/c_types_map.hpp"
#include "common/compiler_workarounds.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/type_helpers.hpp"

#include "cpu/ref_io_helper.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
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
    const dim_t group_count = src_grouped.group_count;
    const dim_t N = dst_d.dims()[dst_d.ndims() - 1];

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

    const void *src_data = CTX_IN_MEM(const void *, DNNL_ARG_SRC, 0);
    const int32_t *src_offsets = CTX_IN_MEM(const int32_t *, DNNL_ARG_SRC, 1);
    const void *wei_data = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS, 0);
    const int32_t *wei_offsets
            = CTX_IN_MEM(const int32_t *, DNNL_ARG_WEIGHTS, 1);
    void *dst_data = CTX_OUT_MEM(void *, DNNL_ARG_DST, 0);
    const int32_t *dst_offsets = CTX_OUT_MEM(const int32_t *, DNNL_ARG_DST, 1);

    const auto src_dt = src_d.data_type();
    const auto wei_dt = wei_d.data_type();
    const auto dst_dt = pd()->dst_md()->data_type;

    const bool with_bias = pd()->with_bias();
    const void *bias_data = CTX_IN_MEM(const void *, DNNL_ARG_BIAS);
    const auto bia_dt = pd()->weights_md(1)->data_type;

    // Attributes (scales/zps/post-ops) are only supported in 2Dx3D and
    // are rejected earlier for 2Dx2D, so all with_* flags below would be false
    const auto &attr_scales = pd()->attr()->scales_;
    const bool with_src_scales = !attr_scales.has_default_values(DNNL_ARG_SRC);
    const auto src_scale_dt = attr_scales.get_data_type(DNNL_ARG_SRC);
    const auto src_scale_group_k = attr_scales.get_group(DNNL_ARG_SRC, -1);
    const auto src_scale_ngroups_k
            = src_scale_group_k > 1 ? K_fixed / src_scale_group_k : 1;
    const void *src_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);

    const bool with_wei_scales
            = !attr_scales.has_default_values(DNNL_ARG_WEIGHTS);
    const auto wei_scale_dt = attr_scales.get_data_type(DNNL_ARG_WEIGHTS);
    const auto wei_scale_group_k = attr_scales.get_group(DNNL_ARG_WEIGHTS, -2);
    const auto wei_scale_ngroups_k
            = wei_scale_group_k > 1 ? K_fixed / wei_scale_group_k : 1;
    const auto wei_scale_group_n = attr_scales.get_group(DNNL_ARG_WEIGHTS, -1);
    const auto wei_scale_ngroups_n = N / wei_scale_group_n;
    const void *wei_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);

    const auto &attr_zps = pd()->attr()->zero_points_;
    const bool with_src_zps = !attr_zps.has_default_values(DNNL_ARG_SRC);
    const auto src_zp_dt = attr_zps.get_data_type(DNNL_ARG_SRC);
    const auto src_zp_group_k = attr_zps.get_group(DNNL_ARG_SRC, -1);
    const dim_t src_zp_ngroups_k
            = src_zp_group_k > 1 ? K_fixed / src_zp_group_k : 1;
    const void *src_zps = CTX_IN_MEM(
            const void *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC);

    const bool with_wei_zps = !attr_zps.has_default_values(DNNL_ARG_WEIGHTS);
    const auto wei_zp_dt = attr_zps.get_data_type(DNNL_ARG_WEIGHTS);
    const auto wei_zp_group_k = attr_zps.get_group(DNNL_ARG_WEIGHTS, -2);
    const dim_t wei_zp_ngroups_k
            = wei_zp_group_k > 1 ? K_fixed / wei_zp_group_k : 1;
    const auto wei_zp_group_n = attr_zps.get_group(DNNL_ARG_WEIGHTS, -1);
    const dim_t wei_zp_ngroups_n = N / wei_zp_group_n;
    const void *wei_zps = CTX_IN_MEM(
            const void *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS);

    // Use the finest K-group granularity across src/wei scales and src/wei ZPs
    const dim_t n_k_groups = is_2dby2d
            ? 1
            : std::max({src_scale_ngroups_k, wei_scale_ngroups_k,
                      src_zp_ngroups_k, wei_zp_ngroups_k});

    const bool use_int_arithmetic
            = utils::one_of(src_dt, data_type::s8, data_type::u8)
            && utils::one_of(wei_dt, data_type::s8, data_type::u8,
                    data_type::s4, data_type::u4);
    const bool use_woq = utils::one_of(src_dt, data_type::f32, data_type::bf16,
                                 data_type::f16)
            && utils::one_of(wei_dt, data_type::s8, data_type::u8,
                    data_type::s4, data_type::u4, data_type::u3)
            && pd()->attr()->fpmath_.apply_to_int_;

    const auto &po = pd()->attr()->post_ops_;
    const bool with_post_ops = !po.has_default_values();

    // Validate runtime group offsets
    for (dim_t group_id = 0; group_id < group_count; ++group_id) {
        if (is_2dby2d) {
            const dim_t total_K = src_d.dims()[1];
            const dim_t k_start
                    = (group_id == 0) ? 0 : src_offsets[group_id - 1];
            const dim_t k_end = src_offsets[group_id];
            // src and wei should share the same offsets
            if (wei_offsets[group_id] != k_end
                    || (group_id > 0 && wei_offsets[group_id - 1] != k_start)
                    || k_start < 0 || k_end > total_K || k_end < k_start)
                return status::invalid_arguments;
        } else {
            const dim_t total_M = src_d.dims()[0];
            const dim_t src_offset_start
                    = (group_id == 0) ? 0 : src_offsets[group_id - 1];
            const dim_t src_offset_end = src_offsets[group_id];
            const dim_t dst_offset_start
                    = (group_id == 0) ? 0 : dst_offsets[group_id - 1];
            const dim_t dst_offset_end = dst_offsets[group_id];
            // src and dst should share the same offsets
            if (src_offset_start < 0 || src_offset_end > total_M
                    || src_offset_end < src_offset_start || dst_offset_start < 0
                    || dst_offset_end > total_M
                    || dst_offset_end < dst_offset_start)
                return status::invalid_arguments;

            // Grouped binary post-op tensors must share the dst offsets
            if (with_post_ops) {
                for (int po_idx = 0; po_idx < po.len(); ++po_idx) {
                    const auto &entry = po.entry_[po_idx];
                    if (!entry.is_binary()) continue;
                    const memory_desc_wrapper bin_d(entry.binary.src1_desc);
                    if (!bin_d.is_grouped_desc()) continue;
                    const int po_arg = DNNL_ARG_ATTR_MULTIPLE_POST_OP(po_idx)
                            | DNNL_ARG_SRC_1;
                    const auto bin_offsets
                            = CTX_IN_MEM(const int32_t *, po_arg, 1);
                    const dim_t bin_offset_start
                            = group_id > 0 ? bin_offsets[group_id - 1] : 0;
                    const dim_t bin_offset_end = bin_offsets[group_id];
                    const dim_t bin_total_m = bin_d.dims()[0];
                    if (bin_offset_start < 0 || bin_offset_end > bin_total_m
                            || bin_offset_end < bin_offset_start
                            || bin_offset_start != dst_offset_start
                            || bin_offset_end != dst_offset_end)
                        return status::invalid_arguments;
                }
            }
        }
    }

    // Parallelize over output rows flattened across all groups to keep the
    // work balanced when group sizes are highly skewed (e.g., "hot" benchdnn
    // profile where one group holds almost all tokens)
    const dim_t total_rows
            = is_2dby2d ? group_count * M_fixed : src_d.dims()[0];
    parallel_nd(total_rows, [= COMPAT_THIS_CAPTURE](dim_t row) {
        dim_t group_id, m, K_g;
        dim_t src_group_start, wei_group_start, dst_group_start;
        dim_t src_attr_row_base = 0;
        dim_t dst_offset_start = 0;

        if (is_2dby2d) {
            group_id = row / M_fixed;
            m = row % M_fixed;
            const dim_t k_start
                    = (group_id == 0) ? 0 : src_offsets[group_id - 1];
            const dim_t k_end = src_offsets[group_id];
            K_g = k_end - k_start;
            src_group_start = k_start;
            wei_group_start = k_start;
            dst_group_start = group_id;
        } else {
            group_id = std::upper_bound(
                               src_offsets, src_offsets + group_count, row)
                    - src_offsets;
            const dim_t src_offset_start
                    = (group_id == 0) ? 0 : src_offsets[group_id - 1];
            m = row - src_offset_start;
            dst_offset_start = (group_id == 0) ? 0 : dst_offsets[group_id - 1];
            K_g = K_fixed;
            src_group_start = src_offset_start;
            wei_group_start = group_id;
            dst_group_start = dst_offset_start;
            src_attr_row_base = src_offset_start;
        }

        const dim_t src_base = src_group_start * src_group_stride;
        const dim_t wei_base = wei_group_start * wei_group_stride;
        const dim_t dst_base = dst_group_start * dst_group_stride;

        // Note, that K_g == 0 must still write zeros
        const dim_t k_group_size = K_g / n_k_groups;

        for (dim_t n = 0; n < N; ++n) {
            float result = 0.0f;

            // int wei path (either int src + int wei or fp src + int wei WOQ)
            // int src follows ref_matmul_int8_t
            // fp  src (WOQ) follows dequantize then multiply
            if (use_int_arithmetic || use_woq) {
                for (dim_t i_group = 0; i_group < n_k_groups; i_group++) {
                    int src_zp_val = 0;
                    float wei_scale = 1.0f;
                    int wei_zp_val = 0;

                    if (with_src_zps) {
                        const dim_t src_k_group
                                = i_group * src_zp_ngroups_k / n_k_groups;
                        const dim_t idx
                                = (src_attr_row_base + m) * src_zp_ngroups_k
                                + src_k_group;
                        src_zp_val
                                = io::load_int_value(src_zp_dt, src_zps, idx);
                    }

                    if (with_wei_scales) {
                        const dim_t wei_k_group
                                = i_group * wei_scale_ngroups_k / n_k_groups;
                        const dim_t idx
                                = (group_id * wei_scale_ngroups_k + wei_k_group)
                                        * wei_scale_ngroups_n
                                + n / wei_scale_group_n;
                        wei_scale = io::load_float_value(
                                wei_scale_dt, wei_scales, idx);
                    }
                    if (with_wei_zps) {
                        const dim_t wei_k_group
                                = i_group * wei_zp_ngroups_k / n_k_groups;
                        const dim_t idx
                                = (group_id * wei_zp_ngroups_k + wei_k_group)
                                        * wei_zp_ngroups_n
                                + n / wei_zp_group_n;
                        wei_zp_val
                                = io::load_int_value(wei_zp_dt, wei_zps, idx);
                    }

                    float acc = 0.0f;
                    if (use_int_arithmetic) {
                        int acc_int = 0;
                        for (dim_t k = 0; k < k_group_size; ++k) {
                            const dim_t k_abs = k + i_group * k_group_size;
                            const dim_t src_idx = src_base + m * src_stride_m
                                    + k_abs * src_stride_k;
                            const dim_t wei_idx = wei_base
                                    + k_abs * wei_stride_k + n * wei_stride_n;
                            const int s = io::load_int_value(
                                    src_dt, src_data, src_idx);
                            const int w = io::load_int_value(
                                    wei_dt, wei_data, wei_idx);
                            acc_int += (s - src_zp_val) * (w - wei_zp_val);
                        }
                        acc = static_cast<float>(acc_int);
                    } else {
                        assert(src_zp_val == 0);
                        for (dim_t k = 0; k < k_group_size; ++k) {
                            const dim_t k_abs = k + i_group * k_group_size;
                            const dim_t src_idx = src_base + m * src_stride_m
                                    + k_abs * src_stride_k;
                            const dim_t wei_idx = wei_base
                                    + k_abs * wei_stride_k + n * wei_stride_n;
                            const float s = io::load_float_value(
                                    src_dt, src_data, src_idx);
                            const int w_int = io::load_int_value(
                                    wei_dt, wei_data, wei_idx);
                            acc += s * static_cast<float>(w_int - wei_zp_val);
                        }
                    }

                    if (with_src_scales) {
                        const dim_t src_k_group
                                = i_group * src_scale_ngroups_k / n_k_groups;
                        const dim_t idx
                                = (src_attr_row_base + m) * src_scale_ngroups_k
                                + src_k_group;
                        const float src_scale = io::load_float_value(
                                src_scale_dt, src_scales, idx);
                        acc *= src_scale;
                    }

                    result += acc * wei_scale;
                }
            } else {
                // fp arithmetic
                for (dim_t i_group = 0; i_group < n_k_groups; i_group++) {
                    float acc = 0.0f;

                    for (dim_t k = 0; k < k_group_size; ++k) {
                        const dim_t k_abs = k + i_group * k_group_size;
                        const dim_t src_idx = src_base + m * src_stride_m
                                + k_abs * src_stride_k;
                        const dim_t wei_idx = wei_base + k_abs * wei_stride_k
                                + n * wei_stride_n;

                        const float s = io::load_float_value(
                                src_dt, src_data, src_idx);
                        const float w = io::load_float_value(
                                wei_dt, wei_data, wei_idx);
                        acc += s * w;
                    }

                    if (with_src_scales) {
                        const dim_t src_k_group
                                = i_group * src_scale_ngroups_k / n_k_groups;
                        const dim_t idx
                                = (src_attr_row_base + m) * src_scale_ngroups_k
                                + src_k_group;
                        const float src_scale = io::load_float_value(
                                src_scale_dt, src_scales, idx);
                        acc *= src_scale;
                    }

                    if (with_wei_scales) {
                        const dim_t wei_k_group
                                = i_group * wei_scale_ngroups_k / n_k_groups;
                        const dim_t idx
                                = (group_id * wei_scale_ngroups_k + wei_k_group)
                                        * wei_scale_ngroups_n
                                + n / wei_scale_group_n;
                        const float wei_scale = io::load_float_value(
                                wei_scale_dt, wei_scales, idx);
                        acc *= wei_scale;
                    }

                    result += acc;
                }
            }

            // Add bias
            if (with_bias) {
                const dim_t bias_idx = group_id * N + n;
                result += io::load_float_value(bia_dt, bias_data, bias_idx);
            }

            const dim_t dst_idx
                    = dst_base + m * dst_stride_m + n * dst_stride_n;

            // Post-Ops support: binary mul with grouped or dense
            // (incl. NVFP4 recipe with global scale) tensor, eltwise
            //
            // Note: ref_post_ops_t won't allow using grouped tensor (with their
            // own per-group offsets) in binary po
            if (with_post_ops) {
                int eltwise_idx = 0, binary_idx = 0;
                for (int po_idx = 0; po_idx < po.len(); ++po_idx) {
                    const auto &entry = po.entry_[po_idx];
                    if (entry.is_eltwise()) {
                        result = eltwise_po_[eltwise_idx++].compute_scalar(
                                result);
                    } else if (entry.is_binary()) {
                        const int po_arg
                                = DNNL_ARG_ATTR_MULTIPLE_POST_OP(po_idx)
                                | DNNL_ARG_SRC_1;
                        const auto bin_data = CTX_IN_MEM(const void *, po_arg);
                        const memory_desc_wrapper bin_d(entry.binary.src1_desc);

                        // For grouped binary tensors, use their own offsets
                        // For dense tensor, use dst offsets
                        dim_t bin_row_base = dst_offset_start;
                        if (bin_d.is_grouped_desc()) {
                            const auto bin_offsets
                                    = CTX_IN_MEM(const int32_t *, po_arg, 1);
                            const dim_t bin_offset_start = group_id > 0
                                    ? bin_offsets[group_id - 1]
                                    : 0;
                            bin_row_base = bin_offset_start;
                        }
                        const dim_t bin_M = bin_d.dims()[0];
                        const dim_t bin_N = bin_d.dims()[1];
                        // Per-group [G, 1]: index is group_id
                        // Per-row [total_M, *]: index is bin_row_base + m
                        const bool per_group
                                = (bin_M == group_count && bin_N == 1);
                        const dim_t eff_m
                                = per_group ? group_id : (bin_row_base + m);
                        const dim_t eff_n = bin_N > 1 ? n : 0;
                        const float val
                                = io::load_float_value(bin_d.data_type(),
                                        bin_data, eff_m * bin_N + eff_n);
                        result = binary_po_[binary_idx++].compute_scalar(
                                result, val, false);
                    }
                }
            }

            io::store_float_value(dst_dt, result, dst_data, dst_idx);
        }
    });

    return status::success;
}

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // DNNL_EXPERIMENTAL_GROUPED_MEMORY
