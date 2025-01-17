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

#include <assert.h>
#include <float.h>
#include <math.h>

#include <algorithm>
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/ref_io_helper.hpp"

#include "cpu/matmul/matmul_utils.hpp"
#include "cpu/matmul/ref_matmul.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace matmul {

status_t ref_matmul_t::execute_ref(const exec_ctx_t &ctx) const {
    status_t status = status::success;
    const auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    const auto weights = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
    const auto bias = CTX_IN_MEM(const void *, DNNL_ARG_BIAS);

    const auto p = CTX_IN_MEM(const float *, DNNL_ARG_ATTR_DROPOUT_PROBABILITY);
    const auto seed = CTX_IN_MEM(const uint32_t *, DNNL_ARG_ATTR_DROPOUT_SEED);
    const auto rnd_seed
            = CTX_IN_MEM(const uint32_t *, DNNL_ARG_ATTR_ROUNDING_SEED);
    auto dropout_mask = CTX_OUT_CLEAN_MEM(
            unsigned char *, DNNL_ARG_ATTR_DROPOUT_MASK, status);
    CHECK(status);

    auto dst = CTX_OUT_CLEAN_MEM(void *, DNNL_ARG_DST, status);
    CHECK(status);

    DEFINE_ARG_SCALES_BUFFER(src_scales, DNNL_ARG_SRC);
    DEFINE_ARG_SCALES_BUFFER(wei_scales, DNNL_ARG_WEIGHTS);
    DEFINE_ARG_SCALES_BUFFER(dst_scales, DNNL_ARG_DST);

    DEFINE_ZERO_POINTS_BUFFER(wei_zero_points, DNNL_ARG_WEIGHTS);

    const auto src_d = ctx.memory_mdw(DNNL_ARG_SRC, pd()->src_md());
    const auto weights_d = ctx.memory_mdw(DNNL_ARG_WEIGHTS, pd()->weights_md());
    const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());
    const auto bia_d = ctx.memory_mdw(DNNL_ARG_BIAS, pd()->weights_md(1));

    const memory_desc_wrapper dropout_mask_d(
            pd()->attr()->dropout_.dropout_desc_);

    if (src_d.has_zero_dim() || weights_d.has_zero_dim()
            || dst_d.has_zero_dim())
        return status::success;

    const bool non_default_attrs = !pd()->attr()->has_default_values();

    matmul_helper_t helper(src_d, weights_d, dst_d);
    const int ndims = pd()->ndims();
    const int batch_ndims = ndims - 2;
    const dim_t M = helper.M();
    const dim_t N = helper.N();
    const dim_t K = helper.K();
    const dim_t batch = helper.batch();

    // Weights decompression
    const bool with_wei_decompression
            = utils::one_of(weights_d.data_type(), data_type::s8, data_type::u8,
                      data_type::s4, data_type::u4)
            && pd()->attr()->fpmath_.apply_to_int_;
    const auto &attr_zps = pd()->attr()->zero_points_;
    const bool with_wei_zero_points
            = !attr_zps.has_default_values(DNNL_ARG_WEIGHTS);
    int wei_zp_mask = attr_zps.get_mask(DNNL_ARG_WEIGHTS);
    const auto &wei_zp_dt = attr_zps.get_data_type(DNNL_ARG_WEIGHTS);
    const auto wei_zp_group_k = attr_zps.get_group(DNNL_ARG_WEIGHTS, 0);
    const auto wei_zp_group_n = attr_zps.get_group(DNNL_ARG_WEIGHTS, 1);
    // Initialize a memory desc for quant entries for easier offset calculation.
    memory_desc_t wei_zp_md {};
    CHECK(matmul_helper_t::get_quant_md(wei_zp_md, ndims, weights_d.dims(),
            wei_zp_mask, wei_zp_group_k, wei_zp_group_n, wei_zp_dt));

    const int src_mask
            = utils::get_dims_mask(dst_d.dims(), src_d.dims(), ndims);
    const int wei_mask
            = utils::get_dims_mask(dst_d.dims(), weights_d.dims(), ndims);
    const int bia_mask
            = utils::get_dims_mask(dst_d.dims(), bia_d.dims(), ndims);

    // arg scales section
    const auto &attr_scales = pd()->attr()->scales_;
    const bool with_src_scales = !attr_scales.has_default_values(DNNL_ARG_SRC);
    const bool with_wei_scales
            = !attr_scales.has_default_values(DNNL_ARG_WEIGHTS);
    const bool with_dst_scales = !attr_scales.has_default_values(DNNL_ARG_DST);
    const auto wei_scale_mask = attr_scales.get_mask(DNNL_ARG_WEIGHTS);
    const dim_t wei_scale_stride_n
            = (wei_scale_mask & pd()->wei_qmask_N()) ? 1 : 0;
    const auto &wei_scale_dt = attr_scales.get_data_type(DNNL_ARG_WEIGHTS);
    const auto wei_scales_d
            = ctx.memory_mdw(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
    const auto wei_scale_group_k = attr_scales.get_group(DNNL_ARG_WEIGHTS, 0);
    const auto wei_scale_group_n = attr_scales.get_group(DNNL_ARG_WEIGHTS, 1);
    // Initialize a memory desc for quant entries for easier offset calculation.
    memory_desc_t wei_scale_md {};
    CHECK(matmul_helper_t::get_quant_md(wei_scale_md, ndims, weights_d.dims(),
            wei_scale_mask, wei_scale_group_k, wei_scale_group_n,
            wei_scale_dt));

    auto dst_rnd_mode = pd()->attr()->rounding_mode_.get(DNNL_ARG_DST);

    // mm kernel
    auto ker = [&](const dims_t dst_dims_idx, dim_t m, dim_t n) {
        float acc = 0;
        dims_t src_dims_idx, weights_dims_idx;
        utils::copy_dims_with_mask(src_dims_idx, dst_dims_idx, ndims, src_mask);
        utils::copy_dims_with_mask(
                weights_dims_idx, dst_dims_idx, ndims, wei_mask);
        src_dims_idx[ndims - 2] = m;
        weights_dims_idx[ndims - 1] = n;
        auto &src_k_dim = src_dims_idx[ndims - 1];
        auto &wei_k_dim = weights_dims_idx[ndims - 2];
        for (dim_t k = 0; k < K; ++k) {
            src_k_dim = k;
            wei_k_dim = k;
            const auto src_off = src_d.off_v(src_dims_idx);
            const auto weights_off = weights_d.off_v(weights_dims_idx);
            const float s
                    = io::load_float_value(src_d.data_type(), src, src_off);
            float w = io::load_float_value(
                    weights_d.data_type(), weights, weights_off);
            // weights decompression should happen before the operation
            if (with_wei_decompression) {
                if (with_wei_zero_points) {
                    const dim_t wei_zp_offset = matmul_helper_t::get_quant_off(
                            weights_dims_idx, ndims, wei_zp_mask,
                            wei_zp_group_k, wei_zp_group_n, wei_zp_md);
                    const auto wei_zp = io::load_int_value(
                            wei_zp_dt, wei_zero_points, wei_zp_offset);
                    w -= wei_zp;
                }
                if (with_wei_scales) {
                    const dim_t wei_scale_offset
                            = matmul_helper_t::get_quant_off(weights_dims_idx,
                                    ndims, wei_scale_mask, wei_scale_group_k,
                                    wei_scale_group_n, wei_scale_md);
                    // Single scale value was already converted into f32.
                    const float wei_scale = wei_scales_d.nelems() == 1
                            ? wei_scales[0]
                            : io::load_float_value(
                                    wei_scale_dt, wei_scales, wei_scale_offset);
                    w *= wei_scale;
                }
            }
            acc += s * w;
        }
        return acc;
    };

    // bias section
    auto ker_bias = [&](const dims_t &dst_dims_idx) -> float {
        dims_t bia_dims_idx;
        utils::copy_dims_with_mask(bia_dims_idx, dst_dims_idx, ndims, bia_mask);
        const auto bias_off = bia_d.off_v(bia_dims_idx);
        return io::load_float_value(bia_d.data_type(), bias, bias_off);
    };

    auto sum_dt = pd()->attr()->post_ops_.get_sum_dt(dst_d.data_type());
    bool with_dropout = !pd()->attr()->dropout_.has_default_values();

    // computations Note: If dst type is < 8 bits, we cannot split a
    // byte during store or we get a race condition. To simplify
    // logic, we limit parallelization on M and N by a factor of 2.
    parallel_nd(batch, utils::div_up(M, 2), utils::div_up(N, 2),
            [&](dim_t mb, dim_t m_, dim_t n_) {
                for_(int m = 2 * m_; m < std::min<int>(2 * (m_ + 1), M); m++)
                for (int n = 2 * n_; n < std::min<int>(2 * (n_ + 1), N); n++) {
                    dims_t dst_dims_idx;
                    // account for M, N dims for index calculations
                    const size_t l_offset = mb * M * N + m * N + n;
                    utils::l_dims_by_l_offset(
                            dst_dims_idx, l_offset, dst_d.dims(), ndims);
                    float d = ker(dst_dims_idx, m, n);
                    if (with_src_scales) d *= src_scales[0];
                    if (with_wei_scales && !with_wei_decompression) {
                        // Single scale value was already converted into f32.
                        const float wei_scale = wei_scales_d.nelems() == 1
                                ? wei_scales[0]
                                : io::load_float_value(wei_scale_dt, wei_scales,
                                        wei_scale_stride_n * n);
                        d *= wei_scale;
                    }
                    if (bias) d += ker_bias(dst_dims_idx);

                    const auto dst_off = dst_d.off_v(dst_dims_idx);
                    if (non_default_attrs) {
                        if (with_dropout)
                            d = ref_dropout(
                                    d, dropout_mask, dst_off, *p, *seed);
                        ref_post_ops_t::args_t args;
                        args.dst_val
                                = io::load_float_value(sum_dt, dst, dst_off);
                        args.ctx = &ctx;
                        args.l_offset = l_offset;
                        args.dst_md = pd()->dst_md();
                        ref_post_ops->execute(d, args);
                    }
                    if (with_dst_scales) d *= dst_scales[0];
                    if (dst_rnd_mode == rounding_mode::stochastic)
                        d = math::stochastic_round_fwd(
                                d, dst_off, rnd_seed[0], dst_d.data_type());
                    io::store_float_value(dst_d.data_type(), d, dst, dst_off);
                    utils::dim_iterator(
                            dst_d.dims(), dst_dims_idx, batch_ndims);
                }
            });

    return status::success;
}

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace dnnl
