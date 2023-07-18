/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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
#include "opdesc.hpp"
#include "primitive_desc_iface.hpp"

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::types;

#define VCHECK_MATMUL(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, matmul, (cond), \
            status::invalid_arguments, msg, ##__VA_ARGS__);

#define VCHECK_MATMUL_UNIMPL(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, matmul, (cond), \
            status::unimplemented, msg, ##__VA_ARGS__);

namespace {
status_t matmul_attr_check(const matmul_desc_t &desc, const engine_t *engine,
        const primitive_attr_t *attr) {
    using smask_t = primitive_attr_t::skip_mask_t;

    if (attr == nullptr) return status::success;
    if (attr->has_default_values()) return status::success;

    // Check attributes
    const data_type_t src_dt = desc.src_desc.data_type;
    const data_type_t dst_dt = desc.dst_desc.data_type;

    // Matmul supports scales for floating point data types
    auto attr_mask
            = smask_t::post_ops | smask_t::sum_dt | smask_t::scales_runtime;

    const bool is_int8 = utils::one_of(src_dt, data_type::s8, data_type::u8);
    if (is_int8) attr_mask |= smask_t::zero_points_runtime;

    VCHECK_MATMUL_UNIMPL(attr->has_default_values(attr_mask, dst_dt),
            VERBOSE_UNSUPPORTED_ATTR);

    // Check scales
    if (!attr->scales_.has_default_values()) {
        const auto &sc = attr->scales_;
        const int mask_src = sc.get(DNNL_ARG_SRC).mask_;
        const int mask_wei = sc.get(DNNL_ARG_WEIGHTS).mask_;
        const int mask_dst = sc.get(DNNL_ARG_DST).mask_;

        VCHECK_MATMUL_UNIMPL(utils::everyone_is(0, mask_src, mask_dst)
                        && utils::one_of(mask_wei, 0,
                                1 << (desc.weights_desc.ndims - 1)),
                VERBOSE_UNSUPPORTED_SCALES_CFG);
    }

    // Check zero points
    if (!attr->zero_points_.has_default_values()) {
        const auto &zp = attr->zero_points_;
        int mask_src = 0, mask_wei = 0, mask_dst = 0;
        zp.get(DNNL_ARG_SRC, &mask_src);
        zp.get(DNNL_ARG_WEIGHTS, &mask_wei);
        zp.get(DNNL_ARG_DST, &mask_dst);

        VCHECK_MATMUL_UNIMPL(mask_wei == 0
                        && (mask_src == 0
                                || (desc.src_desc.ndims == 2
                                        && mask_src == 1 << 1))
                        && (mask_dst == 0
                                || (desc.dst_desc.ndims == 2
                                        && mask_dst == 1 << 1)),
                VERBOSE_UNSUPPORTED_ZP_CFG);
    }

    // Check post-ops
    if (!attr->post_ops_.has_default_values()) {
        const auto &po = attr->post_ops_;
        using namespace primitive_kind;
        VCHECK_MATMUL_UNIMPL(
                po.has_default_values({binary, eltwise, prelu, sum}),
                VERBOSE_UNSUPPORTED_POSTOP);

        // Check sum
        VCHECK_MATMUL_UNIMPL(po.check_sum_consistency(dst_dt, is_int8, true),
                VERBOSE_UNSUPPORTED_POSTOP);
    }

    return status::success;
}

} // namespace

status_t dnnl_matmul_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        const memory_desc_t *src_md, const memory_desc_t *weights_md,
        const memory_desc_t *bias_md, const memory_desc_t *dst_md,
        const primitive_attr_t *attr) {
    VCHECK_MATMUL(!any_null(src_md, weights_md, dst_md), VERBOSE_NULL_ARG);

    auto op_d = matmul_desc_t();
    op_d.primitive_kind = primitive_kind::matmul;

    op_d.src_desc = *src_md;
    op_d.weights_desc = *weights_md;
    if (bias_md) op_d.bias_desc = *bias_md;
    op_d.dst_desc = *dst_md;

    const bool with_bias = op_d.bias_desc.ndims != 0;
    const int ndims = dst_md->ndims;
    VCHECK_MATMUL(ndims >= 2 && ndims <= DNNL_MAX_NDIMS, VERBOSE_BAD_NDIMS,
            "dst", ndims);
    VCHECK_MATMUL(everyone_is(ndims, src_md->ndims, weights_md->ndims),
            VERBOSE_INCONSISTENT_NDIMS, "src", "weights");
    VCHECK_MATMUL(IMPLICATION(with_bias, op_d.bias_desc.ndims == ndims),
            VERBOSE_BAD_NDIMS, "bias", op_d.bias_desc.ndims);

    // check: m, n, k
    const int m_idx = ndims - 2;
    const int k_idx_src = m_idx + 1;
    const int k_idx_wei = m_idx;
    const int n_idx = ndims - 1;
    VCHECK_MATMUL(dst_md->dims[m_idx] == src_md->dims[m_idx],
            VERBOSE_INCONSISTENT_DIM, "dst", m_idx, "src", m_idx);
    VCHECK_MATMUL(dst_md->dims[n_idx] == weights_md->dims[n_idx],
            VERBOSE_INCONSISTENT_DIM, "dst", n_idx, "weights", n_idx);
    VCHECK_MATMUL(src_md->dims[k_idx_src] == weights_md->dims[k_idx_wei],
            VERBOSE_INCONSISTENT_DIM, "src", k_idx_src, "weights", k_idx_wei);
    VCHECK_MATMUL(
            IMPLICATION(with_bias,
                    one_of(op_d.bias_desc.dims[n_idx], 1, dst_md->dims[n_idx])),
            VERBOSE_INCONSISTENT_DIM, "bias", n_idx, "dst", n_idx);
    VCHECK_MATMUL(
            IMPLICATION(with_bias,
                    one_of(op_d.bias_desc.dims[m_idx], 1, dst_md->dims[m_idx])),
            VERBOSE_INCONSISTENT_DIM, "bias", m_idx, "dst", m_idx);

    const int bia_mask = with_bias
            ? utils::get_dims_mask(dst_md->dims, op_d.bias_desc.dims, ndims)
            : 0;

    // check if other dims match.
    for (int d = 0; d < ndims - 2; ++d) {
        const dim_t s_dim = src_md->dims[d];
        const dim_t w_dim = weights_md->dims[d];
        const dim_t d_dim = dst_md->dims[d];
        const dim_t b_dim = with_bias ? op_d.bias_desc.dims[d] : 0;

        if (one_of(DNNL_RUNTIME_DIM_VAL, s_dim, w_dim, d_dim, b_dim)) {

            VCHECK_MATMUL(everyone_is(DNNL_RUNTIME_DIM_VAL, s_dim, w_dim, d_dim)
                            && IMPLICATION((bia_mask & (1 << d)) && with_bias,
                                    b_dim == DNNL_RUNTIME_DIM_VAL),
                    VERBOSE_RUNTIMEDIM_INCONSISTENT, d);
        } else {
            // This follows numpy semantics of broadcasting when 0 is involved.
            VCHECK_MATMUL(IMPLICATION(!everyone_is(s_dim, w_dim, d_dim),
                                  one_of(1, s_dim, w_dim)),
                    VERBOSE_INVALID_BROADCAST, "dst", d);
            VCHECK_MATMUL(IMPLICATION(s_dim == 1, d_dim == w_dim),
                    VERBOSE_INVALID_BROADCAST, "weights", d);
            VCHECK_MATMUL(IMPLICATION(w_dim == 1, d_dim == s_dim),
                    VERBOSE_INVALID_BROADCAST, "src", d);
            VCHECK_MATMUL(IMPLICATION(with_bias, one_of(b_dim, 1, d_dim)),
                    VERBOSE_INCONSISTENT_DIM, "bias", d, "dst", d);
        }
    }

    op_d.accum_data_type = types::default_accum_data_type(src_md->data_type,
            weights_md->data_type, dst_md->data_type, prop_kind::forward);
    VCHECK_MATMUL(op_d.accum_data_type != data_type::undef,
            VERBOSE_INVALID_DATATYPE, "accumulation");
    CHECK(matmul_attr_check(op_d, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&op_d, nullptr, attr);
}
