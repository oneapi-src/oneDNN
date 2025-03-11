/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#include "oneapi/dnnl/dnnl.h"
#include "opdesc.hpp"
#include "primitive_desc_iface.hpp"

#include "c_types_map.hpp"
#include "utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::alg_kind;

#define VCHECK_RED(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, reduction, (cond), \
            status::invalid_arguments, msg, ##__VA_ARGS__);

#define VCHECK_RED_UNIMPL(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, reduction, (cond), \
            status::unimplemented, msg, ##__VA_ARGS__);
namespace dnnl {
namespace impl {

status_t reduction_desc_init(reduction_desc_t *reduction_desc,
        alg_kind_t alg_kind, const memory_desc_t *src_desc,
        const memory_desc_t *dst_desc, float p, float eps) {

    VCHECK_RED(!any_null(src_desc, dst_desc), VERBOSE_NULL_ARG);
    VCHECK_RED(src_desc->format_kind != format_kind::any,
            VERBOSE_UNSUPPORTED_TAG_S, "src");
    VCHECK_RED(one_of(alg_kind, reduction_max, reduction_min, reduction_sum,
                       reduction_mul, reduction_mean, reduction_norm_lp_max,
                       reduction_norm_lp_sum, reduction_norm_lp_power_p_max,
                       reduction_norm_lp_power_p_sum),
            VERBOSE_BAD_ALGORITHM);
    VCHECK_RED(IMPLICATION(one_of(alg_kind, reduction_norm_lp_max,
                                   reduction_norm_lp_sum,
                                   reduction_norm_lp_power_p_max,
                                   reduction_norm_lp_power_p_sum),
                       p >= 1.0f),
            VERBOSE_BAD_PARAM, "p");
    VCHECK_RED(IMPLICATION(one_of(alg_kind, reduction_norm_lp_max,
                                   reduction_norm_lp_sum,
                                   reduction_norm_lp_power_p_max,
                                   reduction_norm_lp_power_p_sum),
                       one_of(src_desc->data_type, data_type::f32,
                               data_type::bf16, data_type::f16)),
            VERBOSE_INVALID_DATATYPE, "src");

    VCHECK_RED(src_desc->ndims == dst_desc->ndims, VERBOSE_INCONSISTENT_NDIMS,
            "src", "dst");

    for (auto d = 0; d < src_desc->ndims; ++d) {
        const auto dst_dim_d = dst_desc->dims[d];
        VCHECK_RED(one_of(dst_dim_d, 1, src_desc->dims[d]),
                VERBOSE_INCONSISTENT_DIM, "src", d, "dst", d);
    }

    // reduction primitive doesn't support identity operation
    VCHECK_RED(!array_cmp(src_desc->dims, dst_desc->dims, src_desc->ndims),
            VERBOSE_INCONSISTENT_DIM, "src", -1, "dst", -1);

    VCHECK_RED(src_desc->format_kind == format_kind::blocked,
            VERBOSE_UNSUPPORTED_TAG_S, "src");
    VCHECK_RED(one_of(dst_desc->format_kind, format_kind::blocked,
                       format_kind::any),
            VERBOSE_UNSUPPORTED_TAG_S, "dst");

    VCHECK_RED(src_desc->extra.flags == 0, VERBOSE_UNSUPPORTED_MD_FLAG, "src");
    VCHECK_RED(IMPLICATION(dst_desc->format_kind == format_kind::blocked,
                       dst_desc->extra.flags == 0),
            VERBOSE_UNSUPPORTED_MD_FLAG, "dst");

    auto rd = reduction_desc_t();
    rd.primitive_kind = primitive_kind::reduction;
    rd.alg_kind = alg_kind;

    rd.src_desc = *src_desc;
    rd.dst_desc = *dst_desc;
    rd.p = p;
    rd.eps = eps;

    (*reduction_desc) = rd;
    return success;
}

status_t reduction_attr_check(const reduction_desc_t &desc,
        const engine_t *engine, const primitive_attr_t *attr) {
    using smask_t = primitive_attr_t::skip_mask_t;

    if (attr == nullptr) return status::success;
    if (attr->has_default_values()) return status::success;

    // Check attributes
    const data_type_t dst_dt = desc.dst_desc.data_type;

    auto attr_mask = smask_t::post_ops;

    VCHECK_RED_UNIMPL(attr->has_default_values(attr_mask, dst_dt),
            VERBOSE_UNSUPPORTED_ATTR);

    // Check post-ops
    if (!attr->post_ops_.has_default_values()) {
        const auto &po = attr->post_ops_;
        using namespace primitive_kind;
        VCHECK_RED_UNIMPL(po.has_default_values({binary, eltwise, sum}),
                VERBOSE_UNSUPPORTED_POSTOP);

        // Check sum
        VCHECK_RED_UNIMPL(po.check_sum_consistency(dst_dt, false, true),
                VERBOSE_UNSUPPORTED_POSTOP);
    }

    return status::success;
}

} // namespace impl
} // namespace dnnl

dnnl_status_t dnnl_reduction_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        alg_kind_t alg_kind, const memory_desc_t *src_desc,
        const memory_desc_t *dst_desc, float p, float eps,
        const primitive_attr_t *attr) {

    auto reduction_desc = reduction_desc_t();
    CHECK(reduction_desc_init(
            &reduction_desc, alg_kind, src_desc, dst_desc, p, eps));
    CHECK(reduction_attr_check(reduction_desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&reduction_desc, nullptr, attr);
}
