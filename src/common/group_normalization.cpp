/*******************************************************************************
* Copyright 2023 Intel Corporation
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
using namespace dnnl::impl::utils;
using namespace dnnl::impl::status;
using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::alg_kind;
using namespace dnnl::impl::types;

#define VCHECK_GNORM(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, gnorm, (cond), \
            status::invalid_arguments, msg, ##__VA_ARGS__);

#define VCHECK_GNORM_UNIMPL(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, gnorm, (cond), status::unimplemented, \
            msg, ##__VA_ARGS__);

namespace {
status_t group_normalization_desc_init(group_normalization_desc_t *desc,
        prop_kind_t prop_kind, const memory_desc_t *src_desc,
        const memory_desc_t *dst_desc, const memory_desc_t *diff_src_desc,
        const memory_desc_t *diff_dst_desc, dnnl_dim_t groups, float epsilon,
        unsigned flags) {
    const bool is_fwd = one_of(prop_kind, forward_training, forward_inference);
    VCHECK_GNORM(!any_null(desc, src_desc), VERBOSE_NULL_ARG);
    VCHECK_GNORM(one_of(prop_kind, forward_training, forward_inference,
                         backward_data, backward),
            VERBOSE_BAD_PROPKIND);
    VCHECK_GNORM(IMPLICATION(is_fwd, dst_desc != nullptr), VERBOSE_NULL_ARG);
    VCHECK_GNORM(IMPLICATION(!is_fwd, !any_null(diff_src_desc, diff_dst_desc)),
            VERBOSE_NULL_ARG);
    VCHECK_GNORM(
            IMPLICATION(is_fwd, !memory_desc_wrapper(src_desc).format_any()),
            VERBOSE_UNSUPPORTED_TAG_S, "src");

    unsigned gnorm_flags = normalization_flags::use_global_stats
            | normalization_flags::use_scale | normalization_flags::use_shift;
    VCHECK_GNORM((~gnorm_flags & flags) == 0, VERBOSE_BAD_FLAGS);

    {
        const dim_t c = src_desc->dims[1];
        VCHECK_GNORM(groups > 0 && groups <= c && c % groups == 0,
                VERBOSE_BAD_PARAM, "groups");
    }

    auto gd = group_normalization_desc_t();
    gd.primitive_kind = primitive_kind::group_normalization;
    gd.prop_kind = prop_kind;

    bool runtime_dims_or_strides
            = memory_desc_wrapper(src_desc).has_runtime_dims_or_strides();
    if (is_fwd) {
        runtime_dims_or_strides = runtime_dims_or_strides
                || memory_desc_wrapper(dst_desc).has_runtime_dims_or_strides();
    } else {
        runtime_dims_or_strides = runtime_dims_or_strides
                || memory_desc_wrapper(diff_src_desc)
                           .has_runtime_dims_or_strides()
                || memory_desc_wrapper(diff_dst_desc)
                           .has_runtime_dims_or_strides();
    }
    VCONDCHECK(primitive, create, check, bnorm, !runtime_dims_or_strides,
            status::unimplemented, VERBOSE_RUNTIMEDIM_UNSUPPORTED);

    gd.src_desc = *src_desc;
    if (is_fwd) gd.dst_desc = *dst_desc;
    if (!is_fwd) {
        gd.diff_src_desc = *diff_src_desc;
        gd.diff_dst_desc = *diff_dst_desc;
    }

    const bool has_scale_or_shift = flags
            & (normalization_flags::use_scale | normalization_flags::use_shift);
    if (has_scale_or_shift) {
        dims_t scaleshift_dims = {src_desc->dims[1]};
        CHECK(memory_desc_init_by_tag(gd.scaleshift_desc, 1, scaleshift_dims,
                data_type::f32, format_tag::x));
        if (!is_fwd) gd.diff_scaleshift_desc = gd.scaleshift_desc;
    }

    dims_t stats_dims = {src_desc->dims[0], groups};
    CHECK(memory_desc_init_by_tag(
            gd.stat_desc, 2, stats_dims, data_type::f32, format_tag::ab));

    VCHECK_GNORM(gd.src_desc.ndims >= 2, VERBOSE_BAD_NDIMS, "src",
            gd.src_desc.ndims);
#define CHECK_DIMS(t1, t2) \
    do { \
        VCHECK_GNORM(gd.t2##_desc.ndims == gd.t1##_desc.ndims, \
                VERBOSE_INCONSISTENT_NDIMS, #t1, #t2); \
        VCHECK_GNORM(array_cmp(gd.t2##_desc.dims, gd.t1##_desc.dims, \
                             gd.t1##_desc.ndims), \
                VERBOSE_INCONSISTENT_DIM, #t1, -1, #t2, -1); \
    } while (0)

    if (is_fwd) {
        CHECK_DIMS(src, dst);
    } else {
        CHECK_DIMS(src, diff_dst);
        CHECK_DIMS(diff_src, diff_dst);
    }
#undef CHECK_DIMS

    gd.groups = groups;
    gd.group_norm_epsilon = epsilon;
    gd.flags = flags;

    *desc = gd;
    return success;
}

status_t group_normalization_attr_check(const group_normalization_desc_t &desc,
        const engine_t *engine, const primitive_attr_t *attr) {
    using smask_t = primitive_attr_t::skip_mask_t;

    if (attr == nullptr) return status::success;
    if (attr->has_default_values()) return status::success;

    // Check attributes
    if (utils::one_of(desc.prop_kind, prop_kind::forward_inference,
                prop_kind::forward_training)) {
        const data_type_t src_dt = desc.src_desc.data_type;
        const data_type_t dst_dt = desc.dst_desc.data_type;

        auto fwd_attr_mask = smask_t::post_ops;

        const bool is_int8 = utils::one_of(src_dt, data_type::s8, data_type::u8)
                || utils::one_of(dst_dt, data_type::s8, data_type::u8);
        if (is_int8) fwd_attr_mask |= smask_t::scales_runtime;

        VCHECK_GNORM_UNIMPL(attr->has_default_values(fwd_attr_mask, dst_dt),
                VERBOSE_UNSUPPORTED_ATTR);

        // Check scales
        if (!attr->scales_.has_default_values()) {
            const auto &sc = attr->scales_;
            const int mask_src = sc.get(DNNL_ARG_SRC).mask_;
            const int mask_dst = sc.get(DNNL_ARG_DST).mask_;

            VCHECK_GNORM_UNIMPL(utils::everyone_is(0, mask_src, mask_dst),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
        }

        // Check post-ops
        if (!attr->post_ops_.has_default_values()) {
            const auto &po = attr->post_ops_;
            using namespace primitive_kind;
            VCHECK_GNORM_UNIMPL(po.has_default_values({binary, eltwise}),
                    VERBOSE_UNSUPPORTED_POSTOP);
        }
    } else {
        VCHECK_GNORM_UNIMPL(false, VERBOSE_UNSUPPORTED_ATTR);
    }

    return status::success;
}

} // namespace

status_t dnnl_group_normalization_forward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        prop_kind_t prop_kind, const memory_desc_t *src_desc,
        const memory_desc_t *dst_desc, dnnl_dim_t groups, float epsilon,
        unsigned flags, const primitive_attr_t *attr) {
    VCHECK_GNORM(one_of(prop_kind, forward_training, forward_inference),
            VERBOSE_BAD_PROPKIND);

    auto desc = group_normalization_desc_t();
    CHECK(group_normalization_desc_init(&desc, prop_kind, src_desc, dst_desc,
            nullptr, nullptr, groups, epsilon, flags));
    CHECK(group_normalization_attr_check(desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&desc, nullptr, attr);
}

status_t dnnl_group_normalization_backward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        prop_kind_t prop_kind, const memory_desc_t *diff_src_desc,
        const memory_desc_t *diff_dst_desc, const memory_desc_t *src_desc,
        dnnl_dim_t groups, float epsilon, unsigned flags,
        const primitive_desc_iface_t *hint_fwd_pd,
        const primitive_attr_t *attr) {
    VCHECK_GNORM(
            one_of(prop_kind, backward, backward_data), VERBOSE_BAD_PROPKIND);

    auto desc = group_normalization_desc_t();
    CHECK(group_normalization_desc_init(&desc, prop_kind, src_desc, nullptr,
            diff_src_desc, diff_dst_desc, groups, epsilon, flags));
    CHECK(group_normalization_attr_check(desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&desc, hint_fwd_pd, attr);
}
