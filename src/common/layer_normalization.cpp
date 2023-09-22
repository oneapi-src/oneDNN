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
#include "oneapi/dnnl/dnnl.h"
#include "opdesc.hpp"
#include "primitive_desc_iface.hpp"

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::status;
using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::types;

#define VCHECK_LNORM(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, lnorm, (cond), \
            status::invalid_arguments, msg, ##__VA_ARGS__);

#define VCHECK_LNORM_UNIMPL(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, lnorm, (cond), status::unimplemented, \
            msg, ##__VA_ARGS__);

namespace {
status_t lnorm_desc_init(layer_normalization_desc_t *lnorm_desc,
        prop_kind_t prop_kind, const memory_desc_t *src_desc,
        const memory_desc_t *dst_desc, const memory_desc_t *stat_desc,
        const memory_desc_t *diff_src_desc, const memory_desc_t *diff_dst_desc,
        float epsilon, unsigned flags) {
    VCHECK_LNORM(!any_null(lnorm_desc, src_desc), VERBOSE_NULL_ARG);
    VCHECK_LNORM(2 <= src_desc->ndims && src_desc->ndims <= 5,
            VERBOSE_BAD_NDIMS, "src", src_desc->ndims);

    VCHECK_LNORM((flags
                         & ~(normalization_flags::use_global_stats
                                 | normalization_flags::use_scale
                                 | normalization_flags::use_shift))
                    == 0,
            VERBOSE_BAD_FLAGS);

    bool is_fwd
            = prop_kind == forward_training || prop_kind == forward_inference;
    VCHECK_LNORM(IMPLICATION(is_fwd, dst_desc != nullptr), VERBOSE_NULL_ARG);
    VCHECK_LNORM(IMPLICATION(!is_fwd, !any_null(diff_src_desc, diff_dst_desc)),
            VERBOSE_NULL_ARG);
    VCHECK_LNORM(
            IMPLICATION(is_fwd, !memory_desc_wrapper(src_desc).format_any()),
            VERBOSE_UNSUPPORTED_TAG_S, "src");

    auto ld = layer_normalization_desc_t();
    ld.primitive_kind = primitive_kind::layer_normalization;
    ld.prop_kind = prop_kind;

    bool runtime_dims_or_strides
            = memory_desc_wrapper(src_desc).has_runtime_dims_or_strides()
            || memory_desc_wrapper(dst_desc).has_runtime_dims_or_strides()
            || (stat_desc
                    && memory_desc_wrapper(stat_desc)
                               .has_runtime_dims_or_strides());
    if (!is_fwd)
        runtime_dims_or_strides = runtime_dims_or_strides
                || memory_desc_wrapper(diff_src_desc)
                           .has_runtime_dims_or_strides()
                || memory_desc_wrapper(diff_dst_desc)
                           .has_runtime_dims_or_strides();
    VCONDCHECK(primitive, create, check, lnorm, !runtime_dims_or_strides,
            status::unimplemented, VERBOSE_RUNTIMEDIM_UNSUPPORTED);

    ld.src_desc = *src_desc;
    if (is_fwd) ld.dst_desc = *dst_desc;
    if (!is_fwd) ld.diff_src_desc = *diff_src_desc;
    if (!is_fwd) ld.diff_dst_desc = *diff_dst_desc;

    if (stat_desc)
        ld.stat_desc = *stat_desc;
    else
        VCHECK_LNORM(
                memory_desc_init_by_tag(ld.stat_desc, ld.src_desc.ndims - 1,
                        ld.src_desc.dims, data_type::f32, format_tag::any)
                        == success,
                VERBOSE_UNSUPPORTED_TAG_S, "stats");

    int ndims = src_desc->ndims;
    ld.data_scaleshift_desc = zero_md();
    if (flags
            & (normalization_flags::use_scale
                    | normalization_flags::use_shift)) {
        dims_t scaleshift_dims = {src_desc->dims[ndims - 1]};
        memory_desc_init_by_tag(ld.data_scaleshift_desc, 1, scaleshift_dims,
                data_type::f32, dnnl_x);
    } else {
        dims_t scaleshift_dims = {2, src_desc->dims[ndims - 1]};
        memory_desc_init_by_tag(ld.data_scaleshift_desc, 2, scaleshift_dims,
                data_type::f32, dnnl_nc);
    }
    if (ld.prop_kind == backward) {
        ld.diff_data_scaleshift_desc = ld.data_scaleshift_desc;
    }

    ld.layer_norm_epsilon = epsilon;

    ld.flags = flags;

#define CHECK_DIMS(t1, t2, off_ndims) \
    do { \
        VCHECK_LNORM(ld.t1##_desc.ndims == ld.t2##_desc.ndims + (off_ndims), \
                VERBOSE_INCONSISTENT_NDIMS, #t1, #t2); \
        VCHECK_LNORM(array_cmp(ld.t1##_desc.dims, ld.t2##_desc.dims, \
                             ld.t2##_desc.ndims), \
                VERBOSE_INCONSISTENT_DIM, #t1, -1, #t2, -1); \
    } while (0)

    if (is_fwd) {
        CHECK_DIMS(src, dst, 0);
    } else {
        CHECK_DIMS(src, diff_src, 0);
        CHECK_DIMS(src, diff_dst, 0);
        CHECK_DIMS(src, stat, 1);
    }
#undef CHECK_DIMS

    *lnorm_desc = ld;
    return success;
}

status_t layer_normalization_attr_check(const layer_normalization_desc_t &desc,
        const engine_t *engine, const primitive_attr_t *attr) {
    using smask_t = primitive_attr_t::skip_mask_t;

    if (attr == nullptr) return status::success;
    if (attr->has_default_values()) return status::success;

    // Check attributes
    if (utils::one_of(desc.prop_kind, prop_kind::forward_inference,
                prop_kind::forward_training)) {
        const data_type_t src_dt = desc.src_desc.data_type;
        const data_type_t dst_dt = desc.dst_desc.data_type;

        auto fwd_attr_mask = smask_t::none;

        const bool is_int8 = utils::one_of(src_dt, data_type::s8, data_type::u8)
                || utils::one_of(dst_dt, data_type::s8, data_type::u8);
        if (is_int8) fwd_attr_mask |= smask_t::scales_runtime;

        VCHECK_LNORM_UNIMPL(attr->has_default_values(fwd_attr_mask, dst_dt),
                VERBOSE_UNSUPPORTED_ATTR);

        // Check scales
        if (!attr->scales_.has_default_values()) {
            const auto &sc = attr->scales_;
            const int mask_src = sc.get(DNNL_ARG_SRC).mask_;
            const int mask_dst = sc.get(DNNL_ARG_DST).mask_;

            VCHECK_LNORM_UNIMPL(utils::everyone_is(0, mask_src, mask_dst),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
        }
    } else {
        VCHECK_LNORM_UNIMPL(false, VERBOSE_UNSUPPORTED_ATTR);
    }

    return status::success;
}

} // namespace

status_t dnnl_layer_normalization_forward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        prop_kind_t prop_kind, const memory_desc_t *src_desc,
        const memory_desc_t *dst_desc, const memory_desc_t *stat_desc,
        float epsilon, unsigned flags, const primitive_attr_t *attr) {
    if (!one_of(prop_kind, forward_training, forward_inference))
        return invalid_arguments;

    auto lnorm_desc = layer_normalization_desc_t();
    CHECK(lnorm_desc_init(&lnorm_desc, prop_kind, src_desc, dst_desc, stat_desc,
            nullptr, nullptr, epsilon, flags));
    CHECK(layer_normalization_attr_check(lnorm_desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&lnorm_desc, nullptr, attr);
}

status_t dnnl_layer_normalization_backward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        prop_kind_t prop_kind, const memory_desc_t *diff_src_desc,
        const memory_desc_t *diff_dst_desc, const memory_desc_t *src_desc,
        const memory_desc_t *stat_desc, float epsilon, unsigned flags,
        const primitive_desc_iface_t *hint_fwd_pd,
        const primitive_attr_t *attr) {
    if (!one_of(prop_kind, backward, backward_data)) return invalid_arguments;

    auto lnorm_desc = layer_normalization_desc_t();
    CHECK(lnorm_desc_init(&lnorm_desc, prop_kind, src_desc, nullptr, stat_desc,
            diff_src_desc, diff_dst_desc, epsilon, flags));
    CHECK(layer_normalization_attr_check(lnorm_desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&lnorm_desc, hint_fwd_pd, attr);
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
