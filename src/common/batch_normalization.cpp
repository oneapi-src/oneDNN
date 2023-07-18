/*******************************************************************************
* Copyright 2016-2023 Intel Corporation
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
using namespace dnnl::impl::alg_kind;
using namespace dnnl::impl::types;

#define VCHECK_BNORM(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, bnorm, (cond), \
            status::invalid_arguments, msg, ##__VA_ARGS__);

#define VCHECK_BNORM_UNIMPL(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, bnorm, (cond), status::unimplemented, \
            msg, ##__VA_ARGS__);

namespace {
status_t bnrm_desc_init(batch_normalization_desc_t *bnrm_desc,
        prop_kind_t prop_kind, const memory_desc_t *src_desc,
        const memory_desc_t *dst_desc, const memory_desc_t *diff_src_desc,
        const memory_desc_t *diff_dst_desc, float epsilon, unsigned flags) {
    const bool is_fwd = one_of(prop_kind, forward_training, forward_inference);
    VCHECK_BNORM(!any_null(bnrm_desc, src_desc), VERBOSE_NULL_ARG);
    VCHECK_BNORM(one_of(prop_kind, forward_training, forward_inference,
                         backward_data, backward),
            VERBOSE_BAD_PROPKIND);
    VCHECK_BNORM(IMPLICATION(is_fwd, dst_desc != nullptr), VERBOSE_NULL_ARG);
    VCHECK_BNORM(IMPLICATION(!is_fwd, !any_null(diff_src_desc, diff_dst_desc)),
            VERBOSE_NULL_ARG);
    VCHECK_BNORM(
            IMPLICATION(is_fwd, !memory_desc_wrapper(src_desc).format_any()),
            VERBOSE_UNSUPPORTED_TAG_S, "src");

    unsigned bnorm_flags = normalization_flags::use_global_stats
            | normalization_flags::fuse_norm_relu
            | normalization_flags::fuse_norm_add_relu
            | normalization_flags::use_scale | normalization_flags::use_shift;
    VCHECK_BNORM((~bnorm_flags & flags) == 0, VERBOSE_BAD_FLAGS);

    auto bd = batch_normalization_desc_t();
    bd.primitive_kind = primitive_kind::batch_normalization;
    bd.prop_kind = prop_kind;

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

    bd.src_desc = *src_desc;
    if (is_fwd) bd.dst_desc = *dst_desc;
    if (!is_fwd) {
        bd.diff_src_desc = *diff_src_desc;
        bd.diff_dst_desc = *diff_dst_desc;
    }

    const bool has_scale_or_shift = flags
            & (normalization_flags::use_scale | normalization_flags::use_shift);
    if (has_scale_or_shift) {
        dims_t scaleshift_dims = {src_desc->dims[1]};
        memory_desc_init_by_tag(bd.scaleshift_desc, 1, scaleshift_dims,
                data_type::f32, format_tag::a);
        if (!is_fwd) bd.diff_scaleshift_desc = bd.scaleshift_desc;
    }

    dims_t stats_dims = {src_desc->dims[1]};
    memory_desc_init_by_tag(
            bd.stat_desc, 1, stats_dims, data_type::f32, format_tag::a);

    bd.batch_norm_epsilon = epsilon;
    bd.flags = flags;

    VCHECK_BNORM(bd.src_desc.ndims >= 2, VERBOSE_BAD_NDIMS, "src",
            bd.src_desc.ndims);
#define CHECK_DIMS(t1, t2) \
    do { \
        VCHECK_BNORM(bd.t2##_desc.ndims == bd.t1##_desc.ndims, \
                VERBOSE_INCONSISTENT_NDIMS, #t1, #t2); \
        VCHECK_BNORM(array_cmp(bd.t2##_desc.dims, bd.t1##_desc.dims, \
                             bd.t1##_desc.ndims), \
                VERBOSE_INCONSISTENT_DIM, #t1, -1, #t2, -1); \
    } while (0)

    if (is_fwd) {
        CHECK_DIMS(src, dst);
    } else {
        CHECK_DIMS(src, diff_dst);
        CHECK_DIMS(diff_src, diff_dst);
    }
#undef CHECK_DIMS

    *bnrm_desc = bd;
    return success;
}

status_t batch_normalization_attr_check(const batch_normalization_desc_t &desc,
        const engine_t *engine, const primitive_attr_t *attr) {
    using smask_t = primitive_attr_t::skip_mask_t;

    if (attr == nullptr) return status::success;
    if (attr->has_default_values()) return status::success;

    if (utils::one_of(desc.prop_kind, prop_kind::forward_inference,
                prop_kind::forward_training)) {
        // Check attributes
        const data_type_t dst_dt = desc.dst_desc.data_type;

        auto attr_mask = smask_t::post_ops;

        VCHECK_BNORM_UNIMPL(attr->has_default_values(attr_mask, dst_dt),
                VERBOSE_UNSUPPORTED_ATTR);

        // Check post-ops
        if (!attr->post_ops_.has_default_values()) {
            const auto &po = attr->post_ops_;
            using namespace primitive_kind;
            VCHECK_BNORM_UNIMPL(po.has_default_values({eltwise}),
                    VERBOSE_UNSUPPORTED_POSTOP);
        }
    } else {
        VCHECK_BNORM_UNIMPL(false, VERBOSE_UNSUPPORTED_ATTR);
    }

    return status::success;
}

} // namespace

status_t dnnl_batch_normalization_forward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        prop_kind_t prop_kind, const memory_desc_t *src_desc,
        const memory_desc_t *dst_desc, float epsilon, unsigned flags,
        const primitive_attr_t *attr) {
    if (!one_of(prop_kind, forward_training, forward_inference))
        return invalid_arguments;

    auto bnrm_desc = batch_normalization_desc_t();
    CHECK(bnrm_desc_init(&bnrm_desc, prop_kind, src_desc, dst_desc, nullptr,
            nullptr, epsilon, flags));
    CHECK(batch_normalization_attr_check(bnrm_desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&bnrm_desc, nullptr, attr);
}

status_t dnnl_batch_normalization_backward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        prop_kind_t prop_kind, const memory_desc_t *diff_src_desc,
        const memory_desc_t *diff_dst_desc, const memory_desc_t *src_desc,
        float epsilon, unsigned flags,
        const primitive_desc_iface_t *hint_fwd_pd,
        const primitive_attr_t *attr) {
    if (!one_of(prop_kind, backward, backward_data)) return invalid_arguments;

    auto bnrm_desc = batch_normalization_desc_t();
    CHECK(bnrm_desc_init(&bnrm_desc, prop_kind, src_desc, nullptr,
            diff_src_desc, diff_dst_desc, epsilon, flags));
    CHECK(batch_normalization_attr_check(bnrm_desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&bnrm_desc, hint_fwd_pd, attr);
}

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
