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
#include "memory_desc_wrapper.hpp"
#include "utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::status;
using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::alg_kind;
using namespace dnnl::impl::types;

#define VCHECK_SOFTMAX(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, softmax, (cond), \
            status::invalid_arguments, msg, ##__VA_ARGS__);

#define VCHECK_SOFTMAX_UNIMPL(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, softmax, (cond), \
            status::unimplemented, msg, ##__VA_ARGS__);

namespace {
status_t softmax_desc_init(softmax_desc_t *softmax_desc, prop_kind_t prop_kind,
        alg_kind_t alg_kind, const memory_desc_t *src_desc,
        const memory_desc_t *dst_desc, const memory_desc_t *diff_src_desc,
        const memory_desc_t *diff_dst_desc, int softmax_axis) {
    const bool is_fwd = one_of(prop_kind, forward_training, forward_inference);
    VCHECK_SOFTMAX(!any_null(softmax_desc, dst_desc), VERBOSE_NULL_ARG);
    VCHECK_SOFTMAX(IMPLICATION(is_fwd, src_desc != nullptr), VERBOSE_NULL_ARG);
    VCHECK_SOFTMAX(
            IMPLICATION(!is_fwd, !any_null(diff_src_desc, diff_dst_desc)),
            VERBOSE_NULL_ARG);
    VCHECK_SOFTMAX(one_of(alg_kind, softmax_accurate, softmax_log),
            VERBOSE_BAD_ALGORITHM);
    VCHECK_SOFTMAX(0 <= softmax_axis && softmax_axis < dst_desc->ndims,
            VERBOSE_BAD_AXIS);
    VCHECK_SOFTMAX(
            IMPLICATION(is_fwd, !memory_desc_wrapper(src_desc).format_any()),
            VERBOSE_UNSUPPORTED_TAG_S, "src");
    VCHECK_SOFTMAX(
            IMPLICATION(!is_fwd, !memory_desc_wrapper(dst_desc).format_any()),
            VERBOSE_UNSUPPORTED_TAG_S, "dst");

    bool runtime_dims_or_strides
            = memory_desc_wrapper(dst_desc).has_runtime_dims_or_strides();
    if (is_fwd) {
        runtime_dims_or_strides = runtime_dims_or_strides
                || memory_desc_wrapper(src_desc).has_runtime_dims_or_strides();
    } else {
        runtime_dims_or_strides = runtime_dims_or_strides
                || memory_desc_wrapper(diff_src_desc)
                           .has_runtime_dims_or_strides()
                || memory_desc_wrapper(diff_dst_desc)
                           .has_runtime_dims_or_strides();
    }
    VCONDCHECK(primitive, create, check, softmax, !runtime_dims_or_strides,
            status::unimplemented, VERBOSE_RUNTIMEDIM_UNSUPPORTED);

    auto sd = softmax_desc_t();
    sd.primitive_kind = primitive_kind::softmax;
    sd.prop_kind = prop_kind;

    if (is_fwd) sd.src_desc = *src_desc;
    if (!is_fwd) sd.diff_src_desc = *diff_src_desc;
    sd.softmax_axis = softmax_axis;
    sd.alg_kind = alg_kind;
    sd.dst_desc = *dst_desc;
    if (!is_fwd) sd.diff_dst_desc = *diff_dst_desc;

    *softmax_desc = sd;
    return success;
}

status_t softmax_attr_check(const softmax_desc_t &desc, const engine_t *engine,
        const primitive_attr_t *attr) {
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

        VCHECK_SOFTMAX_UNIMPL(attr->has_default_values(fwd_attr_mask, dst_dt),
                VERBOSE_UNSUPPORTED_ATTR);

        if (!attr->scales_.has_default_values()) {
            const auto &sc = attr->scales_;
            const int mask_src = sc.get(DNNL_ARG_SRC).mask_;
            const int mask_dst = sc.get(DNNL_ARG_DST).mask_;

            VCHECK_SOFTMAX_UNIMPL(utils::everyone_is(0, mask_src, mask_dst),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
        }

        // Check post-ops
        if (!attr->post_ops_.has_default_values()) {
            const auto &po = attr->post_ops_;
            using namespace primitive_kind;
            VCHECK_SOFTMAX_UNIMPL(po.has_default_values({binary, eltwise}),
                    VERBOSE_UNSUPPORTED_POSTOP);
        }
    } else {
        VCHECK_SOFTMAX_UNIMPL(false, VERBOSE_UNSUPPORTED_ATTR);
    }

    return status::success;
}

} // namespace

status_t dnnl_softmax_forward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        prop_kind_t prop_kind, alg_kind_t alg_kind,
        const memory_desc_t *src_desc, const memory_desc_t *dst_desc, int axis,
        const primitive_attr_t *attr) {
    if (!one_of(prop_kind, forward_inference, forward_training))
        return invalid_arguments;

    auto softmax_desc = softmax_desc_t();
    CHECK(softmax_desc_init(&softmax_desc, prop_kind, alg_kind, src_desc,
            dst_desc, nullptr, nullptr, axis));
    CHECK(softmax_attr_check(softmax_desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&softmax_desc, nullptr, attr);
}

status_t dnnl_softmax_backward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        alg_kind_t alg_kind, const memory_desc_t *diff_src_desc,
        const memory_desc_t *diff_dst_desc, const memory_desc_t *dst_desc,
        int axis, const primitive_desc_iface_t *hint_fwd_pd,
        const primitive_attr_t *attr) {

    auto softmax_desc = softmax_desc_t();
    CHECK(softmax_desc_init(&softmax_desc, prop_kind::backward_data, alg_kind,
            nullptr, dst_desc, diff_src_desc, diff_dst_desc, axis));
    CHECK(softmax_attr_check(softmax_desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&softmax_desc, hint_fwd_pd, attr);
}

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
