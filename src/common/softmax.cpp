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
    VCONDCHECK(create, check, softmax, (cond), status::invalid_arguments, msg, \
            ##__VA_ARGS__);

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
    VCONDCHECK(create, check, softmax, !runtime_dims_or_strides,
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
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&softmax_desc, hint_fwd_pd, attr);
}

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
