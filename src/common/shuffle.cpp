/*******************************************************************************
* Copyright 2018-2023 Intel Corporation
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

#include "c_types_map.hpp"
#include "opdesc.hpp"
#include "primitive_desc_iface.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::status;
using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::types;

#define VCHECK_SHUFFLE(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, shuffle, (cond), \
            status::invalid_arguments, msg, ##__VA_ARGS__);

#define VCHECK_SHUFFLE_UNIMPL(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, shuffle, (cond), \
            status::unimplemented, msg, ##__VA_ARGS__);

namespace {
status_t shuffle_desc_init(shuffle_desc_t *shuffle_desc, prop_kind_t prop_kind,
        const memory_desc_t *src_desc, const memory_desc_t *dst_desc, int axis,
        dim_t group_size) {
    VCHECK_SHUFFLE(
            !any_null(shuffle_desc, src_desc, dst_desc), VERBOSE_NULL_ARG);
    VCHECK_SHUFFLE(one_of(prop_kind, forward_training, forward_inference,
                           backward_data),
            VERBOSE_BAD_PROPKIND);
    VCHECK_SHUFFLE(IMPLICATION(prop_kind != backward_data,
                           src_desc->format_kind != format_kind::any),
            VERBOSE_UNSUPPORTED_TAG_S, "src");
    VCHECK_SHUFFLE(axis >= 0 && axis < src_desc->ndims, VERBOSE_BAD_AXIS);
    VCHECK_SHUFFLE(group_size > 0 && group_size <= src_desc->dims[axis],
            VERBOSE_BAD_PARAM, "group_size");

    VCONDCHECK(primitive, create, check, shuffle,
            !memory_desc_wrapper(src_desc).has_runtime_dims_or_strides(),
            status::unimplemented, VERBOSE_RUNTIMEDIM_UNSUPPORTED);
    VCONDCHECK(primitive, create, check, shuffle,
            !memory_desc_wrapper(dst_desc).has_runtime_dims_or_strides(),
            status::unimplemented, VERBOSE_RUNTIMEDIM_UNSUPPORTED);

    auto sd = shuffle_desc_t();
    sd.primitive_kind = primitive_kind::shuffle;
    sd.prop_kind = prop_kind;
    sd.src_desc = *src_desc;
    sd.dst_desc = *dst_desc;
    sd.axis = axis;
    sd.group_size = group_size;

    VCHECK_SHUFFLE(sd.src_desc.dims[axis] % sd.group_size == 0,
            VERBOSE_INCONSISTENT_DIM, "src", axis, "group_size", 0);
    VCHECK_SHUFFLE(sd.dst_desc.ndims == sd.src_desc.ndims,
            VERBOSE_INCONSISTENT_NDIMS, "src", "dst");
    VCHECK_SHUFFLE(
            array_cmp(sd.dst_desc.dims, sd.src_desc.dims, sd.src_desc.ndims),
            VERBOSE_INCONSISTENT_DIM, "src", -1, "dst", -1);

    *shuffle_desc = sd;
    return success;
}

status_t shuffle_attr_check(const shuffle_desc_t &desc, const engine_t *engine,
        const primitive_attr_t *attr) {

    if (attr == nullptr) return status::success;

    // Check attributes
    VCHECK_SHUFFLE_UNIMPL(attr->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

    return status::success;
}

} // namespace

dnnl_status_t dnnl_shuffle_forward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        prop_kind_t prop_kind, const memory_desc_t *src_desc,
        const memory_desc_t *dst_desc, int axis, dim_t group_size,
        const primitive_attr_t *attr) {
    VCHECK_SHUFFLE(one_of(prop_kind, forward_training, forward_inference),
            VERBOSE_BAD_PROPKIND);

    auto shuffle_desc = shuffle_desc_t();
    CHECK(shuffle_desc_init(
            &shuffle_desc, prop_kind, src_desc, dst_desc, axis, group_size));
    CHECK(shuffle_attr_check(shuffle_desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&shuffle_desc, nullptr, attr);
}

dnnl_status_t dnnl_shuffle_backward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        const memory_desc_t *diff_src_desc, const memory_desc_t *diff_dst_desc,
        int axis, dim_t group_size, const primitive_desc_iface_t *hint_fwd_pd,
        const primitive_attr_t *attr) {

    auto shuffle_desc = shuffle_desc_t();
    CHECK(shuffle_desc_init(&shuffle_desc, backward_data, diff_src_desc,
            diff_dst_desc, axis, group_size));
    CHECK(shuffle_attr_check(shuffle_desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&shuffle_desc, hint_fwd_pd, attr);
}

// vim: et ts=5 sw=4 cindent cino+=l0,\:4,N-s
