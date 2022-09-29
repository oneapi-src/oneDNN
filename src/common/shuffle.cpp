/*******************************************************************************
* Copyright 2018-2022 Intel Corporation
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

namespace {
status_t shuffle_desc_init(shuffle_desc_t *shuffle_desc, prop_kind_t prop_kind,
        const memory_desc_t *src_desc, const memory_desc_t *dst_desc, int axis,
        dim_t group_size) {
    bool args_ok = !any_null(shuffle_desc, src_desc, dst_desc)
            && one_of(prop_kind, forward_training, forward_inference,
                    backward_data)
            && IMPLICATION(prop_kind != backward_data,
                    src_desc->format_kind != format_kind::any)
            && axis >= 0 && axis < src_desc->ndims && group_size > 0
            && group_size <= src_desc->dims[axis];
    if (!args_ok) return invalid_arguments;

    if (memory_desc_wrapper(src_desc).has_runtime_dims_or_strides()
            || memory_desc_wrapper(dst_desc).has_runtime_dims_or_strides())
        return unimplemented;

    auto sd = shuffle_desc_t();
    sd.primitive_kind = primitive_kind::shuffle;
    sd.prop_kind = prop_kind;
    sd.src_desc = *src_desc;
    sd.dst_desc = *dst_desc;
    sd.axis = axis;
    sd.group_size = group_size;

    bool consistency = sd.src_desc.dims[axis] % sd.group_size == 0
            && sd.dst_desc.ndims == sd.src_desc.ndims
            && array_cmp(sd.dst_desc.dims, sd.src_desc.dims, sd.src_desc.ndims);
    if (!consistency) return invalid_arguments;

    *shuffle_desc = sd;
    return success;
}
} // namespace

dnnl_status_t dnnl_shuffle_forward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        prop_kind_t prop_kind, const memory_desc_t *src_desc,
        const memory_desc_t *dst_desc, int axis, dim_t group_size,
        const primitive_attr_t *attr) {
    if (!one_of(prop_kind, forward_training, forward_inference))
        return invalid_arguments;

    auto shuffle_desc = shuffle_desc_t();
    CHECK(shuffle_desc_init(
            &shuffle_desc, prop_kind, src_desc, dst_desc, axis, group_size));
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
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&shuffle_desc, hint_fwd_pd, attr);
}

// vim: et ts=5 sw=4 cindent cino+=l0,\:4,N-s
