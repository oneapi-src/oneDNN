/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#include "nstl.hpp"

#include "type_helpers.hpp"
#include "engine.hpp"
#include "primitive.hpp"

#include "utils.hpp"

using namespace mkldnn::impl;
using namespace mkldnn::impl::status;
using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::alg_kind;

status_t mkldnn_pooling_desc_init(pooling_desc_t *pooling_desc,
        prop_kind_t prop_kind, alg_kind_t alg_kind,
        const memory_desc_t *src_desc, const memory_desc_t *dst_desc,
        const dims_t strides, const dims_t kernel, const nd_offset_t padding,
        padding_kind_t padding_kind)
{
    const bool args_ok = !any_null(pooling_desc,
            src_desc, dst_desc, strides, padding)
        && one_of(prop_kind, forward_training, forward_scoring, backward_data)
        && one_of(alg_kind, pooling_max);
    if (!args_ok)
        return invalid_arguments;

    pooling_desc_t cd;
    cd.prop_kind = prop_kind;
    cd.alg_kind = alg_kind;
    cd.src_desc = *src_desc;
    cd.dst_desc = *dst_desc;
    cd.padding_kind = padding_kind;
    const uint32_t ndims_spatial = src_desc->tensor_desc.ndims - 2;
    array_copy(cd.strides, strides, ndims_spatial);
    array_copy(cd.kernel, kernel, ndims_spatial);
    array_copy(cd.padding, padding, ndims_spatial);

    status_t status = types::pooling_desc_is_ok(cd);
    if (status == success)
        *pooling_desc = cd;

    return status;
}

status_t mkldnn_pooling_primitive_desc_init(
        pooling_primitive_desc_t *pooling_primitive_desc,
        const pooling_desc_t *pooling_desc,
        const engine *engine)
{
    if (any_null(pooling_primitive_desc, pooling_desc, engine))
        return invalid_arguments;

    return primitive_desc_init(
            primitive_desc_t::convert_from_c(pooling_primitive_desc),
            *pooling_desc, *engine);
}

status_t mkldnn_pooling_create(primitive **pooling,
        const pooling_primitive_desc_t *pooling_primitive_desc,
        const primitive_at_t src, const primitive_at_t indices,
        const primitive *dst)
{
    auto ppd = reinterpret_cast<const mkldnn_primitive_desc_t *>(
            pooling_primitive_desc);
    const primitive_at_t inputs[] = {src, indices};
    const primitive *outputs[] = {dst};
    return mkldnn_primitive_create(pooling, ppd, inputs, outputs);
}

status_t mkldnn_pooling_get_primitive_desc(const primitive *pooling,
        pooling_primitive_desc_t *pooling_primitive_desc)
{
    if (any_null(pooling, pooling_primitive_desc)
            || pooling->kind() != primitive_kind::pooling)
        return invalid_arguments;
    *pooling_primitive_desc = pooling->primitive_desc().pooling;
    return success;
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
