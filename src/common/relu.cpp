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

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "engine.hpp"
#include "primitive.hpp"

#include "utils.hpp"

using namespace mkldnn::impl;
using namespace mkldnn::impl::status;
using namespace mkldnn::impl::prop_kind;

status_t mkldnn_relu_desc_init(relu_desc_t *relu_desc,
        prop_kind_t prop_kind, double negative_slope,
        const memory_desc_t *src_desc, const memory_desc_t *dst_desc)
{
    bool args_ok = !any_null(relu_desc, src_desc, dst_desc)
        && one_of(prop_kind, forward_training, forward_scoring, backward_data);
    if (!args_ok) return invalid_arguments;

    relu_desc_t rd;
    rd.prop_kind = prop_kind;
    rd.negative_slope = negative_slope;
    rd.src_desc = *src_desc;
    rd.dst_desc = *dst_desc;

    // XXX: rsdubtso: I do not like this...
    status_t status = types::relu_desc_is_ok(rd);
    if (status == success)
        *relu_desc = rd;

    return status;
}

status_t mkldnn_relu_primitive_desc_init(
        relu_primitive_desc_t *relu_primitive_desc,
        const relu_desc_t *relu_desc, const engine *engine)
{
    if (any_null(relu_primitive_desc, relu_desc, engine))
        return invalid_arguments;

    return primitive_desc_init(
            primitive_desc_t::convert_from_c(relu_primitive_desc),
            *relu_desc, *engine);
}

status_t mkldnn_relu_create(primitive **relu,
        const relu_primitive_desc_t *relu_primitive_desc,
        const primitive_at_t src, const primitive *dst)
{
    auto *rpd = reinterpret_cast<const mkldnn_primitive_desc_t *>(
            relu_primitive_desc);
    // XXX: must check that shapes of in/out memory match what's in the desc (?)
    const primitive_at_t inputs[] = {src};
    const primitive *outputs[] = {dst};
    return mkldnn_primitive_create(relu, rpd, inputs, outputs);
}

status_t mkldnn_relu_get_primitive_desc(const primitive *relu,
        relu_primitive_desc_t *relu_primitive_desc)
{
    if (any_null(relu, relu_primitive_desc)
            || relu->kind() != primitive_kind::relu)
        return invalid_arguments;
    *relu_primitive_desc = relu->primitive_desc().relu;
    return success;
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
