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
#include "engine.hpp"
#include "primitive.hpp"
#include "type_helpers.hpp"

#include "utils.hpp"

using namespace mkldnn::impl;
using namespace mkldnn::impl::status;
using namespace mkldnn::impl::prop_kind;

status_t mkldnn_inner_product_desc_init(
        inner_product_desc_t *inner_product_desc, prop_kind_t prop_kind,
        const memory_desc_t *src_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *bias_desc, const memory_desc_t *dst_desc)
{
    const bool args_ok = !any_null(inner_product_desc, src_desc, weights_desc,
            dst_desc)
        && one_of(prop_kind, forward_training, forward_scoring, backward_data,
                backward_weights, backward_bias)
        && implication(prop_kind == backward_bias, !any_null(bias_desc));
    if (!args_ok)
        return invalid_arguments;

    inner_product_desc_t ipd;
    ipd.prop_kind = prop_kind;
    ipd.src_desc = *src_desc;
    ipd.weights_desc = *weights_desc;
    ipd.bias_desc = bias_desc ? *bias_desc : types::zero<memory_desc_t>();
    ipd.dst_desc = *dst_desc;

    status_t status = types::inner_product_desc_is_ok(ipd);
    if (status == success)
        *inner_product_desc = ipd;

    return status;
}

status_t mkldnn_inner_product_primitive_desc_init(
        inner_product_primitive_desc_t *inner_product_primitive_desc,
        const inner_product_desc_t *inner_product_desc, const engine *engine)
{
    if (any_null(inner_product_primitive_desc, inner_product_desc, engine))
        return invalid_arguments;

    return primitive_desc_init(
            primitive_desc_t::convert_from_c(inner_product_primitive_desc),
            *inner_product_desc, *engine);
}

status_t mkldnn_inner_product_create(primitive **inner_product,
        const inner_product_primitive_desc_t *inner_product_primitive_desc,
        const primitive_at_t src, const primitive_at_t weights,
        const primitive_at_t bias, const primitive *dst)
{
    auto ippd = reinterpret_cast<const mkldnn_primitive_desc_t *>(
            inner_product_primitive_desc);
    // XXX: must check that shapes of in/out memory match what's in the desc (?)
    const primitive_at_t inputs[] = {src, weights, bias};
    const primitive *outputs[] = {dst};
    return mkldnn_primitive_create(inner_product, ippd, inputs, outputs);
}

status_t mkldnn_inner_product_get_primitive_desc(
        const primitive *inner_product,
        inner_product_primitive_desc_t *inner_product_primitive_desc)
{
    if (any_null(inner_product, inner_product_primitive_desc)
            || inner_product->kind() != primitive_kind::inner_product)
        return invalid_arguments;
    *inner_product_primitive_desc =
        inner_product->primitive_desc().inner_product;
    return success;
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
