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

#include <assert.h>
#include "mkldnn.h"

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace mkldnn::impl;
using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::status;
using namespace mkldnn::impl::prop_kind;

namespace {
status_t ip_desc_init(inner_product_desc_t *ip_desc, prop_kind_t prop_kind,
        const memory_desc_t *src_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *bias_desc, const memory_desc_t *dst_desc) {
    bool args_ok = !any_null(ip_desc, src_desc, weights_desc, dst_desc);
    if (!args_ok) return invalid_arguments;

    inner_product_desc_t id = {};
    id.primitive_kind = primitive_kind::inner_product;
    id.prop_kind = prop_kind;

    const bool is_fwd = one_of(prop_kind, forward_training, forward_inference);
    const bool with_bias = bias_desc
        && bias_desc->format != memory_format::undef;

    (is_fwd ? id.src_desc : id.diff_src_desc) = *src_desc;
    (prop_kind == backward_weights ? id.diff_weights_desc : id.weights_desc) =
        *weights_desc;
    if (with_bias) id.bias_desc = *bias_desc;
    else if (is_fwd) id.bias_desc = types::zero_md();
    (prop_kind == backward_data ? id.diff_dst_desc : id.dst_desc)  = *dst_desc;

    /* FIXME: fill-in! */
    bool consistency = true
        && one_of(src_desc->ndims, 2, 4)
        && dst_desc->ndims == 2
        && weights_desc->ndims == src_desc->ndims
        && (with_bias ? bias_desc->ndims == 1 : true)
        && (with_bias ? bias_desc->dims[0] == dst_desc->dims[1] : true)
        && src_desc->dims[0] == dst_desc->dims[0]
        && array_cmp(&src_desc->dims[1], &weights_desc->dims[1],
                src_desc->ndims - 1)
        && dst_desc->dims[1] == weights_desc->dims[0];
    if (!consistency) return invalid_arguments;

    *ip_desc = id;
    return success;
}
}

status_t mkldnn_inner_product_forward_desc_init(inner_product_desc_t *ip_desc,
        prop_kind_t prop_kind, const memory_desc_t *src_desc,
        const memory_desc_t *weights_desc, const memory_desc_t *bias_desc,
        const memory_desc_t *dst_desc) {
    if (!one_of(prop_kind, forward_training, forward_inference))
        return invalid_arguments;
    return ip_desc_init(ip_desc, prop_kind, src_desc, weights_desc, bias_desc,
            dst_desc);
}

status_t mkldnn_inner_product_backward_data_desc_init(
        inner_product_desc_t *ip_desc, const memory_desc_t *diff_src_desc,
        const memory_desc_t *weights_desc, const memory_desc_t *diff_dst_desc)
{
    return ip_desc_init(ip_desc, backward_data, diff_src_desc, weights_desc,
            nullptr, diff_dst_desc);
}

status_t mkldnn_inner_product_backward_weights_desc_init(
        inner_product_desc_t *ip_desc, const memory_desc_t *src_desc,
        const memory_desc_t *diff_weights_desc,
        const memory_desc_t *diff_dst_desc) {
    return ip_desc_init(ip_desc, backward_weights, src_desc, diff_weights_desc,
            nullptr, diff_dst_desc);
}

status_t mkldnn_inner_product_backward_bias_desc_init(
        inner_product_desc_t *ip_desc, const memory_desc_t *diff_bias_desc,
        const memory_desc_t *diff_dst_desc) {
    bool args_ok = !any_null(ip_desc, diff_bias_desc, diff_dst_desc);
    if (!args_ok) return invalid_arguments;

    inner_product_desc_t id = {};
    id.primitive_kind = primitive_kind::inner_product;
    id.prop_kind = backward_bias;

    id.diff_bias_desc = *diff_bias_desc;
    id.diff_dst_desc = *diff_dst_desc;

    bool consistency = true
        && diff_dst_desc->ndims == 2
        && diff_bias_desc->ndims == 1
        && diff_bias_desc->dims[0] == diff_dst_desc->dims[1];
    if (!consistency) return invalid_arguments;

    *ip_desc = id;
    return success;
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
