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

status_t mkldnn_convolution_relu_desc_init(
        mkldnn_convolution_relu_desc_t *convolution_relu_desc,
        const mkldnn_convolution_desc_t *convolution_desc,
        double negative_slope)
{
    const bool args_ok = !any_null(convolution_relu_desc, convolution_desc)
        && one_of(convolution_desc->prop_kind, prop_kind::forward_scoring);
    if (!args_ok)
        return invalid_arguments;

    convolution_relu_desc_t crd;
    crd.convolution_desc = *convolution_desc;
    crd.negative_slope = negative_slope;

    *convolution_relu_desc = crd;
    return success;
}

status_t mkldnn_convolution_relu_primitive_desc_init(
        convolution_relu_primitive_desc_t *convolution_relu_primitive_desc,
        const convolution_relu_desc_t *convolution_relu_desc,
        const engine *engine)
{
    if (any_null(convolution_relu_primitive_desc, convolution_relu_desc,
                engine))
        return invalid_arguments;

    return primitive_desc_init(
            primitive_desc_t::convert_from_c(convolution_relu_primitive_desc),
            *convolution_relu_desc, *engine);
}

status_t mkldnn_convolution_relu_create(primitive **convolution_relu,
        const convolution_relu_primitive_desc_t *conv_relu_primitive_desc,
        const primitive_at_t src, const primitive_at_t weights,
        const primitive_at_t bias, const primitive *dst)
{
    auto crpd = reinterpret_cast<const mkldnn_primitive_desc_t *>(
            conv_relu_primitive_desc);
    // XXX: must check that shapes of in/out memory match what's in the desc (?)
    const primitive_at_t inputs[] = {src, weights, bias};
    const primitive *outputs[] = {dst};
    return mkldnn_primitive_create(convolution_relu, crpd, inputs, outputs);
}

status_t mkldnn_convolution_relu_get_primitive_desc(
        const primitive *convolution_relu,
        convolution_relu_primitive_desc_t *convolution_relu_primitive_desc)
{
    if (any_null(convolution_relu, convolution_relu_primitive_desc)
            || convolution_relu->kind() != primitive_kind::convolution_relu)
        return invalid_arguments;
    *convolution_relu_primitive_desc =
        convolution_relu->primitive_desc().convolution_relu;
    return success;
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
