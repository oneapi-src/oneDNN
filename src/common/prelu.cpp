/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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
#include "dnnl.h"

#include "common/c_types_map.hpp"
#include "opdesc.hpp"
#include "primitive_desc_iface.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "common/broadcast_strategy.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::status;
using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::types;

namespace {
status_t prelu_desc_init(prelu_desc_t *prelu_desc, prop_kind_t prop_kind,
        const memory_desc_t *data_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *diff_data_desc,
        const memory_desc_t *diff_weights_desc) {
    static constexpr int max_supported_ndims = 5;
    bool args_ok = !any_null(prelu_desc, data_desc, weights_desc)
            && one_of(prop_kind, forward_training, forward_inference, backward)
            && data_desc->ndims <= max_supported_ndims
            && data_desc->ndims == weights_desc->ndims
            && IMPLICATION(prop_kind == backward,
                    !any_null(diff_data_desc, diff_weights_desc)
                            && diff_data_desc->ndims == data_desc->ndims
                            && diff_weights_desc->ndims == weights_desc->ndims)
            && IMPLICATION(
                    one_of(prop_kind, forward_training, forward_inference),
                    !memory_desc_wrapper(data_desc).format_any());

    if (!args_ok) return invalid_arguments;

    if (memory_desc_wrapper(data_desc).has_runtime_dims_or_strides()
            || memory_desc_wrapper(weights_desc).has_runtime_dims_or_strides())
        return unimplemented;

    if (prop_kind == backward
            && (memory_desc_wrapper(diff_data_desc)
                            .has_runtime_dims_or_strides()
                    || memory_desc_wrapper(diff_weights_desc)
                               .has_runtime_dims_or_strides()))
        return unimplemented;

    auto pd = prelu_desc_t();

    pd.primitive_kind = primitive_kind::prelu;
    pd.prop_kind = prop_kind;
    pd.data_desc = *data_desc;
    pd.weights_desc = *weights_desc;
    if (pd.prop_kind == backward) {
        pd.diff_data_desc = *diff_data_desc;
        pd.diff_weights_desc = *diff_weights_desc;
    }

    memory_desc_wrapper data_md(*data_desc);
    if (get_rhs_arg_broadcasting_strategy(pd.weights_desc, data_md)
            == broadcasting_strategy_t::unsupported)
        return invalid_arguments;

    *prelu_desc = pd;
    return success;
}
} // namespace

status_t dnnl_prelu_forward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        prop_kind_t prop_kind, const memory_desc_t *data_desc,
        const memory_desc_t *weights_desc, const primitive_attr_t *attr) {

    if (!one_of(prop_kind, forward_training, forward_inference))
        return invalid_arguments;

    auto prelu_desc = prelu_desc_t();
    CHECK(prelu_desc_init(
            &prelu_desc, prop_kind, data_desc, weights_desc, nullptr, nullptr));

    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&prelu_desc, nullptr, attr);
}

status_t dnnl_prelu_backward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        const memory_desc_t *data_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *diff_data_desc,
        const memory_desc_t *diff_weights_desc,
        const primitive_desc_iface_t *hint_fwd_pd,
        const primitive_attr_t *attr) {

    auto prelu_desc = prelu_desc_t();
    CHECK(prelu_desc_init(&prelu_desc, backward, data_desc, weights_desc,
            diff_data_desc, diff_weights_desc));

    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&prelu_desc, hint_fwd_pd, attr);
}

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
