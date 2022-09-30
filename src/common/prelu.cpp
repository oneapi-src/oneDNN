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
        const memory_desc_t *src_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *dst_desc, const memory_desc_t *diff_src_desc,
        const memory_desc_t *diff_weights_desc,
        const memory_desc_t *diff_dst_desc) {
    const bool is_fwd = one_of(prop_kind, forward_training, forward_inference);
    bool args_ok = !any_null(prelu_desc, src_desc, weights_desc)
            && one_of(prop_kind, forward_training, forward_inference, backward)
            && IMPLICATION(is_fwd, dst_desc != nullptr)
            && IMPLICATION(!is_fwd,
                    !any_null(diff_src_desc, diff_weights_desc, diff_dst_desc))
            && IMPLICATION(is_fwd, !memory_desc_wrapper(src_desc).format_any());
    if (!args_ok) return invalid_arguments;

    if (memory_desc_wrapper(src_desc).has_runtime_dims_or_strides()
            || memory_desc_wrapper(weights_desc).has_runtime_dims_or_strides())
        return unimplemented;
    if (prop_kind == backward
            && (memory_desc_wrapper(diff_src_desc).has_runtime_dims_or_strides()
                    || memory_desc_wrapper(diff_weights_desc)
                               .has_runtime_dims_or_strides()))
        return unimplemented;

    auto pd = prelu_desc_t();
    pd.primitive_kind = primitive_kind::prelu;
    pd.prop_kind = prop_kind;
    pd.src_desc = *src_desc;
    pd.weights_desc = *weights_desc;
    if (is_fwd) {
        pd.dst_desc = *dst_desc;
    } else {
        pd.diff_src_desc = *diff_src_desc;
        pd.diff_weights_desc = *diff_weights_desc;
        pd.diff_dst_desc = *diff_dst_desc;
    }

    const memory_desc_wrapper src_mdw(*src_desc);
    if (get_rhs_arg_broadcasting_strategy(pd.weights_desc, src_mdw)
            == broadcasting_strategy_t::unsupported)
        return invalid_arguments;

    static constexpr int max_supported_ndims = 5;
    bool consistency = src_desc->ndims <= max_supported_ndims
            && src_desc->ndims == weights_desc->ndims;
    if (consistency && is_fwd) {
        consistency = pd.dst_desc.ndims == pd.src_desc.ndims
                && array_cmp(
                        pd.dst_desc.dims, pd.src_desc.dims, pd.src_desc.ndims);
    }
    if (consistency && !is_fwd) {
        consistency = pd.diff_dst_desc.ndims == pd.src_desc.ndims
                && pd.diff_dst_desc.ndims == pd.diff_src_desc.ndims
                && array_cmp(pd.diff_dst_desc.dims, pd.src_desc.dims,
                        pd.src_desc.ndims)
                && array_cmp(pd.diff_src_desc.dims, pd.diff_dst_desc.dims,
                        pd.diff_dst_desc.ndims);
    }
    if (!consistency) return invalid_arguments;

    *prelu_desc = pd;
    return success;
}
} // namespace

status_t dnnl_prelu_forward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        prop_kind_t prop_kind, const memory_desc_t *src_desc,
        const memory_desc_t *weights_desc, const memory_desc_t *dst_desc,
        const primitive_attr_t *attr) {

    if (!one_of(prop_kind, forward_training, forward_inference))
        return invalid_arguments;

    auto prelu_desc = prelu_desc_t();
    CHECK(prelu_desc_init(&prelu_desc, prop_kind, src_desc, weights_desc,
            dst_desc, nullptr, nullptr, nullptr));

    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&prelu_desc, nullptr, attr);
}

status_t dnnl_prelu_backward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        const memory_desc_t *src_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *diff_src_desc,
        const memory_desc_t *diff_weights_desc,
        const memory_desc_t *diff_dst_desc,
        const primitive_desc_iface_t *hint_fwd_pd,
        const primitive_attr_t *attr) {

    auto prelu_desc = prelu_desc_t();
    CHECK(prelu_desc_init(&prelu_desc, backward, src_desc, weights_desc,
            nullptr, diff_src_desc, diff_weights_desc, diff_dst_desc));

    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&prelu_desc, hint_fwd_pd, attr);
}

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
