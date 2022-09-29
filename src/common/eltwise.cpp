/*******************************************************************************
* Copyright 2016-2022 Intel Corporation
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
#include "math_utils.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::status;
using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::alg_kind;
using namespace dnnl::impl::types;

namespace {
status_t eltwise_desc_init(eltwise_desc_t *eltwise_desc, prop_kind_t prop_kind,
        alg_kind_t alg_kind, const memory_desc_t *src_desc,
        const memory_desc_t *dst_desc, const memory_desc_t *diff_src_desc,
        const memory_desc_t *diff_dst_desc, float alpha, float beta) {
    const bool is_fwd = one_of(prop_kind, forward_training, forward_inference);
    bool args_ok = !any_null(eltwise_desc, src_desc, dst_desc)
            && one_of(prop_kind, forward_training, forward_inference,
                    backward_data)
            && math::is_eltwise_ok(src_desc->data_type, alg_kind, alpha, beta)
            && IMPLICATION(!is_fwd, !any_null(diff_src_desc, diff_dst_desc))
            && IMPLICATION(alg_kind == eltwise_round, is_fwd)
            && IMPLICATION(is_fwd, !memory_desc_wrapper(src_desc).format_any());
    if (!args_ok) return invalid_arguments;

    bool runtime_dims_or_strides
            = memory_desc_wrapper(src_desc).has_runtime_dims_or_strides()
            || memory_desc_wrapper(dst_desc).has_runtime_dims_or_strides();
    if (!is_fwd)
        runtime_dims_or_strides = runtime_dims_or_strides
                || memory_desc_wrapper(diff_src_desc)
                           .has_runtime_dims_or_strides()
                || memory_desc_wrapper(diff_dst_desc)
                           .has_runtime_dims_or_strides();
    if (runtime_dims_or_strides) return unimplemented;

    auto ed = eltwise_desc_t();
    ed.primitive_kind = primitive_kind::eltwise;
    ed.prop_kind = prop_kind;
    ed.alg_kind = alg_kind;

    ed.src_desc = *src_desc;
    ed.dst_desc = *dst_desc;
    if (!is_fwd) {
        ed.diff_src_desc = *diff_src_desc;
        ed.diff_dst_desc = *diff_dst_desc;
    }

    ed.alpha = alpha;
    ed.beta = beta;

    bool consistency = true;
    if (consistency && is_fwd) {
        consistency = ed.dst_desc.ndims == ed.src_desc.ndims
                && array_cmp(
                        ed.dst_desc.dims, ed.src_desc.dims, ed.src_desc.ndims);
    }
    if (consistency && !is_fwd) {
        consistency = ed.diff_dst_desc.ndims == ed.src_desc.ndims
                && ed.diff_dst_desc.ndims == ed.diff_src_desc.ndims
                && array_cmp(ed.diff_dst_desc.dims, ed.src_desc.dims,
                        ed.src_desc.ndims)
                && array_cmp(ed.diff_src_desc.dims, ed.diff_dst_desc.dims,
                        ed.diff_dst_desc.ndims);
    }
    if (!consistency) return invalid_arguments;

    *eltwise_desc = ed;
    return success;
}
} // namespace

status_t dnnl_eltwise_forward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        prop_kind_t prop_kind, alg_kind_t alg_kind,
        const memory_desc_t *src_desc, const memory_desc_t *dst_desc,
        float alpha, float beta, const primitive_attr_t *attr) {
    if (!one_of(prop_kind, forward_training, forward_inference))
        return invalid_arguments;

    auto eltwise_desc = eltwise_desc_t();
    CHECK(eltwise_desc_init(&eltwise_desc, prop_kind, alg_kind, src_desc,
            dst_desc, nullptr, nullptr, alpha, beta));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&eltwise_desc, nullptr, attr);
}

status_t dnnl_eltwise_backward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        alg_kind_t alg_kind, const memory_desc_t *diff_src_desc,
        const memory_desc_t *diff_dst_desc, const memory_desc_t *data_desc,
        float alpha, float beta, const primitive_desc_iface_t *hint_fwd_pd,
        const primitive_attr_t *attr) {

    auto eltwise_desc = eltwise_desc_t();
    CHECK(eltwise_desc_init(&eltwise_desc, backward_data, alg_kind, data_desc,
            data_desc, diff_src_desc, diff_dst_desc, alpha, beta));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&eltwise_desc, hint_fwd_pd, attr);
}

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
