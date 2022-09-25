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
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::status;
using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::alg_kind;
using namespace dnnl::impl::types;

namespace {
status_t lrn_desc_init(lrn_desc_t *lrn_desc, prop_kind_t prop_kind,
        alg_kind_t alg_kind, const memory_desc_t *src_desc,
        const memory_desc_t *dst_desc, const memory_desc_t *diff_src_desc,
        const memory_desc_t *diff_dst_desc, dim_t local_size, float alpha,
        float beta, float k) {
    const bool is_fwd = one_of(prop_kind, forward_training, forward_inference);
    bool args_ok = !any_null(lrn_desc, src_desc)
            && one_of(alg_kind, lrn_within_channel, lrn_across_channels)
            && one_of(prop_kind, forward_training, forward_inference,
                    backward_data)
            && local_size >= 0 && IMPLICATION(is_fwd, dst_desc != nullptr)
            && IMPLICATION(!is_fwd, !any_null(diff_src_desc, diff_dst_desc))
            && IMPLICATION(is_fwd, !memory_desc_wrapper(src_desc).format_any());
    if (!args_ok) return invalid_arguments;

    auto ld = lrn_desc_t();
    ld.primitive_kind = primitive_kind::lrn;
    ld.prop_kind = prop_kind;
    ld.alg_kind = alg_kind;

    bool runtime_dims_or_strides
            = memory_desc_wrapper(src_desc).has_runtime_dims_or_strides();
    if (is_fwd) {
        runtime_dims_or_strides = runtime_dims_or_strides
                || memory_desc_wrapper(dst_desc).has_runtime_dims_or_strides();
    } else {
        runtime_dims_or_strides = runtime_dims_or_strides
                || memory_desc_wrapper(diff_src_desc)
                           .has_runtime_dims_or_strides()
                || memory_desc_wrapper(diff_dst_desc)
                           .has_runtime_dims_or_strides();
    }
    if (runtime_dims_or_strides) return unimplemented;

    ld.src_desc = *src_desc;
    if (is_fwd) ld.dst_desc = *dst_desc;
    if (!is_fwd) {
        ld.diff_src_desc = *diff_src_desc;
        ld.diff_dst_desc = *diff_dst_desc;
    }

    ld.local_size = local_size;
    ld.lrn_alpha = alpha;
    ld.lrn_beta = beta;
    ld.lrn_k = k;

    bool consistency = ld.src_desc.ndims >= 2;
    if (consistency && is_fwd) {
        consistency = ld.dst_desc.ndims == ld.src_desc.ndims
                && array_cmp(
                        ld.dst_desc.dims, ld.src_desc.dims, ld.src_desc.ndims);
    }
    if (consistency && !is_fwd) {
        consistency = ld.diff_dst_desc.ndims == ld.src_desc.ndims
                && ld.diff_dst_desc.ndims == ld.diff_src_desc.ndims
                && array_cmp(ld.diff_dst_desc.dims, ld.src_desc.dims,
                        ld.src_desc.ndims)
                && array_cmp(ld.diff_src_desc.dims, ld.diff_dst_desc.dims,
                        ld.diff_dst_desc.ndims);
    }
    if (!consistency) return invalid_arguments;

    *lrn_desc = ld;
    return success;
}
} // namespace

status_t dnnl_lrn_forward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        prop_kind_t prop_kind, alg_kind_t alg_kind,
        const memory_desc_t *src_desc, const memory_desc_t *dst_desc,
        dim_t local_size, float alpha, float beta, float k,
        const primitive_attr_t *attr) {
    if (!one_of(prop_kind, forward_training, forward_inference))
        return invalid_arguments;

    auto lrn_desc = lrn_desc_t();
    CHECK(lrn_desc_init(&lrn_desc, prop_kind, alg_kind, src_desc, dst_desc,
            nullptr, nullptr, local_size, alpha, beta, k));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&lrn_desc, nullptr, attr);
}

status_t dnnl_lrn_backward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        alg_kind_t alg_kind, const memory_desc_t *diff_src_desc,
        const memory_desc_t *diff_dst_desc, const memory_desc_t *src_desc,
        dim_t local_size, float alpha, float beta, float k,
        const primitive_desc_iface_t *hint_fwd_pd,
        const primitive_attr_t *attr) {

    auto lrn_desc = lrn_desc_t();
    CHECK(lrn_desc_init(&lrn_desc, backward_data, alg_kind, src_desc, nullptr,
            diff_src_desc, diff_dst_desc, local_size, alpha, beta, k));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&lrn_desc, hint_fwd_pd, attr);
}

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
