/*******************************************************************************
* Copyright 2016-2023 Intel Corporation
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

#define VCHECK_LRN(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, lrn, (cond), \
            status::invalid_arguments, msg, ##__VA_ARGS__)

#define VCHECK_LRN_UNIMPL(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, lrn, (cond), status::unimplemented, \
            msg, ##__VA_ARGS__)

namespace {
status_t lrn_desc_init(lrn_desc_t *lrn_desc, prop_kind_t prop_kind,
        alg_kind_t alg_kind, const memory_desc_t *src_desc,
        const memory_desc_t *dst_desc, const memory_desc_t *diff_src_desc,
        const memory_desc_t *diff_dst_desc, dim_t local_size, float alpha,
        float beta, float k) {
    const bool is_fwd = one_of(prop_kind, forward_training, forward_inference);
    VCHECK_LRN(!any_null(lrn_desc, src_desc), VERBOSE_NULL_ARG);
    VCHECK_LRN(one_of(alg_kind, lrn_within_channel, lrn_across_channels),
            VERBOSE_BAD_ALGORITHM);
    VCHECK_LRN(one_of(prop_kind, forward_training, forward_inference,
                       backward_data),
            VERBOSE_BAD_PROPKIND);
    VCHECK_LRN(local_size >= 0, VERBOSE_BAD_PARAM, "local_size");
    VCHECK_LRN(IMPLICATION(is_fwd, dst_desc != nullptr), VERBOSE_NULL_ARG);
    VCHECK_LRN(IMPLICATION(!is_fwd, !any_null(diff_src_desc, diff_dst_desc)),
            VERBOSE_NULL_ARG);
    VCHECK_LRN(IMPLICATION(is_fwd, !memory_desc_wrapper(src_desc).format_any()),
            VERBOSE_UNSUPPORTED_TAG_S, "src");

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
    VCONDCHECK(primitive, create, check, lrn, !runtime_dims_or_strides,
            status::unimplemented, VERBOSE_RUNTIMEDIM_UNSUPPORTED);

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

#define CHECK_DIMS(t1, t2) \
    do { \
        VCHECK_LRN(ld.t2##_desc.ndims == ld.t1##_desc.ndims, \
                VERBOSE_INCONSISTENT_NDIMS, #t1, #t2); \
        VCHECK_LRN(array_cmp(ld.t2##_desc.dims, ld.t1##_desc.dims, \
                           ld.t1##_desc.ndims), \
                VERBOSE_INCONSISTENT_DIM, #t1, -1, #t2, -1); \
    } while (0)

    VCHECK_LRN(ld.src_desc.ndims >= 2, VERBOSE_BAD_NDIMS, "src",
            ld.src_desc.ndims);
    if (is_fwd) {
        CHECK_DIMS(src, dst);
    } else {
        CHECK_DIMS(src, diff_dst);
        CHECK_DIMS(diff_src, diff_dst);
    }
#undef CHECK_DIMS

    *lrn_desc = ld;
    return success;
}

status_t lrn_attr_check(const lrn_desc_t &desc, const engine_t *engine,
        const primitive_attr_t *attr) {

    if (attr == nullptr) return status::success;

    // Check attributes
    VCHECK_LRN_UNIMPL(attr->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

    return status::success;
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
    CHECK(lrn_attr_check(lrn_desc, engine, attr));
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
    CHECK(lrn_attr_check(lrn_desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&lrn_desc, hint_fwd_pd, attr);
}

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
