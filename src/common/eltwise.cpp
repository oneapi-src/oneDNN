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
#include "math_utils.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::status;
using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::alg_kind;
using namespace dnnl::impl::types;

#define VCHECK_ELTWISE(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, eltwise, (cond), \
            status::invalid_arguments, msg, ##__VA_ARGS__);

#define VCHECK_ELTWISE_IMPL(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, eltwise, (cond), \
            status::unimplemented, msg, ##__VA_ARGS__);

namespace dnnl {
namespace impl {
status_t eltwise_desc_init(eltwise_desc_t *eltwise_desc, prop_kind_t prop_kind,
        alg_kind_t alg_kind, const memory_desc_t *src_desc,
        const memory_desc_t *dst_desc, const memory_desc_t *diff_src_desc,
        const memory_desc_t *diff_dst_desc, float alpha, float beta) {
    const bool is_fwd = one_of(prop_kind, forward_training, forward_inference);
    VCHECK_ELTWISE(
            !any_null(eltwise_desc, src_desc, dst_desc), VERBOSE_NULL_ARG);
    VCHECK_ELTWISE(one_of(prop_kind, forward_training, forward_inference,
                           backward_data),
            VERBOSE_BAD_PROPKIND);
    VCHECK_ELTWISE(
            math::is_eltwise_ok(src_desc->data_type, alg_kind, alpha, beta),
            VERBOSE_INCONSISTENT_ALPHA_BETA);
    VCHECK_ELTWISE(
            IMPLICATION(!is_fwd, !any_null(diff_src_desc, diff_dst_desc)),
            VERBOSE_NULL_ARG);
    VCHECK_ELTWISE(IMPLICATION(alg_kind == eltwise_round, is_fwd),
            VERBOSE_BAD_PROPKIND);
    VCHECK_ELTWISE(
            IMPLICATION(is_fwd, !memory_desc_wrapper(src_desc).format_any()),
            VERBOSE_UNSUPPORTED_TAG_S, "src");

    bool runtime_dims_or_strides
            = memory_desc_wrapper(src_desc).has_runtime_dims_or_strides()
            || memory_desc_wrapper(dst_desc).has_runtime_dims_or_strides();
    if (!is_fwd)
        runtime_dims_or_strides = runtime_dims_or_strides
                || memory_desc_wrapper(diff_src_desc)
                           .has_runtime_dims_or_strides()
                || memory_desc_wrapper(diff_dst_desc)
                           .has_runtime_dims_or_strides();
    VCONDCHECK(primitive, create, check, eltwise, !runtime_dims_or_strides,
            status::unimplemented, VERBOSE_RUNTIMEDIM_UNSUPPORTED);

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

#define CHECK_DIMS(t1, t2) \
    do { \
        VCHECK_ELTWISE(ed.t2##_desc.ndims == ed.t1##_desc.ndims, \
                VERBOSE_INCONSISTENT_NDIMS, #t1, #t2); \
        VCHECK_ELTWISE(array_cmp(ed.t2##_desc.dims, ed.t1##_desc.dims, \
                               ed.t1##_desc.ndims), \
                VERBOSE_INCONSISTENT_DIM, #t1, -1, #t2, -1); \
    } while (0)

    if (is_fwd) {
        CHECK_DIMS(src, dst);
    } else {
        CHECK_DIMS(src, diff_dst);
        CHECK_DIMS(diff_src, diff_dst);
    }
#undef CHECK_DIMS

    *eltwise_desc = ed;
    return success;
}
} // namespace impl
} // namespace dnnl

status_t eltwise_attr_check(const eltwise_desc_t &desc, const engine_t *engine,
        const primitive_attr_t *attr) {
    using smask_t = primitive_attr_t::skip_mask_t;

    if (attr == nullptr) return status::success;
    if (attr->has_default_values()) return status::success;

    // Check attributes
    if (utils::one_of(desc.prop_kind, prop_kind::forward_inference,
                prop_kind::forward_training)) {
        const data_type_t dst_dt = desc.dst_desc.data_type;

        auto fwd_attr_mask = smask_t::post_ops;

        VCHECK_ELTWISE_IMPL(attr->has_default_values(fwd_attr_mask, dst_dt),
                VERBOSE_UNSUPPORTED_ATTR);

        // Check post-ops
        if (!attr->post_ops_.has_default_values()) {
            const auto &po = attr->post_ops_;
            using namespace primitive_kind;
            VCHECK_ELTWISE_IMPL(po.has_default_values({binary}),
                    VERBOSE_UNSUPPORTED_POSTOP);
        }
    } else {
        VCHECK_ELTWISE_IMPL(false, VERBOSE_UNSUPPORTED_ATTR);
    }

    return status::success;
}

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
    CHECK(eltwise_attr_check(eltwise_desc, engine, attr));
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
    CHECK(eltwise_attr_check(eltwise_desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&eltwise_desc, hint_fwd_pd, attr);
}

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
