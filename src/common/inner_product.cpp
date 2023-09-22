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
using namespace dnnl::impl::types;

#define VCHECK_IP(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, ip, (cond), \
            status::invalid_arguments, msg, ##__VA_ARGS__)

#define VCHECK_IP_UNIMPL(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, ip, (cond), status::unimplemented, \
            msg, ##__VA_ARGS__)

namespace dnnl {
namespace impl {
status_t ip_desc_init(inner_product_desc_t *ip_desc, prop_kind_t prop_kind,
        const memory_desc_t *src_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *bias_desc, const memory_desc_t *dst_desc) {
    VCHECK_IP(!any_null(ip_desc, src_desc, weights_desc, dst_desc),
            VERBOSE_NULL_ARG);

    auto id = inner_product_desc_t();
    id.primitive_kind = primitive_kind::inner_product;
    id.prop_kind = prop_kind;

    id.diff_src_desc = id.src_desc = zero_md();
    id.diff_dst_desc = id.dst_desc = zero_md();
    id.diff_weights_desc = id.weights_desc = zero_md();
    id.diff_bias_desc = id.bias_desc = zero_md();

    const bool is_fwd = one_of(prop_kind, forward_training, forward_inference);
    const bool with_bias
            = bias_desc && bias_desc->format_kind != format_kind::undef;

    bool runtime_dims_or_strides
            = memory_desc_wrapper(src_desc).has_runtime_dims_or_strides()
            || memory_desc_wrapper(weights_desc).has_runtime_dims_or_strides()
            || memory_desc_wrapper(dst_desc).has_runtime_dims_or_strides();
    if (with_bias)
        runtime_dims_or_strides = runtime_dims_or_strides
                || memory_desc_wrapper(bias_desc).has_runtime_dims_or_strides();
    VCONDCHECK(primitive, create, check, ip, !runtime_dims_or_strides,
            status::unimplemented, VERBOSE_RUNTIMEDIM_UNSUPPORTED);

    (prop_kind == backward_data ? id.diff_src_desc : id.src_desc) = *src_desc;
    (is_fwd ? id.dst_desc : id.diff_dst_desc) = *dst_desc;
    (prop_kind == backward_weights ? id.diff_weights_desc : id.weights_desc)
            = *weights_desc;
    if (with_bias)
        (prop_kind == backward_weights ? id.diff_bias_desc : id.bias_desc)
                = *bias_desc;

    id.accum_data_type = types::default_accum_data_type(src_desc->data_type,
            weights_desc->data_type, dst_desc->data_type, prop_kind);
    VCHECK_IP(id.accum_data_type != data_type::undef, VERBOSE_INVALID_DATATYPE,
            "accumulation");

    VCHECK_IP(memory_desc_wrapper(weights_desc).nelems(), VERBOSE_EMPTY_TENSOR,
            "weights");
    VCHECK_IP(one_of(src_desc->ndims, 2, 3, 4, 5), VERBOSE_BAD_NDIMS, "src",
            src_desc->ndims);
    VCHECK_IP(dst_desc->ndims == 2, VERBOSE_BAD_NDIMS, "dst", dst_desc->ndims);
    VCHECK_IP(weights_desc->ndims == src_desc->ndims,
            VERBOSE_INCONSISTENT_NDIMS, "weights", "src");
    VCHECK_IP((with_bias ? bias_desc->ndims == 1 : true), VERBOSE_BAD_NDIMS,
            "bias", bias_desc->ndims);
    VCHECK_IP((with_bias ? bias_desc->dims[0] == dst_desc->dims[1] : true),
            VERBOSE_INCONSISTENT_DIM, "bias", 0, "dst", 1);
    VCHECK_IP(src_desc->dims[0] == dst_desc->dims[0], VERBOSE_INCONSISTENT_DIM,
            "src", 0, "dst", 0);
    VCHECK_IP(array_cmp(&src_desc->dims[1], &weights_desc->dims[1],
                      src_desc->ndims - 1),
            VERBOSE_INCONSISTENT_DIM, "src", -1, "weights", -1);
    VCHECK_IP(dst_desc->dims[1] == weights_desc->dims[0],
            VERBOSE_INCONSISTENT_DIM, "dst", 1, "weights", 0);

    *ip_desc = id;
    return success;
}

status_t ip_attr_check(const inner_product_desc_t &desc, const engine_t *engine,
        const primitive_attr_t *attr) {
    using smask_t = primitive_attr_t::skip_mask_t;

    if (attr == nullptr) return status::success;
    if (attr->has_default_values()) return status::success;

    // Check attributes
    if (utils::one_of(desc.prop_kind, prop_kind::forward_inference,
                prop_kind::forward_training)) {
        const data_type_t src_dt = desc.src_desc.data_type;
        const data_type_t dst_dt = desc.dst_desc.data_type;

        auto fwd_attr_mask = smask_t::post_ops | smask_t::sum_dt;

        bool is_int8 = utils::one_of(src_dt, data_type::s8, data_type::u8);
        if (engine->kind() == engine_kind::gpu)
            is_int8 = is_int8
                    || utils::one_of(dst_dt, data_type::s8, data_type::u8,
                            data_type::s32);
        if (is_int8) fwd_attr_mask |= smask_t::scales_runtime;

        VCHECK_IP_UNIMPL(attr->has_default_values(fwd_attr_mask, dst_dt),
                VERBOSE_UNSUPPORTED_ATTR);

        // Check scales
        if (!attr->scales_.has_default_values()) {
            const auto &sc = attr->scales_;
            const int mask_src = sc.get(DNNL_ARG_SRC).mask_;
            const int mask_wei = sc.get(DNNL_ARG_WEIGHTS).mask_;
            const int mask_dst = sc.get(DNNL_ARG_DST).mask_;

            VCHECK_IP_UNIMPL(utils::everyone_is(0, mask_src, mask_dst)
                            && utils::one_of(mask_wei, 0, 1),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
        }

        // Check post-ops
        if (!attr->post_ops_.has_default_values()) {
            const auto &po = attr->post_ops_;
            using namespace primitive_kind;
            VCHECK_IP_UNIMPL(
                    po.has_default_values({binary, eltwise, prelu, sum}),
                    VERBOSE_UNSUPPORTED_POSTOP);

            // Check sum
            VCHECK_IP_UNIMPL(po.check_sum_consistency(dst_dt, is_int8, true),
                    VERBOSE_UNSUPPORTED_POSTOP);
        }
    } else {
        VCHECK_IP_UNIMPL(false, VERBOSE_UNSUPPORTED_ATTR);
    }

    return status::success;
}

} // namespace impl
} // namespace dnnl

status_t dnnl_inner_product_forward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        prop_kind_t prop_kind, const memory_desc_t *src_desc,
        const memory_desc_t *weights_desc, const memory_desc_t *bias_desc,
        const memory_desc_t *dst_desc, const primitive_attr_t *attr) {
    if (!one_of(prop_kind, forward_training, forward_inference))
        return invalid_arguments;

    auto ip_desc = inner_product_desc_t();
    CHECK(ip_desc_init(
            &ip_desc, prop_kind, src_desc, weights_desc, bias_desc, dst_desc));
    CHECK(ip_attr_check(ip_desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&ip_desc, nullptr, attr);
}

status_t dnnl_inner_product_backward_data_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        const memory_desc_t *diff_src_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *diff_dst_desc,
        const primitive_desc_iface_t *hint_fwd_pd,
        const primitive_attr_t *attr) {

    auto ip_desc = inner_product_desc_t();
    CHECK(ip_desc_init(&ip_desc, backward_data, diff_src_desc, weights_desc,
            nullptr, diff_dst_desc));
    CHECK(ip_attr_check(ip_desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&ip_desc, hint_fwd_pd, attr);
}

status_t dnnl_inner_product_backward_weights_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        const memory_desc_t *src_desc, const memory_desc_t *diff_weights_desc,
        const memory_desc_t *diff_bias_desc, const memory_desc_t *diff_dst_desc,
        const primitive_desc_iface_t *hint_fwd_pd,
        const primitive_attr_t *attr) {

    auto ip_desc = inner_product_desc_t();
    CHECK(ip_desc_init(&ip_desc, backward_weights, src_desc, diff_weights_desc,
            diff_bias_desc, diff_dst_desc));
    CHECK(ip_attr_check(ip_desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&ip_desc, hint_fwd_pd, attr);
}

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
