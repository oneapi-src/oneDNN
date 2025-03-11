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

#include "c_types_map.hpp"
#include "opdesc.hpp"
#include "primitive_desc_iface.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::status;
using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::alg_kind;
using namespace dnnl::impl::types;

#define VCHECK_POOLING(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, pooling, (cond), \
            status::invalid_arguments, msg, ##__VA_ARGS__);

#define VCHECK_POOLING_IMPL(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, pooling, (cond), \
            status::unimplemented, msg, ##__VA_ARGS__);

namespace {
status_t pooling_desc_init(pooling_desc_t *pool_desc, prop_kind_t prop_kind,
        alg_kind_t alg_kind, const memory_desc_t *src_desc,
        const memory_desc_t *dst_desc, const dims_t strides,
        const dims_t kernel, const dims_t dilation, const dims_t padding_l,
        const dims_t padding_r) {
    VCHECK_POOLING(!any_null(pool_desc, src_desc, dst_desc, strides, kernel,
                           padding_l),
            VERBOSE_NULL_ARG);
    VCHECK_POOLING(one_of(alg_kind, pooling_max, pooling_avg_include_padding,
                           pooling_avg_exclude_padding),
            VERBOSE_BAD_ALGORITHM);
    VCHECK_POOLING(
            IMPLICATION(one_of(prop_kind, forward_training, forward_inference),
                    !memory_desc_wrapper(src_desc).format_any()),
            VERBOSE_UNSUPPORTED_TAG_S, "src");

    if (padding_r == nullptr) padding_r = padding_l;

    auto pd = pooling_desc_t();
    pd.primitive_kind = primitive_kind::pooling;
    pd.prop_kind = prop_kind;
    pd.alg_kind = alg_kind;
    pd.src_desc.ndims = src_desc->ndims;

    const bool is_fwd = one_of(prop_kind, forward_training, forward_inference);

    const bool rt_dims_or_strides
            = memory_desc_wrapper(src_desc).has_runtime_dims_or_strides()
            || memory_desc_wrapper(dst_desc).has_runtime_dims_or_strides();
    VCONDCHECK(primitive, create, check, pool, !rt_dims_or_strides,
            status::unimplemented, VERBOSE_RUNTIMEDIM_UNSUPPORTED);

    pd.diff_src_desc = pd.src_desc = zero_md();
    pd.diff_dst_desc = pd.dst_desc = zero_md();

    (is_fwd ? pd.src_desc : pd.diff_src_desc) = *src_desc;
    (is_fwd ? pd.dst_desc : pd.diff_dst_desc) = *dst_desc;

    int sp_dims = src_desc->ndims - 2;
    utils::array_copy(pd.strides, strides, sp_dims);
    utils::array_copy(pd.kernel, kernel, sp_dims);
    utils::array_copy(pd.padding[0], padding_l, sp_dims);
    utils::array_copy(pd.padding[1], padding_r, sp_dims);
    utils::array_copy(pd.dilation, dilation, sp_dims);

    if (one_of(alg_kind, pooling_max, pooling_avg_include_padding,
                pooling_avg_exclude_padding)) {
        pd.accum_data_type = types::default_accum_data_type(
                src_desc->data_type, dst_desc->data_type, false);
    } else {
        pd.accum_data_type = dst_desc->data_type;
    }

    VCHECK_POOLING(pd.accum_data_type != data_type::undef,
            VERBOSE_INVALID_DATATYPE, "accumulator");
    VCHECK_POOLING(utils::one_of(src_desc->ndims, 3, 4, 5), VERBOSE_BAD_NDIMS,
            "src", src_desc->ndims);
    VCHECK_POOLING(utils::one_of(dst_desc->ndims, 3, 4, 5), VERBOSE_BAD_NDIMS,
            "dst", dst_desc->ndims);
    for (int i : {0, 1})
        VCHECK_POOLING(src_desc->dims[i] == dst_desc->dims[i],
                VERBOSE_INCONSISTENT_DIM, "src", i, "dst", i);

    for (int i = 2; i < src_desc->ndims; ++i) {
        const dim_t src = src_desc->dims[i];
        const dim_t dst = dst_desc->dims[i];
        const dim_t ker = kernel[i - 2];
        const dim_t dil = dilation ? dilation[i - 2] : 0;
        const dim_t pad_l = padding_l[i - 2];
        const dim_t pad_r = padding_r[i - 2];
        const dim_t str = strides[i - 2];
        const dim_t ker_range = 1 + (ker - 1) * (dil + 1);

        VCHECK_POOLING(str > 0 && dil >= 0 && pad_l >= 0 && (pad_r + str >= 0),
                VERBOSE_INCONSISTENT_PRB);
        VCHECK_POOLING((src - ker_range + pad_l + pad_r) / str + 1 == dst,
                VERBOSE_INCONSISTENT_PRB)

        // It's not allowed for pooling window to be totally placed outside
        // of real source domain for pooling_avg_exclude_padding algorithm
        // due to 0 / 0 ambiguity
        VCHECK_POOLING(
                IMPLICATION(alg_kind == pooling_avg_exclude_padding,
                        (pad_l < ker_range && pad_r < ker_range && dil < src)),
                VERBOSE_INCONSISTENT_PRB);
    }

    *pool_desc = pd;
    return success;
}

status_t pooling_attr_check(const pooling_desc_t &desc, const engine_t *engine,
        const primitive_attr_t *attr) {
    using smask_t = primitive_attr_t::skip_mask_t;

    if (attr == nullptr) return status::success;
    if (attr->has_default_values()) return status::success;

    // Check attributes
    if (utils::one_of(desc.prop_kind, prop_kind::forward_inference,
                prop_kind::forward_training)) {
        const data_type_t dst_dt = desc.dst_desc.data_type;

        auto fwd_attr_mask = smask_t::post_ops;

        VCHECK_POOLING_IMPL(attr->has_default_values(fwd_attr_mask, dst_dt),
                VERBOSE_UNSUPPORTED_ATTR);

        // Check post-ops
        if (!attr->post_ops_.has_default_values()) {
            const auto &po = attr->post_ops_;
            using namespace primitive_kind;
            VCHECK_POOLING_IMPL(po.has_default_values({binary, eltwise}),
                    VERBOSE_UNSUPPORTED_POSTOP);
        }
    } else {
        VCHECK_POOLING_IMPL(false, VERBOSE_UNSUPPORTED_ATTR);
    }

    return status::success;
}

} // namespace

dnnl_status_t dnnl_pooling_forward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        prop_kind_t prop_kind, alg_kind_t alg_kind,
        const memory_desc_t *src_desc, const memory_desc_t *dst_desc,
        const dims_t strides, const dims_t kernel, const dims_t dilation,
        const dims_t padding_l, const dims_t padding_r,
        const primitive_attr_t *attr) {

    if (!one_of(prop_kind, forward_training, forward_inference))
        return invalid_arguments;

    auto pool_desc = pooling_desc_t();
    CHECK(pooling_desc_init(&pool_desc, prop_kind, alg_kind, src_desc, dst_desc,
            strides, kernel, dilation, padding_l, padding_r));
    CHECK(pooling_attr_check(pool_desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&pool_desc, nullptr, attr);
}

dnnl_status_t dnnl_pooling_backward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        alg_kind_t alg_kind, const memory_desc_t *diff_src_desc,
        const memory_desc_t *diff_dst_desc, const dims_t strides,
        const dims_t kernel, const dims_t dilation, const dims_t padding_l,
        const dims_t padding_r, const primitive_desc_iface_t *hint_fwd_pd,
        const primitive_attr_t *attr) {

    auto pool_desc = pooling_desc_t();
    CHECK(pooling_desc_init(&pool_desc, prop_kind::backward_data, alg_kind,
            diff_src_desc, diff_dst_desc, strides, kernel, dilation, padding_l,
            padding_r));
    CHECK(pooling_attr_check(pool_desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&pool_desc, hint_fwd_pd, attr);
}

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
