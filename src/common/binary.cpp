/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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
#include "opdesc.hpp"
#include "primitive_desc_iface.hpp"

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::status;
using namespace dnnl::impl::alg_kind;
using namespace dnnl::impl::types;

#define VCHECK_BINARY(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, binary, (cond), \
            status::invalid_arguments, msg, ##__VA_ARGS__);

#define VCHECK_BINARY_UNIMPL(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, binary, (cond), \
            status::unimplemented, msg, ##__VA_ARGS__);

status_t binary_attr_check(const binary_desc_t &desc, const engine_t *engine,
        const primitive_attr_t *attr) {
    using smask_t = primitive_attr_t::skip_mask_t;

    if (attr == nullptr) return status::success;
    if (attr->has_default_values()) return status::success;

    // Check attributes
    const data_type_t dst_dt = desc.dst_desc.data_type;

    auto attr_mask = smask_t::post_ops | smask_t::scales_runtime;

    VCHECK_BINARY_UNIMPL(attr->has_default_values(attr_mask, dst_dt),
            VERBOSE_UNSUPPORTED_ATTR);

    // Check scales
    if (!attr->scales_.has_default_values()) {
        VCHECK_BINARY_UNIMPL(attr->scales_.has_default_values(
                                     {DNNL_ARG_SRC_0, DNNL_ARG_SRC_1}),
                VERBOSE_UNSUPPORTED_SCALES_CFG);

        const auto &sc = attr->scales_;
        const int mask_src_0 = sc.get(DNNL_ARG_SRC_0).mask_;
        const int mask_src_1 = sc.get(DNNL_ARG_SRC_1).mask_;

        VCHECK_BINARY_UNIMPL(utils::everyone_is(0, mask_src_0, mask_src_1),
                VERBOSE_UNSUPPORTED_SCALES_CFG);
    }

    // Check post-ops
    if (!attr->post_ops_.has_default_values()) {
        const auto &po = attr->post_ops_;
        using namespace primitive_kind;
        VCHECK_BINARY_UNIMPL(po.has_default_values({binary, eltwise, sum}),
                VERBOSE_UNSUPPORTED_POSTOP);

        // Check sum
        VCHECK_BINARY_UNIMPL(po.check_sum_consistency(dst_dt, false, true),
                VERBOSE_UNSUPPORTED_POSTOP);
    }

    return status::success;
}

status_t dnnl_binary_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        alg_kind_t alg_kind, const memory_desc_t *src0_md,
        const memory_desc_t *src1_md, const memory_desc_t *dst_md,
        const primitive_attr_t *attr) {
    VCHECK_BINARY(!any_null(src0_md, src1_md, dst_md), VERBOSE_NULL_ARG);
    VCHECK_BINARY(
            one_of(alg_kind, binary_add, binary_mul, binary_max, binary_min,
                    binary_div, binary_sub, binary_ge, binary_gt, binary_le,
                    binary_lt, binary_eq, binary_ne),
            VERBOSE_BAD_ALGORITHM);
    // TODO - Add support for mutual or bi-directional broadcasts
    VCHECK_BINARY(!memory_desc_wrapper(src0_md).format_any(),
            VERBOSE_UNSUPPORTED_TAG_S, "src0");

    auto bod = binary_desc_t();
    bod.primitive_kind = primitive_kind::binary;
    bod.alg_kind = alg_kind;

    VCONDCHECK(primitive, create, check, binary,
            !memory_desc_wrapper(src0_md).has_runtime_dims_or_strides(),
            status::unimplemented, VERBOSE_RUNTIMEDIM_UNSUPPORTED);
    VCONDCHECK(primitive, create, check, binary,
            !memory_desc_wrapper(src1_md).has_runtime_dims_or_strides(),
            status::unimplemented, VERBOSE_RUNTIMEDIM_UNSUPPORTED);
    VCONDCHECK(primitive, create, check, binary,
            !memory_desc_wrapper(dst_md).has_runtime_dims_or_strides(),
            status::unimplemented, VERBOSE_RUNTIMEDIM_UNSUPPORTED);

    bod.src_desc[0] = *src0_md;
    bod.src_desc[1] = *src1_md;
    bod.dst_desc = *dst_md;

    const int ndims = dst_md->ndims;
    const dims_t &dims = dst_md->dims;

    VCHECK_BINARY(
            src0_md->ndims == ndims, VERBOSE_INCONSISTENT_NDIMS, "src0", "dst");
    VCHECK_BINARY(
            src1_md->ndims == ndims, VERBOSE_INCONSISTENT_NDIMS, "src1", "dst");
    for (int d = 0; d < ndims; ++d) {
        //dims must equal eachother or equal 1 (broadcast)
        VCHECK_BINARY(utils::one_of(src0_md->dims[d], 1, dims[d]),
                VERBOSE_BAD_DIM, "src0", d);
        VCHECK_BINARY(utils::one_of(src1_md->dims[d], 1, dims[d]),
                VERBOSE_BAD_DIM, "src1", d);
        VCHECK_BINARY(IMPLICATION(src0_md->dims[d] != dims[d],
                              src1_md->dims[d] == dims[d]),
                VERBOSE_INCONSISTENT_DIM, "src1", d, "dst", d);
    }

    CHECK(binary_attr_check(bod, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&bod, nullptr, attr);
}
