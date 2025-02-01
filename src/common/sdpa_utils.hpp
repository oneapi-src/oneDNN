/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#ifndef COMMON_SDPA_UTILS_HPP
#define COMMON_SDPA_UTILS_HPP

#include "oneapi/dnnl/dnnl.h"

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "common/sdpa_types.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {

#define VCHECK_SDPA(f, msg, ...) \
    VCHECK(primitive, create, check, sdpa, (f), msg, ##__VA_ARGS__);

#define VCHECK_SDPA_COND(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, sdpa, (cond), \
            status::invalid_arguments, msg, ##__VA_ARGS__);

#define VCHECK_SDPA_ATTR_TYPE( \
        variable_check, variable, attribute_member_name, expected_types) \
    VCONDCHECK(primitive, create, check, sdpa, (variable_check), \
            status::invalid_arguments, VERBOSE_INVALID_DATATYPE, \
            format_verbose_string(#variable attribute_member_name \
                    "(%s). must be " expected_types, \
                    attr2str(variable).c_str()) \
                    .c_str())

#define VCHECK_SDPA_UNIMPL(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, sdpa, (cond), status::unimplemented, \
            msg, ##__VA_ARGS__);

static inline status_t sdpa_attr_check(const memory_desc_t *q_desc,
        const memory_desc_t *k_desc, const memory_desc_t *v_desc,
        const engine_t *engine, const primitive_attr_t *attr,
        const primitive_attr_t *kq_attr, const primitive_attr_t *vs_attr) {
    using smask_t = primitive_attr_t::skip_mask_t;

    if (utils::everyone_is(nullptr, attr, kq_attr, vs_attr))
        return status::success;
    if (attr && attr->has_default_values() && kq_attr
            && kq_attr->has_default_values() && vs_attr
            && vs_attr->has_default_values()) {
        return status::success;
    }

    using namespace dnnl::impl::data_type;
    if (kq_attr && !kq_attr->has_default_values()) {
        const auto &sc = kq_attr->scales_;
        const auto &zp = kq_attr->zero_points_;
        if (!sc.has_default_values()) {
            const auto &scale_dt = sc.get_data_type(DNNL_ARG_WEIGHTS);
            VCHECK_SDPA_ATTR_TYPE(utils::one_of(scale_dt, f16, f32), kq_attr,
                    "scales", "f16 or f32");
        }
        if (!zp.has_default_values()) {
            const auto &zp_dt = zp.get_data_type(DNNL_ARG_WEIGHTS);
            VCHECK_SDPA_ATTR_TYPE(utils::one_of(zp_dt, s4, u4, u8, s8, s32),
                    kq_attr, "zero_points", "u4, s4, u8, s8, or s32");
        }
    }

    if (vs_attr && !vs_attr->has_default_values()) {
        const auto &sc = vs_attr->scales_;
        const auto &zp = vs_attr->zero_points_;

        if (!sc.has_default_values()) {
            const auto &scale_dt = sc.get_data_type(DNNL_ARG_WEIGHTS);
            VCHECK_SDPA_ATTR_TYPE(utils::one_of(scale_dt, f16, f32), vs_attr,
                    "scales", "f16 or f32");
        }
        if (!zp.has_default_values()) {
            const auto &zp_dt = zp.get_data_type(DNNL_ARG_WEIGHTS);
            VCHECK_SDPA_ATTR_TYPE(utils::one_of(zp_dt, s4, u4, u8, s8, s32),
                    vs_attr, "zero_points", "u4, s4, u8, s8, or s32");
        }
    }

    if (attr) {
        smask_t attr_mask = smask_t::none;
        VCHECK_SDPA_UNIMPL(
                attr->has_default_values(attr_mask), VERBOSE_UNSUPPORTED_ATTR);
    }

    return status::success;
}

static inline sdpa_desc_t create_sdpa_desc(const memory_desc_t *q_md,
        const memory_desc_t *k_md, const memory_desc_t *v_md,
        const memory_desc_t *dst_md, const memory_desc_t *attn_mask_md,
        data_type_t scale_dt, bool invert_scale, dim_t kv_head_number,
        bool causal_mask, const primitive_attr_t *kq_attr,
        const primitive_attr_t *vs_attr) {
    auto sdpa_desc = sdpa_desc_t();
    sdpa_desc.primitive_kind = primitive_kind::sdpa;
    sdpa_desc.q_desc = *q_md;
    sdpa_desc.k_desc = *k_md;
    if (kq_attr) {
        sdpa_desc.kq_scales = kq_attr->scales_.get(DNNL_ARG_WEIGHTS);
        sdpa_desc.kq_zero_points = kq_attr->zero_points_;
    }
    if (vs_attr) {
        sdpa_desc.vs_scales = vs_attr->scales_.get(DNNL_ARG_WEIGHTS);
        sdpa_desc.vs_zero_points = vs_attr->zero_points_;
    }
    sdpa_desc.v_desc = *v_md;
    sdpa_desc.dst_desc = *dst_md;
    if (attn_mask_md) sdpa_desc.attn_mask_desc = *attn_mask_md;
    sdpa_desc.scale_dt = scale_dt;
    sdpa_desc.invert_scale = invert_scale;
    sdpa_desc.kv_head_number = kv_head_number;
    sdpa_desc.causal_mask = causal_mask;
    return sdpa_desc;
}

static inline status_t create_sdpa_pd(
        std::shared_ptr<primitive_desc_t> &sdpa_pd_, engine_t *engine,
        const memory_desc_t *q_md, const memory_desc_t *k_md,
        const memory_desc_t *v_md, const memory_desc_t *dst_md,
        const memory_desc_t *attn_mask_md, data_type_t scale_dt,
        bool invert_scale, dim_t kv_head_number, bool causal_mask,
        const primitive_attr_t *attr, const primitive_attr_t *kq_attr = nullptr,
        const primitive_attr_t *vs_attr = nullptr) {
    CHECK(sdpa_attr_check(q_md, k_md, v_md, engine, attr, kq_attr, vs_attr));

    auto sdpa_desc = create_sdpa_desc(q_md, k_md, v_md, dst_md, attn_mask_md,
            scale_dt, invert_scale, kv_head_number, causal_mask, kq_attr,
            vs_attr);

    int ndims = dst_md->ndims;
    int r = ndims - 2, c = ndims - 1;
    VCHECK_SDPA_COND(
            utils::everyone_is(ndims, q_md->ndims, k_md->ndims, v_md->ndims),
            "number of dimensions have to match. expected: %d q: %d k: %d v: "
            "%d",
            ndims, q_md->ndims, k_md->ndims, v_md->ndims);

    VCHECK_SDPA_COND(q_md->dims[c] == k_md->dims[r],
            "q_md->dims[%d](%s) must match k_md->dims[%d](%s)", c,
            md2dim_str(q_md).c_str(), r, md2dim_str(k_md).c_str());
    VCHECK_SDPA_COND(k_md->dims[c] == v_md->dims[r],
            "k_md->dims[%d](%s) must match v_md->dims[%d](%s)", c,
            md2dim_str(k_md).c_str(), r, md2dim_str(v_md).c_str());
    VCHECK_SDPA_COND(dst_md->dims[r] == q_md->dims[r],
            "dst_md->dims[%d](%s) == q_md->dims[%d](%s)", r,
            md2dim_str(dst_md).c_str(), r, md2dim_str(q_md).c_str());
    VCHECK_SDPA_COND(dst_md->dims[c] == v_md->dims[c],
            "dst_md->dims[%d](%s) == v_md->dims[%d](%s)", c,
            md2dim_str(dst_md).c_str(), c, md2dim_str(v_md).c_str());

    primitive_attr_t sdpa_attr = attr ? *attr : default_attr();

    primitive_desc_iterator_t it(
            engine, (op_desc_t *)&sdpa_desc, &sdpa_attr, nullptr);

    sdpa_pd_ = *(++it);
    VCHECK_SDPA_COND(sdpa_pd_, "failed to create the SDPA primitive");

    return status::success;
}

} // namespace impl
} // namespace dnnl

#endif
