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

#include "oneapi/dnnl/dnnl.h"
#include "opdesc.hpp"
#include "primitive_desc_iface.hpp"

#include "c_types_map.hpp"
#include "utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::alg_kind;

namespace dnnl {
namespace impl {

status_t reduction_desc_init(reduction_desc_t *reduction_desc,
        alg_kind_t alg_kind, const dnnl_memory_desc_t *src_desc,
        const dnnl_memory_desc_t *dst_desc, float p, float eps) {

    bool args_ok = !any_null(src_desc, dst_desc)
            && src_desc->format_kind != format_kind::any
            && one_of(alg_kind, reduction_max, reduction_min, reduction_sum,
                    reduction_mul, reduction_mean, reduction_norm_lp_max,
                    reduction_norm_lp_sum, reduction_norm_lp_power_p_max,
                    reduction_norm_lp_power_p_sum)
            && IMPLICATION(one_of(alg_kind, reduction_norm_lp_max,
                                   reduction_norm_lp_sum,
                                   reduction_norm_lp_power_p_max,
                                   reduction_norm_lp_power_p_sum),
                    p >= 1.0f)
            && IMPLICATION(one_of(alg_kind, reduction_norm_lp_max,
                                   reduction_norm_lp_sum,
                                   reduction_norm_lp_power_p_max,
                                   reduction_norm_lp_power_p_sum),
                    one_of(src_desc->data_type, data_type::f32, data_type::bf16,
                            data_type::f16));
    if (!args_ok) return invalid_arguments;

    if (src_desc->ndims != dst_desc->ndims) return invalid_arguments;

    for (auto d = 0; d < src_desc->ndims; ++d) {
        const auto dst_dim_d = dst_desc->dims[d];
        if (!one_of(dst_dim_d, 1, src_desc->dims[d])) return invalid_arguments;
    }

    // reduction primitive doesn't support identity operation
    if (array_cmp(src_desc->dims, dst_desc->dims, src_desc->ndims))
        return invalid_arguments;

    if (src_desc->format_kind != format_kind::blocked
            || !one_of(dst_desc->format_kind, format_kind::blocked,
                    format_kind::any))
        return invalid_arguments;

    if (src_desc->extra.flags != 0
            || !IMPLICATION(dst_desc->format_kind == format_kind::blocked,
                    dst_desc->extra.flags == 0))
        return invalid_arguments;

    auto rd = reduction_desc_t();
    rd.primitive_kind = primitive_kind::reduction;
    rd.alg_kind = alg_kind;

    rd.src_desc = *src_desc;
    rd.dst_desc = *dst_desc;
    rd.p = p;
    rd.eps = eps;

    (*reduction_desc) = rd;
    return success;
}

} // namespace impl
} // namespace dnnl

dnnl_status_t dnnl_reduction_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        alg_kind_t alg_kind, const dnnl_memory_desc_t *src_desc,
        const dnnl_memory_desc_t *dst_desc, float p, float eps,
        const primitive_attr_t *attr) {

    auto reduction_desc = reduction_desc_t();
    CHECK(reduction_desc_init(
            &reduction_desc, alg_kind, src_desc, dst_desc, p, eps));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&reduction_desc, nullptr, attr);
}
