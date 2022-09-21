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
status_t bnrm_desc_init(batch_normalization_desc_t *bnrm_desc,
        prop_kind_t prop_kind, const memory_desc_t *data_desc,
        const memory_desc_t *diff_data_desc, float epsilon, unsigned flags) {
    bool args_ok = !any_null(bnrm_desc, data_desc)
            && one_of(prop_kind, forward_training, forward_inference,
                    backward_data, backward)
            && IMPLICATION(prop_kind & backward, diff_data_desc != nullptr)
            && IMPLICATION(
                    one_of(prop_kind, forward_training, forward_inference),
                    !memory_desc_wrapper(data_desc).format_any());
    if (!args_ok) return invalid_arguments;

    unsigned bnorm_flags = normalization_flags::use_global_stats
            | normalization_flags::fuse_norm_relu
            | normalization_flags::fuse_norm_add_relu
            | normalization_flags::use_scale | normalization_flags::use_shift;
    if ((~bnorm_flags & flags) != 0) return invalid_arguments;

    auto bd = batch_normalization_desc_t();
    bd.primitive_kind = primitive_kind::batch_normalization;
    bd.prop_kind = prop_kind;

    bool runtime_dims_or_strides
            = memory_desc_wrapper(data_desc).has_runtime_dims_or_strides();
    if (one_of(prop_kind, backward_data, backward))
        runtime_dims_or_strides = runtime_dims_or_strides
                || memory_desc_wrapper(diff_data_desc)
                           .has_runtime_dims_or_strides();
    if (runtime_dims_or_strides) return unimplemented;

    bd.data_desc = *data_desc;
    bd.diff_data_desc = zero_md();
    if (one_of(bd.prop_kind, backward_data, backward))
        bd.diff_data_desc = *diff_data_desc;

    bd.data_scaleshift_desc = zero_md();
    if (flags
            & (normalization_flags::use_scale
                    | normalization_flags::use_shift)) {
        dims_t scaleshift_dims = {data_desc->dims[1]};
        memory_desc_init_by_tag(bd.data_scaleshift_desc, 1, scaleshift_dims,
                data_type::f32, dnnl_x);
    }

    bd.diff_data_scaleshift_desc = zero_md();
    if (bd.prop_kind == backward
            && (flags
                    & (normalization_flags::use_scale
                            | normalization_flags::use_shift))) {
        bd.diff_data_scaleshift_desc = bd.data_scaleshift_desc;
    }

    dims_t stats_dims = {data_desc->dims[1]};
    memory_desc_init_by_tag(
            bd.stat_desc, 1, stats_dims, data_type::f32, dnnl_x);
    bd.batch_norm_epsilon = epsilon;

    bd.flags = flags;

    bool consistency = true && utils::one_of(bd.data_desc.ndims, 2, 3, 4, 5);
    if (bd.prop_kind == backward_data)
        consistency = consistency
                && utils::one_of(bd.diff_data_desc.ndims, 2, 3, 4, 5)
                && array_cmp(bd.diff_data_desc.dims, bd.data_desc.dims,
                        bd.diff_data_desc.ndims);
    if (!consistency) return invalid_arguments;

    *bnrm_desc = bd;
    return success;
}
} // namespace

status_t dnnl_batch_normalization_forward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        prop_kind_t prop_kind, const memory_desc_t *data_desc, float epsilon,
        unsigned flags, const primitive_attr_t *attr) {
    if (!one_of(prop_kind, forward_training, forward_inference))
        return invalid_arguments;

    auto bnrm_desc = batch_normalization_desc_t();
    CHECK(bnrm_desc_init(
            &bnrm_desc, prop_kind, data_desc, nullptr, epsilon, flags));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&bnrm_desc, nullptr, attr);
}

status_t dnnl_batch_normalization_backward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        prop_kind_t prop_kind, const memory_desc_t *diff_data_desc,
        const memory_desc_t *data_desc, float epsilon, unsigned flags,
        const primitive_desc_iface_t *hint_fwd_pd,
        const primitive_attr_t *attr) {
    if (!one_of(prop_kind, backward, backward_data)) return invalid_arguments;

    auto bnrm_desc = batch_normalization_desc_t();
    CHECK(bnrm_desc_init(
            &bnrm_desc, prop_kind, data_desc, diff_data_desc, epsilon, flags));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&bnrm_desc, hint_fwd_pd, attr);
}

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
