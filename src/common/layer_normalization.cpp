/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
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
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::status;
using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::types;

namespace {
status_t lnorm_desc_init(layer_normalization_desc_t *lnorm_desc,
        prop_kind_t prop_kind, const memory_desc_t *src_desc,
        const memory_desc_t *dst_desc, const memory_desc_t *stat_desc,
        const memory_desc_t *diff_src_desc, const memory_desc_t *diff_dst_desc,
        float epsilon, unsigned flags) {
    bool args_ok = !any_null(lnorm_desc, src_desc) && 2 <= src_desc->ndims
            && src_desc->ndims <= 5
            && (flags
                       & ~(dnnl_use_global_stats | dnnl_use_scaleshift
                               | dnnl_use_scale | dnnl_use_shift))
                    == 0;
    if (!args_ok) return invalid_arguments;

    bool is_fwd
            = prop_kind == forward_training || prop_kind == forward_inference;
    args_ok = IMPLICATION(is_fwd, dst_desc != nullptr)
            && IMPLICATION(!is_fwd, !any_null(diff_src_desc, diff_dst_desc))
            && IMPLICATION(is_fwd, !memory_desc_wrapper(src_desc).format_any());
    if (!args_ok) return invalid_arguments;

    auto ld = layer_normalization_desc_t();
    ld.primitive_kind = primitive_kind::layer_normalization;
    ld.prop_kind = prop_kind;

    bool runtime_dims_or_strides
            = memory_desc_wrapper(src_desc).has_runtime_dims_or_strides()
            || memory_desc_wrapper(dst_desc).has_runtime_dims_or_strides()
            || (stat_desc
                    && memory_desc_wrapper(stat_desc)
                               .has_runtime_dims_or_strides());
    if (!is_fwd)
        runtime_dims_or_strides = runtime_dims_or_strides
                || memory_desc_wrapper(diff_src_desc)
                           .has_runtime_dims_or_strides()
                || memory_desc_wrapper(diff_dst_desc)
                           .has_runtime_dims_or_strides();
    if (runtime_dims_or_strides) return unimplemented;

    ld.src_desc = *src_desc;
    if (is_fwd) ld.dst_desc = *dst_desc;
    if (!is_fwd) ld.diff_src_desc = *diff_src_desc;
    if (!is_fwd) ld.diff_dst_desc = *diff_dst_desc;

    if (stat_desc)
        ld.stat_desc = *stat_desc;
    else
        CHECK(dnnl_memory_desc_init_by_tag(&ld.stat_desc, ld.src_desc.ndims - 1,
                ld.src_desc.dims, data_type::f32, format_tag::any));

    int ndims = src_desc->ndims;
    ld.data_scaleshift_desc = zero_md();
    if (flags & (dnnl_use_scale | dnnl_use_shift)) {
        dims_t scaleshift_dims = {src_desc->dims[ndims - 1]};
        dnnl_memory_desc_init_by_tag(&ld.data_scaleshift_desc, 1,
                scaleshift_dims, data_type::f32, dnnl_x);
    } else {
        dims_t scaleshift_dims = {2, src_desc->dims[ndims - 1]};
        dnnl_memory_desc_init_by_tag(&ld.data_scaleshift_desc, 2,
                scaleshift_dims, data_type::f32, dnnl_nc);
    }
    if (ld.prop_kind == backward) {
        ld.diff_data_scaleshift_desc = ld.data_scaleshift_desc;
    }

    ld.layer_norm_epsilon = epsilon;

    // dnnl_use_scaleshift can't be mixed with dnnl_use_scale or dnnl_use_shift
    if ((flags & dnnl_use_scaleshift)
            && (flags & (dnnl_use_scale | dnnl_use_shift)))
        return invalid_arguments;

    ld.flags = flags;

    if (is_fwd) {
        bool consistency = ld.src_desc.ndims == ld.dst_desc.ndims
                && array_cmp(
                        ld.src_desc.dims, ld.dst_desc.dims, ld.src_desc.ndims);
        if (!consistency) return invalid_arguments;
    } else {
        bool consistency = ld.diff_src_desc.ndims == ld.src_desc.ndims
                && array_cmp(ld.diff_src_desc.dims, ld.src_desc.dims,
                        ld.diff_src_desc.ndims)
                && ld.diff_src_desc.ndims == ld.diff_dst_desc.ndims
                && array_cmp(ld.diff_src_desc.dims, ld.diff_dst_desc.dims,
                        ld.diff_src_desc.ndims)
                && ld.src_desc.ndims == ld.stat_desc.ndims + 1
                && array_cmp(ld.stat_desc.dims, ld.src_desc.dims,
                        ld.stat_desc.ndims);
        if (!consistency) return invalid_arguments;
    }

    *lnorm_desc = ld;
    return success;
}
} // namespace

status_t dnnl_layer_normalization_forward_desc_init(
        layer_normalization_desc_t *lnorm_desc, prop_kind_t prop_kind,
        const memory_desc_t *src_desc, const memory_desc_t *dst_desc,
        const memory_desc_t *stat_desc, float epsilon, unsigned flags) {
    if (!one_of(prop_kind, forward_training, forward_inference))
        return invalid_arguments;
    return lnorm_desc_init(lnorm_desc, prop_kind, src_desc, dst_desc, stat_desc,
            nullptr, nullptr, epsilon, flags);
}

status_t dnnl_layer_normalization_backward_desc_init(
        layer_normalization_desc_t *lnorm_desc, prop_kind_t prop_kind,
        const memory_desc_t *diff_src_desc, const memory_desc_t *diff_dst_desc,
        const memory_desc_t *src_desc, const memory_desc_t *stat_desc,
        float epsilon, unsigned flags) {
    if (!one_of(prop_kind, backward, backward_data)) return invalid_arguments;
    return lnorm_desc_init(lnorm_desc, prop_kind, src_desc, nullptr, stat_desc,
            diff_src_desc, diff_dst_desc, epsilon, flags);
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
