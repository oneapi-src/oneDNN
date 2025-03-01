/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#ifndef COMMON_GATED_MLP_UTILS_HPP
#define COMMON_GATED_MLP_UTILS_HPP

#include "oneapi/dnnl/dnnl.h"

#include "common/c_types_map.hpp"
#include "common/gated_mlp_types.hpp"
#include "common/nstl.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {

static inline gated_mlp_desc_t create_gated_mlp_desc(
        const memory_desc_t *src_md, const memory_desc_t *W_gate_md,
        const memory_desc_t *W_up_md, const memory_desc_t *W_down_md,
        const memory_desc_t *dst_md, const primitive_attr_t *gate_attr,
        const primitive_attr_t *up_attr, const primitive_attr_t *down_attr) {

    auto gated_mlp_desc = gated_mlp_desc_t();
    gated_mlp_desc.primitive_kind = primitive_kind::gated_mlp;
    gated_mlp_desc.src_desc = *src_md;
    gated_mlp_desc.W_gate_desc = *W_gate_md;
    gated_mlp_desc.W_up_desc = *W_up_md;
    gated_mlp_desc.W_down_desc = *W_down_md;
    gated_mlp_desc.dst_desc = *dst_md;

    if (gate_attr) {
        gated_mlp_desc.wts_gate_scales = gate_attr->scales_.get(DNNL_ARG_SRC_0);
        gated_mlp_desc.wts_gate_zero_points = gate_attr->zero_points_;
    }
    if (up_attr) {
        gated_mlp_desc.wts_up_scales = up_attr->scales_.get(DNNL_ARG_SRC_1);
        gated_mlp_desc.wts_up_zero_points = up_attr->zero_points_;
    }
    if (down_attr) {
        gated_mlp_desc.wts_down_scales = down_attr->scales_.get(DNNL_ARG_SRC_2);
        gated_mlp_desc.wts_down_zero_points = down_attr->zero_points_;
    }
    return gated_mlp_desc;
}

static inline status_t create_gated_mlp_pd(
        std::shared_ptr<primitive_desc_t> &gated_mlp_pd_, engine_t *engine,
        const memory_desc_t *src_md, const memory_desc_t *W_gate_md,
        const memory_desc_t *W_up_md, const memory_desc_t *W_down_md,
        const memory_desc_t *dst_md, const primitive_attr_t *attr,
        const primitive_attr_t *gate_attr = nullptr,
        const primitive_attr_t *up_attr = nullptr,
        const primitive_attr_t *down_attr = nullptr) {

    auto gated_mlp_desc = create_gated_mlp_desc(src_md, W_gate_md, W_up_md,
            W_down_md, dst_md, gate_attr, up_attr, down_attr);

    int ndims = dst_md->ndims;

    if (ndims != 2) { return status::invalid_arguments; }
    if (!utils::everyone_is(ndims, src_md->ndims, W_gate_md->ndims,
                W_up_md->ndims, W_down_md->ndims))
        return status::invalid_arguments;
    long mb = src_md->dims[0];
    long ic = src_md->dims[1];
    long oc = W_gate_md->dims[1];
    if (W_gate_md->dims[0] != ic || W_gate_md->dims[1] != oc)
        return status::invalid_arguments;
    if (W_up_md->dims[0] != ic || W_up_md->dims[1] != oc)
        return status::invalid_arguments;
    if (W_gate_md->dims[0] != ic || W_gate_md->dims[1] != oc)
        return status::invalid_arguments;
    if (W_down_md->dims[0] != oc || W_down_md->dims[1] != ic)
        return status::invalid_arguments;
    if (dst_md->dims[0] != mb || dst_md->dims[1] != ic)
        return status::invalid_arguments;

    primitive_attr_t gated_mlp_attr = attr ? *attr : default_attr();

    primitive_desc_iterator_t it(
            engine, (op_desc_t *)&gated_mlp_desc, &gated_mlp_attr, nullptr);

    gated_mlp_pd_ = *(++it);
    if (!gated_mlp_pd_) return status::unimplemented;

    return status::success;
}

} // namespace impl
} // namespace dnnl

#endif
