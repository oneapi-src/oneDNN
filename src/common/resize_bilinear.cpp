/*******************************************************************************
* Copyright 2018 Intel Corporation
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
#include "mkldnn.h"
#include "patch_mkldnn.h"

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace mkldnn::impl;
using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::status;
using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::types;

namespace {
status_t resize_bilinear_desc_init(resize_bilinear_desc_t *bilinear_desc,
        prop_kind_t prop_kind,
        const memory_desc_t *src_desc, const memory_desc_t *dst_desc,
        const int *align_corners) {
    bool args_ok = true
        && !any_null(bilinear_desc, src_desc, dst_desc);
    if (!args_ok) return invalid_arguments;

    auto pd = resize_bilinear_desc_t();
    pd.primitive_kind = primitive_kind::resize_bilinear;
    pd.prop_kind = prop_kind;
    pd.src_desc.ndims = src_desc->ndims;
    pd.align_corners = *align_corners;

    const bool is_fwd = one_of(prop_kind, forward_training, forward_inference);

    pd.diff_src_desc = pd.src_desc = zero_md();
    pd.diff_dst_desc = pd.dst_desc = zero_md();

    (is_fwd ? pd.src_desc : pd.diff_src_desc) = *src_desc;
    (is_fwd ? pd.dst_desc : pd.diff_dst_desc) = *dst_desc;

    pd.accum_data_type = types::default_accum_data_type(
            src_desc->data_type, dst_desc->data_type);

    bool consistency = true
        && src_desc->dims[0] == dst_desc->dims[0]
        && src_desc->dims[1] == dst_desc->dims[1]
        && utils::one_of(src_desc->ndims, 4, 5)
        && utils::one_of(dst_desc->ndims, 4, 5);
    if (!consistency) return invalid_arguments;

    *bilinear_desc = pd;
    return success;
}
}

status_t mkldnn_resize_bilinear_forward_desc_init(resize_bilinear_desc_t *bilinear_desc,
        prop_kind_t prop_kind,
        const memory_desc_t *src_desc, const memory_desc_t *dst_desc,
        const int *align_corners) {
    if (!one_of(prop_kind, forward_training, forward_inference))
        return invalid_arguments;

    return resize_bilinear_desc_init(bilinear_desc, prop_kind, src_desc, dst_desc, align_corners);
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
