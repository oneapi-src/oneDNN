/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#include "c_types_map.hpp"
#include "utils.hpp"

using namespace mkldnn::impl;
using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::status;
using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::alg_kind;

namespace {
status_t bnrm_desc_init(batch_normalization_desc_t *bnrm_desc,
        prop_kind_t prop_kind, const memory_desc_t *data_desc,
        const memory_desc_t *diff_data_desc, double epsilon) {
    bool args_ok = true
        && !any_null(bnrm_desc, data_desc)
        && one_of(prop_kind, forward_training, forward_inference,
                backward_data, backward)
        && implication(prop_kind & backward, diff_data_desc != nullptr);
    if (!args_ok) return invalid_arguments;

    batch_normalization_desc_t bd = {};
    bd.primitive_kind = primitive_kind::batch_normalization;
    bd.prop_kind = prop_kind;

    bd.data_desc = *data_desc;
    if (bd.prop_kind == backward_data)
        bd.diff_data_desc = *diff_data_desc;

    dims_t scale_shift_dims = { 2, data_desc->dims[1] };
    mkldnn_memory_desc_init(&bd.data_diff_scaleshift_desc, 2, scale_shift_dims,
            data_desc->data_type, mkldnn_nc);

    bd.batch_norm_epsilon = epsilon;

    bool consistency = true
        && bd.data_desc.ndims == 4;
    if (bd.prop_kind == backward_data)
        consistency = consistency
            && bd.diff_data_desc.ndims == 4
            && array_cmp(bd.diff_data_desc.dims, bd.data_desc.dims, 4);
    if (!consistency) return invalid_arguments;

    *bnrm_desc = bd;
    return success;
}
}

status_t mkldnn_batch_normalization_forward_desc_init(
        batch_normalization_desc_t *bnrm_desc, prop_kind_t prop_kind,
        const memory_desc_t *data_desc, double epsilon) {
    if (!one_of(prop_kind, forward_training, forward_inference))
        return invalid_arguments;
    return bnrm_desc_init(bnrm_desc, prop_kind, data_desc, nullptr, epsilon);
}

status_t mkldnn_batch_normalization_backward_data_desc_init(
        batch_normalization_desc_t *bnrm_desc,
        const memory_desc_t *diff_data_desc, const memory_desc_t *data_desc) {
    return bnrm_desc_init(bnrm_desc, backward_data, data_desc, diff_data_desc,
            0);
}

status_t mkldnn_batch_normalization_backward_desc_init(
        batch_normalization_desc_t *bnrm_desc,
        const memory_desc_t *diff_data_desc, const memory_desc_t *data_desc) {
    return bnrm_desc_init(bnrm_desc, backward, data_desc, diff_data_desc, 0);
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
