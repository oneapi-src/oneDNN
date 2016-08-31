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

#include "nstl.hpp"

#include "type_helpers.hpp"
#include "engine.hpp"
#include "primitive.hpp"

#include "utils.hpp"

using namespace mkldnn::impl;
using namespace mkldnn::impl::status;
using namespace mkldnn::impl::prop_kind;

status_t mkldnn_batch_normalization_desc_init(batch_normalization_desc_t *bnd,
        prop_kind_t prop_kind, const memory_desc_t *src_desc,
        const memory_desc_t *dst_desc, const memory_desc_t *scaleshift_desc,
        double epsilon)
{
    bool args_ok = !any_null(bnd, src_desc, dst_desc)
        && one_of(prop_kind, forward_training, forward_scoring, backward_data);
    if (!args_ok) return invalid_arguments;

    batch_normalization_desc_t cd;
    cd.prop_kind = prop_kind;
    cd.src_desc = *src_desc;
    cd.dst_desc = *dst_desc;
    cd.scaleshift_desc = *scaleshift_desc;
    cd.epsilon = epsilon;

    status_t status = types::batch_normalization_desc_is_ok(cd);
    if (status == success) *bnd = cd;

    return status;
}

status_t mkldnn_batch_normalization_primitive_desc_init(
        batch_normalization_primitive_desc_t *bnpd,
        const batch_normalization_desc_t *bnd,
        const engine *engine)
{
    if (any_null(bnpd, bnd, engine))
        return invalid_arguments;

    return primitive_desc_init(
            primitive_desc_t::convert_from_c(bnpd),
            *bnd, *engine);
}

status_t mkldnn_batch_normalization_create(primitive **batch_normalization,
        const batch_normalization_primitive_desc_t *bnpd,
        const primitive_at_t src, const primitive *dst,
        const primitive_at_t scaleshift, const primitive_at_t workspace)
{
    auto *pd = reinterpret_cast<const mkldnn_primitive_desc_t *>(bnpd);
    const primitive_at_t inputs[] = {src, scaleshift, workspace};
    const primitive *outputs[] = {dst};

    return mkldnn_primitive_create(batch_normalization, pd, inputs, outputs);
}

status_t mkldnn_batch_normalization_get_primitive_desc(
        const primitive *batch_normalization,
        batch_normalization_primitive_desc_t *bnpd)
{
    if (any_null(batch_normalization, bnpd)
            || batch_normalization->kind() != primitive_kind::batch_normalization)
        return invalid_arguments;
    *bnpd = batch_normalization->primitive_desc().batch_normalization;
    return success;
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
