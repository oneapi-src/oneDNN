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
using namespace mkldnn::impl::alg_kind;

status_t mkldnn_lrn_desc_init(lrn_desc_t *lrn_desc, prop_kind_t prop_kind,
        alg_kind_t alg_kind, const memory_desc_t *src_desc,
        const memory_desc_t *dst_desc, double alpha, double beta,
        uint32_t local_size)
{
    bool args_ok = !any_null(lrn_desc, src_desc, dst_desc)
        && one_of(prop_kind, forward_training, forward_scoring, backward_data)
        && one_of(alg_kind, lrn_across_channels, lrn_within_channel);
    if (!args_ok) return invalid_arguments;

    lrn_desc_t cd;
    cd.prop_kind    = prop_kind;
    cd.alg_kind     = alg_kind;
    cd.src_desc     = *src_desc;
    cd.dst_desc     = *dst_desc;
    cd.alpha        = alpha;
    cd.beta         = beta;
    cd.local_size   = local_size;

    status_t status = types::lrn_desc_is_ok(cd);
    if (status == success) *lrn_desc = cd;

    return status;
}

status_t mkldnn_lrn_primitive_desc_init(
        lrn_primitive_desc_t *lrn_primitive_desc,
        const lrn_desc_t *lrn_desc,
        const engine *engine)
{
    if (any_null(lrn_primitive_desc, lrn_desc, engine))
        return invalid_arguments;

    return primitive_desc_init(
            primitive_desc_t::convert_from_c(lrn_primitive_desc),
            *lrn_desc, *engine);
}

status_t mkldnn_lrn_create(primitive **lrn,
        const lrn_primitive_desc_t *lrn_primitive_desc,
        const primitive_at_t src, const primitive_at_t scratch,
        const primitive *dst)
{
    auto *lpd = reinterpret_cast<const mkldnn_primitive_desc_t *>(
            lrn_primitive_desc);
    const primitive_at_t inputs[] = {src, scratch};
    const primitive *outputs[] = {dst};
    return mkldnn_primitive_create(lrn, lpd, inputs, outputs);
}

status_t mkldnn_lrn_get_primitive_desc(const primitive *lrn,
        lrn_primitive_desc_t *lrn_primitive_desc)
{
    if (any_null(lrn, lrn_primitive_desc)
            || lrn->kind() != primitive_kind::lrn)
        return invalid_arguments;
    *lrn_primitive_desc = lrn->primitive_desc().lrn;
    return success;
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
