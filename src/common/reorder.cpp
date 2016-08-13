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

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "engine.hpp"
#include "primitive.hpp"
#include "reorder.hpp"

#include "utils.hpp"

using namespace mkldnn::impl;
using namespace mkldnn::impl::status;
using namespace mkldnn::impl::engine_kind;

status_t mkldnn_reorder_primitive_desc_init(
        reorder_primitive_desc_t *reorder_primitive_desc,
        const memory_primitive_desc_t *input,
        const memory_primitive_desc_t *output)
{
    if (any_null(reorder_primitive_desc, input, output))
        return invalid_arguments;

    /* XXX: assumptions:
     * 1. reorders between different engines are possible only via cpu
     * 2. reorder from/to non-cpu engine to/from cpu should be implemented in
     *    non-cpu engine */
    const bool possible_engine_mismatch = implication(
            input->base.engine != output->base.engine,
            one_of(input->base.engine->kind(), cpu, cpu_lazy)
            || one_of(output->base.engine->kind(), cpu, cpu_lazy));
    if (!possible_engine_mismatch) return invalid_arguments;
    auto engine = one_of(input->base.engine->kind(), cpu, cpu_lazy)
        ? input->base.engine : output->base.engine;

    for (auto init = engine->get_reorder_inits(); *init; ++init) {
        status_t status = (*init)(
                primitive_desc_t::convert_from_c(reorder_primitive_desc), input,
                output);
        if (status == success)
            return success;
    }

    return unimplemented;
}

status_t mkldnn_reorder_create(primitive **reorder,
        const reorder_primitive_desc_t *reorder_primitive_desc,
        const primitive_at_t input, const primitive *output) {
    auto rpd = reinterpret_cast<const mkldnn_primitive_desc_t *>(
            reorder_primitive_desc);
    // XXX: must check that shapes of in/out memory match what's in the desc (?)
    const primitive_at_t inputs[] = {input};
    const primitive *outputs[] = {output};
    return mkldnn_primitive_create(reorder, rpd, inputs, outputs);
}

status_t mkldnn_reorder_get_primitive_desc(const primitive *reorder,
        reorder_primitive_desc_t *reorder_primitive_desc)
{
    if (any_null(reorder, reorder_primitive_desc)
            || reorder->kind() != primitive_kind::reorder)
        return invalid_arguments;
    *reorder_primitive_desc = reorder->primitive_desc().reorder;
    return success;
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
