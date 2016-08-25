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

#include "c_types_map.hpp"
#include "primitive.hpp"
#include "engine.hpp"
#include "type_helpers.hpp"

using namespace mkldnn::impl;
using namespace mkldnn::impl::status;
using namespace mkldnn::impl::primitive_kind;

status_t mkldnn_primitive_create(primitive **aprimitive,
        const_mkldnn_primitive_desc_t primitive_desc,
        const primitive_at_t inputs[], const primitive *outputs[]) {
    if (any_null(aprimitive, primitive_desc, inputs, outputs))
        return invalid_arguments;

    auto &pd = *static_cast<const primitive_desc_t*>(primitive_desc);

    /* FIXME: singularity :( */
    if (pd.base.primitive_kind == memory
            && memory_desc_wrapper(pd.memory.memory_desc).is_zero()) {
        *aprimitive = nullptr;
        return success;
    }

    if (!pd.base.engine->is_ok())
        return invalid_arguments;

    auto impl = static_cast<const primitive_impl*>(pd.base.implementation);
    return impl->primitive_create(aprimitive, &pd, inputs, outputs);
}

status_t mkldnn_primitive_get_primitive_desc(const primitive *primitive,
        mkldnn_primitive_desc_t *primitive_desc) {
    if (any_null(primitive, primitive_desc))
        return invalid_arguments;

    auto &pd = *reinterpret_cast<primitive_desc_t*>(primitive_desc);

    switch (pd.base.primitive_kind) {
#   define CASE(x) case x: pd.x = primitive->primitive_desc().x; break
    CASE(relu);
    CASE(lrn);
    CASE(memory);
    CASE(reorder);
    CASE(pooling);
    CASE(convolution);
    CASE(inner_product);
#   undef CASE
    default: assert(!"unregistered primitive_desc");
    }

    return success;
}

mkldnn_status_t mkldnn_primitive_get_input_at(const primitive *aprimitive,
        size_t index, primitive_at_t *input)
{
    if (index >= aprimitive->input_count())
        return invalid_arguments;

    *input = aprimitive->input()[index];
    return success;
}

status_t mkldnn_primitive_get_output(const primitive *aprimitive, size_t index,
        const primitive **output) {
    if (index >= aprimitive->output_count())
        return invalid_arguments;

    *output = aprimitive->output()[index];
    return success;
}

status_t mkldnn_primitive_destroy(primitive *aprimitive) {
    if (aprimitive != NULL)
        delete aprimitive;
    return success;
}

primitive_at_t mkldnn_primitive_at(const primitive *aprimitive,
        size_t output_index) {
    primitive_at_t result = {aprimitive, output_index};
    return result;
}

namespace mkldnn {
namespace impl {

status_t primitive_desc_init(primitive_desc_t *primitive_desc,
        const op_desc_t &op_desc, const engine &aengine) {
    if (op_desc._kind == primitive_kind::reorder)
        return invalid_arguments;
    for (auto init = aengine.get_primitive_inits(); *init; ++init) {
        status_t status = (*init)(primitive_desc, op_desc, aengine);
        if (status == success)
            return success;
    }

    return status::unimplemented;
}

}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
