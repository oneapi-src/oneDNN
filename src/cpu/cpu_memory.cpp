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

#include "mkldnn_types.h"

#include "c_types_map.hpp"
#include "cpu/cpu_memory.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::precision;
using namespace mkldnn::impl::memory_format;

status_t cpu_memory::memory_desc_init(primitive_desc_t *primitive_desc,
        const op_desc_t &op_desc, const mkldnn::impl::engine &aengine) {
    if (op_desc._kind != primitive_kind::memory)
        return invalid_arguments;
    auto memory_desc = op_desc.memory;

    memory_primitive_desc_t mpd;
    mpd.base.primitive_kind = primitive_kind::memory;
    mpd.base.engine = &aengine;
    mpd.base.implementation
        = reinterpret_cast<const void*>(&memory_implementation);
    mpd.memory_desc = memory_desc;

    // if (!memory_primitive_desc_is_ok(mpd)) return invalid_arguments; // ???
    primitive_desc->memory = mpd;

    return success;
}

namespace {
status_t memory_create(primitive **aprimitive,
        const primitive_desc_t *primitive_desc, const primitive_at_t inputs[],
        const primitive *outputs[]) {
    if (any_null(aprimitive, primitive_desc, inputs, outputs))
        return invalid_arguments;

    assert(primitive_desc->base.primitive_kind == primitive_kind::memory);
    assert(inputs[0].primitive == outputs[0]);

    char* ptr = const_cast<char *>(reinterpret_cast<const char*>(outputs[0]));
    *aprimitive = new cpu_memory(primitive_desc->memory, ptr);
    return *aprimitive ? success : out_of_memory;
}
}

const primitive_impl cpu_memory::memory_implementation = { memory_create };

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
