#include <assert.h>

#include "mkl_dnn_types.h"

#include "c_types_map.hpp"
#include "cpu/cpu_memory.hpp"

namespace mkl_dnn {
namespace impl {
namespace cpu {

using namespace mkl_dnn::impl::status;
using namespace mkl_dnn::impl::precision;
using namespace mkl_dnn::impl::memory_format;

status_t cpu_memory::memory_desc_init(primitive_desc_t *primitive_desc,
        const op_desc_t &op_desc, const mkl_dnn::impl::engine &aengine) {
    if (op_desc._kind != primitive_kind::memory)
        return invalid_arguments;
    auto memory_desc = op_desc.memory;

    memory_primitive_desc_t mpd = {
        .base = {
            .primitive_kind = primitive_kind::memory,
            .engine = &aengine,
            .implementation =
                reinterpret_cast<const void*>(&memory_implementation),
        },
        .memory_desc = memory_desc
    };

    // if (!memory_primitive_desc_is_ok(mpd)) return invalid_arguments; // ???
    primitive_desc->memory = mpd;

    return success;
}

namespace {
status_t memory_create(primitive **aprimitive,
        const primitive_desc_t *primitive_desc, const primitive_at_t inputs[],
        const primitive *outputs[]) {
    assert(primitive_desc->base.primitive_kind == primitive_kind::memory);
    assert(inputs[0].primitive == outputs[0]);

    char* ptr = const_cast<char *>(reinterpret_cast<const char*>(outputs[0]));
    *aprimitive = new cpu_memory(primitive_desc->memory, ptr);
    return aprimitive ? success : out_of_memory;
}
}

const primitive_impl cpu_memory::memory_implementation = {
    .primitive_create = memory_create
};

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
