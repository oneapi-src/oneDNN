#include <assert.h>

#include "c_types_map.hpp"
#include "primitive.hpp"
#include "engine.hpp"

using namespace mkl_dnn::impl;
using namespace mkl_dnn::impl::status;
using namespace mkl_dnn::impl::primitive_kind;

status_t mkl_dnn_primitive_create(primitive **aprimitive,
        const_mkl_dnn_primitive_desc_t primitive_desc,
        const primitive_at_t inputs[], primitive *outputs[]) {
    if (any_null(aprimitive, primitive_desc, inputs, outputs))
        return invalid_arguments;

    auto &pd = *static_cast<const primitive_desc_t*>(primitive_desc);

    if (!pd.base.engine->is_ok())
        return invalid_arguments;

    auto impl = static_cast<const primitive_impl*>(pd.base.implementation);
    return impl->primitive_create(aprimitive, &pd, inputs, outputs);
}

status_t mkl_dnn_primitive_get_primitive_desc(const primitive *primitive,
        mkl_dnn_primitive_desc_t *primitive_desc) {
    auto &pd = *reinterpret_cast<primitive_desc_t*>(primitive_desc);

    switch (pd.base.primitive_kind) {
#   define CASE(x) case x: pd.x = primitive->primitive_desc().x; break
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

status_t mkl_dnn_primitive_get_output(const primitive *aprimitive, size_t index,
        primitive **output) {
    if (index >= aprimitive->output_count())
        return invalid_arguments;

    *output = aprimitive->output()[index];
    return success;
}

status_t mkl_dnn_primitive_destroy(primitive *aprimitive) {
    if (aprimitive != NULL)
        delete aprimitive;
    return success;
}

primitive_at_t mkl_dnn_primitive_at(const primitive *aprimitive,
        size_t output_index) {
    primitive_at_t result = {aprimitive, output_index};
    return result;
}

namespace mkl_dnn {
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
