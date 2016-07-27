#include "c_types_map.hpp"
#include "primitive.hpp"
#include "engine.hpp"

using namespace mkl_dnn::impl;
using namespace mkl_dnn::impl::status;

status_t mkl_dnn_primitive_create(primitive **aprimitive,
        const_primitive_desc_t primitive_desc,
        const primitive_at_t *inputs, primitive **outputs) {
    if (any_null(aprimitive, primitive_desc, inputs, outputs))
        return invalid_arguments;

    auto base_pd = static_cast<const primitive_base_desc_t*>(primitive_desc);
    if (!base_pd->engine->is_ok())
        return invalid_arguments;

    typedef const primitive_impl *impl;
    return reinterpret_cast<impl>(base_pd->implementation)->primitive_create(
            aprimitive, primitive_desc, inputs, outputs);
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

// vim: et ts=4 sw=4 cindent cino^=l0,\:0
