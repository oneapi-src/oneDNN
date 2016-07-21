#include "primitive.hpp"
#include "engine.hpp"

status_t primitive_create(dnn_primitive_t *primitive,
        const_primitive_desc_t primitive_desc, const_dnn_primitive_t *inputs,
        const_dnn_primitive_t *outputs) {
    if (mkl_dnn::impl::any_null(primitive, primitive_desc, inputs, outputs))
        return invalid_arguments;

    auto base_pd = static_cast<const primitive_base_desc_t*>(primitive_desc);
    if (!base_pd->engine->is_ok()) return invalid_arguments;

    typedef const mkl_dnn::impl::primitive_impl *impl;
    return reinterpret_cast<impl>(base_pd->implementation)->primitive_create(
            primitive, primitive_desc, inputs, outputs);
}

status_t primitive_destroy(dnn_primitive_t primitive) {
    if (primitive != NULL) delete primitive;
    return success;
}
