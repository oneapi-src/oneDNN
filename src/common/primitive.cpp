#include "primitive.hpp"
#include "engine.hpp"

status_t primitive_create(primitive_t *primitive,
        const_primitive_desc_t primitive_desc, const_primitive_t *inputs,
        const_primitive_t *outputs) {
    if (mkl_dnn::impl::any_null(primitive, primitive_desc, inputs, outputs))
        return invalid_arguments;

    auto base_pd = static_cast<const primitive_base_desc_t*>(primitive_desc);
    if (!base_pd->engine->is_ok()) return invalid_arguments;

    typedef const mkl_dnn::impl::primitive_impl *impl;
    return reinterpret_cast<impl>(base_pd->implementation)->primitive_create(
            reinterpret_cast<mkl_dnn::impl::primitive**>(primitive),
            primitive_desc,
            reinterpret_cast<const mkl_dnn::impl::primitive**>(inputs),
            reinterpret_cast<const mkl_dnn::impl::primitive**>(outputs));
}

status_t primitive_destroy(primitive_t primitive)
{
    if (primitive != NULL)
        delete static_cast<mkl_dnn::impl::primitive*>(primitive);
    return success;
}
