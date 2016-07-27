#include "primitive.hpp"
#include "engine.hpp"

using namespace mkl_dnn::impl;

mkl_dnn_status_t mkl_dnn_primitive_create(mkl_dnn_primitive_t *primitive,
        const_mkl_dnn_primitive_desc_t primitive_desc,
        const mkl_dnn_primitive_at_t *inputs,
		const_mkl_dnn_primitive_t *outputs) {
    if (any_null(primitive, primitive_desc, inputs, outputs))
        return mkl_dnn_invalid_arguments;

    auto base_pd = static_cast<const primitive_base_desc_t*>(primitive_desc);
    if (!base_pd->engine->is_ok())
		return mkl_dnn_invalid_arguments;

    typedef const primitive_impl *impl;
    return reinterpret_cast<impl>(base_pd->implementation)->primitive_create(
            primitive, primitive_desc, inputs, outputs);
}

status_t mkl_dnn_primitive_destroy(mkl_dnn_primitive_t primitive) {
    if (primitive != NULL)
		delete primitive;
    return mkl_dnn_success;
}

mkl_dnn_primitive_at_t mkl_dnn_primitive_at(const_mkl_dnn_primitive_t primitive,
        size_t output_index) {
    mkl_dnn_primitive_at_t result = {primitive, output_index};
    return result;
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0