#include "nstl.hpp"

#include "type_helpers.hpp"
#include "engine.hpp"
#include "primitive.hpp"

#include "pooling.hpp"
#include "utils.hpp"

using namespace mkl_dnn::impl;
using namespace mkl_dnn::impl::status;
using namespace mkl_dnn::impl::prop_kind;
using namespace mkl_dnn::impl::alg_kind;

status_t mkl_dnn_pooling_desc_init(pooling_desc_t *pooling_desc,
        prop_kind_t prop_kind, alg_kind_t alg_kind,
        const memory_desc_t *input_desc, const memory_desc_t *indices_desc,
        const memory_desc_t *output_desc,
        const dims_t strides, const dims_t kernel, const nd_offset_t padding,
        padding_kind_t padding_kind)
{
    const bool args_ok = !any_null(pooling_desc,
            input_desc, indices_desc, output_desc, strides, padding)
        && one_of(prop_kind, forward, backward_data)
        && one_of(alg_kind, pooling_max);
    if (!args_ok)
        return invalid_arguments;

    pooling_desc_t cd;
    cd.prop_kind = prop_kind;
    cd.alg_kind = alg_kind;
    cd.input_desc = *input_desc;
    cd.indices_desc = *indices_desc;
    cd.output_desc = *output_desc;
    cd.padding_kind = padding_kind;
    const uint32_t ndims_spatial = input_desc->tensor_desc.ndims_spatial;
    array_copy(cd.strides, strides, ndims_spatial);
    array_copy(cd.kernel, kernel, ndims_spatial);
    array_copy(cd.padding, padding, ndims_spatial);

    status_t status = types::pooling_desc_is_ok(cd);
    if (status == success)
        *pooling_desc = cd;

    return status;
}

status_t mkl_dnn_pooling_primitive_desc_init(
        pooling_primitive_desc_t *pooling_primitive_desc,
        const pooling_desc_t *pooling_desc,
        const engine *engine)
{
    if (any_null(pooling_primitive_desc, pooling_desc, engine))
        return invalid_arguments;

    for (auto init = engine->get_pooling_inits(); *init; ++init) {
        status_t status = (*init)(
                reinterpret_cast<primitive_desc_t*>(pooling_primitive_desc),
                static_cast<const_op_desc_t>(pooling_desc), *engine);
        if (status == success)
            return success;
    }

    return unimplemented;
}

mkl_dnn_status_t mkl_dnn_pooling_create(mkl_dnn_primitive_t *pooling,
        const mkl_dnn_pooling_primitive_desc_t *pooling_primitive_desc,
        const mkl_dnn_primitive_at_t input, const mkl_dnn_primitive_at_t indices,
        mkl_dnn_primitive_t output) {
    const mkl_dnn_primitive_desc_t *cpd =
        reinterpret_cast<const mkl_dnn_primitive_desc_t *>(
                pooling_primitive_desc);
    const mkl_dnn_primitive_at_t inputs[] = { input, indices };
    mkl_dnn_primitive_t outputs[] = { output };

    return mkl_dnn_primitive_create(pooling, cpd, inputs, outputs);
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
