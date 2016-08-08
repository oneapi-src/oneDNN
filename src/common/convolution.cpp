#include "nstl.hpp"

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "engine.hpp"
#include "primitive.hpp"

#include "utils.hpp"

using namespace mkl_dnn::impl;
using namespace mkl_dnn::impl::status;
using namespace mkl_dnn::impl::prop_kind;
using namespace mkl_dnn::impl::alg_kind;

status_t mkl_dnn_convolution_desc_init(convolution_desc_t *convolution_desc,
        prop_kind_t prop_kind, alg_kind_t alg_kind,
        const memory_desc_t *src_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *bias_desc, const memory_desc_t *dst_desc,
        const dims_t strides, const nd_offset_t padding,
        padding_kind_t padding_kind)
{
    const bool args_ok = !any_null(convolution_desc,
            src_desc, weights_desc, bias_desc, dst_desc, strides, padding)
        && one_of(prop_kind, forward, backward_data,
                backward_weights, backward_bias)
        && one_of(alg_kind, convolution_direct);
    if (!args_ok)
        return invalid_arguments;

    convolution_desc_t cd;
    cd.prop_kind = prop_kind;
    cd.alg_kind = alg_kind;
    cd.src_desc = *src_desc;
    cd.weights_desc = *weights_desc;
    cd.bias_desc = *bias_desc;
    cd.dst_desc = *dst_desc;
    cd.padding_kind = padding_kind;
    const uint32_t ndims_spatial = src_desc->tensor_desc.ndims_spatial;
    array_copy(cd.strides, strides, ndims_spatial);
    array_copy(cd.padding, padding, ndims_spatial);

    status_t status = types::convolution_desc_is_ok(cd);
    if (status == success)
        *convolution_desc = cd;

    return status;
}

status_t mkl_dnn_convolution_primitive_desc_init(
        convolution_primitive_desc_t *convolution_primitive_desc,
        const convolution_desc_t *convolution_desc,
        const engine *engine)
{
    if (any_null(convolution_primitive_desc, convolution_desc, engine))
        return invalid_arguments;

    return primitive_desc_init(
            primitive_desc_t::convert_from_c(convolution_primitive_desc),
            *convolution_desc, *engine);
}

mkl_dnn_status_t mkl_dnn_convolution_create(mkl_dnn_primitive_t *convolution,
        const mkl_dnn_convolution_primitive_desc_t *convolution_primitive_desc,
        const mkl_dnn_primitive_at_t src, const mkl_dnn_primitive_at_t weights,
        const mkl_dnn_primitive_at_t bias, mkl_dnn_primitive_t dst) {
    const mkl_dnn_primitive_desc_t *cpd =
        reinterpret_cast<const mkl_dnn_primitive_desc_t *>(
                convolution_primitive_desc);
    // XXX: must check that shapes of in/out memory match what's in the desc (?)
    const mkl_dnn_primitive_at_t inputs[] = {src, weights, bias};
    mkl_dnn_primitive_t outputs[] = {dst};
    return mkl_dnn_primitive_create(convolution, cpd, inputs, outputs);
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
