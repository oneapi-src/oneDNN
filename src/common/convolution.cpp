#include "nstl.hpp"

#include "type_helpers.hpp"
#include "engine.hpp"
#include "primitive.hpp"

#include "convolution.hpp"
#include "utils.hpp"

using namespace mkl_dnn::impl;

status_t mkl_dnn_convolution_desc_init(convolution_desc_t *convolution_desc,
        prop_kind_t prop_kind, alg_kind_t alg_kind,
        const memory_desc_t *input_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *bias_desc, const memory_desc_t *output_desc,
        const dims_t strides, const nd_pos_t padding,
        padding_kind_t padding_kind)
{
    const bool args_ok = !any_null(convolution_desc,
            input_desc, weights_desc, bias_desc, output_desc, strides, padding)
        && one_of(prop_kind, mkl_dnn_forward, mkl_dnn_backward_data,
                mkl_dnn_backward_weights, mkl_dnn_backward_bias)
        && one_of(alg_kind, mkl_dnn_convolution_direct);
    if (!args_ok)
		return mkl_dnn_invalid_arguments;

    convolution_desc_t cd;
    cd.prop_kind = prop_kind;
    cd.alg_kind = alg_kind;
    cd.input_desc = *input_desc;
    cd.weights_desc = *weights_desc;
    cd.bias_desc = *bias_desc;
    cd.output_desc = *output_desc;
    cd.padding_kind = padding_kind;
    const uint32_t ndims_spatial = input_desc->tensor_desc.ndims_spatial;
    array_copy(cd.strides, strides, ndims_spatial);
    array_copy(cd.padding, padding, ndims_spatial);

    status_t status = types::convolution_desc_is_ok(cd);
    if (status == mkl_dnn_success)
		*convolution_desc = cd;

    return status;
}

status_t mkl_dnn_convolution_primitive_desc_init(
        convolution_primitive_desc_t *convolution_primitive_desc,
        const convolution_desc_t *convolution_desc,
		const mkl_dnn_engine *engine)
{
    if (any_null(convolution_primitive_desc, convolution_desc, engine))
		return mkl_dnn_invalid_arguments;

    for (auto init = engine->get_convolution_inits(); *init; ++init) {
        status_t status = (*init)(
                reinterpret_cast<primitive_desc_t*>(convolution_primitive_desc),
                static_cast<const_op_desc_t>(convolution_desc), *engine);
        if (status == mkl_dnn_success)
			return mkl_dnn_success;
    }

    return mkl_dnn_unimplemented;
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0