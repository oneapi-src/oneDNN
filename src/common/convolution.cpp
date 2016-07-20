#include "nstl.hpp"

#include "type_helpers.hpp"
#include "engine.hpp"
#include "primitive.hpp"

#include "convolution.hpp"
#include "utils.hpp"

status_t convolution_desc_init(convolution_desc_t *convolution_desc,
        prop_kind_t prop_kind, alg_kind_t alg_kind,
        const memory_desc_t *input_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *bias_desc, const memory_desc_t *output_desc,
        const dims_t strides, const nd_pos_t padding,
        padding_kind_t padding_kind)
{
    const bool args_ok = !mkl_dnn::impl::any_null(convolution_desc,
            input_desc, weights_desc, bias_desc, output_desc, strides, padding)
        && mkl_dnn::impl::one_of(prop_kind, forward, backward_data,
                backward_weights, backward_bias)
        && mkl_dnn::impl::one_of(alg_kind, convolution_direct);
    if (!args_ok) return invalid_arguments;

    convolution_desc_t cd;
    cd.prop_kind = prop_kind;
    cd.alg_kind = alg_kind;
    cd.input_desc = *input_desc;
    cd.weights_desc = *weights_desc;
    cd.bias_desc = *bias_desc;
    cd.output_desc = *output_desc;
    cd.padding_kind = padding_kind;
    const uint32_t ndims_spatial = input_desc->tensor_desc.ndims_spatial;
    mkl_dnn::impl::array_copy(cd.strides, strides, ndims_spatial);
    mkl_dnn::impl::array_copy(cd.padding, padding, ndims_spatial);

    status_t status = mkl_dnn::impl::types::convolution_desc_is_ok(cd);
    if (status == success) *convolution_desc = cd;

    return status;
}

status_t convolution_primitive_desc_init(
        convolution_primitive_desc_t *convolution_primitive_desc,
        const convolution_desc_t *convolution_desc, const_engine_t engine)
{
    if (mkl_dnn::impl::any_null(convolution_primitive_desc, convolution_desc,
                engine)) return invalid_arguments;

    auto e = static_cast<const mkl_dnn::impl::engine*>(engine);

    for (auto i = e->get_convolution_inits(); *i; ++i) {
        using mkl_dnn::impl::const_op_desc_t;
        status_t status = (*i)(
                reinterpret_cast<primitive_desc_t*>(convolution_primitive_desc),
                static_cast<const_op_desc_t>(convolution_desc), *e);
        if (status == success) return success;
    }

    return unimplemented;
}
