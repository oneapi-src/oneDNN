#include "nstl.hpp"

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "engine.hpp"
#include "primitive.hpp"

#include "utils.hpp"

using namespace mkl_dnn::impl;
using namespace mkl_dnn::impl::status;
using namespace mkl_dnn::impl::prop_kind;

status_t mkl_dnn_relu_desc_init(relu_desc_t *relu_desc,
        prop_kind_t prop_kind, double negative_slope,
        const memory_desc_t *src_desc, const memory_desc_t *dst_desc)
{
    if (any_null(relu_desc, src_desc, dst_desc)
            || !one_of(prop_kind,
                forward, backward_data, backward_weights, backward_bias))
        return invalid_arguments;

    relu_desc_t rd;
    rd.prop_kind = prop_kind;
    rd.negative_slope = negative_slope;
    rd.src_desc = *src_desc;
    rd.dst_desc = *dst_desc;

    // XXX: rsdubtso: I do not like this...
    status_t status = types::relu_desc_is_ok(rd);
    if (status == success)
        *relu_desc = rd;

    return status;
}

status_t mkl_dnn_relu_primitive_desc_init(
        relu_primitive_desc_t *relu_primitive_desc,
        const relu_desc_t *relu_desc, const engine *engine)
{
    if (any_null(relu_primitive_desc, relu_desc, engine))
        return invalid_arguments;

    return primitive_desc_init(
            primitive_desc_t::convert_from_c(relu_primitive_desc),
            *relu_desc, *engine);
}

mkl_dnn_status_t mkl_dnn_relu_create(mkl_dnn_primitive_t *relu,
        const mkl_dnn_relu_primitive_desc_t *relu_primitive_desc,
        const mkl_dnn_primitive_at_t src, mkl_dnn_primitive_t dst)
{
    auto *rpd = reinterpret_cast<const mkl_dnn_primitive_desc_t *>(
            relu_primitive_desc);
    // XXX: must check that shapes of in/out memory match what's in the desc (?)
    const mkl_dnn_primitive_at_t inputs[] = {src};
    mkl_dnn_primitive_t outputs[] = {dst};
    return mkl_dnn_primitive_create(relu, rpd, inputs, outputs);
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
