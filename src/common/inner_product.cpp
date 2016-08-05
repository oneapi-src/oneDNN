#include "nstl.hpp"

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "engine.hpp"
#include "primitive.hpp"

#include "inner_product.hpp"
#include "utils.hpp"

using namespace mkl_dnn::impl;
using namespace mkl_dnn::impl::status;
using namespace mkl_dnn::impl::prop_kind;

status_t mkl_dnn_inner_product_desc_init(
        inner_product_desc_t *inner_product_desc,
        prop_kind_t prop_kind, const memory_desc_t *src_desc,
        const memory_desc_t *weights_desc, const memory_desc_t *dst_desc)
{
    const bool args_ok = !any_null(inner_product_desc,
            src_desc, weights_desc, dst_desc)
        && one_of(prop_kind, forward, backward_data,
                backward_weights);
    if (!args_ok)
        return invalid_arguments;

    inner_product_desc_t ipd;
    ipd.prop_kind = prop_kind;
    ipd.src_desc = *src_desc;
    ipd.weights_desc = *weights_desc;
    ipd.dst_desc = *dst_desc;

    status_t status = types::inner_product_desc_is_ok(ipd);
    if (status == success)
        *inner_product_desc = ipd;

    return status;
}

status_t mkl_dnn_inner_product_primitive_desc_init(
        inner_product_primitive_desc_t *inner_product_primitive_desc,
        const inner_product_desc_t *inner_product_desc,
        const engine *engine)
{
    if (any_null(inner_product_primitive_desc, inner_product_desc, engine))
        return invalid_arguments;

    return primitive_desc_init(
            primitive_desc_t::convert_from_c(inner_product_primitive_desc),
            *inner_product_desc, *engine);
}

mkl_dnn_status_t mkl_dnn_inner_product_create(
        mkl_dnn_primitive_t *inner_product,
        const mkl_dnn_inner_product_primitive_desc_t *inner_product_primitive_desc,
        const mkl_dnn_primitive_at_t src, const mkl_dnn_primitive_at_t weights,
        mkl_dnn_primitive_t dst) {
    const mkl_dnn_primitive_desc_t *ippd =
        reinterpret_cast<const mkl_dnn_primitive_desc_t *>(
                inner_product_primitive_desc);
    // XXX: must check that shapes of in/out memory match what's in the desc (?)
    const mkl_dnn_primitive_at_t inputs[] = {src, weights};
    mkl_dnn_primitive_t outputs[] = {dst};
    return mkl_dnn_primitive_create(inner_product, ippd, inputs, outputs);
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
