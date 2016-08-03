#include "mkl_dnn_types.h"

#include "c_types_map.hpp"
#include "reference_reorder.hpp"
#include "type_helpers.hpp"

#define CHECK(f) do { \
    status_t status = f; \
    if (status != success) return status; \
} while(0)

namespace mkl_dnn { namespace impl { namespace cpu {

using namespace mkl_dnn::impl::status;
using namespace mkl_dnn::impl::precision;
using namespace mkl_dnn::impl::memory_format;
using namespace mkl_dnn::impl::primitive_kind;

template <precision_t prec_i, precision_t prec_o>
status_t reference_reorder<prec_i, prec_o>::execute() {
    return unimplemented;
}

template <precision_t prec_i, precision_t prec_o>
status_t reference_reorder<prec_i, prec_o>::reorder_primitive_desc_init(
        primitive_desc_t *primitive_desc, const memory_primitive_desc_t *input,
        const memory_primitive_desc_t *output) {
    bool args_ok = input->memory_desc.format != any
        && input->memory_desc.precision == prec_i
        && output->memory_desc.format != any
        && output->memory_desc.precision == prec_o
        && input->base.engine == output->base.engine;
    if (!args_ok)
        return invalid_arguments;

    reorder_primitive_desc_t rpd = {
        .base = {
            .primitive_kind = reorder,
            .engine = input->base.engine,
            .implementation = reinterpret_cast<const void*>(&implementation),
        },
        .input = *input,
        .output = *output,
    };

    // if (!reorder_primitive_desc_is_ok(cpd)) return invalid; // ???
    primitive_desc->reorder = rpd;

    return success;
}

template <precision_t prec_i, precision_t prec_o>
status_t reference_reorder<prec_i, prec_o>::create(primitive **primitive,
        const primitive_desc_t *primitive_desc,
        const primitive_at_t inputs[], mkl_dnn::impl::primitive *outputs[]) {
    assert(primitive_desc->base.primitive_kind == reorder);

    auto& rpd = primitive_desc->reorder;
    // TODO: some checks here.

    *primitive = new reference_reorder(rpd, inputs, outputs);
    return primitive ? success : out_of_memory;
}

template <precision_t prec_i, precision_t prec_o>
const primitive_impl reference_reorder<prec_i, prec_o>::implementation = {
    .primitive_desc_init = nullptr,
    .primitive_create = reference_reorder::create,
};

template class reference_reorder<f32, f32>;

}}}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
