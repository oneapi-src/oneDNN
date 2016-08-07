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
    const size_t oi = this->input()[0].output_index;
    const data_i_t *input = reinterpret_cast<const data_i_t*>(
                this->input()[0].primitive->output()[oi]->memory_const());
    data_o_t *output = reinterpret_cast<data_o_t*>(this->output()[0]->memory());

    const memory_desc_wrapper input_d(this->_rpd.input.memory_desc);
    const memory_desc_wrapper output_d(this->_rpd.output.memory_desc);

#   pragma omp parallel for
    for (size_t i = 0; i < input_d.nelems(); ++i) {
        output[output_d.off_l(i)] =
            static_cast<data_o_t>(input[input_d.off_l(i)]);
    }

    return success;
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

    reorder_primitive_desc_t rpd;
    rpd.base.primitive_kind = reorder;
    rpd.base.engine = input->base.engine;
    rpd.base.implementation = reinterpret_cast<const void*>(&implementation);
    rpd.input = *input;
    rpd.output = *output;

    // if (!reorder_primitive_desc_is_ok(rpd)) return invalid_arguments; // ???
    primitive_desc->reorder = rpd;

    return success;
}

namespace {
template <precision_t prec_i, precision_t prec_o>
status_t create(primitive **aprimitive, const primitive_desc_t *primitive_desc,
        const primitive_at_t inputs[], const primitive *outputs[]) {
    assert(primitive_desc->base.primitive_kind == reorder);

    auto &rpd = primitive_desc->reorder;
    // TODO: some checks here.

    *aprimitive = new reference_reorder<prec_i, prec_o>(rpd, inputs, outputs);
    return aprimitive ? success : out_of_memory;
}
}

template <precision_t prec_i, precision_t prec_o>
const primitive_impl reference_reorder<prec_i, prec_o>::implementation = {
    create<prec_i, prec_o>,
};

template class reference_reorder<f32, f32>;

}}}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
