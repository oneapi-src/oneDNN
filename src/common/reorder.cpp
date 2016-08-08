#include "nstl.hpp"

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "engine.hpp"
#include "primitive.hpp"
#include "reorder.hpp"

#include "utils.hpp"

using namespace mkl_dnn::impl;
using namespace mkl_dnn::impl::status;
using namespace mkl_dnn::impl::engine_kind;

status_t mkl_dnn_reorder_primitive_desc_init(
        reorder_primitive_desc_t *reorder_primitive_desc,
        const memory_primitive_desc_t *input,
        const memory_primitive_desc_t *output)
{
    if (any_null(reorder_primitive_desc, input, output))
        return invalid_arguments;

    /* XXX: assumptions:
     * 1. reorders between different engines are possible only via cpu
     * 2. reorder from/to non-cpu engine to/from cpu should be implemented in
     *    non-cpu engine */
    const bool possible_engine_mismatch = implication(
            input->base.engine != output->base.engine,
            one_of(input->base.engine->kind(), cpu, cpu_lazy)
            || one_of(output->base.engine->kind(), cpu, cpu_lazy));
    if (!possible_engine_mismatch) return invalid_arguments;
    auto engine = one_of(input->base.engine->kind(), cpu, cpu_lazy)
        ? input->base.engine : output->base.engine;

    for (auto init = engine->get_reorder_inits(); *init; ++init) {
        status_t status = (*init)(
                primitive_desc_t::convert_from_c(reorder_primitive_desc), input,
                output);
        if (status == success)
            return success;
    }

    return unimplemented;
}

status_t mkl_dnn_reorder_create(primitive **reorder,
        const reorder_primitive_desc_t *reorder_primitive_desc,
        const primitive_at_t input, const primitive *output) {
    auto rpd = reinterpret_cast<const mkl_dnn_primitive_desc_t *>(
            reorder_primitive_desc);
    // XXX: must check that shapes of in/out memory match what's in the desc (?)
    const primitive_at_t inputs[] = {input};
    const primitive *outputs[] = {output};
    return mkl_dnn_primitive_create(reorder, rpd, inputs, outputs);
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
