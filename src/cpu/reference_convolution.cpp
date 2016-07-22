#include "mkl_dnn_types.h"
#include "reference_convolution.hpp"

#define CHECK(f) do { \
    status_t status = f; \
    if (status != success) return status; \
} while(0)

namespace mkl_dnn { namespace impl { namespace cpu {

status_t reference_convolution::execute_forward() {
    printf("conv does forward :)\n");
    return success;
}

status_t reference_convolution::execute_backward_data() {
    return unimplemented;
}

status_t reference_convolution::execute_backward_weights() {
    return unimplemented;
}

status_t reference_convolution::execute_backward_bias() {
    return unimplemented;
}

status_t reference_convolution::primitive_desc_init(
        primitive_desc_t *primitive_desc, const_op_desc_t op_desc,
        const dnn_engine &engine) {
    auto conv_pd =
        reinterpret_cast<convolution_primitive_desc_t*>(primitive_desc);
    auto conv_d = *static_cast<const convolution_desc_t*>(op_desc);

    // TODO: f32 ?
    if (conv_d.prop_kind != forward) return unimplemented;
    if (conv_d.alg_kind != convolution_direct) return unimplemented;

    /* memory descriptors check and fill-in */
    if (conv_d.input_desc.format == memory_format_any)
        CHECK(memory_desc_init(&conv_d.input_desc,
                    &conv_d.input_desc.tensor_desc, memory_format_nchw_f32));
    if (conv_d.weights_desc.format == memory_format_any)
        CHECK(memory_desc_init(&conv_d.weights_desc,
                &conv_d.weights_desc.tensor_desc, memory_format_oihw_f32));
    if (conv_d.bias_desc.format == memory_format_any)
        CHECK(memory_desc_init(&conv_d.bias_desc, &conv_d.bias_desc.tensor_desc,
                    memory_format_n_f32));
    if (conv_d.output_desc.format == memory_format_any)
        CHECK(memory_desc_init(&conv_d.output_desc,
                    &conv_d.output_desc.tensor_desc, memory_format_nchw_f32));

    /* memory primitive descriptors check */
    memory_primitive_desc_t input_pd, weights_pd, bias_pd, output_pd;
    CHECK(memory_primitive_desc_init(&input_pd, &conv_d.input_desc, &engine));
    CHECK(memory_primitive_desc_init(&weights_pd, &conv_d.weights_desc,
                &engine));
    CHECK(memory_primitive_desc_init(&bias_pd, &conv_d.bias_desc, &engine));
    CHECK(memory_primitive_desc_init(&output_pd, &conv_d.output_desc, &engine));

    /* final stage */
    convolution_primitive_desc_t cpd = {
        .base = {
            .primitive_kind = primitive_kind_convolution,
            .engine = &engine,
            .implementation = reinterpret_cast<const void*>(&implementation),
        },
        .convolution_desc = conv_d,
        .input_primitive_desc   = input_pd,
        .weights_primitive_desc = weights_pd,
        .bias_primitive_desc    = bias_pd,
        .output_primitive_desc  = output_pd,
    };

    // if (!convolution_primitive_desc_is_ok(cpd)) return invalid; // ???

    *conv_pd = cpd;

    return success;
}

status_t reference_convolution::create(dnn_primitive **primitive,
        const_primitive_desc_t primitive_desc,
        const dnn_primitive_at_t inputs[], const dnn_primitive *outputs[]) {
    auto& cpd =
        *static_cast<const convolution_primitive_desc_t*>(primitive_desc);
    assert(cpd.base.primitive_kind == primitive_kind_convolution);

    // TODO: some checks here.

    *primitive = new reference_convolution(cpd);
    if (primitive) return success;
    return out_of_memory;
}

const primitive_impl reference_convolution::implementation = {
    .primitive_desc_init = reference_convolution::primitive_desc_init,
    .primitive_create = reference_convolution::create,
};

}}}
