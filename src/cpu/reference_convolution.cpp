#include "mkl_dnn_types.h"

#include "reference_convolution.hpp"

#define CHECK(f) do { \
    status_t status = f; \
    if (status != mkl_dnn_success) return status; \
} while(0)

namespace mkl_dnn { namespace impl { namespace cpu {

status_t reference_convolution::execute_forward() {
    printf("conv does forward :)\n");
    return mkl_dnn_success;
}

status_t reference_convolution::execute_backward_data() {
    return mkl_dnn_unimplemented;
}

status_t reference_convolution::execute_backward_weights() {
    return mkl_dnn_unimplemented;
}

status_t reference_convolution::execute_backward_bias() {
    return mkl_dnn_unimplemented;
}

status_t reference_convolution::primitive_desc_init(
        primitive_desc_t *primitive_desc, const_op_desc_t op_desc,
        const mkl_dnn_engine &engine) {
    auto conv_pd =
        reinterpret_cast<convolution_primitive_desc_t*>(primitive_desc);
    auto conv_d = *static_cast<const convolution_desc_t*>(op_desc);

    // TODO: f32 ?
    if (conv_d.prop_kind != mkl_dnn_forward)
		return mkl_dnn_unimplemented;
    if (conv_d.alg_kind != mkl_dnn_convolution_direct)
		return mkl_dnn_unimplemented;

    /* memory descriptors check and fill-in */
    if (conv_d.input_desc.format == mkl_dnn_any_f32)
        CHECK(mkl_dnn_memory_desc_init(&conv_d.input_desc,
                    &conv_d.input_desc.tensor_desc, mkl_dnn_nchw_f32));
    if (conv_d.weights_desc.format == mkl_dnn_any_f32)
        CHECK(mkl_dnn_memory_desc_init(&conv_d.weights_desc,
                &conv_d.weights_desc.tensor_desc, mkl_dnn_oihw_f32));
    if (conv_d.bias_desc.format == mkl_dnn_any_f32)
        CHECK(mkl_dnn_memory_desc_init(&conv_d.bias_desc,
					&conv_d.bias_desc.tensor_desc, mkl_dnn_n_f32));
    if (conv_d.output_desc.format == mkl_dnn_any_f32)
        CHECK(mkl_dnn_memory_desc_init(&conv_d.output_desc,
                    &conv_d.output_desc.tensor_desc, mkl_dnn_nchw_f32));

    /* memory primitive descriptors check */
    memory_primitive_desc_t input_pd, weights_pd, bias_pd, output_pd;
    CHECK(mkl_dnn_memory_primitive_desc_init(&input_pd,
				&conv_d.input_desc, &engine));
    CHECK(mkl_dnn_memory_primitive_desc_init(&weights_pd,
				&conv_d.weights_desc, &engine));
    CHECK(mkl_dnn_memory_primitive_desc_init(&bias_pd,
				&conv_d.bias_desc, &engine));
    CHECK(mkl_dnn_memory_primitive_desc_init(&output_pd,
				&conv_d.output_desc, &engine));

    /* final stage */
    convolution_primitive_desc_t cpd = {
        .base = {
            .primitive_kind = mkl_dnn_convolution,
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

    return mkl_dnn_success;
}

status_t reference_convolution::create(primitive **primitive,
        const_primitive_desc_t primitive_desc,
        const primitive_at_t inputs[], mkl_dnn_primitive *outputs[]) {
    auto& cpd =
        *static_cast<const convolution_primitive_desc_t*>(primitive_desc);
    assert(cpd.base.primitive_kind == mkl_dnn_convolution);

    // TODO: some checks here.

    *primitive = new reference_convolution(cpd);
    if (primitive)
		return mkl_dnn_success;
    return mkl_dnn_out_of_memory;
}

const primitive_impl reference_convolution::implementation = {
    .primitive_desc_init = reference_convolution::primitive_desc_init,
    .primitive_create = reference_convolution::create,
};

}}}
