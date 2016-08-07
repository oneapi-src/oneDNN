#include "mkl_dnn_types.h"

#include "c_types_map.hpp"
#include "reference_relu.hpp"
#include "type_helpers.hpp"

// XXX: move this to utils.hpp / somewhere else
#define CHECK(f) do { \
    status_t status = f; \
    if (status != success) return status; \
} while(0)

namespace mkl_dnn { namespace impl { namespace cpu {

using namespace mkl_dnn::impl::status;
using namespace mkl_dnn::impl::prop_kind;
using namespace mkl_dnn::impl::precision;
using namespace mkl_dnn::impl::memory_format;
using namespace mkl_dnn::impl::primitive_kind;

template <impl::precision_t prec>
status_t reference_relu<prec>::execute_forward() {
    auto obtain_ptr = [this](uint32_t idx) {
        const size_t oi = this->input()[idx].output_index;
        return reinterpret_cast<const data_t*>(
                this->input()[idx].primitive->output()[oi]->memory_const());
    };
    const data_t *src = obtain_ptr(0);
    data_t *dst = reinterpret_cast<data_t*>(this->output()[0]->memory());

    const memory_desc_wrapper src_d(this->_rpd.src_primitive_desc.memory_desc);
    const uint32_t N = src_d.dims()[0];
    const uint32_t C = src_d.dims()[1];
    const uint32_t H = src_d.dims()[2];
    const uint32_t W = src_d.dims()[3];
    const double negative_slope = this->_rpd.relu_desc.negative_slope;

#   pragma omp parallel for collapse(4)
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t c = 0; c < C; ++c) {
            for (uint32_t h = 0; h < H; ++h) {
                for (uint32_t w = 0; w < W; ++w) {
                    data_t s = src[src_d.off(n, c, h, w)];
                    data_t &d = dst[src_d.off(n, c, h, w)];
                    d = (s > 0) ? s : s * negative_slope; // alpha, beta, etc ?
                }
            }
        }
    }

    return success;
}

template <impl::precision_t prec>
status_t reference_relu<prec>::execute_backward_data() {
    return unimplemented;
}

template <impl::precision_t prec>
status_t reference_relu<prec>::primitive_desc_init(
        primitive_desc_t *primitive_desc, const op_desc_t &op_desc,
        const mkl_dnn::impl::engine &engine) {
    if (op_desc._kind != primitive_kind::relu)
        return invalid_arguments;
    auto relu_d = op_desc.relu;

    // TODO: f32 ?
    if (relu_d.prop_kind != forward)
        return unimplemented;

    /* memory descriptors check and fill-in */
    /* XXX: code duplication */
    if (relu_d.src_desc.format == any)
        CHECK(mkl_dnn_memory_desc_init(&relu_d.src_desc,
                    &relu_d.src_desc.tensor_desc, f32, nchw));
    if (relu_d.dst_desc.format == any)
        CHECK(mkl_dnn_memory_desc_init(&relu_d.dst_desc,
                    &relu_d.dst_desc.tensor_desc, f32, nchw));

    /* memory primitive descriptors check */
    /* XXX: code duplication */
    memory_primitive_desc_t src_pd, dst_pd;
    CHECK(mkl_dnn_memory_primitive_desc_init(&src_pd,
                &relu_d.src_desc, &engine));
    CHECK(mkl_dnn_memory_primitive_desc_init(&dst_pd,
                &relu_d.dst_desc, &engine));

    /* final stage */
    relu_primitive_desc_t rpd;
    rpd.base.primitive_kind = relu;
    rpd.base.engine = &engine;
    rpd.base.implementation = reinterpret_cast<const void*>(&implementation);
    rpd.relu_desc = relu_d;
    rpd.src_primitive_desc   = src_pd;
    rpd.dst_primitive_desc  = dst_pd;

    primitive_desc->relu = rpd;

    return success;
}

namespace {
template <impl::precision_t prec>
status_t create(primitive **aprimitive, const primitive_desc_t *primitive_desc,
        const primitive_at_t inputs[], const primitive *outputs[]) {
    assert(aprimitive);
    assert(inputs);
    assert(outputs);
    assert(primitive_desc);
    assert(primitive_desc->base.primitive_kind == relu);

    auto& rpd = primitive_desc->relu;
    // TODO: some checks here (asserts: all the error checks must have been
    // done in the upper layers)

    *aprimitive = new reference_relu<prec>(rpd, inputs, outputs);
    return aprimitive ? success : out_of_memory;
}
}

template <impl::precision_t prec>
const primitive_impl reference_relu<prec>::implementation = { create<prec> };

template class reference_relu<f32>;

}}}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
