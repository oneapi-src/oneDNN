#include "mkl_dnn_types.h"

#include "c_types_map.hpp"
#include "reference_convolution.hpp"
#include "type_helpers.hpp"

#define CHECK(f) do { \
    status_t status = f; \
    if (status != success) return status; \
} while(0)

namespace mkl_dnn { namespace impl { namespace cpu {

using namespace mkl_dnn::impl::status;
using namespace mkl_dnn::impl::prop_kind;
using namespace mkl_dnn::impl::alg_kind;
using namespace mkl_dnn::impl::precision;
using namespace mkl_dnn::impl::memory_format;
using namespace mkl_dnn::impl::primitive_kind;

template <impl::precision_t prec>
status_t reference_convolution<prec>::execute_forward() {
    auto obtain_ptr = [this](uint32_t idx) {
        const size_t oi = this->input()[idx].output_index;
        return reinterpret_cast<const data_t*>(
                this->input()[idx].primitive->output()[oi]->memory_const());
    };
    const data_t *src = obtain_ptr(0);
    const data_t *weights = obtain_ptr(1);
    const data_t *bias = obtain_ptr(2);
    data_t *dst = reinterpret_cast<data_t*>(this->output()[0]->memory());

    const memory_desc_wrapper
        src_d(this->_cpd.input_primitive_desc.memory_desc),
        weights_d(this->_cpd.weights_primitive_desc.memory_desc),
        bias_d(this->_cpd.bias_primitive_desc.memory_desc),
        dst_d(this->_cpd.output_primitive_desc.memory_desc);

    const bool w_groups = weights_d.tensor().ndims_batch == 1;
    const uint32_t w_idx_base = w_groups ? 1 : 0;
    const uint32_t G = w_groups ? weights_d.dims()[0] : 1;
    const uint32_t MB = src_d.dims()[0];
    const uint32_t OC = weights_d.dims()[w_idx_base + 0];
    const uint32_t OH = dst_d.dims()[2];
    const uint32_t OW = dst_d.dims()[3];

    auto ker = [=](data_t *d, uint32_t g, uint32_t mb, uint32_t oc, uint32_t oh,
            uint32_t ow)
    {
        const uint32_t IH = src_d.dims()[2];
        const uint32_t IW = src_d.dims()[3];
        const uint32_t IC = weights_d.dims()[w_idx_base + 1];
        const uint32_t KH = weights_d.dims()[w_idx_base + 2];
        const uint32_t KW = weights_d.dims()[w_idx_base + 3];
        const uint32_t KSH = this->_cpd.convolution_desc.strides[0];
        const uint32_t KSW = this->_cpd.convolution_desc.strides[1];
        const uint32_t padH = this->_cpd.convolution_desc.padding[0];
        const uint32_t padW = this->_cpd.convolution_desc.padding[1];

        for (uint32_t ic = 0; ic < IC; ++ic) {
            for (uint32_t kh = 0; kh < KH; ++kh) {
                for (uint32_t kw = 0; kw < KW; ++kw) {
                    if (oh*KSH + kh < padH) continue;
                    if (ow*KSW + kw < padW) continue;

                    if (oh*KSH + kh >= IH + padH) continue;
                    if (ow*KSW + kw >= IW + padW) continue;

                    const uint32_t ih = oh * KSH - padH + kh;
                    const uint32_t iw = ow * KSW - padW + kw;

                    if (w_groups) {
                        *d += src[src_d.off(mb, g*IC + ic, ih, iw)] *
                            weights[weights_d.off(g, oc, ic, kh, kw)];
                    } else {
                        *d += src[src_d.off(mb, g*IC + ic, ih, iw)] *
                            weights[weights_d.off(oc, ic, kh, kw)];
                    }
                }
            }
        }
    };

#   pragma omp parallel for collapse(5)
    for (uint32_t g = 0; g < G; ++g) {
        for (uint32_t mb = 0; mb < MB; ++mb) {
            for (uint32_t oc = 0; oc < OC; ++oc) {
                for (uint32_t oh = 0; oh < OH; ++oh) {
                    for (uint32_t ow = 0; ow < OW; ++ow) {
                        data_t *d = &dst[dst_d.off(mb, g*OC + oc, oh, ow)];
                        *d = bias[bias_d.off(g*OC + oc)];
                        ker(d, g, mb, oc, oh, ow);
                    }
                }
            }
        }
    }

    return success;
}

template <impl::precision_t prec>
status_t reference_convolution<prec>::execute_backward_data() {
    return unimplemented;
}

template <impl::precision_t prec>
status_t reference_convolution<prec>::execute_backward_weights() {
    return unimplemented;
}

template <impl::precision_t prec>
status_t reference_convolution<prec>::execute_backward_bias() {
    return unimplemented;
}

template <impl::precision_t prec>
status_t reference_convolution<prec>::primitive_desc_init(
        primitive_desc_t *primitive_desc, const_op_desc_t op_desc,
        const mkl_dnn::impl::engine &engine) {
    auto conv_pd =
        reinterpret_cast<convolution_primitive_desc_t*>(primitive_desc);
    auto conv_d = *static_cast<const convolution_desc_t*>(op_desc);

    if (conv_d.prop_kind != forward)
        return unimplemented;
    if (conv_d.alg_kind != convolution_direct)
        return unimplemented;

    /* memory descriptors check and fill-in */
    if (conv_d.input_desc.format == any)
        CHECK(mkl_dnn_memory_desc_init(&conv_d.input_desc,
                    &conv_d.input_desc.tensor_desc, f32, nchw));
    const bool groups = conv_d.weights_desc.tensor_desc.ndims_batch != 0;
    if (conv_d.weights_desc.format == any)
        CHECK(mkl_dnn_memory_desc_init(&conv_d.weights_desc,
                &conv_d.weights_desc.tensor_desc, f32, groups ? goihw : oihw));
    if (conv_d.bias_desc.format == any)
        CHECK(mkl_dnn_memory_desc_init(&conv_d.bias_desc,
                    &conv_d.bias_desc.tensor_desc, f32, n));
    if (conv_d.output_desc.format == any)
        CHECK(mkl_dnn_memory_desc_init(&conv_d.output_desc,
                    &conv_d.output_desc.tensor_desc, f32, nchw));

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
            .primitive_kind = convolution,
            .engine = &engine,
            .implementation = reinterpret_cast<const void*>(&implementation),
        },
        .convolution_desc = conv_d,
        .input_primitive_desc = input_pd,
        .weights_primitive_desc = weights_pd,
        .bias_primitive_desc = bias_pd,
        .output_primitive_desc = output_pd,
    };

    // if (!convolution_primitive_desc_is_ok(cpd)) return invalid; // ???

    *conv_pd = cpd;

    return success;
}

template <impl::precision_t prec>
status_t reference_convolution<prec>::create(primitive **primitive,
        const primitive_desc_t *primitive_desc,
        const primitive_at_t inputs[], mkl_dnn::impl::primitive *outputs[]) {
    assert(primitive_desc->base.primitive_kind == convolution);

    auto& cpd = primitive_desc->convolution;
    // TODO: some checks here.

    *primitive = new reference_convolution(cpd, inputs, outputs);
    return primitive ? success : out_of_memory;
}

template <impl::precision_t prec>
const primitive_impl reference_convolution<prec>::implementation = {
    .primitive_desc_init = reference_convolution::primitive_desc_init,
    .primitive_create = reference_convolution::create,
};

template class reference_convolution<f32>;

}}}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
