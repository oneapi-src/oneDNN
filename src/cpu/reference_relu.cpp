/*******************************************************************************
* Copyright 2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "mkldnn_types.h"

#include "c_types_map.hpp"
#include "reference_relu.hpp"
#include "type_helpers.hpp"

namespace mkldnn { namespace impl { namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::precision;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::primitive_kind;

template <impl::precision_t prec>
status_t reference_relu<prec>::execute_forward_generic() {
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

#   pragma omp parallel for collapse(4) schedule(static)
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
status_t reference_relu<prec>::execute_forward_dense() {
    auto obtain_ptr = [this](uint32_t idx) {
        const size_t oi = this->input()[idx].output_index;
        return reinterpret_cast<const data_t*>(
                this->input()[idx].primitive->output()[oi]->memory_const());
    };
    const data_t *src = obtain_ptr(0);
    data_t *dst = reinterpret_cast<data_t*>(this->output()[0]->memory());

    const double negative_slope = _rpd.relu_desc.negative_slope;
    const memory_desc_wrapper src_d(_rpd.src_primitive_desc.memory_desc);
    const memory_desc_wrapper dst_d(_rpd.dst_primitive_desc.memory_desc);

    const size_t nelems = src_d.nelems();

    src += src_d.blocking_desc().offset_padding;
    dst += dst_d.blocking_desc().offset_padding;

#   pragma omp parallel for schedule(static)
    for (size_t e = 0; e < nelems; ++e) {
        dst[e] = src[e] * ((src[e] > 0) ? 1. : negative_slope);
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
        const mkldnn::impl::engine &engine) {
    auto relu_d = op_desc.relu;

    bool args_ok = op_desc._kind == primitive_kind::relu
        && one_of(relu_d.prop_kind, forward_training, forward_scoring);
    if (!args_ok) return unimplemented;

    /* memory descriptors check and fill-in */
    /* XXX: code duplication */
    if (relu_d.src_desc.format == any)
        CHECK(mkldnn_memory_desc_init(&relu_d.src_desc,
                    &relu_d.src_desc.tensor_desc, prec, nchw));
    if (relu_d.dst_desc.format == any)
        CHECK(mkldnn_memory_desc_init(&relu_d.dst_desc,
                    &relu_d.dst_desc.tensor_desc, prec, relu_d.src_desc.format));

    /* memory primitive descriptors check */
    /* XXX: code duplication */
    memory_primitive_desc_t src_pd, dst_pd;
    CHECK(mkldnn_memory_primitive_desc_init(&src_pd,
                &relu_d.src_desc, &engine));
    CHECK(mkldnn_memory_primitive_desc_init(&dst_pd,
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
