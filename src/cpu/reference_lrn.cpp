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
#include "reference_lrn.hpp"
#include "type_helpers.hpp"
#include <cmath>

namespace mkldnn { namespace impl { namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::alg_kind;
using namespace mkldnn::impl::precision;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::primitive_kind;

template <impl::precision_t prec>
status_t reference_lrn<prec>::execute_forward() {
    const data_t *src =
        reinterpret_cast<const data_t *>(this->input()[0].primitive->output()[this->input()[0].output_index]->memory_const());
    data_t *scratch =
        reinterpret_cast<data_t *>(this->input()[1].primitive->output()[this->input()[1].output_index]->memory());
    data_t *dst =
        reinterpret_cast<data_t *>(this->output()[0]->memory());

    const memory_desc_wrapper
        src_d(this->_ppd.src_primitive_desc.memory_desc),
        scratch_d(this->_ppd.scratch_primitive_desc.memory_desc),
        dst_d(this->_ppd.dst_primitive_desc.memory_desc);

    const uint32_t C = src_d.dims()[1];
    const uint32_t H = src_d.dims()[2];
    const uint32_t W = src_d.dims()[3];

    auto ker = [=](data_t *d, uint32_t n, uint32_t oc, uint32_t oh, uint32_t ow)
    {
        const double alpha = this->_ppd.lrn_desc.alpha;
        const double beta = this->_ppd.lrn_desc.beta;

        const uint32_t size = this->_ppd.lrn_desc.local_size;
        const uint32_t CSIZE = this->_ppd.lrn_desc.alg_kind == lrn_across_channels ? size : 1;
        const uint32_t HWSIZE = size + 1 - CSIZE;

        data_t sum = 0.0;
        uint32_t summands = this->_ppd.lrn_desc.alg_kind == lrn_across_channels ? size : size*size;
        for (uint32_t c = oc; c < oc + CSIZE; ++c) {
            if (c < (CSIZE - 1) / 2) continue;
            if (c >= C + (CSIZE - 1) / 2) continue;
            for (uint32_t h = oh; h < oh + HWSIZE; ++h) {
                if (h < (HWSIZE - 1) / 2) continue;
                if (h >= H + (HWSIZE - 1) / 2) continue;
                for (uint32_t w = ow; w < ow + HWSIZE; ++w) {
                    if (w < (HWSIZE - 1) / 2) continue;
                    if (w >= W + (HWSIZE - 1) / 2) continue;
                    data_t s = src[src_d.off(n, c - (CSIZE - 1) / 2, h - (HWSIZE - 1) / 2, w - (HWSIZE - 1) / 2)];
                    sum += s * s;
                }
            }
        }
        data_t k = pow(1 + alpha * sum / summands, beta);
        d[0] = src[src_d.off(n, oc, oh, ow)] / k;
        scratch[scratch_d.off(n, oc, oh, ow)] = 1 / (k * (1 + alpha * sum / summands)); // for back prop
    };

    const uint32_t N = src_d.dims()[0];
#   pragma omp parallel for collapse(4)
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t c = 0; c < C; ++c) {
            for (uint32_t h = 0; h < H; ++h) {
                for (uint32_t w = 0; w < W; ++w) {
                    ker(&dst[dst_d.off(n, c, h, w)], n, c, h, w);
                }
            }
        }
    }

    return success;
}

template <impl::precision_t prec>
status_t reference_lrn<prec>::execute_backward_data() {
    return unimplemented;
}

template <impl::precision_t prec>
status_t reference_lrn<prec>::primitive_desc_init(
        primitive_desc_t *primitive_desc, const op_desc_t &op_desc,
        const mkldnn::impl::engine &engine) {
    if (op_desc._kind != primitive_kind::lrn)
        return invalid_arguments;
    auto lrn_d = op_desc.lrn;

    if (lrn_d.prop_kind != forward)
        return unimplemented;

    /* memory descriptors check and fill-in */
    if (lrn_d.src_desc.format == any)
        CHECK(mkldnn_memory_desc_init(&lrn_d.src_desc,
        &lrn_d.src_desc.tensor_desc, prec, nchw));
    if (lrn_d.dst_desc.format == any)
        CHECK(mkldnn_memory_desc_init(&lrn_d.dst_desc,
        &lrn_d.dst_desc.tensor_desc, prec, lrn_d.src_desc.format));

    memory_desc_t scratch_desc;
    CHECK(mkldnn_memory_desc_init(&scratch_desc,
        &lrn_d.dst_desc.tensor_desc, prec, lrn_d.dst_desc.format));

    /* memory primitive descriptors check */
    memory_primitive_desc_t src_pd, scratch_pd, dst_pd;
    CHECK(mkldnn_memory_primitive_desc_init(&src_pd,
        &lrn_d.src_desc, &engine));
    CHECK(mkldnn_memory_primitive_desc_init(&dst_pd,
        &lrn_d.dst_desc, &engine));
    CHECK(mkldnn_memory_primitive_desc_init(&scratch_pd,
        &scratch_desc, &engine));

    /* final stage */
    lrn_primitive_desc_t lpd;
    lpd.base.primitive_kind = lrn;
    lpd.base.engine = &engine;
    lpd.base.implementation = reinterpret_cast<const void*>(&implementation);
    lpd.lrn_desc = lrn_d;
    lpd.src_primitive_desc   = src_pd;
    lpd.scratch_primitive_desc = scratch_pd;
    lpd.dst_primitive_desc  = dst_pd;

    // if (!lrn_primitive_desc_is_ok(lpd)) return invalid_arguments; // ???

    primitive_desc->lrn = lpd;

    return success;
}

namespace {
template <impl::precision_t prec>
status_t create(primitive **aprimitive, const primitive_desc_t *primitive_desc,
        const primitive_at_t inputs[], const primitive *outputs[]) {
    assert(primitive_desc->base.primitive_kind == lrn);

    auto& ppd = primitive_desc->lrn;
    // TODO: some checks here.

    *aprimitive = new reference_lrn<prec>(ppd, inputs, outputs);
    return aprimitive ? success : out_of_memory;
}
}

template <impl::precision_t prec>
const primitive_impl reference_lrn<prec>::implementation = { create<prec> };

template class reference_lrn<f32>;

}}}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
