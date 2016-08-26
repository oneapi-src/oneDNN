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

#include <math.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"

#include "reference_lrn.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;

template <impl::precision_t prec>
status_t reference_lrn<prec>::execute_forward() {
    auto src = reinterpret_cast<const data_t *>(
            this->input()[0].primitive->output()[
            this->input()[0].output_index]->memory_const());
    data_t *scratch = this->_is_training
        ? reinterpret_cast<data_t *>(this->input()[1].primitive->output()[
                this->input()[1].output_index]->memory())
        : nullptr;
    auto dst = reinterpret_cast<data_t *>(this->output()[0]->memory());

    const memory_desc_wrapper
        src_d(this->_lpd.src_primitive_desc.memory_desc),
        scratch_d(this->_lpd.scratch_primitive_desc.memory_desc),
        dst_d(this->_lpd.dst_primitive_desc.memory_desc);

    const uint32_t C = src_d.dims()[1];
    const uint32_t H = src_d.dims()[2];
    const uint32_t W = src_d.dims()[3];
    const bool across_channels =
        this->_lpd.lrn_desc.alg_kind == alg_kind::lrn_across_channels;

    auto ker = [=](data_t *d, uint32_t n, uint32_t oc, uint32_t oh, uint32_t ow)
    {
        const double alpha = this->_lpd.lrn_desc.alpha;
        const double beta = this->_lpd.lrn_desc.beta;

        const uint32_t size = this->_lpd.lrn_desc.local_size;
        const uint32_t CSIZE = across_channels ? size : 1;
        const uint32_t HWSIZE = size + 1 - CSIZE;

        data_t sum = 0.0;
        uint32_t summands = across_channels ? size : size*size;
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
        if (this->_is_training)
            scratch[scratch_d.off(n, oc, oh, ow)] =
                1 / (k * (1 + alpha * sum / summands)); // for back prop
    };

    const uint32_t N = src_d.dims()[0];
#   pragma omp parallel for collapse(4) schedule(static)
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
const primitive_impl reference_lrn<prec>::implementation = {
    reference_lrn<prec>::create
};

template class reference_lrn<precision::f32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
