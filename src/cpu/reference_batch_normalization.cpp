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

#include <cmath>

#include "c_types_map.hpp"
#include "type_helpers.hpp"

#include "reference_batch_normalization.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;

template <impl::precision_t prec>
status_t reference_batch_normalization<prec>::execute_forward() {
    auto src = reinterpret_cast<const data_t *>(
            this->input()[0].primitive->output()[
            this->input()[0].output_index]->memory_const());
    auto scaleshift = reinterpret_cast<data_t *>(
            this->input()[1].primitive->output()[
            this->input()[1].output_index]->memory());
    auto workspace = this->_is_training
        ? reinterpret_cast<data_t *>(
            this->input()[2].primitive->output()[
            this->input()[2].output_index]->memory())
        : nullptr;
    auto dst = reinterpret_cast<data_t*>(this->output()[0]->memory());

    const memory_desc_wrapper
        src_d(this->_bnpd.src_primitive_desc.memory_desc),
        dst_d(this->_bnpd.dst_primitive_desc.memory_desc),
        scaleshift_d(this->_bnpd.scaleshift_primitive_desc.memory_desc);

    const int N = src_d.dims()[0];
    const int C = src_d.dims()[1];
    const int H = src_d.dims()[2];
    const int W = src_d.dims()[3];

    data_t *mean = nullptr, 
           *variance = nullptr;
    data_t v_mean, v_variance;
    if (this->_is_training) {
        mean        = &workspace[0];
        variance    = &workspace[C];
    }

#   pragma omp parallel for schedule(static)
    for (uint32_t c = 0; c < C; ++c)
    {
        data_t *_l_mean = this->_is_training ? &mean[c] : &v_mean;
        data_t *_l_variance = this->_is_training ? &variance[c] : &v_variance;

        *_l_mean = 0.0;
        for (uint32_t n = 0; n < N; ++n)
        for (uint32_t h = 0; h < H; ++h)
        for (uint32_t w = 0; w < W; ++w)
        {
            *_l_mean += src[src_d.off(n,c,h,w)];
        }
        *_l_mean /= W * N * H;

        *_l_variance = 0.0;
        for (uint32_t n = 0; n < N; ++n)
        for (uint32_t h = 0; h < H; ++h)
        for (uint32_t w = 0; w < W; ++w) {
            data_t _tmp = src[src_d.off(n,c,h,w)] - *_l_mean;
            *_l_variance += _tmp * _tmp;
        }
        *_l_variance = *_l_variance/(W * H * N) +
                        this->_bnpd.batch_normalization_desc.epsilon;
        *_l_variance = (data_t)1.0/std::sqrt(*_l_variance);

        for (uint32_t n = 0; n < N; ++n)
        for (uint32_t h = 0; h < H; ++h)
        for (uint32_t w = 0; w < W; ++w)
            dst[dst_d.off(n,c,h,w)] = scaleshift[scaleshift_d.off(0u,c)] *
              (src[src_d.off(n,c,h,w)] - *_l_mean)*(*_l_variance) +
                scaleshift[scaleshift_d.off(1u,c)];
    }

    return success;
}

template <impl::precision_t prec>
const primitive_impl reference_batch_normalization<prec>::implementation = {
    reference_batch_normalization<prec>::create
};

template class reference_batch_normalization<precision::f32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
