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
#include "reference_pooling.hpp"
#include "type_helpers.hpp"

#include <limits>

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;

template <impl::precision_t prec>
status_t reference_pooling<prec>::execute_forward() {
    auto src = reinterpret_cast<const data_t *>(
            this->input()[0].primitive->output()[
            this->input()[0].output_index]->memory_const());
    index_t *indices = this->_is_training
        ? reinterpret_cast<index_t*>(this->input()[1].primitive->output()[
                this->input()[1].output_index]->memory())
        : nullptr;
    auto dst = reinterpret_cast<data_t*>(this->output()[0]->memory());

    const memory_desc_wrapper
        src_d(this->_ppd.src_primitive_desc.memory_desc),
        indices_d(this->_ppd.indices_primitive_desc.memory_desc),
        dst_d(this->_ppd.dst_primitive_desc.memory_desc);

    const uint32_t IH = src_d.dims()[2];
    const uint32_t IW = src_d.dims()[3];
    const uint32_t KH = this->_ppd.pooling_desc.kernel[0];
    const uint32_t KW = this->_ppd.pooling_desc.kernel[1];
    const uint32_t SH = this->_ppd.pooling_desc.strides[0];
    const uint32_t SW = this->_ppd.pooling_desc.strides[1];
    const int32_t PH = this->_ppd.pooling_desc.padding[0];
    const int32_t PW = this->_ppd.pooling_desc.padding[1];

    auto ker = [=](data_t *d, uint32_t mb, uint32_t oc, uint32_t oh,
            uint32_t ow)
    {
        for (uint32_t kh = 0; kh < KH; ++kh) {
            for (uint32_t kw = 0; kw < KW; ++kw) {
                if (oh*SH + kh < (uint32_t)nstl::max(0, PH)) continue;
                if (ow*SW + kw < (uint32_t)nstl::max(0, PW)) continue;

                if (oh*SH + kh >= IH + PH) continue;
                if (ow*SW + kw >= IW + PW) continue;

                const uint32_t ih = oh * SH - PH + kh;
                const uint32_t iw = ow * SW - PW + kw;

                if (src[src_d.off(mb, oc, ih, iw)] > d[0]) {
                    d[0] = src[src_d.off(mb, oc, ih, iw)];
                    if (this->_is_training)
                        indices[indices_d.off(mb, oc, oh, ow)] = kh*KW + kw;
                }
            }
        }
    };

    const uint32_t MB = src_d.dims()[0];
    const uint32_t OC = dst_d.dims()[1];
    const uint32_t OH = dst_d.dims()[2];
    const uint32_t OW = dst_d.dims()[3];

#   pragma omp parallel for collapse(4) schedule(static)
    for (uint32_t mb = 0; mb < MB; ++mb) {
        for (uint32_t oc = 0; oc < OC; ++oc) {
            for (uint32_t oh = 0; oh < OH; ++oh) {
                for (uint32_t ow = 0; ow < OW; ++ow) {
                    data_t *d = &dst[dst_d.off(mb, oc, oh, ow)];
                    d[0] = -std::numeric_limits<data_t>::infinity();
                    ker(d, mb, oc, oh, ow);
                }
            }
        }
    }

    return success;
}

template <impl::precision_t prec>
status_t reference_pooling<prec>::constraint(const pooling_desc_t &pool_d) {
    bool args_ok = true
        && one_of(pool_d.prop_kind, prop_kind::forward_training,
                prop_kind::forward_scoring)
        && pool_d.alg_kind == alg_kind::pooling_max;
    return args_ok ? success : unimplemented;
}

template <impl::precision_t prec>
const primitive_impl reference_pooling<prec>::implementation = {
    reference_pooling<prec>::create
};

template class reference_pooling<precision::f32>;

}}}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
