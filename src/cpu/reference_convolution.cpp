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

#include "c_types_map.hpp"
#include "type_helpers.hpp"

#include "reference_convolution.hpp"

namespace mkldnn { namespace impl { namespace cpu {

template <impl::precision_t prec>
status_t reference_convolution<prec>::execute_forward() {
    auto obtain_ptr = [this](uint32_t idx) {
        const size_t oi = this->input()[idx].output_index;
        return reinterpret_cast<const data_t*>(
                this->input()[idx].primitive->output()[oi]->memory_const());
    };
    const data_t *src = obtain_ptr(0);
    const data_t *weights = obtain_ptr(1);
    const data_t *bias = this->_with_bias ? obtain_ptr(2) : nullptr;
    data_t *dst = reinterpret_cast<data_t*>(this->output()[0]->memory());

    const memory_desc_wrapper
        src_d(this->_cpd.src_primitive_desc.memory_desc),
        weights_d(this->_cpd.weights_primitive_desc.memory_desc),
        bias_d(this->_cpd.bias_primitive_desc.memory_desc),
        dst_d(this->_cpd.dst_primitive_desc.memory_desc);

    const uint32_t w_idx_base = this->_with_groups ? 1 : 0;
    const uint32_t G = this->_with_groups ? weights_d.dims()[0] : 1;

    const uint32_t MB = src_d.dims()[0];
    const uint32_t OH = dst_d.dims()[2];
    const uint32_t OW = dst_d.dims()[3];
    const uint32_t IH = src_d.dims()[2];
    const uint32_t IW = src_d.dims()[3];

    const uint32_t OC = weights_d.dims()[w_idx_base + 0];
    const uint32_t IC = weights_d.dims()[w_idx_base + 1];
    const uint32_t KH = weights_d.dims()[w_idx_base + 2];
    const uint32_t KW = weights_d.dims()[w_idx_base + 3];

    const uint32_t KSH = this->_cpd.convolution_desc.strides[0];
    const uint32_t KSW = this->_cpd.convolution_desc.strides[1];

    const int32_t padH = this->_cpd.convolution_desc.padding[0];
    const int32_t padW = this->_cpd.convolution_desc.padding[1];

    auto ker = [=](data_t *d, uint32_t g, uint32_t mb, uint32_t oc, uint32_t oh,
            uint32_t ow)
    {
        for (uint32_t ic = 0; ic < IC; ++ic) {
            for (uint32_t kh = 0; kh < KH; ++kh) {
                for (uint32_t kw = 0; kw < KW; ++kw) {
                    if (oh*KSH + kh < (uint32_t)nstl::max(0, padH)) continue;
                    if (ow*KSW + kw < (uint32_t)nstl::max(0, padW)) continue;

                    if (oh*KSH + kh >= IH + padH) continue;
                    if (ow*KSW + kw >= IW + padW) continue;

                    const uint32_t ih = oh * KSH - padH + kh;
                    const uint32_t iw = ow * KSW - padW + kw;

                    if (this->_with_groups) {
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

#   pragma omp parallel for collapse(5) schedule(static)
    for (uint32_t g = 0; g < G; ++g) {
        for (uint32_t mb = 0; mb < MB; ++mb) {
            for (uint32_t oc = 0; oc < OC; ++oc) {
                for (uint32_t oh = 0; oh < OH; ++oh) {
                    for (uint32_t ow = 0; ow < OW; ++ow) {
                        data_t *d = &dst[dst_d.off(mb, g*OC + oc, oh, ow)];
                        *d = bias ? bias[bias_d.off(g*OC + oc)] : data_t(0);
                        ker(d, g, mb, oc, oh, ow);
                    }
                }
            }
        }
    }

    return status::success;
}

template <impl::precision_t prec>
status_t reference_convolution<prec>::constraint(
        const convolution_desc_t &conv_d) {
    bool args_ok = true
        && one_of(conv_d.prop_kind, prop_kind::forward_training,
                prop_kind::forward_scoring)
        && conv_d.alg_kind == alg_kind::convolution_direct;
    return args_ok ? success : unimplemented;
}

template <impl::precision_t prec>
const primitive_impl reference_convolution<prec>::implementation = {
    reference_convolution<prec>::create
};

template class reference_convolution<precision::f32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
