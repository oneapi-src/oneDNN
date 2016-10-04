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

#include "ref_inner_product.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t data_type>
void ref_inner_product_fwd_t<data_type>::execute_forward() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<data_t*>(this->memory());

    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper dst_d(conf_.dst_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));
    const memory_desc_wrapper bias_d(conf_.weights_pd(1));

    const int MB = conf_.MB();
    const int OC = conf_.OC();
    const int IC = conf_.IC();

    const bool src_has_spatial = src_d.ndims() == 4;
    auto ker_has_spatial = [=](data_t *d, int mb, int oc) {
        const int KH = conf_.KH();
        const int KW = conf_.KW();
        for (int ic = 0; ic < IC; ++ic) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    *d += src[src_d.off(mb, ic, kh, kw)]
                        * weights[weights_d.off(oc, ic, kh, kw)];
                }
            }
        }
    };

    auto ker_no_spatial = [=](data_t *d, int mb, int oc) {
        for (int ic = 0; ic < IC; ++ic) {
            *d += src[src_d.off(mb, ic)] * weights[weights_d.off(oc, ic)];
        }
    };

#   pragma omp parallel for collapse(2) schedule(static)
    for (int mb = 0; mb < MB; ++mb) {
        for (int oc = 0; oc < OC; ++oc) {
            data_t *d = &dst[dst_d.off(mb, oc)];
            *d = bias ? bias[bias_d.off(oc)] : data_t(0);
            if (src_has_spatial) {
                ker_has_spatial(d, mb, oc);
            } else {
                ker_no_spatial(d, mb, oc);
            }
        }
    }
}

template struct ref_inner_product_fwd_t<data_type::f32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

