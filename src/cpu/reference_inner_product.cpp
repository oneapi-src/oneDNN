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

#include "reference_inner_product.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::precision_t prec>
status_t reference_inner_product<prec>::execute_forward() {
    auto obtain_ptr = [this](uint32_t idx) {
        const size_t oi = this->input()[idx].output_index;
        return reinterpret_cast<const data_t *>(
                this->input()[idx].primitive->output()[oi]->memory_const());
    };
    const data_t *src = obtain_ptr(0);
    const data_t *weights = obtain_ptr(1);
    const data_t *bias = this->_with_bias ? obtain_ptr(2) : nullptr;
    data_t *dst = reinterpret_cast<data_t *>(this->output()[0]->memory());

    const memory_desc_wrapper src_d(this->_ippd.src_primitive_desc.memory_desc),
            weights_d(this->_ippd.weights_primitive_desc.memory_desc),
            bias_d(this->_ippd.bias_primitive_desc.memory_desc),
            dst_d(this->_ippd.dst_primitive_desc.memory_desc);

    const uint32_t MB = src_d.dims()[0];
    const uint32_t OC = weights_d.dims()[0];
    const uint32_t IC = weights_d.dims()[1];

    const bool src_has_spatial = src_d.ndims() == 4;
    auto ker_has_spatial = [=](data_t *d, uint32_t mb, uint32_t oc) {
        const uint32_t KH = weights_d.dims()[2];
        const uint32_t KW = weights_d.dims()[3];
        for (uint32_t ic = 0; ic < IC; ++ic) {
            for (uint32_t kh = 0; kh < KH; ++kh) {
                for (uint32_t kw = 0; kw < KW; ++kw) {
                    *d += src[src_d.off(mb, ic, kh, kw)]
                        * weights[weights_d.off(oc, ic, kh, kw)];
                }
            }
        }
    };

    auto ker_no_spatial = [=](data_t *d, uint32_t mb, uint32_t oc) {
        for (uint32_t ic = 0; ic < IC; ++ic) {
            *d += src[src_d.off(mb, ic)] * weights[weights_d.off(oc, ic)];
        }
    };

#   pragma omp parallel for collapse(2) schedule(static)
    for (uint32_t mb = 0; mb < MB; ++mb) {
        for (uint32_t oc = 0; oc < OC; ++oc) {
            data_t *d = &dst[dst_d.off(mb, oc)];
            *d = bias ? bias[bias_d.off(oc)] : data_t(0);
            if (src_has_spatial) {
                ker_has_spatial(d, mb, oc);
            } else {
                ker_no_spatial(d, mb, oc);
            }
        }
    }

    return status::success;
}

template <impl::precision_t prec>
status_t reference_inner_product<prec>::constraint(
        const inner_product_desc_t &ip_d) {
    bool args_ok = one_of(ip_d.prop_kind, prop_kind::forward_training,
            prop_kind::forward_scoring);
    return args_ok ? success : unimplemented;
}


template <impl::precision_t prec>
const primitive_impl reference_inner_product<prec>::implementation = {
    reference_inner_product<prec>::create
};

template class reference_inner_product<precision::f32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
