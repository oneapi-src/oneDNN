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
#include "jit_avx2_convolution_relu.hpp"
#include "type_helpers.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;

template <impl::precision_t prec>
jit_avx2_convolution_relu<prec>::jit_avx2_convolution_relu(
        const convolution_relu_primitive_desc_t &crpd,
        const primitive_at_t *inputs, const primitive *outputs[])
    : convolution_relu<jit_avx2_convolution_relu<prec>>(crpd, inputs, outputs)
    , generator(new jit_avx2_conv_generator_f32(crpd)) {}

template <impl::precision_t prec>
status_t jit_avx2_convolution_relu<prec>::execute_forward() {
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
        src_d(this->_crpd.src_primitive_desc),
        weights_d(this->_crpd.weights_primitive_desc),
        bias_d(this->_crpd.bias_primitive_desc),
        dst_d(this->_crpd.dst_primitive_desc);

    const auto &jcp = this->generator->jcp;

    auto ker = [&](uint32_t g, uint32_t n, uint32_t oc, uint32_t ic,
            uint32_t oh) {
        jit_convolution_kernel_t par_conv = {};

        const int ij = oh * jcp.stride_h;
        const int i_t_overflow = nstl::max(0, jcp.t_pad - ij);
        const int i_b_overflow = nstl::max(jcp.ih, ij + jcp.kh - jcp.t_pad)
            - jcp.ih;

        const uint32_t ih = nstl::max(ij - jcp.t_pad, 0);
        par_conv.src = const_cast<data_t *>(&src[src_d.blk_off(n,
                    jcp.ic == 3 ? 0 : g * jcp.nb_ic + ic, ih, 0)]);

        par_conv.dst = &dst[dst_d.blk_off(n,
                g * jcp.nb_oc + oc * jcp.nb_oc_blocking, oh, 0)];

        const uint32_t wcb = jcp.nb_oc_blocking*oc;
        const uint32_t wh = i_t_overflow;
        par_conv.filt = &weights[this->_with_groups
            ? weights_d.blk_off(g, wcb, jcp.ic == 3 ? 0 : ic, wh, 0u)
            : weights_d.blk_off(wcb, jcp.ic == 3 ? 0 : ic, wh, 0u)];

        if (ic == 0) {
            if (bias) {
                const size_t _c = g*jcp.nb_oc + jcp.nb_oc_blocking*oc;
                par_conv.bias = &bias[bias_d.blk_off(_c*jcp.oc_block)];
            }
            par_conv.ic_flag |= jit_avx2_conv_generator_f32::IC_FLAG_FIRST;
        }
        if (ic + 1 == jcp.nb_ic) {
            par_conv.ic_flag |= jit_avx2_conv_generator_f32::IC_FLAG_LAST;
        }

        par_conv.kh_padding = jcp.kh - i_t_overflow - i_b_overflow;
        par_conv.kw_padding = 0;

        this->generator->jit_ker((void*)&par_conv);
    };

#   pragma omp parallel for collapse(3) schedule(static)
    for (uint32_t g = 0; g < jcp.ngroups; ++g) {
        for (uint32_t n = 0; n < jcp.mb; ++n) {
            for (uint32_t oc = 0; oc < (jcp.nb_oc/jcp.nb_oc_blocking); ++oc) {
                for (uint32_t ic = 0; ic < jcp.nb_ic; ++ic) {
                    for (uint32_t oh = 0; oh < jcp.oh; ++oh) {
                        ker(g, n, oc, ic, oh);
                    }
                }
            }
        }
    }

    return success;
}

template <impl::precision_t prec>
status_t jit_avx2_convolution_relu<prec>::set_default_parameters(
        convolution_relu_desc_t &conv_relu_d) {
    auto &conv_d = conv_relu_d.convolution_desc;
    const bool flat = conv_d.src_desc.tensor_desc.dims[1] == 3;
    const bool with_groups = conv_d.weights_desc.tensor_desc.ndims
            == (conv_d.src_desc.tensor_desc.ndims + 1);

    if (conv_d.src_desc.format == any) {
        CHECK(types::set_default_format<prec>(conv_d.src_desc,
                    flat ? nchw : nChw8c));
    }
    if (conv_d.weights_desc.format == any) {
        CHECK(types::set_default_format<prec>(conv_d.weights_desc,
                    with_groups ? gOIhw8i8o : (flat ? Ohwi8o : OIhw8i8o)));
    }
    if (conv_d.dst_desc.format == any) {
        CHECK(types::set_default_format<prec>(conv_d.dst_desc, nChw8c));
    }

    return convolution_relu<jit_avx2_convolution_relu<prec>>::template
        set_default_parameters<void>(conv_relu_d);
}

template <impl::precision_t prec>
status_t jit_avx2_convolution_relu<prec>::constraint(
        const convolution_relu_desc_t &conv_relu_d) {
    const auto &conv_d = conv_relu_d.convolution_desc;
    bool args_ok = true
        && one_of(conv_d.prop_kind, prop_kind::forward_scoring)
        && conv_d.alg_kind == alg_kind::convolution_direct
        && jit_avx2_conv_generator_f32::is_applicable(conv_relu_d);
    return args_ok ? success : unimplemented;
}

template <impl::precision_t prec>
const primitive_impl jit_avx2_convolution_relu<prec>::implementation = {
    jit_avx2_convolution_relu<prec>::create
};

template class jit_avx2_convolution_relu<precision::f32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
