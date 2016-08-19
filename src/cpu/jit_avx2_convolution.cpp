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
#include "jit_avx2_convolution.hpp"
#include "type_helpers.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;

template <impl::precision_t prec>
jit_avx2_convolution<prec>::jit_avx2_convolution(
        const convolution_primitive_desc_t &cpd, const primitive_at_t *inputs,
        const primitive *outputs[])
    : convolution<jit_avx2_convolution<prec>>(cpd, inputs, outputs) {
    const memory_desc_wrapper
        src_d(this->_cpd.src_primitive_desc),
        weights_d(this->_cpd.weights_primitive_desc),
        dst_d(this->_cpd.dst_primitive_desc);

    const uint32_t w_idx_base = this->_with_groups ? 1 : 0;
    jcp.ngroups = this->_with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
    jcp.ic = weights_d.dims()[w_idx_base + 1];
    jcp.oc = weights_d.dims()[w_idx_base + 0];

    jcp.ih = src_d.dims()[2]; jcp.iw = src_d.dims()[3];
    jcp.oh = dst_d.dims()[2]; jcp.ow = dst_d.dims()[3];

    jcp.t_pad = this->_cpd.convolution_desc.padding[0];
    jcp.l_pad = this->_cpd.convolution_desc.padding[1];
    jcp.kh = weights_d.dims()[w_idx_base + 2];
    jcp.kw = weights_d.dims()[w_idx_base + 3];
    jcp.stride_h = this->_cpd.convolution_desc.strides[0];
    jcp.stride_w = this->_cpd.convolution_desc.strides[1];

    const uint32_t simd_w = 8;
    jcp.ic_block = (jcp.ic % simd_w != 0) ? jcp.ic : simd_w;
    jcp.nb_ic = jcp.ic / jcp.ic_block;

    jcp.oc_block = simd_w;
    jcp.nb_oc = jcp.oc / jcp.oc_block;
    jcp.ur_h = 1; /* no code-unrolling by h so far */
    jcp.ur_w = 3;
    jcp.nb_ic_blocking =  jcp.nb_oc_blocking = 1;
    for (int b = 4; b > 1; b--)
        if (jcp.nb_oc % b == 0) {
            jcp.nb_oc_blocking = b;
            break;
        }
    jcp.ur_w_tail = jcp.ow % jcp.ur_w;
    jcp.src_fmt = src_d.format();

    generator = new jit_avx2_conv_generator_f32(&jcp);
    //TODO: if(generator == nullptr) return nullptr;
    jit_ker = (void (*)(void*))generator->getCode();
    //TODO: if(jit_ker == nullptr) return nullptr;
}

template <impl::precision_t prec>
status_t jit_avx2_convolution<prec>::execute_forward() {
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
        src_d(this->_cpd.src_primitive_desc),
        weights_d(this->_cpd.weights_primitive_desc),
        bias_d(this->_cpd.bias_primitive_desc),
        dst_d(this->_cpd.dst_primitive_desc);

    auto ker = [&](uint32_t g, uint32_t n, uint32_t oc, uint32_t ic,
            uint32_t oh) {
        if (ic == 0) {
            for (uint32_t ow = 0; ow < jcp.ow; ++ow) {
                for (uint32_t b = 0; b < jcp.nb_oc_blocking; ++b) {
                    const uint32_t _c = g*jcp.nb_oc + jcp.nb_oc_blocking*oc + b;
                    data_t *__tmp_dst = &dst[dst_d.blk_off(n, _c, oh, ow)];
                    if (bias) {
                        const data_t *__tmp_bias = &bias[bias_d.blk_off(
                                _c*jcp.oc_block)];
                        for (uint32_t i = 0; i < jcp.oc_block; ++i)
                            __tmp_dst[i] = __tmp_bias[i];
                    } else {
                        for (uint32_t i = 0; i < jcp.oc_block; ++i)
                        __tmp_dst[i] = 0;
                    }
                }
            }
        }

        jit_convolution_kernel_t par_conv;
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

        par_conv.src_prf = NULL;
        par_conv.dst_prf = NULL;
        par_conv.filt_prf = NULL;

        par_conv.kh_padding = jcp.kh - i_t_overflow - i_b_overflow;
        par_conv.kw_padding = 0;
        this->jit_ker((void*)&par_conv);
    };

#   pragma omp parallel for collapse(3)
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
status_t jit_avx2_convolution<prec>::set_default_parameters(
        convolution_desc_t &conv_d) {
    const bool flat = conv_d.src_desc.tensor_desc.dims[1] == 3;
    const bool with_groups = conv_d.weights_desc.tensor_desc.ndims
            == (conv_d.src_desc.tensor_desc.ndims + 1);

    if (conv_d.src_desc.format == any) {
        CHECK(conv_set_default_format<prec>(conv_d.src_desc,
                    flat ? nchw : nChw8c));
    }
    if (conv_d.weights_desc.format == any) {
        CHECK(conv_set_default_format<prec>(conv_d.weights_desc,
                    with_groups ? gOIhw8i8o : (flat ? Ohwi8o : OIhw8i8o)));
    }
    if (conv_d.dst_desc.format == any) {
        CHECK(conv_set_default_format<prec>(conv_d.dst_desc, nChw8c));
    }

    return convolution<jit_avx2_convolution<prec>>::template
        set_default_parameters<void>(conv_d);
}

template <impl::precision_t prec>
status_t jit_avx2_convolution<prec>::constraint(
        const convolution_desc_t &conv_d) {
    const memory_desc_wrapper src_d(conv_d.src_desc),
          weights_d(conv_d.weights_desc), dst_d(conv_d.dst_desc);

    const bool flat = src_d.dims()[1] == 3;
    const bool mimo = !flat;
    const bool with_groups = weights_d.ndims() == (src_d.ndims() + 1);

    bool args_ok = true
        && conv_d.prop_kind == prop_kind::forward
        && conv_d.alg_kind == alg_kind::convolution_direct
        && implication(flat, one_of(src_d.format(), nchw, nhwc))
        && implication(mimo, src_d.format() == nChw8c)
        && weights_d.format() ==
                (with_groups ? gOIhw8i8o : (flat ? Ohwi8o : OIhw8i8o))
        && one_of(conv_d.bias_desc.format, memory_format::undef, x)
        && dst_d.format() == nChw8c;
    if (!args_ok) return unimplemented;

    const uint32_t w_idx_base = with_groups ? 1 : 0;
    int ic = weights_d.dims()[w_idx_base + 1];
    int oc = weights_d.dims()[w_idx_base + 0];
    int iw = src_d.dims()[3];
    int ow = dst_d.dims()[3];

    int t_pad = conv_d.padding[0];
    int l_pad = conv_d.padding[1];
    int kw = weights_d.dims()[w_idx_base + 3];
    uint32_t stride_h = conv_d.strides[0];
    uint32_t stride_w = conv_d.strides[1];

    int ur_w = 3;
    int ur_w_tail = ow % ur_w;

    const uint32_t simd_w = 8;

    args_ok = true
        && stride_w == stride_h
        && implication(mimo, true
                && stride_w == 1 && stride_h == 1
                && ic % simd_w == 0 && oc % simd_w == 0
                && l_pad <= ur_w)
        && implication(flat, t_pad == 0 && l_pad == 0);
    if (!args_ok) return unimplemented;

    if (mimo) {
        int r_pad_step0 = nstl::max(0,
                ((ow == ur_w_tail ? ur_w_tail : ur_w) - 1)
                + (kw - 1) - (iw + l_pad - 1));
        int r_pad_no_tail = nstl::max(0, (ow - ur_w_tail - 1)
                + (kw - 1) - (iw + l_pad - 1));

        /* no steps with both left and right padding so far */
        if (l_pad > 0 && r_pad_step0 > 0) return unimplemented;

        /* maximum 1 ur_w block with r_pad so far */
        if (r_pad_no_tail > ur_w) return unimplemented;
    }

    return success;
}

template <impl::precision_t prec>
const primitive_impl jit_avx2_convolution<prec>::implementation = {
    jit_avx2_convolution<prec>::create
};

template class jit_avx2_convolution<precision::f32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
