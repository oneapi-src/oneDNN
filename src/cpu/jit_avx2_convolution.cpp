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

#include <stdio.h>

#define CHECK(f) do { \
    status_t status = f; \
    if (status != success) return status; \
} while(0)

namespace mkldnn { namespace impl { namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::alg_kind;
using namespace mkldnn::impl::precision;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::primitive_kind;

template <impl::precision_t prec>
status_t jit_avx2_convolution<prec>::execute_forward()
{
    auto obtain_ptr = [this](uint32_t idx) {
        const size_t oi = this->input()[idx].output_index;
        return reinterpret_cast<const data_t*>(
                this->input()[idx].primitive->output()[oi]->memory_const());
    };
    const data_t *src = obtain_ptr(0);
    const data_t *weights = obtain_ptr(1);
    const data_t *bias = _with_bias ? obtain_ptr(2) : nullptr;

    data_t *dst = reinterpret_cast<data_t*>(this->output()[0]->memory());

    const memory_desc_wrapper
        src_d(this->_cpd.src_primitive_desc.memory_desc),
        weights_d(this->_cpd.weights_primitive_desc.memory_desc),
        bias_d(this->_cpd.bias_primitive_desc.memory_desc),
        dst_d(this->_cpd.dst_primitive_desc.memory_desc);

    const bool w_groups = weights_d.ndims() == (src_d.ndims() + 1);

#   pragma omp parallel for collapse(3)
    for (uint32_t g = 0; g < jcp.ngroups; ++g) {
        for (uint32_t n = 0; n < jcp.mb; ++n) {
            for (uint32_t oc = 0; oc < (jcp.nb_oc/jcp.nb_oc_blocking); ++oc) {
                for (uint32_t ic = 0; ic < jcp.nb_ic; ++ic) {
                    for (uint32_t oh = 0; oh < jcp.oh; ++oh) {
                        if (ic == 0) {
                            for (uint32_t ow = 0; ow < jcp.ow; ++ow) {
                                for (uint32_t b = 0; b < jcp.nb_oc_blocking; ++b) {
                                    const uint32_t _c = g*jcp.nb_oc +
                                        jcp.nb_oc_blocking*oc + b;
                                    data_t *__tmp_dst  = &dst[dst_d.blk_off(
                                            n, _c, oh, ow)];
                                    const data_t *__tmp_bias = &bias[
                                        bias_d.blk_off(_c*jcp.oc_block)];
                                    for (uint32_t i = 0; i < jcp.oc_block; ++i) {
                                        __tmp_dst[i] = bias ? __tmp_bias[i] : 0;
                                    }
                                }
                            }
                        }

                        jit_convolution_kernel_t par_conv;
                        const int ij = oh * jcp.stride_h;
                        const int i_t_overflow = nstl::max(0, jcp.t_pad - ij);
                        const int i_b_overflow = nstl::max(jcp.ih,
                                ij + jcp.kh - jcp.t_pad) - jcp.ih;

                        const uint32_t ih = nstl::max(ij - jcp.t_pad, 0);
                        par_conv.src = const_cast<data_t *>(&src[src_d.blk_off(
                                n, jcp.ic == 3 ? 0 : g * jcp.nb_ic + ic, ih,
                                0)]);

                        par_conv.dst = &dst[dst_d.blk_off(n,
                                g * jcp.nb_oc + oc * jcp.nb_oc_blocking, oh,
                                0)];

                        const uint32_t wcb = jcp.nb_oc_blocking*oc;
                        const uint32_t wh = i_t_overflow;
                        par_conv.filt = const_cast<data_t *>(&weights[w_groups ?
                                        weights_d.blk_off(g, wcb,
                                                jcp.ic == 3 ? 0 : ic, wh, 0u) :
                                        weights_d.blk_off(wcb,
                                                jcp.ic == 3 ? 0 : ic, wh, 0u)]);

                        par_conv.src_prf = NULL;
                        par_conv.dst_prf = NULL;
                        par_conv.filt_prf = NULL;

                        par_conv.kh_padding = jcp.kh - i_t_overflow
                            - i_b_overflow;
                        par_conv.kw_padding = 0;
                        this->jit_ker((void*)&par_conv);
                    }
                }
            }
        }
    }

    return success;
}

template <impl::precision_t prec>
status_t jit_avx2_convolution<prec>::execute_backward_data() {
    return unimplemented;
}

template <impl::precision_t prec>
status_t jit_avx2_convolution<prec>::execute_backward_weights() {
    return unimplemented;
}

template <impl::precision_t prec>
status_t jit_avx2_convolution<prec>::execute_backward_bias() {
    return unimplemented;
}

template <impl::precision_t prec>
status_t jit_avx2_convolution<prec>::primitive_desc_init(
        primitive_desc_t *primitive_desc, const op_desc_t &op_desc,
        const mkldnn::impl::engine &engine) {
    if (op_desc._kind != primitive_kind::convolution)
        return mkldnn_invalid_arguments;
    auto conv_d = op_desc.convolution;

    if (conv_d.prop_kind != forward)
        return unimplemented;
    if (conv_d.alg_kind != convolution_direct)
        return unimplemented;

    /* memory descriptors check and fill-in */
    if (conv_d.src_desc.format == any) {
        if (conv_d.src_desc.tensor_desc.dims[1] == 3) {
            // maybe 1st convolution
            CHECK(mkldnn_memory_desc_init(
                    &conv_d.src_desc, &conv_d.src_desc.tensor_desc, f32, nhwc));
        } else {
            CHECK(mkldnn_memory_desc_init(&conv_d.src_desc,
                    &conv_d.src_desc.tensor_desc, f32, nChw8c));
        }
    }

    const bool groups = conv_d.weights_desc.tensor_desc.ndims
            == (conv_d.src_desc.tensor_desc.ndims + 1);
    if (conv_d.weights_desc.format == any) {
        if (!groups && conv_d.weights_desc.tensor_desc.dims[1] == 3) {
            // maybe 1st convolution
            CHECK(mkldnn_memory_desc_init(&conv_d.weights_desc,
                    &conv_d.weights_desc.tensor_desc, f32, Ohwi8o));
        } else {
            CHECK(mkldnn_memory_desc_init(&conv_d.weights_desc,
                    &conv_d.weights_desc.tensor_desc, f32,
                    groups ? gOIhw8i8o : OIhw8i8o));
        }
    }

    if (conv_d.bias_desc.format == any)
        CHECK(mkldnn_memory_desc_init(&conv_d.bias_desc,
                    &conv_d.bias_desc.tensor_desc, f32, x));
    else if (conv_d.bias_desc.format != mkldnn_x)
        return mkldnn_invalid_arguments;

    if (conv_d.dst_desc.format == any)
        CHECK(mkldnn_memory_desc_init(&conv_d.dst_desc,
                &conv_d.dst_desc.tensor_desc, f32, nChw8c));
    else if (conv_d.dst_desc.format != mkldnn_nChw8c)
        return mkldnn_invalid_arguments;

    const bool optimize_1st_conv = conv_d.src_desc.format == mkldnn_nhwc
            && !groups && conv_d.weights_desc.format == mkldnn_Ohwi8o;
    const bool optimize_iochannels_blocked_conv
            = conv_d.src_desc.format == mkldnn_nChw8c
            && ((!groups && conv_d.weights_desc.format == mkldnn_OIhw8i8o)
                       || (groups
                                  && conv_d.weights_desc.format
                                          == mkldnn_gOIhw8i8o));

    if (!optimize_1st_conv && !optimize_iochannels_blocked_conv)
        return mkldnn_invalid_arguments;

    /* memory primitive descriptors check */
    memory_primitive_desc_t src_pd, weights_pd, bias_pd, dst_pd;
    CHECK(mkldnn_memory_primitive_desc_init(&src_pd,
                &conv_d.src_desc, &engine));
    CHECK(mkldnn_memory_primitive_desc_init(&weights_pd,
                &conv_d.weights_desc, &engine));
    CHECK(mkldnn_memory_primitive_desc_init(&bias_pd,
                &conv_d.bias_desc, &engine));
    CHECK(mkldnn_memory_primitive_desc_init(&dst_pd,
                &conv_d.dst_desc, &engine));

    /* Check input parameters */
    const memory_desc_wrapper
        src_d(src_pd), weights_d(weights_pd), dst_d(dst_pd);
    const bool w_groups = weights_d.ndims() == (src_d.ndims() + 1);
    const uint32_t w_idx_base = w_groups ? 1 : 0;
    auto      ic = weights_d.dims()[w_idx_base + 1];
    auto      oc = weights_d.dims()[w_idx_base + 0];
    auto      iw = src_d.dims()[3];
    auto      ow = dst_d.dims()[3];

    auto t_pad = conv_d.padding[0];
    auto l_pad = conv_d.padding[1];
    auto    kw = weights_d.dims()[w_idx_base + 3];
    auto stride_h = conv_d.strides[0];
    auto stride_w = conv_d.strides[1];

    if (stride_w != stride_h) {
        return unimplemented;
    }
    if (optimize_iochannels_blocked_conv && (stride_w > 1 || stride_h > 1)) {
        return unimplemented;
    }

    if ((optimize_iochannels_blocked_conv && ic % 8 != 0) || (oc % 8 != 0)) {
        return unimplemented;
    }

    auto ur_w = 3;
    auto ur_w_tail = ow % ur_w;

    if (optimize_1st_conv && (t_pad > 0 || l_pad > 0)) {
        return unimplemented;
    }

    if (optimize_iochannels_blocked_conv) {
        if (l_pad > (int)ur_w) { // maximum 1 step with l_pad so far
            return unimplemented;
        }

        int r_pad_step0
                = nstl::max(0, (int)(((ow == ur_w_tail ? ur_w_tail : ur_w) - 1)
                                       + kw - 1 - (iw + l_pad - 1)));
        if (l_pad > 0 && r_pad_step0 > 0) { // no steps with both left and right
                                            // padding so far
            return unimplemented;
        }

        int r_pad_no_tail = nstl::max(
                0, (int)((ow - ur_w_tail - 1) + kw - 1 - (iw + l_pad - 1)));
        if (r_pad_no_tail
                > (int)ur_w) { // maximum 1 ur_w block with r_pad so far
            return unimplemented;
        }
    }

    /* final stage */
    convolution_primitive_desc_t cpd;
    cpd.base.primitive_kind = convolution;
    cpd.base.engine = &engine;
    cpd.base.implementation = reinterpret_cast<const void*>(&implementation);
    cpd.convolution_desc = conv_d;
    cpd.src_primitive_desc = src_pd;
    cpd.weights_primitive_desc = weights_pd;
    cpd.bias_primitive_desc = bias_pd;
    cpd.dst_primitive_desc = dst_pd;

    // if (!convolution_primitive_desc_is_ok(cpd)) return invalid; // ???

    primitive_desc->convolution = cpd;

    return success;
}

namespace {
template <impl::precision_t prec>
status_t create(primitive **aprimitive, const primitive_desc_t *primitive_desc,
        const primitive_at_t inputs[], const primitive *outputs[]) {
    assert(primitive_desc->base.primitive_kind == convolution);

    auto& cpd = primitive_desc->convolution;

    *aprimitive = new jit_avx2_convolution<prec>(cpd, inputs, outputs);
    return aprimitive ? success : out_of_memory;
}
}


template <impl::precision_t prec>
const primitive_impl jit_avx2_convolution<prec>::implementation = {
    create<prec>
};

template class jit_avx2_convolution<f32>;

}}}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
