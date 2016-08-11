#include "mkl_dnn_types.h"

#include "c_types_map.hpp"
#include "jit_avx2_convolution.hpp"
#include "type_helpers.hpp"

#include <stdio.h>

#define CHECK(f) do { \
    status_t status = f; \
    if (status != success) return status; \
} while(0)

namespace mkl_dnn { namespace impl { namespace cpu {

using namespace mkl_dnn::impl::status;
using namespace mkl_dnn::impl::prop_kind;
using namespace mkl_dnn::impl::alg_kind;
using namespace mkl_dnn::impl::precision;
using namespace mkl_dnn::impl::memory_format;
using namespace mkl_dnn::impl::primitive_kind;

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
    const uint32_t w_idx_base = w_groups ? 1 : 0;


#   pragma omp parallel for collapse(3)
    for (uint32_t g = 0; g < jcp.ngroups; ++g) {
        for (uint32_t n = 0; n < jcp.mb; ++n) {
            for (uint32_t oc = 0; oc < (jcp.nb_oc/jcp.nb_oc_blocking); ++oc) {
                for (uint32_t ic = 0; ic < jcp.nb_ic; ++ic) {
                    for (uint32_t oj = 0; oj < jcp.ohp; ++oj) {
                        uint32_t _n, _c, _w, _h;
                        uint32_t _C, _O, _I;
                        if (ic == 0)
                            for (uint32_t oi = 0; oi <  jcp.owp; ++oi)
                                if (bias != nullptr)
                                    for (uint32_t b = 0; b < jcp.nb_oc_blocking; ++b) {
                                        for (uint32_t i = 0; i < jcp.oc_block; ++i) {
                                            _n = n;
                                            _c = g*jcp.oc + (jcp.nb_oc_blocking*oc + b)*jcp.oc_block + i;
                                            _h = oj;
                                            _w = oi;
                                            dst[dst_d.off(_n, _c, _h, _w)] = bias[bias_d.off(_c)];
                                        }
                                     }
                                else
                                    for (uint32_t b = 0; b < jcp.nb_oc_blocking; ++b){
                                        for (uint32_t i = 0; i < jcp.oc_block; ++i) {
                                            _n = n;
                                            _c = g*jcp.oc + (jcp.nb_oc_blocking*oc + b)*jcp.oc_block + i;
                                            _h = oj;
                                            _w = oi;
                                            dst[dst_d.off(_n, _c, _h, _w)] = 0.0;
                                        }
                                    }

                        hnk_conv_kernel_t par_conv;

                        uint32_t ij = oj * jcp.stride_h;
                        uint32_t i_t_overflow = nstl::max(0, (int)(jcp.t_pad - ij));
                        uint32_t i_b_overflow = nstl::max((int)(ij + jcp.kh - jcp.t_pad), (int)jcp.ih) - jcp.ih;

                        _n = n;
                        _C = g*jcp.ic + ic*jcp.ic_block;
                        _h = (uint32_t)((int)ij-jcp.t_pad+(int)i_t_overflow);
                        _w = 0u;
                        par_conv.src  = (data_t*)&(src[src_d.off(_n, _C, _h, _w)]);

                        _n = n;
                        _C = g*jcp.oc + jcp.nb_oc_blocking*oc*jcp.oc_block;
                        _h = oj;
                        _w = 0u;
                        par_conv.dst  = (data_t*)&(dst[dst_d.off(_n, _C, _h, _w)]);

                        _O = jcp.nb_oc_blocking*oc*jcp.oc_block;
                        _I = ic*jcp.ic_block;
                        _h = i_t_overflow;
                        _w = 0u;
                        if (w_idx_base)
                            par_conv.filt = (data_t*)&(weights[weights_d.off(g, _O, _I, _h, _w)]);
                        else
                            par_conv.filt = (data_t*)&(weights[weights_d.off(_O, _I, _h, _w)]);

                        par_conv.src_prf  = NULL;
                        par_conv.dst_prf  = NULL;
                        par_conv.filt_prf = NULL;

                        par_conv.kh_padding = jcp.kh - i_t_overflow - i_b_overflow;
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
        const mkl_dnn::impl::engine &engine) {
    if (op_desc._kind != primitive_kind::convolution)
        return mkl_dnn_invalid_arguments;
    auto conv_d = op_desc.convolution;

    if (conv_d.prop_kind != forward)
        return unimplemented;
    if (conv_d.alg_kind != convolution_direct)
        return unimplemented;

    /* memory descriptors check and fill-in */
    if (conv_d.src_desc.format == any)
        CHECK(mkl_dnn_memory_desc_init(&conv_d.src_desc,
                &conv_d.src_desc.tensor_desc, f32, nChw8c));
    else if (conv_d.src_desc.format != mkl_dnn_nChw8c)
        return mkl_dnn_invalid_arguments;

    const bool groups = conv_d.weights_desc.tensor_desc.ndims
        == (conv_d.src_desc.tensor_desc.ndims + 1);
    if (conv_d.weights_desc.format == any)
        CHECK(mkl_dnn_memory_desc_init(&conv_d.weights_desc,
                &conv_d.weights_desc.tensor_desc, f32, groups ? gOIhw8i8o : OIhw8i8o));
    else {
        if (groups) {
            if (conv_d.weights_desc.format != mkl_dnn_gOIhw8i8o)
                return mkl_dnn_invalid_arguments;
        } else {
            if (conv_d.weights_desc.format != mkl_dnn_OIhw8i8o)
                return mkl_dnn_invalid_arguments;
        }
    }

    if (conv_d.bias_desc.format == any)
        CHECK(mkl_dnn_memory_desc_init(&conv_d.bias_desc,
                    &conv_d.bias_desc.tensor_desc, f32, x));
    else if (conv_d.bias_desc.format != mkl_dnn_x)
        return mkl_dnn_invalid_arguments;

    if (conv_d.dst_desc.format == any)
        CHECK(mkl_dnn_memory_desc_init(&conv_d.dst_desc,
                &conv_d.dst_desc.tensor_desc, f32, nChw8c));
    else if (conv_d.dst_desc.format != mkl_dnn_nChw8c)
        return mkl_dnn_invalid_arguments;

    /* memory primitive descriptors check */
    memory_primitive_desc_t src_pd, weights_pd, bias_pd, dst_pd;
    CHECK(mkl_dnn_memory_primitive_desc_init(&src_pd,
                &conv_d.src_desc, &engine));
    CHECK(mkl_dnn_memory_primitive_desc_init(&weights_pd,
                &conv_d.weights_desc, &engine));
    CHECK(mkl_dnn_memory_primitive_desc_init(&bias_pd,
                &conv_d.bias_desc, &engine));
    CHECK(mkl_dnn_memory_primitive_desc_init(&dst_pd,
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

    auto l_pad = conv_d.padding[1];
    auto    kw = weights_d.dims()[w_idx_base + 3];
    auto stride_h = conv_d.strides[0];
    auto stride_w = conv_d.strides[1];

    if (stride_w != stride_h) {
        return unimplemented;
    }
    if (stride_w > 1 || stride_h > 1) {
        return unimplemented;
    }

    if ((ic % 8) || (oc % 8)) {
        return mkl_dnn_unimplemented;
    }

    auto ur_w = 3;
    auto ur_w_tail = ow % ur_w;

    if (l_pad > (int)ur_w) { // maximum 1 step with l_pad so far
       return mkl_dnn_unimplemented;
    }

    int r_pad_step0 = nstl::max(0,
                          (int)(((ow == ur_w_tail ? ur_w_tail : ur_w)-1) +
                                  kw - 1 - (iw + l_pad - 1 )));
    if (l_pad > 0 && r_pad_step0 > 0) {// no steps with both left and right padding so far
        return mkl_dnn_unimplemented;
    }

    int r_pad_no_tail = nstl::max(0,
                            (int)((ow - ur_w_tail-1) + kw - 1 - (iw + l_pad - 1 )));
    if (r_pad_no_tail > (int)ur_w) { // maximum 1 ur_w block with r_pad so far
        return mkl_dnn_unimplemented;
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
