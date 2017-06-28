/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
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
#include "utils.hpp"
#include "type_helpers.hpp"
#include "gemm_convolution_utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

namespace jit_gemm_convolution_utils {

void im2col(
    jit_gemm_conv_conf_t &jcp, const float *im, float *col) {
    const size_t im_step = jcp.ih * jcp.iw;
    const size_t col_step = jcp.ks * jcp.os;

    int num_thr = (jcp.mb != 1) ? omp_get_max_threads() : 1;
#pragma omp parallel for  num_threads(num_thr)
    for (int ic = 0; ic < jcp.ic; ++ic) {
        for (int kh = 0; kh < jcp.kh; ++kh) {
        for (int oh = 0; oh < jcp.oh; ++oh) {
            const int ih = oh * jcp.stride_h - jcp.t_pad + kh * (1 + jcp.dilate_h);
            if (ih < 0 || ih >= jcp.ih) continue;

            for (int kw = 0; kw < jcp.kw; ++kw) {
            for (int ow = 0; ow < jcp.ow; ++ow) {
                const int iw = ow * jcp.stride_w - jcp.l_pad + kw * (1 + jcp.dilate_w);
                if (iw < 0 || iw >= jcp.iw) continue;

                const size_t col_idx = ((kh*jcp.kw + kw)*jcp.oh+oh)*jcp.ow+ow;
                const size_t im_idx = ih*jcp.iw + iw;
                col[col_idx] = im[im_idx];
            }
            }
        }
        }
        im += im_step;
        col += col_step;
    }
}

void col2im(
    jit_gemm_conv_conf_t &jcp, const float *col, float *im) {
    const size_t col_step = jcp.ks * jcp.os;
    const size_t im_step = jcp.ih * jcp.iw;
    const int iS = jcp.ih * jcp.iw;

    int num_thr = (jcp.mb != 1) ? omp_get_max_threads() : 1;
#pragma omp parallel for  num_threads(num_thr)
    for (int ic = 0; ic < jcp.ic; ++ic) {
        for (int is = 0; is < iS; ++is) im[is] = 0.;

        for (int oh = 0; oh < jcp.oh; ++oh) {
        for (int kh = 0; kh < jcp.kh; ++kh) {
            const int ih = oh * jcp.stride_h - jcp.t_pad + kh * (1 + jcp.dilate_h);
            if (ih < 0 || ih >= jcp.ih) continue;

            for (int ow = 0; ow < jcp.ow; ++ow) {
            for (int kw = 0; kw < jcp.kw; ++kw) {
                const int iw = ow * jcp.stride_w - jcp.l_pad + kw * (1 + jcp.dilate_w);
                if (iw < 0 || iw >= jcp.iw) continue;

                const size_t col_idx = ((kh*jcp.kw + kw)*jcp.oh+oh)*jcp.ow+ow;
                const size_t im_idx = ih*jcp.iw + iw;
                im[im_idx] += col[col_idx];
            }
            }
        }
        }
        col += col_step;
        im += im_step;
    }
}

void init_conf(
    jit_gemm_conv_conf_t &jcp, const convolution_desc_t &cd,
    const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
    const memory_desc_wrapper &dst_d,
    bool with_relu, double relu_negative_slope) {

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    jcp.prop_kind = cd.prop_kind;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];

    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;

    jcp.ih = src_d.dims()[2];
    jcp.iw = src_d.dims()[3];
    jcp.oh = dst_d.dims()[2];
    jcp.ow = dst_d.dims()[3];

    jcp.kh = weights_d.dims()[with_groups + 2];
    jcp.kw = weights_d.dims()[with_groups + 3];

    jcp.t_pad = cd.padding[0][0];
    jcp.l_pad = cd.padding[0][1];

    jcp.stride_h = cd.strides[0];
    jcp.stride_w = cd.strides[1];

    jcp.dilate_h = cd.dilates[0];
    jcp.dilate_w = cd.dilates[1];

    jcp.src_fmt = src_d.format();
    jcp.with_bias
        = cd.bias_desc.format != memory_format::undef
        || cd.diff_bias_desc.format != memory_format::undef;
    jcp.with_relu = with_relu;
    jcp.relu_negative_slope = relu_negative_slope;

    jcp.os = jcp.oh * jcp.ow;
    jcp.ks = jcp.kh * jcp.kw;
    jcp.need_im2col = !(jcp.oh == jcp.ih && jcp.ow == jcp.iw && jcp.ks == 1);
}

status_t prepare_workspace(
        jit_gemm_conv_conf_t &jcp, float **ws, bool is_bwd_weights,
        const size_t weights_size) {
    const size_t nthr = omp_get_max_threads();
    if (jcp.need_im2col) {
        const size_t sz_per_thread = jcp.ic*jcp.ks*jcp.os;
        jcp.im2col_size = utils::rnd_up(nthr*sz_per_thread, 16);
    } else {
        jcp.im2col_size = 0;
    }
    size_t weights_reduce_size = 0;
    if (is_bwd_weights && jcp.mb != 1 && nthr != 1) {
        const size_t sz_per_thread = jcp.ngroups * weights_size;
        weights_reduce_size = nthr * sz_per_thread;
    }
    *ws = 0;
    const size_t ws_size = sizeof(float)*jcp.im2col_size + weights_reduce_size;
    if (ws_size != 0) {
        *ws = (float*)malloc(ws_size, 64);
        if (*ws == NULL) return status::out_of_memory;
        for (size_t i = 0; i < jcp.im2col_size; ++i) (*ws)[i] = 0.;
    }
    return status::success;
}

void bwd_weights_balance(int ithr, int nthr, int ngroups, int mb, int &ithr_g,
        int &nthr_g, int &ithr_mb, int &nthr_mb) {
    nthr_g = nstl::min(ngroups, nthr);
    nthr_mb = nstl::min(mb, nthr / nthr_g);
    if (ithr / nthr_mb >= ngroups) {
        ithr_g = ithr_mb = -1;
    } else {
        ithr_g = ithr / nthr_mb;
        ithr_mb = ithr % nthr_mb;
    }
}

void bwd_weights_reduction_par(int ithr, int nthr, const jit_gemm_conv_conf_t &jcp,
        const float *weights_reduce_ws, float *weights) {
    const size_t weights_g_size = jcp.ic * jcp.oc * jcp.ks;

    size_t weights_start{0}, weights_end{0};
    balance211(weights_g_size, nthr, ithr, weights_start, weights_end);

    for (int i = 0; i < nthr; ++i) {
        const float *ws_i = weights_reduce_ws + i * weights_g_size;
        for (size_t s = weights_start; s < weights_end; ++s)
            weights[s] = (i == 0 ? 0 : weights[s]) + ws_i[s];
    }
}

};

}
}
}
