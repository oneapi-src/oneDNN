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

#include "jit_avx2_conv_generator_f32.hpp"
#include "nstl.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

inline void jit_avx2_conv_generator_f32::oh_step_unroll(
        jit_convolution_param_t *params, uint32_t ur_w, int pad_l, int pad_r)
{
    using Xbyak::Ymm;

    uint32_t iw = params->iw;
    uint32_t ih = params->ih;
    uint32_t kw = params->kw;
    uint32_t kh = params->kh;
    uint32_t nb_ic = params->nb_ic;
    uint32_t stride_w = params->stride_w;
    uint32_t nb_oc_block = params->nb_oc_blocking;
    uint32_t ic_blk = params->ic_block;
    uint32_t oc_blk = params->oc_block;

    for (uint32_t ki = 0; ki < kw; ki++) {
        uint32_t jj_start = (uint32_t)nstl::max(0, pad_l - (int)ki);
        uint32_t jj_end = ur_w
                - (uint32_t)nstl::max(0, (int)ki + pad_r - (int)(kw - 1));
        for (uint32_t ifm2 = 0; ifm2 < ic_blk; ifm2++) {
            for (uint32_t jj = jj_start; jj < jj_end; jj++) {
                int inp_off;
                if (params->src_fmt == mkldnn_nchw)
                    inp_off = ifm2 * ih * iw + (jj * stride_w - pad_l);
                else
                    inp_off = (ki + jj * stride_w - pad_l) * ic_blk + ifm2;
                vbroadcastss(Ymm(nb_oc_block * ur_w + jj),
                        ptr[aux_reg_input + sizeof(float) * inp_off]);
            }
            for (uint32_t ii = 0; ii < nb_oc_block; ii++) {
                int ker_off = ii * nb_ic * kh * kw * ic_blk * oc_blk
                        + ki * ic_blk * oc_blk + ifm2 * oc_blk;
                vmovups(ymm15, ptr[aux_reg_kernel + sizeof(float) * ker_off]);
                for (uint32_t jj = jj_start; jj < jj_end; jj++)
                    vfmadd231ps(Ymm(ur_w * ii + jj),
                            Ymm(nb_oc_block * ur_w + jj), ymm15);
            }
        }
    }
}

inline void jit_avx2_conv_generator_f32::oh_step_nopad(
        jit_convolution_param_t *params, uint32_t ur_w, int pad_l, int pad_r,
        const char *kw_lable)
{
    using Xbyak::Ymm;

    uint32_t iw = params->iw;
    uint32_t ih = params->ih;
    uint32_t kw = params->kw;
    uint32_t kh = params->kh;
    uint32_t nb_ic = params->nb_ic;
    uint32_t stride_w = params->stride_w;
    uint32_t nb_oc_block = params->nb_oc_blocking;
    uint32_t ic_blk = params->ic_block;
    uint32_t oc_blk = params->oc_block;

    xor_(ki_iter, ki_iter);
    L(kw_lable);
    {
        uint32_t jj_start = 0;
        uint32_t jj_end = ur_w;
        for (uint32_t ifm2 = 0; ifm2 < ic_blk; ifm2++) {
            for (uint32_t jj = jj_start; jj < jj_end; jj++) {
                int inp_off;
                if (params->src_fmt == mkldnn_nchw)
                    inp_off = ifm2 * ih * iw + (jj * stride_w - pad_l);
                else
                    inp_off = (jj * stride_w - pad_l) * ic_blk + ifm2;
                vbroadcastss(Ymm(nb_oc_block * ur_w + jj),
                        ptr[aux_reg_input + sizeof(float) * inp_off]);
            }
            for (uint32_t ii = 0; ii < nb_oc_block; ii++) {
                int aux_kernel_offset = ii * nb_ic * kh * kw * ic_blk * oc_blk
                        + ifm2 * oc_blk;
                vmovups(ymm15, ptr[aux_reg_kernel
                                       + sizeof(float) * aux_kernel_offset]);
                for (uint32_t jj = jj_start; jj < jj_end; jj++)
                    vfmadd231ps(Ymm(ur_w * ii + jj),
                            Ymm(nb_oc_block * ur_w + jj), ymm15);
            }
        }
        add(aux_reg_kernel, sizeof(float) * oc_blk * ic_blk);
        if (params->src_fmt == mkldnn_nchw)
            add(aux_reg_input, sizeof(float));
        else
            add(aux_reg_input, sizeof(float) * ic_blk);

        inc(ki_iter);
        cmp(ki_iter, kw);
        jl(kw_lable, T_NEAR);
    }
}

inline void jit_avx2_conv_generator_f32::width_blk_step(
        jit_convolution_param_t *params, uint32_t ur_w, int pad_l, int pad_r,
        const char *kh_lable, const char *kw_lable)
{
    using Xbyak::Ymm;

    uint32_t iw = params->iw;
    uint32_t kw = params->kw;
    uint32_t ow = params->ow;
    uint32_t oh = params->oh;
    uint32_t nb_oc_block = params->nb_oc_blocking;
    uint32_t ic_blk = params->ic_block;
    uint32_t oc_blk = params->oc_block;

    for (uint32_t ii = 0; ii < nb_oc_block; ii++)
        for (uint32_t jj = 0; jj < ur_w; jj++)
            vmovups(Ymm(ur_w * ii + jj),
                    YWORD[reg_output
                            + sizeof(float) * (ii * oh * ow + jj) * oc_blk]);

    mov(aux_reg_input, reg_input);
    mov(aux_reg_kernel, reg_kernel);

    mov(kj, reg_kh);
    L(kh_lable);
    {
        if (params->kw < 5 || pad_l > 0 || pad_r > 0) {
            oh_step_unroll(params, ur_w, pad_l, pad_r);
            add(aux_reg_kernel, sizeof(float) * kw * oc_blk * ic_blk);
            if (params->src_fmt == mkldnn_nchw)
                add(aux_reg_input, sizeof(float) * iw);
            else
                add(aux_reg_input, sizeof(float) * iw * ic_blk);
        } else {
            oh_step_nopad(params, ur_w, pad_l, pad_r, kw_lable);
            if (params->src_fmt == mkldnn_nchw) {
                sub(aux_reg_input, sizeof(float) * kw);
                add(aux_reg_input, sizeof(float) * iw);
            } else {
                sub(aux_reg_input, sizeof(float) * kw * ic_blk);
                add(aux_reg_input, sizeof(float) * iw * ic_blk);
            }
        }

        dec(kj);
        cmp(kj, 0);
        jg(kh_lable, T_NEAR);
    }

    for (uint32_t ii = 0; ii < nb_oc_block; ii++)
        for (uint32_t jj = 0; jj < ur_w; jj++)
            vmovups(YWORD[reg_output
                            + sizeof(float) * (ii * oh * ow + jj) * oc_blk],
                    Ymm(ur_w * ii + jj));
}

jit_avx2_conv_generator_f32::jit_avx2_conv_generator_f32(
        jit_convolution_param_t *params, void *code_ptr, size_t code_size)
    : jit_generator(code_ptr, code_size)
{
    using Xbyak::Ymm;
    this->preamble();

    mov(reg_input, ptr[this->param1]);
    mov(reg_output, ptr[this->param1 + 8]);
    mov(reg_kernel, ptr[this->param1 + 16]);
    mov(reg_kh, ptr[this->param1 + 48]);

    // NB: works only for params->ur_w == 3 && params->nb_oc % 4 == 0
    uint32_t ur_w = params->ur_w;
    uint32_t ur_w_tail = params->ur_w_tail;
    uint32_t n_oi = params->ow / ur_w;
    uint32_t iw = params->iw;
    uint32_t kw = params->kw;
    uint32_t ic_blk = params->ic_block;
    uint32_t oc_blk = params->oc_block;
    uint32_t str_w = params->stride_w;

    xor_(oi_iter, oi_iter);
    int l_pad = (int)params->l_pad;
    int r_pad = nstl::max(0, (int)(((int)params->ow - 1) * str_w + (int)kw - 1
                                     - ((int)iw + (int)l_pad - 1)));
    int r_pad1 = 0;
    if (l_pad > 0) {
        width_blk_step(params, ur_w, l_pad, 0, ".kh_loop_oimain_padwl",
                ".kw_loop_oimain_padwl");
        if (params->src_fmt == mkldnn_nchw)
            add(reg_input, sizeof(float) * (ur_w * str_w - l_pad));
        else
            add(reg_input, sizeof(float) * (ur_w * str_w - l_pad) * ic_blk);
        add(reg_output, sizeof(float) * ur_w * oc_blk);
        inc(oi_iter);

        r_pad1 = ((int)ur_w * n_oi - 1) * str_w + (int)kw - 1
                - ((int)iw + (int)l_pad - 1);
        if (r_pad1 > 0)
            n_oi--;
    }

    if ((l_pad <= 0 && n_oi > 0) || (l_pad > 0 && n_oi > 1)) {
        L(".ow_loop");
        {
            width_blk_step(
                    params, ur_w, 0, 0, ".kh_loop_oimain", ".kw_loop_oimain");
            if (params->src_fmt == mkldnn_nchw)
                add(reg_input, sizeof(float) * ur_w * str_w);
            else
                add(reg_input, sizeof(float) * ur_w * str_w * ic_blk);
            add(reg_output, sizeof(float) * ur_w * oc_blk);

            inc(oi_iter);
            cmp(oi_iter, n_oi);
            jl(".ow_loop", T_NEAR);
        }
        L(".ow_loop_end");
    }

    if (r_pad1 > 0) {
        width_blk_step(params, ur_w, 0, r_pad1, ".kh_loop_oimain_padwr",
                ".kw_loop_oimain_padwr");
        if (params->src_fmt == mkldnn_nchw)
            add(reg_input, sizeof(float) * ur_w * str_w);
        else
            add(reg_input, sizeof(float) * ur_w * str_w * ic_blk);
        add(reg_output, sizeof(float) * ur_w * oc_blk);
    }

    if (ur_w_tail != 0)
        width_blk_step(params, ur_w_tail, 0, r_pad, ".kh_loop_oitail",
                ".kw_loop_oitail");

    this->postamble();
    return;
}
}
}
}
