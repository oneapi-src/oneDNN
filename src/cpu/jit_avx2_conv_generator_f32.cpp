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
#include "nstl.hpp"
#include "type_helpers.hpp"

#include "jit_avx2_conv_generator_f32.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::memory_format;

inline void jit_avx2_conv_generator_f32::oh_step_unroll(
        jit_convolution_param_t *params, int ur_w, int pad_l, int pad_r)
{
    using Xbyak::Ymm;

    int iw = params->iw;
    int ih = params->ih;
    int kw = params->kw;
    int kh = params->kh;
    int nb_ic = params->nb_ic;
    int stride_w = params->stride_w;
    int nb_oc_block = params->nb_oc_blocking;
    int ic_blk = params->ic_block;
    int oc_blk = params->oc_block;

    for (int ki = 0; ki < kw; ki++) {
        int jj_start = nstl::max(0, pad_l - ki);
        int jj_end = ur_w
                - (int)nstl::max(0, (int)ki + pad_r - (int)(kw - 1));
        for (int ifm2 = 0; ifm2 < ic_blk; ifm2++) {
            for (int jj = jj_start; jj < jj_end; jj++) {
                int inp_off;
                if (_src_in_nchw)
                    inp_off = ifm2 * ih * iw + (jj * stride_w - pad_l);
                else
                    inp_off = (ki + jj * stride_w - pad_l) * ic_blk + ifm2;
                vbroadcastss(Ymm(nb_oc_block * ur_w + jj),
                        ptr[aux_reg_input + sizeof(float) * inp_off]);
            }
            for (int ii = 0; ii < nb_oc_block; ii++) {
                int ker_off = ii * nb_ic * kh * kw * ic_blk * oc_blk
                        + ki * ic_blk * oc_blk + ifm2 * oc_blk;
                vmovups(ymm15, ptr[aux_reg_kernel + sizeof(float) * ker_off]);
                for (int jj = jj_start; jj < jj_end; jj++)
                    vfmadd231ps(Ymm(ur_w * ii + jj),
                            Ymm(nb_oc_block * ur_w + jj), ymm15);
            }
        }
    }
}

inline void jit_avx2_conv_generator_f32::oh_step_nopad(
        jit_convolution_param_t *params, int ur_w, int pad_l, int pad_r,
        const char *kw_lable)
{
    using Xbyak::Ymm;

    int iw = params->iw;
    int ih = params->ih;
    int kw = params->kw;
    int kh = params->kh;
    int nb_ic = params->nb_ic;
    int stride_w = params->stride_w;
    int nb_oc_block = params->nb_oc_blocking;
    int ic_blk = params->ic_block;
    int oc_blk = params->oc_block;

    xor_(ki_iter, ki_iter);
    L(kw_lable);
    {
        int jj_start = 0;
        int jj_end = ur_w;
        for (int ifm2 = 0; ifm2 < ic_blk; ifm2++) {
            for (int jj = jj_start; jj < jj_end; jj++) {
                int inp_off;
                if (_src_in_nchw)
                    inp_off = ifm2 * ih * iw + (jj * stride_w - pad_l);
                else
                    inp_off = (jj * stride_w - pad_l) * ic_blk + ifm2;
                vbroadcastss(Ymm(nb_oc_block * ur_w + jj),
                        ptr[aux_reg_input + sizeof(float) * inp_off]);
            }
            for (int ii = 0; ii < nb_oc_block; ii++) {
                int aux_kernel_offset = ii * nb_ic * kh * kw * ic_blk * oc_blk
                    + ifm2 * oc_blk;
                vmovups(ymm15, ptr[aux_reg_kernel
                        + sizeof(float) * aux_kernel_offset]);
                for (int jj = jj_start; jj < jj_end; jj++)
                    vfmadd231ps(Ymm(ur_w * ii + jj),
                            Ymm(nb_oc_block * ur_w + jj), ymm15);
            }
        }
        add(aux_reg_kernel, sizeof(float) * oc_blk * ic_blk);
        add(aux_reg_input, sizeof(float) * (_src_in_nchw ? 1 : ic_blk));

        inc(ki_iter);
        cmp(ki_iter, kw);
        jl(kw_lable, T_NEAR);
    }
}

inline void jit_avx2_conv_generator_f32::width_blk_step(
        jit_convolution_param_t *params, int ur_w, int pad_l, int pad_r,
        const char *kh_lable, const char *kw_lable)
{
    using Xbyak::Ymm;

    int iw = params->iw;
    int kw = params->kw;
    int ow = params->ow;
    int oh = params->oh;
    int nb_oc_block = params->nb_oc_blocking;
    int ic_blk = params->ic_block;
    int oc_blk = params->oc_block;
    const int inp_mult = _src_in_nchw ? 1 : ic_blk;

    for (int ii = 0; ii < nb_oc_block; ii++)
        for (int jj = 0; jj < ur_w; jj++)
            vmovups(Ymm(ur_w * ii + jj), YWORD[reg_output
                    + sizeof(float) * (ii * oh * ow + jj) * oc_blk]);

    mov(aux_reg_input, reg_input);
    mov(aux_reg_kernel, reg_kernel);

    mov(kj, reg_kh);
    L(kh_lable);
    {
        if (params->kw < 5 || pad_l > 0 || pad_r > 0) {
            oh_step_unroll(params, ur_w, pad_l, pad_r);
            add(aux_reg_kernel, sizeof(float) * kw * oc_blk * ic_blk);
            add(aux_reg_input, sizeof(float) * iw * inp_mult);
        } else {
            oh_step_nopad(params, ur_w, pad_l, pad_r, kw_lable);
            sub(aux_reg_input, sizeof(float) * kw * inp_mult);
            add(aux_reg_input, sizeof(float) * iw * inp_mult);
        }

        dec(kj);
        cmp(kj, 0);
        jg(kh_lable, T_NEAR);
    }

    for (int ii = 0; ii < nb_oc_block; ii++)
        for (int jj = 0; jj < ur_w; jj++)
            vmovups(YWORD[reg_output
                    + sizeof(float) * (ii * oh * ow + jj) * oc_blk],
                    Ymm(ur_w * ii + jj));
}

void jit_avx2_conv_generator_f32::generate() {
    auto params = &this->jcp;
    using Xbyak::Ymm;
    this->preamble();

    mov(reg_input, ptr[this->param1]);
    mov(reg_output, ptr[this->param1 + 8]);
    mov(reg_kernel, ptr[this->param1 + 16]);
    mov(reg_kh, ptr[this->param1 + 48]);

    // NB: works only for params->ur_w == 3 && params->nb_oc % 4 == 0
    int ur_w = params->ur_w;
    int ur_w_tail = params->ur_w_tail;
    int n_oi = params->ow / ur_w;
    int iw = params->iw;
    int kw = params->kw;
    int ic_blk = params->ic_block;
    int oc_blk = params->oc_block;
    int str_w = params->stride_w;
    const int inp_mult = _src_in_nchw ? 1 : ic_blk;

    xor_(oi_iter, oi_iter);
    int l_pad = params->l_pad;
    int r_pad = nstl::max(0, (int(params->ow) - 1) * str_w + kw - 1
            - (iw + l_pad - 1));
    int r_pad1 = 0;
    if (l_pad > 0) {
        width_blk_step(params, ur_w, l_pad, 0, ".kh_loop_oimain_padwl",
                ".kw_loop_oimain_padwl");
        add(reg_input, sizeof(float) * (ur_w * str_w - l_pad) * inp_mult);
        add(reg_output, sizeof(float) * ur_w * oc_blk);
        inc(oi_iter);

        r_pad1 = (ur_w * n_oi - 1) * str_w + kw - 1 - (iw + l_pad - 1);
        if (r_pad1 > 0) n_oi--;
    }

    if ((l_pad <= 0 && n_oi > 0) || (l_pad > 0 && n_oi > 1)) {
        L(".ow_loop");

        width_blk_step(params, ur_w, 0, 0, ".kh_loop_oimain",
                ".kw_loop_oimain");
        add(reg_input, sizeof(float) * ur_w * str_w * inp_mult);
        add(reg_output, sizeof(float) * ur_w * oc_blk);

        inc(oi_iter);
        cmp(oi_iter, n_oi);
        jl(".ow_loop", T_NEAR);

        L(".ow_loop_end");
    }

    if (r_pad1 > 0) {
        width_blk_step(params, ur_w, 0, r_pad1, ".kh_loop_oimain_padwr",
                ".kw_loop_oimain_padwr");
        add(reg_input, sizeof(float) * ur_w * str_w * inp_mult);
        add(reg_output, sizeof(float) * ur_w * oc_blk);
    }

    if (ur_w_tail != 0)
        width_blk_step(params, ur_w_tail, 0, r_pad, ".kh_loop_oitail",
                ".kw_loop_oitail");

    this->postamble();
}

void jit_avx2_conv_generator_f32::init_jit_params(
        const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d)
{
    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    const uint32_t w_idx_base = with_groups ? 1 : 0;
    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
    jcp.ic = weights_d.dims()[w_idx_base + 1];
    jcp.oc = weights_d.dims()[w_idx_base + 0];

    jcp.ih = src_d.dims()[2]; jcp.iw = src_d.dims()[3];
    jcp.oh = dst_d.dims()[2]; jcp.ow = dst_d.dims()[3];

    jcp.t_pad = cd.padding[0];
    jcp.l_pad = cd.padding[1];
    jcp.kh = weights_d.dims()[w_idx_base + 2];
    jcp.kw = weights_d.dims()[w_idx_base + 3];
    jcp.stride_h = cd.strides[0];
    jcp.stride_w = cd.strides[1];

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
}

jit_avx2_conv_generator_f32::jit_avx2_conv_generator_f32(
        const convolution_primitive_desc_t &cpd, void *code_ptr,
        size_t code_size)
    : jit_generator(code_ptr, code_size)
    , _src_in_nchw(cpd.src_primitive_desc.memory_desc.format == nchw)
{
    this->init_jit_params(cpd.convolution_desc, cpd.src_primitive_desc,
            cpd.weights_primitive_desc, cpd.dst_primitive_desc);
    this->generate();
    jit_ker = (void (*)(void*))this->getCode();
    //TODO: if(jit_ker == nullptr) return nullptr;
}

bool jit_avx2_conv_generator_f32::is_applicable(
        const convolution_desc_t &conv_d)
{
    const memory_desc_wrapper src_d(conv_d.src_desc),
          weights_d(conv_d.weights_desc), dst_d(conv_d.dst_desc);

    const bool flat = src_d.dims()[1] == 3;
    const bool mimo = !flat;
    const bool with_groups = weights_d.ndims() == (src_d.ndims() + 1);

    bool args_ok = true
        && implication(flat, one_of(src_d.format(), nchw, nhwc))
        && implication(mimo, src_d.format() == nChw8c)
        && weights_d.format() ==
                (with_groups ? gOIhw8i8o : (flat ? Ohwi8o : OIhw8i8o))
        && one_of(conv_d.bias_desc.format, memory_format::undef, x)
        && dst_d.format() == nChw8c;
    if (!args_ok) return false;

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
    if (!args_ok) return false;

    if (mimo) {
        int r_pad_step0 = nstl::max(0,
                ((ow == ur_w_tail ? ur_w_tail : ur_w) - 1)
                + (kw - 1) - (iw + l_pad - 1));
        int r_pad_no_tail = nstl::max(0, (ow - ur_w_tail - 1)
                + (kw - 1) - (iw + l_pad - 1));

        /* no steps with both left and right padding so far */
        if (l_pad > 0 && r_pad_step0 > 0) return false;

        /* maximum 1 ur_w block with r_pad so far */
        if (r_pad_no_tail > ur_w) return false;
    }

    return true;
}

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
