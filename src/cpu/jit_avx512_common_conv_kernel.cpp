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

#include "c_types_map.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "jit_avx512_common_conv_kernel.hpp"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

namespace {

constexpr auto small_spatial = 14;

inline void pick_loop_order(jit_conv_conf_t &jcp) {
    using namespace prop_kind;
    assert(one_of(jcp.prop_kind,
                forward_training, forward_inference, backward_data));
    auto w = (jcp.prop_kind == backward_data) ? jcp.iw : jcp.ow;
    auto h = (jcp.prop_kind == backward_data) ? jcp.ih : jcp.oh;
    switch (jcp.ver) {
    case ver_fma:
        jcp.loop_order = loop_cgn;
    case ver_4vnni:
    case ver_4fma:
        jcp.loop_order
            = (w <= small_spatial && h <= small_spatial) ? loop_cgn : loop_gnc;
        break;
    default:
        assert(!"unsupported convolution version");
    }
}

}

void jit_avx512_common_conv_fwd_kernel::prepare_output(int ur_w)
{
    for (int k = 0; k < jcp.nb_oc_blocking; k++)
        for (int j = 0; j < ur_w; j++) {
            Zmm zmm = zmm_out(j, k);
            vpxord(zmm, zmm, zmm);
            int aux_output_offset = get_output_offset(j, k);
            mic_prefetcht1(EVEX_compress_addr(reg_out_prf, aux_output_offset));
        }
}

void jit_avx512_common_conv_fwd_kernel::store_output(int ur_w)
{
    Label no_update_label, store_label, relu_label;

    mov(reg_channel, ptr[param1 + GET_OFF(channel)]);
    if (jcp.with_bias) {
        mov(reg_bias, ptr[param1 + GET_OFF(bias)]);
    }
    cmp(reg_channel, 0);
    je(no_update_label, T_NEAR);

    for (int k = 0; k < jcp.nb_oc_blocking; k++)
        for (int j = 0; j < ur_w; j++) {
            Zmm zmm = zmm_out(j, k);
            int aux_output_offset = get_output_offset(j, k);
            vadd(zmm, reg_out, aux_output_offset);
        }
    jmp(relu_label, T_NEAR);

    L(no_update_label);
    if (jcp.with_bias) {
        for (int k = 0; k < jcp.nb_oc_blocking; k++) {
            int bias_offset = jcp.typesize_out * k * jcp.oc_block;
            for (int j = 0; j < ur_w; j++) {
                Zmm zmm = zmm_out(j, k);
                vadd(zmm, reg_bias, bias_offset);
            }
            mic_prefetcht1(EVEX_compress_addr(reg_bias, bias_offset + 64));
        }
    }

    L(relu_label);
    if (jcp.with_relu) {
        vpxord(zmm_zero, zmm_zero, zmm_zero);
        if (jcp.relu_negative_slope == 0 || jcp.ver == ver_4vnni) {
            zmm_relu_ns = zmm_zero;
        } else {
            mov(reg_relu_ns,
                reinterpret_cast<size_t>(&jcp.relu_negative_slope));
            vbroadcastss(zmm_relu_ns, ptr[reg_relu_ns]);
        }
        cmp(reg_channel, jcp.nb_ic - 1);
        jl(store_label, T_NEAR);
        const unsigned char _cmp_lt_os = 1;
        for (int k = 0; k < jcp.nb_oc_blocking; k++)
            for (int j = 0; j < ur_w; j++){
                Opmask kmask = Opmask(7);
                Zmm zmm = zmm_out(j, k);
                vcmp(kmask, zmm, zmm_zero, _cmp_lt_os);
                vmul(zmm, kmask, zmm, zmm_relu_ns);
            }
    }

    L(store_label);
    for (int k = 0; k < jcp.nb_oc_blocking; k++)
        for (int j = 0; j < ur_w; j++) {
            Zmm zmm = zmm_out(j, k);
            int aux_output_offset
                = typesize * (k * jcp.oh * jcp.ow + j) * jcp.oc_block;
            vmovups(EVEX_compress_addr(reg_out, aux_output_offset), zmm);
            prefetcht0(EVEX_compress_addr(reg_out_prf, aux_output_offset));
        }
}

void jit_avx512_common_conv_fwd_kernel::compute_loop_4fma(int ur_w,
        int pad_l, int pad_r)
{
    int iw = jcp.iw;
    int kw = jcp.kw;
    int stride_w = jcp.stride_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    Label kh_label;

    assert(jcp.oc % jcp.nb_oc_blocking == 0);

    mov(aux_reg_inp, reg_inp);
    mov(aux_reg_ker, reg_ker);
    mov(aux_reg_ker_prf, reg_ker_prf);
    mov(aux_reg_inp_prf, reg_inp_prf);

    auto kernel_offset = [=](int ocb, int ic, int ki) {
        int blk_idx = ocb * jcp.nb_ic * jcp.kh * jcp.kw + ki;
        int blk_offset = blk_idx * jcp.oc_block * jcp.ic_block;
        int ic_offset = ic * jcp.oc_block;
        return typesize * (blk_offset + ic_offset);
    };

    prepare_output(ur_w);

    mov(reg_kj, reg_kh);
    L(kh_label); {
        for (int ki = 0; ki < kw; ki++)
        for (int ic = 0; ic < ic_block; ic += 4)
        for (int kk = 0; kk < jcp.nb_oc_blocking; kk++) {
            for (int ii = 0; ii < 4; ii++) {
                int aux_kernel_offset = kernel_offset(kk, ic + ii, ki);
                vmovups(zmm_ker(ii),
                        EVEX_compress_addr(aux_reg_ker, aux_kernel_offset));
            }

            int jj_start = nstl::max(0,
                    (pad_l - ki + stride_w - 1) / stride_w);
            int jj_end = ur_w - nstl::max(0,
                    (ki + pad_r - (kw - 1) + stride_w - 1) / stride_w);
            for (int jj = jj_start, prf_count = 0; jj  < jj_end; jj++) {
                int aux_input_offset =  typesize
                    * ((ki + jj * stride_w -pad_l) * ic_block + ic);
                v4fmaddps(zmm_out(jj, kk), zmm_ker(0),
                        EVEX_compress_addr(aux_reg_inp,aux_input_offset));
                if ((jj % 2) && (prf_count < 4)) {
                    int aux_kernel_prf = kernel_offset(kk, ic + prf_count, ki);
                    mic_prefetcht0(EVEX_compress_addr(
                                aux_reg_ker_prf, aux_kernel_prf));
                    prf_count++;
                }
                if (!(jj % 2) && ki == 0 && ic == 0 && kk == 0) {
                    mic_prefetcht1(EVEX_compress_addr(aux_reg_inp_prf,
                        aux_input_offset));
                }
                if (!(jj % 2) && ki == 1 && ic == 0 && kk == 0) {
                    mic_prefetcht0(EVEX_compress_addr(aux_reg_inp,
                        aux_input_offset + typesize * iw * ic_block));
                }
            }
        }

        add(aux_reg_ker, typesize * kw * oc_block * ic_block);
        add(aux_reg_inp, typesize * iw * ic_block);
        add(aux_reg_ker_prf, typesize * kw * oc_block * ic_block);
        add(aux_reg_inp_prf, typesize * iw * ic_block);

        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }

    store_output(ur_w);
}

void jit_avx512_common_conv_fwd_kernel::compute_loop_fma(int ur_w, int pad_l,
        int pad_r)
{
    bool prf_ker = true;
    bool prf_inp = true;
    int iw = jcp.iw;
    int ih = jcp.ih;
    int kw = jcp.kw;
    int stride_w = jcp.stride_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int nb_oc_block = jcp.nb_oc_blocking;
    Label kh_label;

    int ker_pipeline_depth = 4;
    assert(ker_reg_base_idx + ker_pipeline_depth <= 32);
    assert(oc_block >= ker_pipeline_depth);

    int num_ker_loads = ic_block * nb_oc_block * kw;
    const int simd_w = 16;
    int num_ker_prfs = prf_ker ? num_ker_loads : 0;
    int num_inp_prfs = prf_inp ?
            ur_w * nstl::min(kw, stride_w) + nstl::max(0, kw - stride_w) :
            0;
    if (jcp.is_1stconv && prf_inp) {
        num_inp_prfs = div_up(num_inp_prfs, simd_w) * ic_block;
    }
    int num_prfs = num_ker_prfs + num_inp_prfs;
    int num_fmas = num_ker_loads * ur_w;
    int prf_inst_spacing
            = (prf_ker || prf_inp) ? nstl::max(1, num_fmas / num_prfs) : 1;
    int prf_inst_trigger = (num_fmas % prf_inst_spacing) / 2;

    mov(aux_reg_inp, reg_inp);
    mov(aux_reg_ker, reg_ker);

    prepare_output(ur_w);

    mov(aux_reg_inp_prf, reg_inp_prf);
    mov(aux_reg_ker_prf, reg_ker_prf);
    mov(reg_kj, reg_kh);
    align(16);
    L(kh_label);
    {
        int step = 0;
        int ker_prfs = 0;
        for (int ki = 0; ki < kw; ki++) {
            for (int ic = 0; ic < ic_block; ic++) {
                int aux_kernel_offset = 0;
                if (step == 0) {
                    for (int i = 0; i < ker_pipeline_depth; i++) {
                        aux_kernel_offset = get_kernel_offset(ki, ic, 0, i);
                        vmovups(zmm_ker(i), EVEX_compress_addr(
                                        aux_reg_ker, aux_kernel_offset));
                    }
                } else if (step < num_ker_loads - ker_pipeline_depth + 1) {
                    int load_offset = ker_pipeline_depth - 1;
                    int ker_load_reg_idx
                        = (step + load_offset) % ker_pipeline_depth;
                    aux_kernel_offset = get_kernel_offset(ki,ic,0,load_offset);
                    vmovups(zmm_ker(ker_load_reg_idx),
                            EVEX_compress_addr(aux_reg_ker, aux_kernel_offset));
                }

                bool ker_prf_inserted = false;
                Zmm zmm_kernel = zmm_ker(step % ker_pipeline_depth);
                int j_start = get_ow_start(ki, pad_l);
                int j_end = get_ow_end(ki, pad_r);
                for (int j = j_start; j < j_end; j++) {
                    int aux_input_offset = get_input_offset(ki, ic, j, pad_l);
                    vfmadd231ps(zmm_out(j, 0), zmm_kernel,
                            EVEX_compress_addr(
                                aux_reg_inp, aux_input_offset, true));

                    int fma_idx = step * ur_w + j;
                    int prf_slot_idx = fma_idx / prf_inst_spacing;
                    if (fma_idx % prf_inst_spacing == prf_inst_trigger) {
                        if (prf_ker && !ker_prf_inserted
                                && ker_prfs < num_ker_prfs) {
                            int ker_prf_offset
                                    = jcp.typesize_in * ker_prfs * jcp.oc_block;
                            mic_prefetcht2(EVEX_compress_addr(
                                    aux_reg_ker_prf, ker_prf_offset));
                            ker_prf_inserted = true;
                            ker_prfs++;
                        } else if (prf_inp) {
                            int inp_prf_idx = prf_slot_idx - ker_prfs;
                            if (inp_prf_idx < num_inp_prfs) {
                                int inp_prf_stride = nstl::max(kw, stride_w);
                                int inp_prf_offset;
                                if (!jcp.is_1stconv) {
                                    inp_prf_offset
                                            = ic_block * jcp.typesize_in
                                            * ((inp_prf_idx / kw)
                                            * inp_prf_stride
                                            + (inp_prf_idx % kw));
                                } else {
                                    int ic_prf_stride = jcp.typesize_in*iw*ih;
                                    int iw_prf_stride = jcp.typesize_in*simd_w;
                                    inp_prf_offset = ((inp_prf_idx / ic_block)
                                            * iw_prf_stride
                                            + (inp_prf_idx % ic_block)
                                            * ic_prf_stride);
                                }
                                mic_prefetcht0(EVEX_compress_addr(
                                        aux_reg_inp_prf, inp_prf_offset));
                            }
                        }
                    }
                }
                step++;
            }
        }
        add(aux_reg_ker, jcp.typesize_in * kw * oc_block * ic_block);
        if (prf_ker)
            add(aux_reg_ker_prf, jcp.typesize_in * kw * oc_block * ic_block);
        int inp_mul = !jcp.is_1stconv ? ic_block : 1;
        add(aux_reg_inp, jcp.typesize_in * iw * inp_mul);
        if (prf_inp)
            add(aux_reg_inp_prf, jcp.typesize_in * iw * inp_mul);

        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }

    store_output(ur_w);
}

void jit_avx512_common_conv_fwd_kernel::compute_loop_4vnni(
        int ur_w, int pad_l, int pad_r)
{
    Label kh_label;
    const int ker_reg_base_idx = 28;
    const int ker_load_number = 4;
    const int shift_kernel_ptr = jcp.typesize_in * jcp.kw
                               * jcp.oc_block * jcp.ic_block;
    const int shift_input_ptr = jcp.typesize_in * jcp.iw * jcp.ic_block;

    mov(aux_reg_inp, reg_inp);
    mov(aux_reg_ker, reg_ker);
    mov(aux_reg_ker_prf, reg_ker_prf);
    mov(aux_reg_inp_prf, reg_inp_prf);

    prepare_output(ur_w);

    mov(reg_kj, reg_kh);
    L(kh_label); {
        for (int ki = 0; ki < jcp.kw; ki++) {
            for (int ic = 0; ic < jcp.ic_block / 2; ic += 4) {
                for (int kk = 0; kk < jcp.nb_oc_blocking; kk++) {
                    for (int ii = 0; ii < ker_load_number; ii++) {
                        int kernel_offset = get_kernel_offset(ki, ic, kk, ii);
                        vmovups(Zmm(ker_reg_base_idx+ii),
                            EVEX_compress_addr(aux_reg_ker, kernel_offset));
                    }
                    int ow_start = get_ow_start(ki, pad_l);
                    int ow_end = get_ow_end(ki, pad_r);
                    for (int oi = ow_start, prf_count = 0; oi  < ow_end; oi++) {
                        int input_offset = get_input_offset(ki,ic,oi,pad_l);
                        vp4dpwssd(Zmm(ur_w*kk + oi), Zmm(ker_reg_base_idx),
                            EVEX_compress_addr(aux_reg_inp, input_offset));
                        if ((oi % 2) && (prf_count < ker_load_number)) {
                            int kernel_offset = get_kernel_offset(
                                ki, ic, kk, prf_count++);
                            prefetcht0(EVEX_compress_addr(aux_reg_ker_prf,
                                kernel_offset));
                        }
                        if (!(oi % 2) && ki == 0 && ic==0 && kk==0) {
                            prefetcht1(EVEX_compress_addr(aux_reg_inp_prf,
                                input_offset));
                        }
                        if (!(oi % 2) && ki == 1 && ic==0 && kk==0) {
                            prefetcht0(EVEX_compress_addr(aux_reg_inp,
                                input_offset + shift_input_ptr));
                        }
                    }
                }
            }
        }
        add(aux_reg_ker_prf, shift_kernel_ptr);
        add(aux_reg_inp_prf, shift_input_ptr);
        add(aux_reg_ker, shift_kernel_ptr);
        add(aux_reg_inp, shift_input_ptr);

        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }
    store_output(ur_w);
}

void jit_avx512_common_conv_fwd_kernel::compute_loop(int ur_w,
        int pad_l, int pad_r)
{
    if (jcp.ver == ver_4vnni)
        compute_loop_4vnni(ur_w, pad_l, pad_r);
    else if (jcp.ver == ver_4fma)
        compute_loop_4fma(ur_w, pad_l, pad_r);
    else if (jcp.ver == ver_fma)
        compute_loop_fma(ur_w, pad_l, pad_r);
    else
        assert(!"unknown convolution version");
}

void jit_avx512_common_conv_fwd_kernel::generate()
{
    int iw = jcp.iw;
    int ow = jcp.ow;
    int kw = jcp.kw;
    int l_pad = jcp.l_pad;
    int ur_w = jcp.ur_w;
    int ur_w_tail = jcp.ur_w_tail;
    int stride_w = jcp.stride_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;

    int inp_mult = !jcp.is_1stconv ? ic_block : 1;
    int inp_shift_pad = jcp.typesize_in * (ur_w * stride_w - l_pad) * inp_mult;
    int inp_shift = jcp.typesize_in * (ur_w * stride_w * inp_mult);
    int out_shift = jcp.typesize_out * (ur_w * oc_block);

    preamble();
    mov(reg_inp, ptr[param1 + GET_OFF(src)]);
    mov(reg_out, ptr[param1 + GET_OFF(dst)]);
    mov(reg_ker, ptr[param1 + GET_OFF(filt)]);
    mov(reg_ker_prf, ptr[param1 + GET_OFF(filt_prf)]);
    mov(reg_kh, ptr[param1 + GET_OFF(kh_padding)]);

    int r_pad = nstl::max(0, (ow - 1) * stride_w + (kw - 1) - (iw + l_pad - 1));
    if (ow == ur_w) {
        mov(reg_inp_prf, ptr[param1 + GET_OFF(src_prf)]);
        mov(reg_out_prf, ptr[param1 + GET_OFF(dst_prf)]);
        compute_loop(ur_w, l_pad, r_pad);
    } else {
        mov(reg_inp_prf, reg_inp);
        mov(reg_out_prf, reg_out);
        int n_oi = ow / ur_w;

        int r_pad1 = (ur_w * n_oi - 1) * stride_w + kw - 1 - (iw + l_pad - 1);
        xor_(reg_oi, reg_oi);
        if (l_pad > 0) {
            add(reg_inp_prf, inp_shift_pad);
            add(reg_out_prf, out_shift);
            compute_loop(ur_w, l_pad, 0);
            add(reg_inp, inp_shift_pad);
            add(reg_out, out_shift);
            inc(reg_oi);

            if (r_pad1 > 0)
                n_oi--;
        }
        if ((l_pad <= 0 && n_oi > 0) || (l_pad > 0 && n_oi > 1)) {
            if (l_pad <= 0 && r_pad1 > 0)
                n_oi--;
            Label ow_loop_label;
            L(ow_loop_label);
            {
                add(reg_inp_prf, inp_shift);
                add(reg_out_prf, out_shift);
                compute_loop(ur_w, 0, 0);
                add(reg_inp, inp_shift);
                add(reg_out, out_shift);
                inc(reg_oi);
                cmp(reg_oi, n_oi);
                jl(ow_loop_label, T_NEAR);
            }
        }
        if (r_pad1 > 0) {
            add(reg_inp_prf, inp_shift);
            add(reg_out_prf, out_shift);
            compute_loop(ur_w, 0, r_pad1);
            add(reg_inp, inp_shift);
            add(reg_out, out_shift);
        }
        if (ur_w_tail != 0) {
            add(reg_inp_prf, inp_shift);
            add(reg_out_prf, out_shift);
            compute_loop(ur_w_tail, 0, r_pad);
        }
    }

    postamble();
}

status_t jit_avx512_common_conv_fwd_kernel::init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d,
        bool with_relu, double relu_negative_slope)
{
    using namespace prop_kind;

    const int simd_w = 16;
    const int regs = 28;
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
    jcp.src_fmt = src_d.format();
    jcp.with_bias = cd.bias_desc.format != memory_format::undef;
    jcp.with_relu = with_relu;
    jcp.relu_negative_slope = relu_negative_slope;
    jcp.ur_h = 1;

    jcp.dilate_h = cd.dilates[0];
    jcp.dilate_w = cd.dilates[1];
    if (jcp.dilate_h != 0 || jcp.dilate_w != 0)
        return status::unimplemented;

    // TODO: simplify
    if (jcp.ic % simd_w != 0) {
        if ((jcp.ic == 3 || jcp.ic == 1) && jcp.src_fmt == nchw)
            jcp.is_1stconv = true;
        else
            return status::unimplemented;
    } else
        jcp.is_1stconv = false;

    const bool flat = jcp.ic == 3;
    const bool mimo = !flat;
    if (mayiuse(avx512_mic_4ops) &&
            src_d.data_type() == data_type::s16
         && weights_d.data_type() == data_type::s16
         && dst_d.data_type() == data_type::s32)
    {
        bool args_ok = true
                && mimo // TODO: add support of first convolution
                && src_d.format() == nChw16c
                && weights_d.format() == (with_groups ? gOIhw8i16o2i : OIhw8i16o2i)
                && one_of(cd.bias_desc.format, memory_format::undef, any, x)
                && dst_d.format() == nChw16c;
        if (!args_ok)
            return status::unimplemented;
        jcp.ver = ver_4vnni;
        jcp.typesize_in = sizeof(int16_t);
        jcp.typesize_out = sizeof(int32_t);
    } else if (mayiuse(avx512_common)) {
        bool args_ok = true
            && implication(flat, one_of(src_d.format(), nchw, nhwc)
                    && one_of(weights_d.format(), Ohwi16o, gOhwi16o))
            && implication(mimo, src_d.format() == nChw16c
                    && one_of(weights_d.format(), OIhw16i16o, gOIhw16i16o))
            && one_of(cd.bias_desc.format, memory_format::undef, any, x)
            && dst_d.format() == nChw16c;
        if (!args_ok)
            return status::unimplemented;
        jcp.ver = ver_fma;
        jcp.typesize_in = sizeof(float);
        jcp.typesize_out = sizeof(float);
        if (mayiuse(avx512_mic_4ops) && !jcp.is_1stconv)
            jcp.ver = ver_4fma;
    } else {
            return status::unimplemented;
    }

    for (int ur_w = regs; ur_w > 0; --ur_w) {
        if (jcp.ow % ur_w == 0) {
            jcp.ur_w = ur_w;
            break;
        }
    }

    // TODO (Tanya): currenly applied to Segnet convolutions only.
    // Need to try for other topologies
    if (jcp.ow > 150 && jcp.ur_w < regs/2)
        jcp.ur_w = regs;

    int n_oi = (jcp.ow / jcp.ur_w);
    int r_pad = (jcp.ur_w * n_oi - 1) * jcp.stride_w + jcp.kw - jcp.iw
            - jcp.l_pad;
    if (jcp.l_pad > 0 && r_pad > 0)
        n_oi--;

    bool large_code_size = jcp.ur_w != jcp.ow && jcp.l_pad > 0 && r_pad > 0
            && ((jcp.l_pad <= 0 && n_oi > 0) || (jcp.l_pad > 0 && n_oi > 1));
    if (large_code_size) {
        const int max_code_size = 24 * 1024;
        const int num_ops_per_reg = 6 + jcp.ic_block * jcp.kw;
        int mult = 1;
        if (jcp.l_pad > 0) mult += 1;
        if (r_pad > 0) mult += 1;
        for (int ur_w = jcp.ur_w; ur_w > regs/2; --ur_w) {
            if (ur_w * mult * num_ops_per_reg * 9.0 < max_code_size) {
                jcp.ur_w = ur_w;
                break;
            }
        }
    }
    jcp.ur_w_tail = jcp.ow % jcp.ur_w;

    bool args_ok = true
        && jcp.oc % simd_w == 0
        && jcp.l_pad <= jcp.ur_w
        && implication(mimo, jcp.ic % simd_w == 0);
    if (!args_ok)
        return status::unimplemented;

    int r_pad_no_tail = nstl::max(0, (jcp.ow - jcp.ur_w_tail - 1) * jcp.stride_w
                    + jcp.kw - jcp.iw - jcp.l_pad);
    if (r_pad_no_tail > jcp.ur_w)
        return status::unimplemented;

    jcp.oc_block = simd_w;
    jcp.ic_block = (jcp.ic % simd_w != 0) ? jcp.ic : simd_w;
    jcp.nb_ic = jcp.ic / jcp.ic_block;
    jcp.nb_oc = jcp.oc / jcp.oc_block;
    jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;
    if (one_of(jcp.ver, ver_4vnni, ver_4fma))
        for (int i = jcp.nb_oc; i > 0; i--)
            if (i * jcp.ur_w <= regs && jcp.nb_oc % i == 0) {
                jcp.nb_oc_blocking = i;
                break;
            }

    pick_loop_order(jcp);

    return status::success;
}

void jit_avx512_common_conv_bwd_data_kernel_f32::prepare_output(int ur_w)
{
    for (int k = 0; k < jcp.nb_ic_blocking; k++) {
        for (int j = 0; j  < ur_w; j++) {
            Zmm zmm = zmm_out(j, k);
            vpxord(zmm, zmm, zmm);
            int aux_src_offset
                = typesize * (k * jcp.ih * jcp.iw + j) * jcp.ic_block;
            prefetcht1(EVEX_compress_addr(reg_src_prf, aux_src_offset));
        }
    }
}

void jit_avx512_common_conv_bwd_data_kernel_f32::store_output(int ur_w)
{
    Label no_update_label;

    mov(reg_channel, ptr[param + GET_OFF(channel)]);
    cmp(reg_channel, 0);
    je(no_update_label, T_NEAR);
    for (int k = 0; k < jcp.nb_ic_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            Zmm zmm = zmm_out(j, k);
            int aux_src_offset
                = typesize * (k * jcp.ih * jcp.iw + j) * jcp.ic_block;
            vadd(zmm, reg_src, aux_src_offset);
        }
    }

    L(no_update_label);
    for (int k = 0; k < jcp.nb_ic_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            Zmm zmm = zmm_out(j, k);
            int aux_src_offset
                = typesize * (k * jcp.ih * jcp.iw + j) * jcp.ic_block;
            vmovups(EVEX_compress_addr(reg_src, aux_src_offset), zmm);
            mic_prefetcht0(EVEX_compress_addr(reg_src_prf, aux_src_offset));
        }
    }
}

void jit_avx512_common_conv_bwd_data_kernel_f32::compute_loop_4fma(int ur_w,
        int l_overflow, int r_overflow)
{
    int ow = jcp.ow;
    int kw = jcp.kw;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    Label kh_label;

    mov(aux_reg_dst, reg_dst);
    mov(aux_reg_ker, reg_ker);
    mov(aux_reg_dst_prf, reg_dst_prf);
    mov(aux_reg_ker_prf, reg_ker_prf);

    auto kernel_offset = [=](int icb, int oc, int ki) {
        int blk_idx = icb * jcp.kh * jcp.kw + ki;
        int blk_offset = blk_idx * jcp.oc_block * jcp.ic_block;
        int oc_offset = oc * jcp.oc_block;
        return typesize * (blk_offset + oc_offset);
    };

    prepare_output(ur_w);

    mov(reg_kj, reg_kh);
    L(kh_label); {
        for (int ki = 0; ki < kw; ki++)
        for (int oc = 0; oc < oc_block; oc += 4)
        for (int kk = 0; kk < jcp.nb_ic_blocking; kk++) {
            for (int ii = 0; ii < 4; ii++) {
                int aux_kernel_offset = kernel_offset(kk, oc + ii, ki);
                vmovups(zmm_ker(ii),
                        EVEX_compress_addr(aux_reg_ker, aux_kernel_offset));
            }

            int jj_start = nstl::max(0, l_overflow - (kw - 1) + ki);
            int jj_end = ur_w - nstl::max(0, r_overflow - ki);
            for (int jj = jj_start, prf_count = 0; jj  < jj_end; jj++) {
                int aux_dst_offset = typesize
                    * ((jj + jcp.l_pad - ki) * oc_block + oc);
                v4fmaddps(zmm_out(jj, kk), zmm_ker(0),
                        EVEX_compress_addr(aux_reg_dst, aux_dst_offset));

                if ((jj % 2) && (prf_count < 4)) {
                    int aux_kernel_prf = kernel_offset(kk, oc + prf_count, ki);
                    mic_prefetcht1(EVEX_compress_addr(
                        aux_reg_ker_prf, aux_kernel_prf));
                    prf_count++;
                }
                if (!(jj % 2) && ki == 0 && oc == 0 && kk == 0) {
                    mic_prefetcht1(EVEX_compress_addr(aux_reg_dst_prf,
                        aux_dst_offset));
                }
                if (!(jj % 2) && ki == 1 && oc == 0 && kk == 0) {
                    mic_prefetcht0(EVEX_compress_addr(aux_reg_dst,
                        aux_dst_offset + typesize * ow * oc_block));
                }
            }
        }

        add(aux_reg_ker, typesize * kw * oc_block * ic_block);
        sub(aux_reg_dst, typesize * ow * oc_block);
        add(aux_reg_ker_prf, typesize * kw * oc_block * ic_block);
        sub(aux_reg_dst_prf, typesize * ow * oc_block);

        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }

    store_output(ur_w);
}

void jit_avx512_common_conv_bwd_data_kernel_f32::compute_loop_4vnni(int ur_w,
        int l_overflow, int r_overflow)
{
    int ow = jcp.ow;
    int kw = jcp.kw;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    Label kh_label;

    mov(aux_reg_dst, reg_dst);
    mov(aux_reg_ker, reg_ker);
    mov(aux_reg_dst_prf, reg_dst_prf);
    mov(aux_reg_ker_prf, reg_ker_prf);

    auto kernel_offset = [=](int icb, int oc, int ki) {
        int blk_idx = icb * jcp.kh * jcp.kw + ki;
        int blk_offset = blk_idx * jcp.oc_block * jcp.ic_block;
        int oc_offset = oc * jcp.oc_block;
        return jcp.typesize_in * (blk_offset + oc_offset);
    };

    prepare_output(ur_w);

    mov(reg_kj, reg_kh);
    L(kh_label); {
        for (int ki = 0; ki < kw; ki++)
        for (int oc = 0; oc < oc_block / 2; oc += 4)
        for (int kk = 0; kk < jcp.nb_ic_blocking; kk++) {
            for (int ii = 0; ii < 4; ii++) {
                int aux_kernel_offset = kernel_offset(kk, 2 * (oc + ii), ki);
                vmovups(zmm_ker(ii),
                        EVEX_compress_addr(aux_reg_ker, aux_kernel_offset));
            }

            int jj_start = nstl::max(0, l_overflow - (kw - 1) + ki);
            int jj_end = ur_w - nstl::max(0, r_overflow - ki);
            for (int jj = jj_start, prf_count = 0; jj  < jj_end; jj++) {
                int aux_dst_offset = jcp.typesize_in
                    * ((jj + jcp.l_pad - ki) * oc_block + 2 * oc);
                vp4dpwssd(zmm_out(jj, kk), zmm_ker(0),
                        EVEX_compress_addr(aux_reg_dst, aux_dst_offset));

                if ((jj % 2) && (prf_count < 4)) {
                    int aux_kernel_prf = kernel_offset(kk, oc + prf_count, ki);
                    mic_prefetcht1(EVEX_compress_addr(
                        aux_reg_ker_prf, aux_kernel_prf));
                    prf_count++;
                }
                if (!(jj % 2) && ki == 0 && oc == 0 && kk == 0) {
                    mic_prefetcht1(EVEX_compress_addr(aux_reg_dst_prf,
                        aux_dst_offset));
                }
                if (!(jj % 2) && ki == 1 && oc == 0 && kk == 0) {
                    mic_prefetcht0(EVEX_compress_addr(aux_reg_dst,
                        aux_dst_offset + jcp.typesize_in * ow * oc_block));
                }
            }
        }

        add(aux_reg_ker, jcp.typesize_in * kw * oc_block * ic_block);
        sub(aux_reg_dst, jcp.typesize_in * ow * oc_block);
        add(aux_reg_ker_prf, jcp.typesize_in * kw * oc_block * ic_block);
        sub(aux_reg_dst_prf, jcp.typesize_in * ow * oc_block);

        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }

    store_output(ur_w);
}

void jit_avx512_common_conv_bwd_data_kernel_f32::compute_loop_fma(int ur_w,
        int l_overflow, int r_overflow)
{
    Label kh_label;
    int kw    = jcp.kw;
    int ow    = jcp.ow;

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int l_pad    = jcp.l_pad;
    int stride_w = jcp.stride_w;

    int ker_pipeline_depth = 4;
    assert(ker_reg_base_idx + ker_pipeline_depth <= 32);
    assert(oc_block >= ker_pipeline_depth);

    int num_ker_loads = oc_block * kw;
    int num_inp_prfs = ur_w * nstl::min(kw, stride_w)
                       + nstl::max(0, kw - stride_w);
    int num_prfs = num_ker_loads + num_inp_prfs;
    int num_fmas = num_ker_loads * ur_w;
    int prf_inst_spacing = nstl::max(1, num_fmas / num_prfs);
    int prf_inst_trigger = (num_fmas % prf_inst_spacing) / 2;

    prepare_output(ur_w);

    mov(aux_reg_dst, reg_dst);
    mov(aux_reg_ker, reg_ker);

    mov(aux_reg_dst_prf, reg_dst_prf);
    mov(aux_reg_ker_prf, reg_ker_prf);

    mov(reg_kj, reg_kh);
    L(kh_label); {
        int step = 0;
        int ker_prfs = 0;
        for (int ki = 0; ki < kw; ki++) {
            for (int oc = 0; oc < oc_block; oc++) {
                if (step == 0) {
                    for (int i = 0; i < ker_pipeline_depth; i++) {
                        int aux_kernel_offset = typesize * ((oc + i) * oc_block
                                + ki * ic_block * oc_block);
                        vmovups(zmm_ker(i), EVEX_compress_addr(
                                    aux_reg_ker, aux_kernel_offset));
                    }
                } else if (step < num_ker_loads - ker_pipeline_depth + 1) {
                    int load_offset = ker_pipeline_depth - 1;
                    int ker_load_reg_idx
                        = (step + load_offset) % ker_pipeline_depth;
                    int aux_kernel_offset = typesize * ((oc + load_offset)
                            * oc_block + ki * ic_block * oc_block);
                    vmovups(zmm_ker(ker_load_reg_idx),
                            EVEX_compress_addr(aux_reg_ker, aux_kernel_offset));
                }

                bool ker_prf_inserted = false;
                auto zmm_kernel = zmm_ker(step % ker_pipeline_depth);

                int jj_start = nstl::max(0, l_overflow - (kw - 1) + ki);
                int jj_end   = ur_w - nstl::max(0, r_overflow - ki);
                for (int jj = jj_start; jj  < jj_end; jj++) {
                    int aux_dst_offset = typesize * ((jj + l_pad - ki)
                            * jcp.oc_block + oc);
                    vfmadd231ps(zmm_out(jj, 0), zmm_kernel,
                        EVEX_compress_addr(aux_reg_dst, aux_dst_offset, true));

                    int fma_idx = step * ur_w + jj;
                    int prf_slot_idx = fma_idx / prf_inst_spacing;
                    if (fma_idx % prf_inst_spacing == prf_inst_trigger) {
                        if (!ker_prf_inserted && ker_prfs < num_ker_loads) {
                            int ker_prf_offset = typesize
                                * ker_prfs * jcp.oc_block;
                            mic_prefetcht1(EVEX_compress_addr(
                                        aux_reg_ker_prf, ker_prf_offset));
                            ker_prf_inserted = true;
                            ker_prfs++;
                        } else {
                            int inp_prf_idx = prf_slot_idx - ker_prfs;
                            if (inp_prf_idx < num_inp_prfs) {
                                int inp_prf_offset
                                    = ic_block * typesize
                                    * ((inp_prf_idx / kw) * kw
                                            + (inp_prf_idx % kw));
                                mic_prefetcht0(EVEX_compress_addr(
                                            aux_reg_dst_prf, inp_prf_offset));
                            }
                        }
                    }
                }
                step++;
            }
        }

        add(aux_reg_ker, typesize * kw * oc_block * ic_block);
        sub(aux_reg_dst, typesize * ow * oc_block);
        add(aux_reg_ker_prf, typesize * kw * oc_block * ic_block);
        sub(aux_reg_dst_prf, typesize * ow * oc_block);

        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }

    store_output(ur_w);
}

inline void jit_avx512_common_conv_bwd_data_kernel_f32::compute_loop(int ur_w,
        int l_overflow, int r_overflow)
{
    if (jcp.ver == ver_4vnni)
        compute_loop_4vnni(ur_w, l_overflow, r_overflow);
    else if (jcp.ver == ver_4fma)
        compute_loop_4fma(ur_w, l_overflow, r_overflow);
    else if (jcp.ver == ver_fma)
        compute_loop_fma(ur_w, l_overflow, r_overflow);
    else
        assert("!unknown convolution version");
}

void jit_avx512_common_conv_bwd_data_kernel_f32::generate()
{
    int iw    = jcp.iw;
    int ow    = jcp.ow;
    int kw    = jcp.kw;
    int l_pad = jcp.l_pad;
    int ur_w      = jcp.ur_w;
    int ic_block  = jcp.ic_block;
    int oc_block  = jcp.oc_block;
    int ur_w_tail = jcp.ur_w_tail;

    int dst_shift = jcp.typesize_in * ur_w * ic_block;
    int src_shift = jcp.typesize_out * ur_w * oc_block;

    preamble();

    mov(reg_src, ptr[param + GET_OFF(src)]);
    mov(reg_dst, ptr[param + GET_OFF(dst)]);
    mov(reg_ker, ptr[param + GET_OFF(filt)]);

    mov(reg_kh, ptr[param + GET_OFF(kh_padding)]);
    mov(reg_src_prf, ptr[param + GET_OFF(src_prf)]);
    mov(reg_dst_prf, ptr[param + GET_OFF(dst_prf)]);
    mov(reg_ker_prf, ptr[param + GET_OFF(filt_prf)]);

    int l_overflow = nstl::max(0, ((kw - 1) - l_pad));
    int r_pad      = nstl::max(0, ((ow - 1) + kw - iw - l_pad));
    int r_overflow = nstl::max(0, ((kw - 1) - r_pad));
    int n_oi = iw / ur_w;
    int r_overflow1 = nstl::max(0, (kw-1) - (iw - ur_w * n_oi) - r_pad);
    if (r_overflow1 > 0) n_oi--;

    if (ur_w == iw) {
        compute_loop(ur_w, l_overflow, r_overflow);
    } else if (n_oi == 0) {
        compute_loop(ur_w, l_overflow, r_overflow1);
        add(reg_src, src_shift);
        add(reg_dst, dst_shift);
        add(reg_src_prf, src_shift);
        add(reg_dst_prf, dst_shift);
        if (ur_w_tail != 0)
            compute_loop(ur_w_tail, 0, r_overflow);
    } else {
        xor_(reg_oi, reg_oi);
        if (l_overflow > 0) {
            compute_loop(ur_w, l_overflow, 0);
            add(reg_src, src_shift);
            add(reg_dst, dst_shift);
            add(reg_src_prf, src_shift);
            add(reg_dst_prf, dst_shift);

            inc(reg_oi);
        }
        if ((l_overflow <= 0 && n_oi > 0)
            || (l_overflow >  0 && n_oi > 1)) {
            Label ow_loop_label;
            L(ow_loop_label); {
                compute_loop(ur_w, 0, 0);
                add(reg_src, src_shift);
                add(reg_dst, dst_shift);
                add(reg_src_prf, src_shift);
                add(reg_dst_prf, dst_shift);

                inc(reg_oi);
                cmp(reg_oi, n_oi);
                jl(ow_loop_label, T_NEAR);
            }
        }
        if (r_overflow1 > 0) {
            compute_loop(ur_w, 0, r_overflow1);
            add(reg_src, src_shift);
            add(reg_dst, dst_shift);
            add(reg_src_prf, src_shift);
            add(reg_dst_prf, dst_shift);
        }
        if (ur_w_tail != 0) {
            compute_loop(ur_w_tail, 0, r_overflow);
        }
    }

    postamble();
}

status_t jit_avx512_common_conv_bwd_data_kernel_f32::init_conf(
        jit_conv_conf_t &jcp,
        const convolution_desc_t &cd,
        const memory_desc_wrapper &diff_src_d,
        const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &diff_dst_d)
{
    if (!mayiuse(avx512_common)) return status::unimplemented;

    const bool with_groups = weights_d.ndims() == diff_src_d.ndims() + 1;

    jcp.prop_kind = cd.prop_kind;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = diff_src_d.dims()[0];

    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.ic = diff_src_d.dims()[1] / jcp.ngroups;

    jcp.ih = diff_src_d.dims()[2];
    jcp.iw = diff_src_d.dims()[3];
    jcp.oh = diff_dst_d.dims()[2];
    jcp.ow = diff_dst_d.dims()[3];

    jcp.kh = weights_d.dims()[with_groups + 2];
    jcp.kw = weights_d.dims()[with_groups + 3];

    jcp.t_pad = cd.padding[0][0];
    jcp.l_pad = cd.padding[0][1];

    jcp.stride_h = cd.strides[0];
    jcp.stride_w = cd.strides[1];
    if (jcp.stride_w != jcp.stride_h)
        return status::unimplemented;

    jcp.dilate_h = cd.dilates[0];
    jcp.dilate_w = cd.dilates[1];
    if (jcp.dilate_h != 0 || jcp.dilate_w != 0)
        return status::unimplemented;

    jcp.r_pad = nstl::max(0, (jcp.ow - 1) * jcp.stride_w + jcp.kw - jcp.iw
                          - jcp.l_pad);
    jcp.b_pad = nstl::max(0, (jcp.oh - 1) * jcp.stride_h + jcp.kh - jcp.ih
                          - jcp.t_pad);
    jcp.ihp = jcp.ih + jcp.t_pad + jcp.b_pad;
    jcp.iwp = jcp.iw + jcp.l_pad + jcp.r_pad;
    jcp.ohp = jcp.oh;
    jcp.owp = jcp.ow;

    bool args_ok = true
        && diff_src_d.format() == nChw16c
        && diff_dst_d.format() == nChw16c;
    if (!args_ok)
        return status::unimplemented;

    const int simd_w = 16;

    // TODO: can we please have a standard way to compute this?
    jcp.is_1stconv = false;
    if (jcp.ic % simd_w != 0 ) {
        if (jcp.ic == 3 || jcp.ic == 1)
            jcp.is_1stconv = true;
        else
            return status::unimplemented;
    }

    jcp.ic_block = (jcp.ic % simd_w) ? jcp.ic : simd_w;
    jcp.nb_ic = jcp.ic / jcp.ic_block;

    if (jcp.is_1stconv) {
        jcp.ic_block = jcp.ic;
        jcp.nb_ic = 1;
    }

    jcp.oc_block = simd_w;
    if (jcp.oc % jcp.oc_block)
        return status::unimplemented;
    jcp.nb_oc = jcp.oc / jcp.oc_block;

    jcp.ur_h = 1;
    jcp.ur_w = 1;

    if (jcp.is_1stconv || jcp.stride_w > 1 || jcp.stride_h > 1)
        return status::unimplemented;

    int regs = 28;
    jcp.ur_w = (jcp.iw <= regs) ? jcp.iw : regs;
    int n_oi  = (jcp.iw / jcp.ur_w);
    int l_overflow  = nstl::max(0, ((jcp.kw-1) - jcp.l_pad));
    int r_overflow1 = nstl::max(0, ((jcp.kw-1) - (jcp.iw - jcp.ur_w * n_oi)
                                - jcp.r_pad));
    if (r_overflow1 > 0) n_oi--;

    if (mayiuse(avx512_mic_4ops) && !jcp.is_1stconv
           && jcp.stride_w == 1 && jcp.stride_h == 1
           && diff_dst_d.data_type() == data_type::s16
           && weights_d.data_type() == data_type::s16
           && diff_src_d.data_type() == data_type::s32) {
        if (weights_d.format() != (with_groups ? gOIhw8o16i2o : OIhw8o16i2o))
            return status::unimplemented;
        jcp.ver = ver_4vnni;
        jcp.typesize_in = sizeof(int16_t);
        jcp.typesize_out = sizeof(int32_t);
    } else if (mayiuse(avx512_common)) {
        if (weights_d.format() != (with_groups ? gOIhw16o16i : OIhw16o16i))
            return status::unimplemented;
        jcp.ver = ver_fma;
        jcp.typesize_in = sizeof(float);
        jcp.typesize_out = sizeof(float);
        if (mayiuse(avx512_mic_4ops) && !jcp.is_1stconv
            && jcp.stride_w == 1 && jcp.stride_h == 1) {
                jcp.ver = ver_4fma;
            }
    }

    jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;
    if (jcp.ver == ver_4fma)
        for (int i = jcp.nb_ic; i > 0; i--)
            if (i * jcp.ur_w <= regs && jcp.nb_ic % i == 0) {
                jcp.nb_ic_blocking = i;
                break;
            }

    jcp.loop_order = loop_gnc;

    bool large_code_size = (jcp.ur_w != jcp.ow)
         && ((l_overflow <= 0 && n_oi > 0) ||(l_overflow >  0 && n_oi > 1))
         && (r_overflow1 > 0) && (l_overflow > 0);
    if (large_code_size) {
        const int max_code_size = 24 * 1024;
        const int num_ops_per_reg = 6 + jcp.oc_block * jcp.kw;
        int mult = 1;
        if (l_overflow > 0) mult += 1;
        if (r_overflow1 > 0) mult += 1;
        for (int ur_w = jcp.ur_w; ur_w > regs/2; --ur_w) {
            if (ur_w * mult * num_ops_per_reg * 9.2 < max_code_size) {
                jcp.ur_w = ur_w;
                break;
            }
        }
    }
    jcp.ur_w_tail = jcp.iw % jcp.ur_w;

    if (l_overflow > jcp.ur_w)
        return status::unimplemented;
    int r_overflow_no_tail = nstl::max(0, jcp.kw - 1 - jcp.ur_w_tail
                                       - jcp.r_pad);
    if (r_overflow_no_tail > jcp.ur_w)
        return status::unimplemented;
    int r_pad_no_tail = nstl::max(0, (jcp.ow - jcp.ur_w_tail - 1) + jcp.kw
                                  - jcp.iw - jcp.l_pad);
    if (r_pad_no_tail > jcp.ur_w)
        return status::unimplemented;

    pick_loop_order(jcp);

    return status::success;
}

const int jit_avx512_common_conv_bwd_weights_kernel_f32::max_ur_w = 28;

void jit_avx512_common_conv_bwd_weights_kernel_f32::oh_step_comeback_pointers()
{
    Label kh_comeback_label;

    mov(kj, reg_kh);
    L(kh_comeback_label); {
        int inp_mult = jcp.is_1stconv ? 1 : jcp.ic_block;
        int iw = jcp.transpose_src ? jcp.tr_iw : jcp.iw;
        sub(reg_input, typesize * iw * inp_mult);
        sub(reg_kernel, typesize * jcp.kw * jcp.ic_block * jcp.oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kh_comeback_label, T_NEAR);
    }
}

void jit_avx512_common_conv_bwd_weights_kernel_f32::compute_ic_block_step_fma(
    int ur_w, int pad_l, int pad_r,
    int ic_block_step, int input_offset, int kernel_offset,
    int output_offset, bool input_wraparound)
{

    int kw  = jcp.kw;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    for (int i_kw = 0; i_kw < kw; i_kw++)
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++)
            vmovups(Zmm(i_kw * ic_block_step + i_ic),
                EVEX_compress_addr(reg_kernel, typesize * (i_kw * ic_block
                + i_ic) * jcp.oc_block + kernel_offset));

    for (int i_ur = 0; i_ur < ur_w; i_ur++) {
        if (i_ur == 0) {
            vmovups(Zmm(kw * ic_block_step + (i_ur + 0) % 4),
                EVEX_compress_addr(reg_output, typesize * (i_ur + 0)
                * oc_block + output_offset));
            if (ur_w > 1) vmovups(Zmm(kw * ic_block_step + (i_ur + 1) % 4),
                EVEX_compress_addr(reg_output, typesize * (i_ur + 1) * oc_block
                + output_offset));
            if (ur_w > 2) vmovups(Zmm(kw * ic_block_step + (i_ur + 2) % 4),
                EVEX_compress_addr(reg_output, typesize * (i_ur + 2) * oc_block
                + output_offset));
            if (ur_w > 3) vmovups(Zmm(kw * ic_block_step + (i_ur + 3) % 4),
                EVEX_compress_addr(reg_output, typesize * (i_ur + 3) * oc_block
                + output_offset));
        } else if (i_ur + 3 < ur_w)
            vmovups(Zmm(kw * ic_block_step + (i_ur + 3) % 4),
                EVEX_compress_addr(reg_output, typesize * (i_ur + 3) * oc_block
                + output_offset));

        for (int  i_kw = 0; i_kw < kw; i_kw++) {
            int i_iw = i_ur * jcp.stride_w + i_kw;
            if (i_iw - pad_l < 0 || i_iw > (ur_w - 1) * jcp.stride_w + kw - 1
                - pad_r) continue;
            for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
                const int i_offset = input_offset + typesize * (jcp.transpose_src
                    ? (i_iw - pad_l + i_ic * jcp.tr_iw)
                    : (jcp.is_1stconv
                        ? (i_iw - pad_l) + i_ic * (jcp.ih * jcp.iw)
                        : (i_iw - pad_l) * ic_block + i_ic));
                vfmadd231ps(Zmm(i_kw * ic_block_step + i_ic),
                    Zmm(kw * ic_block_step + i_ur % 4),
                    EVEX_compress_addr(reg_input, i_offset, true));
            }
        }
    }

    for (int i_kw = 0; i_kw < kw; i_kw++)
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++)
            vmovups(EVEX_compress_addr(reg_kernel, typesize
                * (i_kw * ic_block + i_ic) * jcp.oc_block + kernel_offset),
                Zmm(i_kw * ic_block_step + i_ic));
}

void jit_avx512_common_conv_bwd_weights_kernel_f32::compute_ic_block_step_4fma(
    int ur_w, int pad_l, int pad_r,
    int ic_block_step, int input_offset, int kernel_offset,
    int output_offset, bool input_wraparound)
{
    // TODO: add prefetches to fma version as well

    assert(jcp.transpose_src);

    int kw  = jcp.kw;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;

    auto zmm_ker = [=](int i_kw, int i_ic) {
        return Zmm(i_kw * ic_block_step + i_ic);
    };

    auto ker_addr = [=](int i_kw, int i_ic) {
        size_t local_offset
            = typesize * (i_kw * ic_block + i_ic) * jcp.oc_block;
        return EVEX_compress_addr(reg_kernel, local_offset + kernel_offset);
    };

    auto inp_addr = [=](int i_iw, int i_ic, ptrdiff_t extra_offset = 0) {
        int stride = jcp.tr_iw * (jcp.is_1stconv ? jcp.ih : 1);
        int local_offset = typesize * (i_iw + i_ic * stride);
        return EVEX_compress_addr(reg_input,
                local_offset + input_offset + extra_offset);
    };

    auto zmm_out = [=](int i_iw) {
        // TODO: move reg calc to global member funcs
        const int out_zmm_base_idx = 28;
        return Zmm(out_zmm_base_idx + i_iw % 4);
    };

    auto out_addr = [=](int i_ur) {
        return EVEX_compress_addr(reg_output,
                typesize * i_ur * oc_block + output_offset);
    };

    auto pf_callback = [=](int i_ur, int i_kw, int i_ic) {
        assert(i_ur % 4 == 0);
        if (i_ur == 0)
            prefetcht1(ker_addr(i_kw, i_ic));
        if (i_ur + 4 >= ur_w)
            prefetcht0(ker_addr(i_kw, i_ic));

        const ptrdiff_t next_input_block_offset
            = typesize * ic_block_step * jcp.tr_iw;
        if (i_ur % 16 == 4 && i_kw == 0) {
            if (i_ur + 16 < ur_w)
                prefetcht0(inp_addr(i_ur + 16, i_ic));
            else
                prefetcht0(inp_addr(0, i_ic, next_input_block_offset));
        }
        if (i_ur % 16 == 4 && i_kw == 1) {
            if (input_wraparound)
                prefetcht1(inp_addr(i_ur, i_ic, -input_offset));
            else
                prefetcht1(inp_addr(i_ur, i_ic, next_input_block_offset));
        }
    };

    for (int i_kw = 0; i_kw < kw; i_kw++)
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
            auto zmm = zmm_ker(i_kw, i_ic);
            vpxord(zmm, zmm, zmm);
        }

    for (int i_ur = 0; i_ur < ur_w; i_ur += 4) {

        for (int i = 0; i < 4; i++) {
            auto zmm = zmm_out(i_ur + i);
            if (i_ur + i < ur_w)
                vmovups(zmm, out_addr(i_ur + i));
            else
                vpxord(zmm, zmm, zmm);
            prefetcht0(out_addr(i_ur + i + 4));
        }

        for (int  i_kw = 0; i_kw < kw; i_kw++)
            for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
                int i_iw = i_ur + i_kw;
                v4fmaddps(zmm_ker(i_kw, i_ic),
                        zmm_out(i_ur), inp_addr(i_iw, i_ic));
                pf_callback(i_ur, i_kw, i_ic);
            }
    }

    for (int i_kw = 0; i_kw < kw; i_kw++)
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
            auto addr = ker_addr(i_kw, i_ic);
            auto zmm = zmm_ker(i_kw, i_ic);
            vaddps(zmm, zmm, addr);
            vmovups(addr, zmm);
        }
}

void jit_avx512_common_conv_bwd_weights_kernel_f32::compute_ic_block_step(
    int ur_w, int pad_l, int pad_r,
    int ic_block_step, int input_offset, int kernel_offset,
    int output_offset, bool input_wraparound)
{
    if (jcp.ver == ver_4fma)
        compute_ic_block_step_4fma(ur_w, pad_l, pad_r,
                ic_block_step, input_offset, kernel_offset, output_offset,
                input_wraparound);
    else if (jcp.ver == ver_fma)
        compute_ic_block_step_fma(ur_w, pad_l, pad_r,
                ic_block_step, input_offset, kernel_offset, output_offset,
                input_wraparound);
    else
        assert(!"unknown convolution version");
}


void
jit_avx512_common_conv_bwd_weights_kernel_f32::compute_oh_step_unroll_ow_icblock(
    int ic_block_step, int max_ur_w)
{
    UNUSED(max_ur_w);

    Label kh_label;

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int inp_mul = !jcp.is_1stconv ? ic_block : 1;
    int iw = jcp.transpose_src ? jcp.tr_iw : jcp.iw;

    int r_pad = nstl::max(0, (jcp.ow - 1) * jcp.stride_w + jcp.kw - 1
        - (jcp.iw + jcp.l_pad - 1));
    int l_pad = jcp.l_pad;

    mov(kj, reg_kh);
    L(kh_label);
    {
        for (int i_b_ic = 0; i_b_ic < jcp.ic_block; i_b_ic += ic_block_step) {
            const int input_offset = typesize
                * (jcp.transpose_src ? i_b_ic * iw : i_b_ic);
            compute_ic_block_step(jcp.ur_w, l_pad, r_pad, ic_block_step,
                input_offset, typesize * i_b_ic * jcp.oc_block, 0,
                i_b_ic + ic_block_step >= jcp.ic_block);
        }
        add(reg_input, typesize * iw * inp_mul);
        add(reg_kernel, typesize * (jcp.kw) * ic_block * oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kh_label, T_NEAR);
    }
}

void jit_avx512_common_conv_bwd_weights_kernel_f32::compute_oh_step_unroll_ow(
    int ic_block_step, int max_ur_w)
{
    Label kh_label, ic_block_label;

    UNUSED(max_ur_w);

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;

    int r_pad = nstl::max(0,
        (jcp.ow - 1) * jcp.stride_w + jcp.kw - 1
        - (jcp.iw + jcp.l_pad - 1));
    int l_pad = jcp.l_pad;

    mov(kj, reg_kh);
    L(kh_label);
    {
        xor_(b_ic, b_ic);
        L(ic_block_label); {
            compute_ic_block_step(jcp.ow, l_pad, r_pad, ic_block_step,
                0, 0, 0);
            int inp_icblk_stride = jcp.is_1stconv
                ? jcp.ih * jcp.iw
                : (jcp.transpose_src ? jcp.tr_iw : 1);
            add(reg_input, typesize * ic_block_step * inp_icblk_stride);
            add(reg_kernel,  typesize * ic_block_step * oc_block);
            add(b_ic, ic_block_step);
            cmp(b_ic, jcp.ic_block);
            jl(ic_block_label, T_NEAR);
        }

        if (jcp.is_1stconv) {
            sub(reg_input, typesize * jcp.ih * jcp.iw * ic_block);
            add(reg_input, typesize * jcp.iw);
        } else if (!jcp.transpose_src) {
            add(reg_input, typesize * (jcp.iw - 1) * ic_block);
        }
        add(reg_kernel,  typesize * (jcp.kw - 1) * ic_block * oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kh_label, T_NEAR);
    }
}

void jit_avx512_common_conv_bwd_weights_kernel_f32::compute_oh_step_common(
    int ic_block_step, int max_ur_w)
{
    Label kh_label, ic_block_label, ow_block_label;

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;

    int r_pad = nstl::max(0, (jcp.ow - 1) * jcp.stride_w + jcp.kw - 1
        - (jcp.iw + jcp.l_pad - 1));
    int l_pad = (jcp.transpose_src && jcp.ver == ver_4fma) ? 0 : jcp.l_pad;

    int ur_w     = nstl::min(jcp.ow, max_ur_w);
    int ur_w_trips = jcp.ow / ur_w;
    int ur_w_tail  = jcp.ow % ur_w;
    if ((ur_w_tail == 0 && r_pad != 0)
        || r_pad >= ur_w_tail) {
        if (ur_w_trips > 1) {
            ur_w_tail += ur_w;
            ur_w_trips--;
        } else {
            ur_w_tail += (ur_w - ur_w / 2);
            ur_w = ur_w / 2;
        }
    }

    int inp_mult = (jcp.is_1stconv || jcp.transpose_src) ? 1 : ic_block;
    int input_comeback = (ur_w_trips * ur_w * jcp.stride_w - l_pad) * inp_mult;
    int output_comeback = ur_w_trips * ur_w * oc_block;

    mov(kj, reg_kh);
    L(kh_label); {
        xor_(b_ic, b_ic);
        L(ic_block_label); {
            if (l_pad != 0) {
                ur_w_trips--;
                compute_ic_block_step(ur_w, l_pad, 0, ic_block_step, 0, 0, 0);
                add(reg_input, typesize * (ur_w * jcp.stride_w - l_pad)
                    * inp_mult);
                add(reg_output, typesize * ur_w * oc_block);
            }

            if (ur_w_trips > 0) {
                xor_(reg_ur_w_trips, reg_ur_w_trips);
                L(ow_block_label); {
                    compute_ic_block_step(ur_w, 0, 0, ic_block_step, 0, 0, 0);
                    add(reg_input, typesize * ur_w * jcp.stride_w * inp_mult);
                    add(reg_output, typesize * ur_w * oc_block);

                    inc(reg_ur_w_trips);
                    cmp(reg_ur_w_trips, ur_w_trips);
                    jl(ow_block_label, T_NEAR);
                }
            }

            if (ur_w_tail > 0) compute_ic_block_step(ur_w_tail, 0, r_pad,
                ic_block_step, 0, 0, 0);

            sub(reg_input, typesize * input_comeback);
            sub(reg_output, typesize * output_comeback);
            int inp_icblk_stride = jcp.is_1stconv
                ? jcp.ih * jcp.iw
                : (jcp.transpose_src ? jcp.tr_iw : 1);
            add(reg_input, typesize * ic_block_step * inp_icblk_stride);
            add(reg_kernel, typesize * ic_block_step * oc_block);

            add(b_ic, ic_block_step);
            cmp(b_ic, jcp.ic_block);
            jl(ic_block_label, T_NEAR);
        }
        if (jcp.is_1stconv) {
            sub(reg_input, typesize * jcp.ih * jcp.iw * ic_block);
            add(reg_input, typesize * jcp.iw);
        } else if (!jcp.transpose_src) {
            add(reg_input, typesize * (jcp.iw - 1) * ic_block);
        }
        add(reg_kernel, typesize * (jcp.kw - 1) * ic_block * oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kh_label, T_NEAR);
    }
}

void jit_avx512_common_conv_bwd_weights_kernel_f32::compute_oh_step_disp()
{
    int ic_block_step = jcp.kw <= 3 ? 8 : (jcp.kw <= 7 ? 4 : 2);
    if (jcp.is_1stconv) {
        bool large_code = jcp.kw >= 7 && (jcp.l_pad > 0 || jcp.t_pad > 0);
        ic_block_step
            = (jcp.kw * jcp.ic_block <= 28 && !large_code) ? jcp.ic_block : 1;
    }

    bool too_large_to_unroll
        = (jcp.kw > 1 || jcp.kh > 1) && (jcp.stride_w > 1 || jcp.stride_h > 1);

    if (jcp.kw <= 3 && jcp.ow <= 16 && !too_large_to_unroll)
        compute_oh_step_unroll_ow_icblock(ic_block_step, max_ur_w);
    else if (jcp.ow <= max_ur_w)
        compute_oh_step_unroll_ow(ic_block_step, max_ur_w);
    else
        compute_oh_step_common(ic_block_step, max_ur_w);

    oh_step_comeback_pointers();
}

void jit_avx512_common_conv_bwd_weights_kernel_f32::compute_oh_loop_common()
{
    int b_pad = nstl::max(0, (jcp.oh - 1) * jcp.stride_h + jcp.kh - 1
        - (jcp.ih + jcp.t_pad - 1));
    int t_pad = jcp.t_pad;
    int stride_h = jcp.stride_h;
    const int inp_mult = jcp.is_1stconv ? 1 : jcp.ic_block;
    int iw = jcp.transpose_src ? jcp.tr_iw : jcp.iw;
    Label oh_label, oh_label_end, oh_tpad_label, oh_bpad_label,
        oh_bpad_label_end;

    mov(reg_kh, jcp.kh);
    xor_(reg_ih_count, reg_ih_count);
    xor_(reg_oj, reg_oj);
    if (t_pad > 0) {
        assert(jcp.kh <= t_pad + jcp.ih); /* [bwd_w:r1] */
        mov(reg_kh, jcp.kh <= t_pad + jcp.ih ? jcp.kh - t_pad : jcp.ih);
        add(reg_kernel, typesize * t_pad * jcp.kw * jcp.ic_block
            * jcp.oc_block);

        L(oh_tpad_label); {
            compute_oh_step_disp();
            add(reg_output, typesize * jcp.ow * jcp.oc_block);
            sub(reg_kernel, typesize * stride_h * jcp.kw * jcp.ic_block
                * jcp.oc_block);

            inc(reg_oj);
            add(reg_ih_count, stride_h);
            add(reg_kh, stride_h);

            /* the overlap between input and kernel may not reach kernel size.
             * so far we do not support that (until we put constant here) */
            const int final_inp_ker_overlap = jcp.kh; /* [bwd_w:r2] */
            cmp(reg_kh, final_inp_ker_overlap);
            jl(oh_tpad_label, T_NEAR);
        }
        if (t_pad % stride_h != 0) {
            int inp_corr = stride_h -  t_pad % stride_h;
            add(reg_kernel, typesize * inp_corr * jcp.kw * jcp.ic_block
                * jcp.oc_block);
            add(reg_input, typesize * inp_corr * iw * inp_mult);
        }

    }

    cmp(reg_ih_count, jcp.ihp - b_pad - jcp.kh + 1);
    jge(oh_label_end, T_NEAR);
    cmp(reg_oj, jcp.oh);
    jge(oh_label, T_NEAR);

    mov(reg_kh, jcp.kh);
    L(oh_label); {
        compute_oh_step_disp();
        add(reg_input, typesize * stride_h * iw * inp_mult);
        add(reg_output, typesize * jcp.ow * jcp.oc_block);

        inc(reg_oj);
        add(reg_ih_count, stride_h);

        cmp(reg_ih_count, jcp.ihp - b_pad - jcp.kh + 1);
        jge(oh_label_end, T_NEAR);

        cmp(reg_oj, jcp.oh);
        jl(oh_label, T_NEAR);
    }
    L(oh_label_end);

    if (b_pad > 0) {
        cmp(reg_oj, jcp.oh);
        jge(oh_bpad_label_end, T_NEAR);

        mov(reg_kh,  jcp.ihp - b_pad);
        sub(reg_kh, reg_ih_count);
        L(oh_bpad_label);
        {
            compute_oh_step_disp();
            add(reg_input, typesize * stride_h * iw * inp_mult);
            add(reg_output, typesize * jcp.ow * jcp.oc_block);

            sub(reg_kh, stride_h);
            cmp(reg_kh, 0);
            jle(oh_bpad_label_end, T_NEAR);

            inc(reg_oj);
            cmp(reg_oj, jcp.oh);
            jl(oh_bpad_label, T_NEAR);
        }
        L(oh_bpad_label_end);
    }
}

void jit_avx512_common_conv_bwd_weights_kernel_f32::generate()
{
    preamble();

    mov(reg_input, ptr[param + GET_OFF(src)]);
    mov(reg_output, ptr[param + GET_OFF(dst)]);
    mov(reg_kernel, ptr[param + GET_OFF(filt)]);

    compute_oh_loop_common();

    postamble();
}

status_t jit_avx512_common_conv_bwd_weights_kernel_f32::init_conf(
    jit_conv_conf_t &jcp,
    const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
    const memory_desc_wrapper &diff_weights_d,
    const memory_desc_wrapper &diff_dst_d)
{
    if (!mayiuse(avx512_common))
        return status::unimplemented;
    const bool with_groups = diff_weights_d.ndims() == src_d.ndims() + 1;

    jcp.prop_kind = cd.prop_kind;

    jcp.ngroups = with_groups ? diff_weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];

    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;

    jcp.ih = src_d.dims()[2];
    jcp.iw = src_d.dims()[3];
    jcp.oh = diff_dst_d.dims()[2];
    jcp.ow = diff_dst_d.dims()[3];

    jcp.kh = diff_weights_d.dims()[with_groups + 2];
    jcp.kw = diff_weights_d.dims()[with_groups + 3];

    jcp.t_pad = cd.padding[0][0];
    jcp.l_pad = cd.padding[0][1];

    jcp.stride_h = cd.strides[0];
    jcp.stride_w = cd.strides[1];

    jcp.dilate_h = cd.dilates[0];
    jcp.dilate_w = cd.dilates[1];
    if (jcp.dilate_h != 0 || jcp.dilate_w != 0)
        return status::unimplemented;

    jcp.r_pad = nstl::max(0, (jcp.ow - 1) * jcp.stride_w + jcp.kw - jcp.iw
        - jcp.l_pad);
    jcp.b_pad = nstl::max(0, (jcp.oh - 1) * jcp.stride_h + jcp.kh - jcp.ih
        - jcp.t_pad);

    jcp.src_fmt = src_d.format();
    jcp.with_bias = cd.diff_bias_desc.format != memory_format::undef;
    jcp.with_relu = 0;
    jcp.relu_negative_slope = 0;

    const bool flat = jcp.ic == 3;
    const bool mimo = !flat;

    const int simd_w = 16;

    jcp.ihp = jcp.ih + jcp.t_pad + jcp.b_pad;
    jcp.iwp = jcp.iw + jcp.l_pad + jcp.r_pad;
    jcp.ohp = jcp.oh;
    jcp.owp = jcp.ow;

    bool args_ok = true
        && implication(flat, one_of(src_d.format(), nchw, nhwc))
        && implication(flat, one_of(diff_weights_d.format(), Ohwi16o, gOhwi16o))
        && implication(mimo, src_d.format() == nChw16c)
        && implication(mimo, one_of(diff_weights_d.format(), OIhw16i16o, gOIhw16i16o))
        && one_of(cd.bias_desc.format, memory_format::undef, any, x)
        && diff_dst_d.format() == nChw16c;
    if (!args_ok)
        return status::unimplemented;

    jcp.is_1stconv = false;
    if (jcp.ic % simd_w != 0) {
        if ((jcp.ic == 3 || jcp.ic == 1) && jcp.src_fmt == nchw)
            jcp.is_1stconv = true;
        else return status::unimplemented;
    }
    jcp.ic_block = (jcp.ic % simd_w) ? jcp.ic : simd_w;
    jcp.nb_ic = jcp.ic / jcp.ic_block;

    if (jcp.is_1stconv) {
        jcp.ic_block = jcp.ic;
        jcp.nb_ic = 1;
    }

    jcp.oc_block = simd_w;
    if (jcp.oc % jcp.oc_block) return status::unimplemented;
    jcp.nb_oc = jcp.oc / jcp.oc_block;

    jcp.ur_h = 1; /* no code-unrolling by h so far */
    jcp.nb_ic_blocking = 1;
    jcp.nb_oc_blocking = 1;

    if (jcp.kw > 14) return status::unimplemented;
    if (jcp.is_1stconv && jcp.ngroups > 1) return status::unimplemented;

    bool args_ok1 = true
        && jcp.t_pad <= jcp.kh / 2
        && jcp.b_pad <= jcp.kh / 2
        && jcp.kh <= jcp.t_pad + jcp.ih /* [bwd_w:r1] */
        && jcp.kh <= jcp.ih; /* [bwd_w:r2] */
    if (!args_ok1) return status::unimplemented;

    for (int ur_w = nstl::min(max_ur_w, jcp.ow); ur_w > 0; --ur_w) {
        if (jcp.ow % ur_w == 0) {
            jcp.ur_w = ur_w;
            break;
        }
    }

    if (mayiuse(avx512_mic_4ops)
            && jcp.iw >= small_spatial
            && jcp.ih >= small_spatial
            && jcp.stride_w == 1 // transposing output and diff_filter can help
            && !jcp.is_1stconv)
        jcp.ver = ver_4fma;
    else
        jcp.ver = ver_fma;

    jcp.transpose_src = (jcp.ver == ver_4fma);
    if (jcp.transpose_src) {
        jcp.ur_w = jcp.ow;
        if (jcp.ver == ver_4fma) // double check
            jcp.tr_iw = rnd_up(jcp.iw + jcp.l_pad + 4, 4);
        else
            jcp.tr_iw = jcp.iw;
    }

    return status::success;
}

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
