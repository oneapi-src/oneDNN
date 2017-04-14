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

#include "jit_avx512_mic_conv_kernel_f32.hpp"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

void jit_avx512_mic_conv_fwd_kernel_f32::prepare_output(int ur_w)
{
    for (int i = 0; i < ur_w; i++) {
        Zmm zmm(i);
        vpxord(zmm, zmm, zmm);
        int aux_output_offset = typesize * (i)*jcp.oc_block;
        prefetcht1(EVEX_compress_addr(reg_out_prf, aux_output_offset));
    }
}

void jit_avx512_mic_conv_fwd_kernel_f32::store_output(int ur_w)
{
    Label no_update_label, store_label, relu_label;

    mov(reg_current_ic, ptr[this->param1 + GET_OFF(current_ic)]);
    if (jcp.with_bias)
        mov(reg_bias, ptr[this->param1 + GET_OFF(bias)]);

    cmp(reg_current_ic, 0);
    je(no_update_label, T_NEAR);
    for (int i = 0; i < ur_w; i++) {
        Zmm zmm(i);
        int aux_output_offset = typesize * (i)*jcp.oc_block;
        vaddps(zmm, zmm, EVEX_compress_addr(reg_out, aux_output_offset));
    }
    jmp(relu_label, T_NEAR);

    L(no_update_label);
    if (jcp.with_bias) {
        for (int i = 0; i < ur_w; i++) {
            Zmm zmm(i);
            vaddps(zmm, zmm, zword[reg_bias]);
        }
    }
    if (jcp.with_bias)
        prefetcht1(EVEX_compress_addr(reg_bias, 64));

    L(relu_label);
    if (jcp.with_relu) {
        cmp(reg_current_ic, jcp.nb_ic-1);
        jl(store_label, T_NEAR);
        const unsigned char _cmp_lt_os = 1;
        for (int i = 0; i < ur_w; i++){
            Opmask kmask = Opmask(7);
            Zmm zmm(i);
            vcmpps(kmask, zmm, zmm_zero, _cmp_lt_os);
            vmulps(zmm | kmask, zmm, zmm_relu_ns);
        }
    }

    L(store_label);
    for (int i = 0; i < ur_w; i++) {
        Zmm zmm(i);
        int aux_output_offset = typesize * (i)*jcp.oc_block;
        vmovups(EVEX_compress_addr(reg_out, aux_output_offset), zmm);
        prefetcht0(EVEX_compress_addr(reg_out_prf, aux_output_offset));
    }
}

int jit_avx512_mic_conv_fwd_kernel_f32::compute_loop(int ur_w, int pad_l,
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
    assert(oc_block >= ker_pipeline_depth);
    int ker_reg_base_idx = 28;

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
                        aux_kernel_offset = typesize
                                * ((ic + i) * oc_block
                                                    + (ki)*ic_block * oc_block);
                        vmovups(Zmm(i + ker_reg_base_idx),
                                EVEX_compress_addr(
                                        aux_reg_ker, aux_kernel_offset));
                    }
                } else if (step < num_ker_loads - ker_pipeline_depth + 1) {
                    int load_offset = ker_pipeline_depth - 1;
                    int ker_load_reg_idx = ker_reg_base_idx
                            + (step + load_offset) % ker_pipeline_depth;
                    aux_kernel_offset
                            = typesize * ((ic + load_offset) * oc_block
                                                 + (ki)*ic_block * oc_block);
                    vmovups(Zmm(ker_load_reg_idx),
                            EVEX_compress_addr(aux_reg_ker, aux_kernel_offset));
                }
                bool ker_prf_inserted = false;
                auto zmm_ker
                        = Zmm(ker_reg_base_idx + step % ker_pipeline_depth);

                int j_start
                        = nstl::max(0, (pad_l - ki + stride_w - 1) / stride_w);
                int j_end = ur_w
                        - nstl::max(0, (ki + pad_r - (kw - 1) + stride_w - 1)
                        / stride_w);

                for (int j = j_start; j < j_end; j++) {
                    int iw_str = !jcp.is_1stconv ? ic_block : 1;
                    int ic_str = !jcp.is_1stconv ? 1 : iw * ih;
                    int aux_input_offset
                            = typesize * ((ki + j * stride_w - pad_l) * iw_str
                              + ic * ic_str);
                    vfmadd231ps(
                            Zmm(j), zmm_ker, EVEX_compress_addr(aux_reg_inp,
                            aux_input_offset, true));
                    int fma_idx = step * ur_w + j;
                    int prf_slot_idx = fma_idx / prf_inst_spacing;
                    if (fma_idx % prf_inst_spacing == prf_inst_trigger) {
                        if (prf_ker && !ker_prf_inserted
                                && ker_prfs < num_ker_prfs) {
                            int ker_prf_offset
                                    = typesize * ker_prfs * jcp.oc_block;
                            prefetcht2(EVEX_compress_addr(
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
                                            = ic_block * typesize
                                            * ((inp_prf_idx / kw)
                                            * inp_prf_stride
                                            + (inp_prf_idx % kw));
                                } else {
                                    int ic_prf_stride = typesize * iw * ih;
                                    int iw_prf_stride = typesize * simd_w;
                                    inp_prf_offset = ((inp_prf_idx / ic_block)
                                            * iw_prf_stride
                                            + (inp_prf_idx % ic_block)
                                            * ic_prf_stride);
                                }
                                prefetcht0(EVEX_compress_addr(
                                        aux_reg_inp_prf, inp_prf_offset));
                            }
                        }
                    }
                }
                step++;
            }
        }
        add(aux_reg_ker, typesize * kw * oc_block * ic_block);
        if (prf_ker)
            add(aux_reg_ker_prf, typesize * kw * oc_block * ic_block);
        int inp_mul = !jcp.is_1stconv ? ic_block : 1;
        add(aux_reg_inp, typesize * iw * inp_mul);
        if (prf_inp)
            add(aux_reg_inp_prf, typesize * iw * inp_mul);
        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }
    store_output(ur_w);

    return 0;
}

void jit_avx512_mic_conv_fwd_kernel_f32::generate()
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
    int inp_shift_pad = typesize * (ur_w * stride_w - l_pad) * inp_mult;
    int inp_shift = typesize * (ur_w * stride_w * inp_mult);
    int out_shift = typesize * (ur_w * oc_block);

    this->preamble();

    mov(reg_inp, ptr[this->param1 + GET_OFF(src)]);
    mov(reg_out, ptr[this->param1 + GET_OFF(dst)]);
    mov(reg_ker, ptr[this->param1 + GET_OFF(filt)]);
    mov(reg_kh, ptr[this->param1 + GET_OFF(kh_padding)]);
    mov(reg_ker_prf, ptr[this->param1 + GET_OFF(filt_prf)]);
    mov(reg_relu_ns, reinterpret_cast<size_t>(&jcp.relu_negative_slope));

    vbroadcastss(zmm_relu_ns, ptr[reg_relu_ns]);
    vpxord(zmm_zero, zmm_zero, zmm_zero);

    int r_pad = nstl::max(0, (ow - 1) * stride_w + (kw - 1) - (iw + l_pad - 1));
    if (ow == ur_w) {
        mov(reg_inp_prf, ptr[this->param1 + GET_OFF(src_prf)]);
        mov(reg_out_prf, ptr[this->param1 + GET_OFF(dst_prf)]);
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

    this->postamble();
}

status_t jit_avx512_mic_conv_fwd_kernel_f32::init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d,
        bool with_relu, double relu_negative_slope)
{
    if (!mayiuse(avx512_mic))
        return status::unimplemented;

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;

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

    const bool flat = jcp.ic == 3;
    const bool mimo = !flat;

    bool args_ok = true && implication(flat, one_of(src_d.format(), nchw, nhwc))
            && implication(mimo, src_d.format() == nChw16c)
            && weights_d.format()
                    == (with_groups ? gOIhw16i16o :
                                      (flat ? Ohwi16o : OIhw16i16o))
            && one_of(cd.bias_desc.format, memory_format::undef, any, x)
            && dst_d.format() == nChw16c;
    if (!args_ok)
        return status::unimplemented;

    const int simd_w = 16;
    jcp.ur_h = 1;

    const int regs = 28;
    for (int ur_w = regs; ur_w > 0; --ur_w) {
        if (jcp.ow % ur_w == 0) {
            jcp.ur_w = ur_w;
            break;
        }
    }

    int n_oi = (jcp.ow / jcp.ur_w);
    int r_pad = (jcp.ur_w * n_oi - 1) * jcp.stride_w + jcp.kw - jcp.iw
            - jcp.l_pad;
    if (jcp.l_pad > 0 && r_pad > 0)
        n_oi--;

    jcp.ic_block = (jcp.ic % simd_w != 0) ? jcp.ic : simd_w;
    jcp.nb_ic = jcp.ic / jcp.ic_block;

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

    args_ok = true && jcp.oc % simd_w == 0 && jcp.l_pad <= jcp.ur_w
            && implication(mimo, jcp.ic % simd_w == 0);
    if (!args_ok)
        return status::unimplemented;

    int r_pad_no_tail = nstl::max(0, (jcp.ow - jcp.ur_w_tail - 1) * jcp.stride_w
                    + jcp.kw - jcp.iw - jcp.l_pad);
    if (r_pad_no_tail > jcp.ur_w)
        return status::unimplemented;

    jcp.oc_block = simd_w;
    jcp.nb_oc = jcp.oc / jcp.oc_block;
    jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;

    if (jcp.ic % simd_w != 0) {
        if (jcp.ic == 3 || jcp.ic == 1)
            jcp.is_1stconv = true;
        else
            return status::unimplemented;
    } else
        jcp.is_1stconv = false;

    return status::success;
}

void jit_avx512_mic_conv_bwd_data_kernel_f32::prepare_output(int ur_w) {
    for (int k = 0; k < jcp.nb_ic_blocking; k++) {
        for (int j = 0; j  < ur_w; j++) {
            Zmm zmm(ur_w * k + j);
            vpxord(zmm, zmm, zmm);
            int aux_src_offset = typesize *
                (k * jcp.ih * jcp.iw + j) * jcp.ic_block;
            prefetcht1(EVEX_compress_addr(reg_src_prf, aux_src_offset));
        }
    }
}

void jit_avx512_mic_conv_bwd_data_kernel_f32::store_output(int ur_w) {
    Label no_update_label;

    mov(reg_current_ic, ptr[param + GET_OFF(current_ic)]);
    cmp(reg_current_ic, 0);
    je(no_update_label, T_NEAR);
    for (int k = 0; k < jcp.nb_ic_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            Zmm zmm(ur_w * k + j);
            int aux_src_offset = typesize *
                (k * jcp.ih * jcp.iw + j) * jcp.ic_block;
            vaddps(zmm, zmm, EVEX_compress_addr(reg_src, aux_src_offset));
        }
    }
    L(no_update_label);
    for (int k = 0; k < jcp.nb_ic_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            Zmm zmm(ur_w * k + j);
            int aux_src_offset = typesize *
                (k * jcp.ih * jcp.iw + j) * jcp.ic_block;
            vmovups(EVEX_compress_addr(reg_src, aux_src_offset), zmm);
            prefetcht0(EVEX_compress_addr(reg_src_prf, aux_src_offset));
        }
    }
}

int jit_avx512_mic_conv_bwd_data_kernel_f32::compute_loop(int ur_w,
        int l_overflow, int r_overflow) {
    Label kh_label;
    int kw    = jcp.kw;
    int ow    = jcp.ow;

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int l_pad    = jcp.l_pad;
    int stride_w = jcp.stride_w;

    int ker_pipeline_depth = 4;
    assert(oc_block >= ker_pipeline_depth);
    int ker_reg_base_idx = 28;

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
            int jj_start = nstl::max(0, l_overflow - (kw - 1) + ki);
            int jj_end   = ur_w - nstl::max(0, r_overflow - ki);
            for (int oc = 0; oc < oc_block; oc++) {
                int aux_kernel_offset = 0;
                if (step == 0) {
                    for (int i = 0; i < ker_pipeline_depth; i++) {
                        aux_kernel_offset = typesize * ((oc + i) * oc_block
                                + (ki)*ic_block * oc_block);
                        vmovups(Zmm(i + ker_reg_base_idx),
                            EVEX_compress_addr(aux_reg_ker, aux_kernel_offset));
                    }
                } else if (step < num_ker_loads - ker_pipeline_depth + 1) {
                    int load_offset = ker_pipeline_depth - 1;
                    int ker_load_reg_idx = ker_reg_base_idx
                        + (step + load_offset) % ker_pipeline_depth;
                    aux_kernel_offset = typesize * ((oc + load_offset)
                            * oc_block + (ki)*ic_block * oc_block);
                    vmovups(Zmm(ker_load_reg_idx),
                            EVEX_compress_addr(aux_reg_ker, aux_kernel_offset));
                }
                bool ker_prf_inserted = false;
                auto zmm_ker = Zmm(ker_reg_base_idx
                                   + step % ker_pipeline_depth);
                for (int jj = jj_start; jj  < jj_end; jj++) {
                    int aux_dst_offset = typesize * ((jj + l_pad - ki)
                            * jcp.oc_block + oc);
                    vfmadd231ps(Zmm(jj), zmm_ker,
                        EVEX_compress_addr(aux_reg_dst, aux_dst_offset, true));

                    int fma_idx = step * ur_w + jj;
                    int prf_slot_idx = fma_idx / prf_inst_spacing;
                    if (fma_idx % prf_inst_spacing == prf_inst_trigger) {
                        if (!ker_prf_inserted && ker_prfs < num_ker_loads) {
                            int ker_prf_offset = typesize
                                * ker_prfs * jcp.oc_block;
                            prefetcht1(EVEX_compress_addr(aux_reg_ker_prf,
                                                          ker_prf_offset));
                            ker_prf_inserted = true;
                            ker_prfs++;
                        } else {
                            int inp_prf_idx = prf_slot_idx - ker_prfs;
                            if (inp_prf_idx < num_inp_prfs) {
                                int inp_prf_offset = ic_block
                                    * typesize * ((inp_prf_idx / kw)
                                                * kw + (inp_prf_idx % kw));
                                prefetcht0(EVEX_compress_addr(aux_reg_dst_prf,
                                                              inp_prf_offset));
                            }
                        }
                    }
                }
                step++;
            }
        }
        add(aux_reg_ker, typesize * kw * oc_block * ic_block);
        add(aux_reg_ker_prf, typesize * kw * oc_block * ic_block);

        sub(aux_reg_dst, typesize * ow * oc_block);
        sub(aux_reg_dst_prf, typesize * ow * oc_block);

        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }
    store_output(ur_w);

    return 0;
}

void jit_avx512_mic_conv_bwd_data_kernel_f32::generate() {
    int iw    = jcp.iw;
    int ow    = jcp.ow;
    int kw    = jcp.kw;
    int l_pad = jcp.l_pad;
    int ur_w      = jcp.ur_w;
    int ic_block  = jcp.ic_block;
    int oc_block  = jcp.oc_block;
    int ur_w_tail = jcp.ur_w_tail;

    int dst_shift = typesize * ur_w * ic_block;
    int src_shift = typesize * ur_w * oc_block;

    this->preamble();

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
    this->postamble();
}

status_t jit_avx512_mic_conv_bwd_data_kernel_f32::init_conf(
        jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &diff_src_d,
        const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &diff_dst_d) {
    if (!mayiuse(avx512_mic)) return status::unimplemented;

    const bool with_groups = weights_d.ndims() == diff_src_d.ndims() + 1;

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
        && weights_d.format() == with_groups ? gOIhw16o16i : OIhw16o16i
        && diff_dst_d.format() == nChw16c;
    if (!args_ok)
        return status::unimplemented;

    const int simd_w = 16;

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

    jcp.nb_ic_blocking = 1;
    jcp.nb_oc_blocking = 1;

    int regs = 28;
    jcp.ur_w = (jcp.iw <= regs) ? jcp.iw : regs;
    int n_oi  = (jcp.iw / jcp.ur_w);
    int l_overflow  = nstl::max(0, ((jcp.kw-1) - jcp.l_pad));
    int r_overflow1 = nstl::max(0, ((jcp.kw-1) - (jcp.iw - jcp.ur_w*n_oi)
                                - jcp.r_pad));
    if (r_overflow1 > 0) n_oi--;


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
    int r_overflow_no_tail = nstl::max(0,jcp.kw - 1 - jcp.ur_w_tail
                                       - jcp.r_pad);
    if (r_overflow_no_tail > jcp.ur_w)
        return status::unimplemented;
    int r_pad_no_tail = nstl::max(0, (jcp.ow - jcp.ur_w_tail - 1) + jcp.kw
                                  - jcp.iw - jcp.l_pad);
    if (r_pad_no_tail > jcp.ur_w)
        return status::unimplemented;
    return status::success;
}

void jit_avx512_mic_conv_bwd_weights_kernel_f32::oh_step_comeback_pointers() {
    Label kh_comeback_label;

    mov(kj, reg_kh);
    L(kh_comeback_label); {
        int inp_mult = jcp.is_1stconv ? 1 : jcp.ic_block;
        sub(reg_input, typesize * jcp.iw * inp_mult);
        sub(reg_kernel, typesize * jcp.kw * jcp.ic_block * jcp.oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kh_comeback_label, T_NEAR);
    }
}

void jit_avx512_mic_conv_bwd_weights_kernel_f32::compute_ic_block_step(
    int ur_w, int pad_l, int pad_r,
    int ic_block_step, int input_offset, int kernel_offset,
    int output_offset) {

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
                const int i_offset = input_offset +
                    typesize * (jcp.is_1stconv
                    ? (i_iw - pad_l) + i_ic * (jcp.ih * jcp.iw)
                    : (i_iw - pad_l) * ic_block + i_ic);
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

void
jit_avx512_mic_conv_bwd_weights_kernel_f32::compute_oh_step_unroll_ow_icblock(
    int ic_block_step, int max_ur_w) {
    UNUSED(max_ur_w);

    Label kh_label;

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int inp_mul = !jcp.is_1stconv ? ic_block : 1;

    int r_pad = nstl::max(0, (jcp.ow - 1) * jcp.stride_w + jcp.kw - 1
        - (jcp.iw + jcp.l_pad - 1));
    int l_pad = jcp.l_pad;

    mov(kj, reg_kh);
    L(kh_label);
    {
        for (int i_b_ic = 0; i_b_ic < jcp.ic_block; i_b_ic += ic_block_step)
            compute_ic_block_step(jcp.ur_w, l_pad, r_pad, ic_block_step,
                typesize * i_b_ic, typesize * i_b_ic * jcp.oc_block, 0);

        add(reg_input, typesize * (jcp.iw) * inp_mul);
        add(reg_kernel, typesize * (jcp.kw) * ic_block * oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kh_label, T_NEAR);
    }
}

void jit_avx512_mic_conv_bwd_weights_kernel_f32::compute_oh_step_unroll_ow(
    int ic_block_step, int max_ur_w) {
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
            int inp_icblk_stride = jcp.is_1stconv ? jcp.ih * jcp.iw : 1;
            add(reg_input, typesize * ic_block_step * inp_icblk_stride);
            add(reg_kernel,  typesize * ic_block_step * oc_block);
            add(b_ic, ic_block_step);
            cmp(b_ic, jcp.ic_block);
            jl(ic_block_label, T_NEAR);
        }

        if (jcp.is_1stconv) {
            sub(reg_input, typesize * jcp.ih * jcp.iw * ic_block);
            add(reg_input, typesize * jcp.iw);
        } else {
            add(reg_input, typesize * (jcp.iw - 1) * ic_block);
        }
        add(reg_kernel,  typesize * (jcp.kw - 1) * ic_block * oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kh_label, T_NEAR);
    }
}

void jit_avx512_mic_conv_bwd_weights_kernel_f32::compute_oh_step_common(
    int ic_block_step, int max_ur_w) {

    Label kh_label, ic_block_label, ow_block_label;

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;

    int r_pad = nstl::max(0, (jcp.ow - 1) * jcp.stride_w + jcp.kw - 1
        - (jcp.iw + jcp.l_pad - 1));
    int l_pad = jcp.l_pad;

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
    int inp_mult = jcp.is_1stconv ? 1 : ic_block;
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
            int inp_icblk_stride = jcp.is_1stconv ? jcp.ih * jcp.iw : 1;
            add(reg_input, typesize * ic_block_step * inp_icblk_stride);
            add(reg_kernel, typesize * ic_block_step * oc_block);

            add(b_ic, ic_block_step);
            cmp(b_ic, jcp.ic_block);
            jl(ic_block_label, T_NEAR);
        }
        if (jcp.is_1stconv) {
            sub(reg_input, typesize * jcp.ih * jcp.iw * ic_block);
            add(reg_input, typesize * jcp.iw);
        } else {
            add(reg_input, typesize * (jcp.iw - 1) * ic_block);
        }
        add(reg_kernel, typesize * (jcp.kw - 1) * ic_block * oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kh_label, T_NEAR);
    }
}

void jit_avx512_mic_conv_bwd_weights_kernel_f32::compute_oh_step_disp() {

    int ic_block_step = jcp.kw <= 7 ? 4 : 2;
    if (jcp.is_1stconv) {
        bool large_code = jcp.kw >= 7 && (jcp.l_pad > 0 || jcp.t_pad > 0);
        ic_block_step = (jcp.kw * jcp.ic_block <= 28 && !large_code)
            ? jcp.ic_block
            : 1;
    }
    int max_ur_w = 28;

    if (jcp.kw <= 3 && jcp.ow <= 16)
        compute_oh_step_unroll_ow_icblock(ic_block_step, max_ur_w);
    else if (jcp.ow <= max_ur_w)
        compute_oh_step_unroll_ow(ic_block_step, max_ur_w);
    else compute_oh_step_common(ic_block_step, max_ur_w);
    oh_step_comeback_pointers();
}

void jit_avx512_mic_conv_bwd_weights_kernel_f32::compute_oh_loop_common() {
    int b_pad = nstl::max(0, (jcp.oh - 1) * jcp.stride_h + jcp.kh - 1
        - (jcp.ih + jcp.t_pad - 1));
    int t_pad = jcp.t_pad;
    int stride_h = jcp.stride_h;
    const int inp_mult = jcp.is_1stconv ? 1 : jcp.ic_block;
    Label oh_label, oh_label_end, oh_tpad_label, oh_bpad_label,
        oh_bpad_label_end;

    mov(reg_kh, jcp.kh);
    xor_(reg_ih_count, reg_ih_count);
    xor_(reg_oj, reg_oj);
    if (t_pad > 0) {
        mov(reg_kh,  jcp.kh - t_pad);
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
            cmp(reg_kh, jcp.kh);
            jl(oh_tpad_label, T_NEAR);
        }
        if (t_pad % stride_h != 0) {
            int inp_corr = stride_h -  t_pad % stride_h;
            add(reg_kernel, typesize * inp_corr * jcp.kw * jcp.ic_block
                * jcp.oc_block);
            add(reg_input, typesize * inp_corr * jcp.iw * inp_mult);
        }

    }

    cmp(reg_ih_count, jcp.ihp - b_pad - jcp.kh + 1);
    jge(oh_label_end, T_NEAR);
    cmp(reg_oj, jcp.oh);
    jge(oh_label, T_NEAR);

    mov(reg_kh, jcp.kh);
    L(oh_label); {
        compute_oh_step_disp();
        add(reg_input, typesize * stride_h * jcp.iw * inp_mult);
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
            add(reg_input, typesize * stride_h * jcp.iw * inp_mult);
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

void jit_avx512_mic_conv_bwd_weights_kernel_f32::generate() {
    this->preamble();

    mov(reg_input, ptr[param + GET_OFF(src)]);
    mov(reg_output, ptr[param + GET_OFF(dst)]);
    mov(reg_kernel, ptr[param + GET_OFF(filt)]);

    compute_oh_loop_common();

    this->postamble();
}

status_t jit_avx512_mic_conv_bwd_weights_kernel_f32::init_conf(
    jit_conv_conf_t &jcp,
    const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
    const memory_desc_wrapper &diff_weights_d,
    const memory_desc_wrapper &diff_dst_d) {
    if (!mayiuse(avx512_mic)) return status::unimplemented;
    const bool with_groups = diff_weights_d.ndims() == src_d.ndims() + 1;

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
        && implication(mimo, src_d.format() == nChw16c)
        && diff_weights_d.format() == (with_groups ? gOIhw16i16o : (flat ?
              Ohwi16o : OIhw16i16o))
        && one_of(cd.bias_desc.format, memory_format::undef, any, x)
        && diff_dst_d.format() == nChw16c;
    if (!args_ok) return status::unimplemented;

    jcp.is_1stconv = false;
    if (jcp.ic % simd_w != 0) {
        if (jcp.ic == 3 || jcp.ic == 1) jcp.is_1stconv = true;
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
    jcp.ur_w = 1;

    if (jcp.kw > 14) return status::unimplemented;
    if (jcp.is_1stconv && jcp.ngroups > 1) return status::unimplemented;

    bool args_ok1 = true
        && jcp.t_pad <= jcp.kh / 2
        && jcp.b_pad <= jcp.kh / 2;
    if (!args_ok1) return status::unimplemented;

    const int regs = 28;
    for (int ur_w = regs; ur_w > 0; --ur_w) {
        if (jcp.ow % ur_w == 0) {
            jcp.ur_w = ur_w;
            break;
        }
    }
    return status::success;
}

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
