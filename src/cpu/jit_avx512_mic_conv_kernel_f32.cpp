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
        prefetcht2(EVEX_compress_addr(reg_out_prf, aux_output_offset));
    }
}

void jit_avx512_mic_conv_fwd_kernel_f32::store_output(int ur_w, char pad_label)
{
    char no_update_label[] = { '.', 'n', pad_label, '\0' };
    char store_label[] = { '.', 's', pad_label, '\0' };

    mov(reg_current_ic, ptr[this->param1 + GET_OFF(current_ic)]);
    mov(reg_bias, ptr[this->param1 + GET_OFF(bias)]);

    cmp(reg_current_ic, 0);
    je(no_update_label, T_NEAR);
    for (int i = 0; i < ur_w; i++) {
        Zmm zmm(i);
        int aux_output_offset = typesize * (i)*jcp.oc_block;
        vaddps(zmm, zmm, EVEX_compress_addr(reg_out, aux_output_offset));
    }
    jmp(store_label, T_NEAR);

    L(no_update_label);
    if (jcp.with_bias) {
        for (int i = 0; i < ur_w; i++) {
            Zmm zmm(i);
            vaddps(zmm, zmm, zword[reg_bias]);
        }
    }
    if (jcp.with_bias)
        prefetcht1(EVEX_compress_addr(reg_bias, 64));
    L(store_label);
    for (int i = 0; i < ur_w; i++) {
        Zmm zmm(i);
        int aux_output_offset = typesize * (i)*jcp.oc_block;
        vmovups(EVEX_compress_addr(reg_out, aux_output_offset), zmm);
        prefetcht0(EVEX_compress_addr(reg_out_prf, aux_output_offset));
    }
}

int jit_avx512_mic_conv_fwd_kernel_f32::compute_loop(
        int ur_w, int pad_l, int pad_r, const char *kh_label, char pad_label)
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
    store_output(ur_w, pad_label);

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

    int r_pad = nstl::max(0, (ow - 1) * stride_w + (kw - 1) - (iw + l_pad - 1));
    if (ow == ur_w) {
        const char *kh_loop_label = ".kh_loop";
        mov(reg_inp_prf, ptr[this->param1 + GET_OFF(src_prf)]);
        mov(reg_out_prf, ptr[this->param1 + GET_OFF(dst_prf)]);
        compute_loop(ur_w, l_pad, r_pad, kh_loop_label, 'b');
    } else {
        mov(reg_inp_prf, reg_inp);
        mov(reg_out_prf, reg_out);
        int n_oi = ow / ur_w;
        int r_pad1 = 0;

        xor_(reg_oi, reg_oi);
        if (l_pad > 0) {
            const char *kh_loop_l_pad_label = ".kh_loop_l_pad";
            add(reg_inp_prf, inp_shift_pad);
            add(reg_out_prf, out_shift);
            compute_loop(ur_w, l_pad, 0, kh_loop_l_pad_label, 'l');
            add(reg_inp, inp_shift_pad);
            add(reg_out, out_shift);
            inc(reg_oi);

            r_pad1 = (ur_w * n_oi - 1) * stride_w + kw - 1 - (iw + l_pad - 1);
            if (r_pad1 > 0)
                n_oi--;
        }
        if ((l_pad <= 0 && n_oi > 0) || (l_pad > 0 && n_oi > 1)) {
            const char *kh_loop_main_label = ".kh_loop_main";
            const char *ow_loop_label = ".ow_loop";
            L(ow_loop_label);
            {
                add(reg_inp_prf, inp_shift);
                add(reg_out_prf, out_shift);
                compute_loop(ur_w, 0, 0, kh_loop_main_label, 'n');
                add(reg_inp, inp_shift);
                add(reg_out, out_shift);
                inc(reg_oi);
                cmp(reg_oi, n_oi);
                jl(ow_loop_label, T_NEAR);
            }
        }
        if (r_pad1 > 0) {
            const char *kh_loop_r_pad_label = ".kh_loop_r_pad";
            add(reg_inp_prf, inp_shift);
            add(reg_out_prf, out_shift);
            compute_loop(ur_w, 0, r_pad1, kh_loop_r_pad_label, 'r');
            add(reg_inp, inp_shift);
            add(reg_out, out_shift);
        }
        if (ur_w_tail != 0) {
            const char *kh_loop_tail_label = ".kh_loop_tail";
            add(reg_inp_prf, inp_shift);
            add(reg_out_prf, out_shift);
            compute_loop(ur_w_tail, 0, r_pad, kh_loop_tail_label, 't');
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

    bool large_code_size = jcp.ur_w != jcp.ow && jcp.l_pad > 0 && r_pad > 0
            && ((jcp.l_pad <= 0 && n_oi > 0) || (jcp.l_pad > 0 && n_oi > 1));
    jcp.ur_w = (large_code_size) ? 16 : jcp.ur_w;
    jcp.ur_w_tail = jcp.ow % jcp.ur_w;

    args_ok = true && jcp.oc % simd_w == 0 && jcp.l_pad <= jcp.ur_w
            && implication(mimo, jcp.ic % simd_w == 0);
    if (!args_ok)
        return status::unimplemented;

    int r_pad_no_tail = nstl::max(0, (jcp.ow - jcp.ur_w_tail - 1) * jcp.stride_w
                    + jcp.kw - jcp.iw - jcp.l_pad);
    if (r_pad_no_tail > jcp.ur_w)
        return status::unimplemented;

    jcp.ic_block = (jcp.ic % simd_w != 0) ? jcp.ic : simd_w;
    jcp.nb_ic = jcp.ic / jcp.ic_block;

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
}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
