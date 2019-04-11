/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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
#include "cpu_memory.hpp"

#include "jit_avx512_common_conv_kernel.hpp"

#include <iostream>
#include <cmath>

#define GET_OFF(field) offsetof(jit_conv_call_s, field)
#define KNx_L2_EFFECTIVE_CAPACITY ((512-64)*1024)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

using std::cout;
using std::endl;

namespace {

constexpr auto small_spatial = 14;
unsigned int L1_cache_size = get_cache_size(1, true);

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
    case ver_vnni:
        // TBD: Tune on HW
    case ver_4fma:
        jcp.loop_order
            = (w <= small_spatial && h <= small_spatial) ? loop_cgn : loop_gnc;
        break;
    default:
        assert(!"unsupported convolution version");
    }
}

inline bool is_1stconv(const jit_conv_conf_t &jcp) {
    if (mayiuse(avx512_core) && !mayiuse(avx512_core_vnni))
        return jcp.ic < 16;
    else
        return one_of(jcp.ic, 1, 3);
}

}

void jit_avx512_common_conv_fwd_kernel::prepare_output(int ur_w)
{
    for (int k = 0; k < jcp.nb_oc_blocking; k++)
        for (int j = 0; j < ur_w; j++) {
            Zmm zmm = zmm_out(j, k);
            vpxord(zmm, zmm, zmm);
            size_t aux_output_offset = get_output_offset(j, k);
            mic_prefetcht1(EVEX_compress_addr_safe(reg_out_prf,
                        aux_output_offset, reg_out_long_offt));
        }
}

void jit_avx512_common_conv_fwd_kernel::store_output(int ur_w)
{
    Label no_update_label, store_label, relu_label;

    mov(reg_channel, ptr[param1 + GET_OFF(channel)]);
    if (jcp.with_bias) {
        mov(reg_bias, ptr[param1 + GET_OFF(bias)]);
    }

    if (!jcp.with_sum) {
        cmp(reg_channel, 0);
        je(no_update_label, T_NEAR);
    }

    for (int k = 0; k < jcp.nb_oc_blocking; k++)
        for (int j = 0; j < ur_w; j++) {
            Zmm zmm = zmm_out(j, k);
            size_t aux_output_offset = get_output_offset(j, k);
            vadd(zmm,
                make_safe_addr(reg_out, aux_output_offset, reg_out_long_offt));
        }

    if (!jcp.with_sum) {
        jmp(relu_label, T_NEAR);
    } else {
        cmp(reg_channel, 0);
        jne(relu_label, T_NEAR);
    }

    L(no_update_label);
    if (jcp.with_bias) {
        for (int k = 0; k < jcp.nb_oc_blocking; k++) {
            int bias_offset = jcp.typesize_out * k * jcp.oc_block;
            for (int j = 0; j < ur_w; j++) {
                Zmm zmm = zmm_out(j, k);
                vadd(zmm, EVEX_compress_addr(reg_bias, bias_offset));
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
            mov(imm_addr64, float2int(jcp.relu_negative_slope));
            vmovq(xmm_relu_ns, imm_addr64);
            vbroadcastss(zmm_relu_ns, xmm_relu_ns);
        }
        cmp(reg_channel, jcp.nb_ic - 1);
        jl(store_label, T_NEAR);
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
            size_t aux_output_offset = (size_t)typesize *
                ((size_t)k * jcp.od * jcp.oh * jcp.ow + j) * jcp.oc_block;
            vmovups(EVEX_compress_addr_safe(reg_out, aux_output_offset,
                        reg_out_long_offt), zmm);
            mic_prefetcht0(EVEX_compress_addr_safe(reg_out_prf,
                        aux_output_offset, reg_out_long_offt));
        }
}

void jit_avx512_common_conv_fwd_kernel::compute_loop_4fma_1st(int ur_w,
        int pad_l, int pad_r)
{

    int iw = jcp.iw;
    int ih = jcp.ih;
    int kw = jcp.kw;
    int stride_w = jcp.stride_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;

    Label kh_label, kd_label, skip_kd_loop;

    prepare_output(ur_w);

    if (jcp.ndims == 4) {
        mov(aux_reg_inp, reg_inp);
        mov(aux_reg_ker, reg_ker);
        mov(aux_reg_inp_prf, reg_inp_prf);
    }

    size_t max_input_offset = (size_t)jcp.typesize_in
        * ((size_t)(kw * (jcp.dilate_w + 1) + ur_w * stride_w - pad_l)
                + (size_t)ic_block * iw * ih * jcp.id);
    assert(reg_inp_prf == reg_long_offt);
    if (max_input_offset > INT_MAX) push(reg_inp_prf);

    if (jcp.ndims == 5) {
        push(reg_out_prf);
        push(reg_out);

        mov(reg_ki, ptr[param1 + GET_OFF(kd_padding)]);
        mov(aux_reg_ker_d, ptr[param1 + GET_OFF(filt)]);
        mov(aux_reg_inp_d, reg_inp);
        mov(aux_reg_inp_d_prf, reg_inp_prf);

        if ((jcp.kd - 1) * (jcp.dilate_d + 1) < nstl::max(jcp.f_pad, jcp.back_pad)) {
            cmp(reg_ki, 0);
            je(skip_kd_loop, T_NEAR);
        }
        L(kd_label);
    }
    mov(reg_kj, reg_kh);
    Label skip_kh_loop;
    if ((jcp.kh - 1) * (jcp.dilate_h + 1) < nstl::max(jcp.t_pad, jcp.b_pad)) {
        cmp(reg_kj, 0);
        je(skip_kh_loop, T_NEAR);
    }
    if (jcp.ndims == 5) {
        mov(aux_reg_inp, aux_reg_inp_d);
        mov(aux_reg_ker, aux_reg_ker_d);
        mov(aux_reg_inp_prf, aux_reg_inp_d_prf);
    }

    L(kh_label);
    for (int ki = 0; ki < kw; ki += 4) {
        for (int ic = 0; ic < ic_block; ic++) {
            for (int i = 0; i < 4; i++) {
                int aux_ker_offset
                        = jcp.typesize_in
                        * ((ki + i) * oc_block
                                  + ic * kw * jcp.kh * jcp.kd * oc_block);
                if (ki + i < kw)
                    vmovups(zmm_ker(i),
                        EVEX_compress_addr(aux_reg_ker, aux_ker_offset));
                else
                    vpxord(zmm_ker(i), zmm_ker(i), zmm_ker(i));
            }

            int j_start = get_ow_start(ki, pad_l);
            int j_end = get_ow_end(ur_w, ki, pad_r);

            for (int j = j_start, prf_count=0; j < j_end; j++) {
                size_t aux_input_offset = (size_t)jcp.typesize_in
                        * ((size_t)(ki * (jcp.dilate_w + 1) + j * stride_w
                            - pad_l) + (size_t)ic * iw * ih * jcp.id);
                v4fmaddps(zmm_out(j, 0), zmm_ker(0),
                        EVEX_compress_addr_safe(aux_reg_inp, aux_input_offset,
                        reg_long_offt));
                if (ki + prf_count < kw && prf_count < 4
                    && ((ki < 2 && j % 4) || j % 2)) {
                    int aux_ker_offset = jcp.typesize_in
                        * ((ki + prf_count) * oc_block
                        + ic * kw * jcp.kh * jcp.kd * oc_block + kw * oc_block);
                    mic_prefetcht0(EVEX_compress_addr(aux_reg_ker,
                        aux_ker_offset));
                    prf_count++;
                }
                if (ki == 0
                    && j % (64 / (stride_w * jcp.typesize_in)) == 0) {
                    mic_prefetcht0(EVEX_compress_addr_safe(aux_reg_inp_prf,
                        aux_input_offset, reg_long_offt));
                }
                if (ki == 1
                    && j % (64 / (stride_w * jcp.typesize_in)) == 0) {
                    mic_prefetcht0(EVEX_compress_addr_safe(aux_reg_inp,
                        aux_input_offset+jcp.typesize_in * iw, reg_long_offt));
                }
            }
        }
    }
    add(aux_reg_ker, jcp.typesize_in * kw * oc_block);
    add(aux_reg_inp, jcp.typesize_in * (jcp.dilate_h + 1) * iw);
    add(aux_reg_inp_prf, jcp.typesize_in * (jcp.dilate_h + 1) * iw);

    dec(reg_kj);
    cmp(reg_kj, 0);
    jg(kh_label, T_NEAR);

    L(skip_kh_loop);

    if (jcp.ndims == 5) {
        add(aux_reg_inp_d, typesize * (jcp.dilate_d + 1) * jcp.ih * jcp.iw);
        add(aux_reg_ker_d, typesize * jcp.kw * jcp.kh * oc_block);
        add(aux_reg_inp_d_prf, typesize * (jcp.dilate_d + 1) * jcp.ih * jcp.iw);

        dec(reg_ki);
        cmp(reg_ki, 0);
        jg(kd_label, T_NEAR);
        L(skip_kd_loop);

        pop(reg_out);
        pop(reg_out_prf);
    }

    store_output(ur_w);
    if (max_input_offset > INT_MAX) pop(reg_inp_prf);
}

void jit_avx512_common_conv_fwd_kernel::compute_loop_4fma(int ur_w,
        int pad_l, int pad_r)
{
    int stride_w = jcp.stride_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    Label kh_label, last_iter_label, loop_end_label, kd_label, skip_kd_loop;
    int ker_load_number = 4;
    int shift_kernel_ptr = typesize * jcp.kw * jcp.oc_block * jcp.ic_block;
    int shift_input_ptr = typesize * (jcp.dilate_h + 1) * jcp.iw * jcp.ic_block;

    bool check_last_kh = (jcp.kh > 3);
    bool pref_current_inp = (jcp.iw < 14 || jcp.iw > 28);

    int oi_ipref_t0 = get_ow_start(0, pad_l);
    int ow_end_ipref = get_ow_end(ur_w, 0, pad_r);

    assert(jcp.oc % jcp.nb_oc_blocking == 0);

    auto kernel_offset = [=](int ocb, int ic, int ki) {
        int blk_idx = ocb * jcp.nb_ic * jcp.kh * jcp.kw * jcp.kd + ki;
        int blk_offset = blk_idx * jcp.oc_block * jcp.ic_block;
        int ic_offset = ic * jcp.oc_block;
        return typesize * (blk_offset + ic_offset);
    };
    auto kernel_loads = [=](int ki, int ic, int kk) {
        for (int ii = 0; ii < ker_load_number; ii++) {
            int aux_kernel_offset = kernel_offset(kk, ic + ii, ki);
            vmovups(zmm_ker(ii),
                EVEX_compress_addr(aux_reg_ker, aux_kernel_offset));
        }
    };
    auto prefetch_inp_next_kh = [&](int ki, int ki_start, int cnt0, int cnt1) {
        if (cnt1 >= ker_load_number && cnt0 >= ker_load_number
            && ki >= ki_start && oi_ipref_t0 < ow_end_ipref) {
            int aux_inp_offset
                    = typesize
                    * ((oi_ipref_t0 * stride_w - pad_l) * ic_block
                              + (jcp.dilate_h + 1) * jcp.iw * ic_block);
            prefetcht0(EVEX_compress_addr(aux_reg_inp,
                    aux_inp_offset));
            oi_ipref_t0++;
        }
    };

    prepare_output(ur_w);

    if (jcp.ndims == 4) {
        mov(aux_reg_inp, reg_inp);
        mov(aux_reg_ker, reg_ker);
        mov(aux_reg_ker_prf, reg_ker_prf);
        mov(aux_reg_inp_prf, reg_inp_prf);
    }

    if (jcp.ndims == 5) {
        push(reg_out_prf);
        push(reg_out);

        mov(reg_ki, ptr[param1 + GET_OFF(kd_padding)]);
        mov(aux_reg_ker_d, ptr[param1 + GET_OFF(filt)]);
        mov(aux_reg_inp_d, reg_inp);
        mov(aux_reg_inp_d_prf, reg_inp_prf);
        mov(aux_reg_ker_d_prf, reg_ker_prf);

        if ((jcp.kd - 1) * (jcp.dilate_d + 1) < nstl::max(jcp.f_pad, jcp.back_pad)) {
            cmp(reg_ki, 0);
            je(skip_kd_loop, T_NEAR);
        }
        L(kd_label);
        mov(reg_kj, ptr[param1 + GET_OFF(kh_padding)]);
    } else {
        mov(reg_kj, reg_kh);
    }
    Label skip_kh_loop;
    if ((jcp.kh - 1) * (jcp.dilate_h + 1) < nstl::max(jcp.t_pad, jcp.b_pad)) {
        cmp(reg_kj, 0);
        je(skip_kh_loop, T_NEAR);
    }
    if (jcp.ndims == 5) {
        mov(aux_reg_inp, aux_reg_inp_d);
        mov(aux_reg_ker, aux_reg_ker_d);
        mov(aux_reg_ker_prf, aux_reg_ker_d_prf);
        mov(aux_reg_inp_prf, aux_reg_inp_d_prf);
    }

    align(16);
    L(kh_label);
    int kw = jcp.kw;
    if (check_last_kh) {
        for (int ki = 0; ki < kw; ki++)
            for (int ic = 0; ic < ic_block; ic += 4)
                for (int kk = 0; kk < jcp.nb_oc_blocking; kk++) {
                    bool last_kernel_loads = (kk == jcp.nb_oc_blocking - 1
                        && ki == kw - 1 && (ic + 4) == ic_block);

                    if (last_kernel_loads) {
                        cmp(reg_kj, 1);
                        je(last_iter_label, T_NEAR);
                    }

                    kernel_loads(ki, ic, kk);
                    for (int oi = get_ow_start(ki, pad_l), prf_count_t1 = 0,
                             prf_count_t0 = 0;
                            oi < get_ow_end(ur_w, ki, pad_r); oi++) {
                        int aux_input_offset = typesize
                                * ((ki * (jcp.dilate_w + 1) + oi * stride_w
                                           - pad_l) * ic_block
                                                       + ic);
                        v4fmaddps(zmm_out(oi, kk), zmm_ker(0),
                            EVEX_compress_addr(aux_reg_inp, aux_input_offset));

                        if (oi % 2) {
                            if (prf_count_t0 < 4) {
                                int aux_kernel_prf;
                                if (last_kernel_loads)
                                    aux_kernel_prf= kernel_offset(0,
                                        prf_count_t0 + ic + 4
                                        - ic_block, 0) + typesize * kw
                                        * oc_block * ic_block;
                                else
                                    aux_kernel_prf = kernel_offset(kk, ic + 4
                                        + prf_count_t0, ki);
                                mic_prefetcht0(EVEX_compress_addr(aux_reg_ker,
                                    aux_kernel_prf));
                                prf_count_t0++;
                            } else if (prf_count_t1 < 4) {
                                mic_prefetcht1(EVEX_compress_addr(
                                    aux_reg_ker_prf, kernel_offset(kk, ic
                                    + prf_count_t1, ki)));
                                prf_count_t1++;
                            }
                        } else
                           prefetch_inp_next_kh(ki, 2, prf_count_t0,
                               prf_count_t1);
                    }

                    if (last_kernel_loads) {
                        jmp(loop_end_label, T_NEAR);

                        L(last_iter_label);

                        kernel_loads(ki, ic, kk);
                        for (int oi = get_ow_start(ki, pad_l), prf_count_t1 = 0,
                                 prf_count_t0 = 0;
                                oi < get_ow_end(ur_w, ki, pad_r); oi++) {
                            int aux_input_offset = typesize
                                    * ((ki * (jcp.dilate_w + 1) + oi * stride_w
                                               - pad_l) * ic_block
                                                           + ic);
                            v4fmaddps(zmm_out(oi, kk), zmm_ker(0),
                                EVEX_compress_addr(aux_reg_inp,
                                    aux_input_offset));
                            if (oi % 2) {
                                if (prf_count_t0 < 4) {
                                    mic_prefetcht0(EVEX_compress_addr(
                                        aux_reg_ker_prf, kernel_offset(0,
                                        prf_count_t0, 0)));
                                    prf_count_t0++;
                                } else if (prf_count_t1 < 4) {
                                    mic_prefetcht1(EVEX_compress_addr(
                                        aux_reg_ker_prf, kernel_offset(kk,
                                        ic + prf_count_t1, ki)));
                                    prf_count_t1++;
                                }
                            }
                        }
                        L(loop_end_label);
                    }
                }
    } else {
        for (int ki = 0; ki < kw; ki++)
            for (int ic = 0; ic < ic_block; ic += 4)
                for (int kk = 0; kk < jcp.nb_oc_blocking; kk++) {
                    kernel_loads(ki, ic, kk);
                    for (int oi = get_ow_start(ki, pad_l), prf_count_t1 = 0;
                            oi < get_ow_end(ur_w, ki, pad_r); oi++) {
                        int aux_input_offset = typesize
                                * ((ki * (jcp.dilate_w + 1) + oi * stride_w
                                           - pad_l) * ic_block
                                                       + ic);
                        v4fmaddps(zmm_out(oi, kk), zmm_ker(0),
                            EVEX_compress_addr(aux_reg_inp,
                                aux_input_offset));
                        if ((oi % 2) && (prf_count_t1 < 4)) {
                            mic_prefetcht1(EVEX_compress_addr(
                                aux_reg_ker_prf, kernel_offset(kk,
                                ic + prf_count_t1, ki)));
                            prf_count_t1++;
                        }
                        if (pref_current_inp) {
                            if (ki == 0 && ic == 0 && kk == 0)
                                mic_prefetcht0(EVEX_compress_addr(
                                    aux_reg_inp,
                                    aux_input_offset+shift_input_ptr));
                        } else {
                            if (ki == 1 && ic == 0 && kk == 0)
                                mic_prefetcht1(EVEX_compress_addr(
                                    aux_reg_inp_prf, aux_input_offset));
                        }
                    }
                }
    }

    add(aux_reg_ker, shift_kernel_ptr);
    add(aux_reg_inp, shift_input_ptr);
    add(aux_reg_ker_prf, shift_kernel_ptr);
    add(aux_reg_inp_prf, shift_input_ptr);

    dec(reg_kj);
    cmp(reg_kj, 0);
    jg(kh_label, T_NEAR);

    L(skip_kh_loop);

    if (jcp.ndims == 5) {
        add(aux_reg_inp_d,
                typesize * (jcp.dilate_d + 1) * jcp.ih * jcp.iw * jcp.ic_block);
        add(aux_reg_ker_d, typesize * jcp.kw * jcp.kh * jcp.oc_block
                * jcp.ic_block);
        add(aux_reg_inp_d_prf,
                typesize * (jcp.dilate_d + 1) * jcp.ih * jcp.iw * jcp.ic_block);
        add(aux_reg_ker_d_prf, typesize * jcp.kw * jcp.kh * jcp.oc_block
                * jcp.ic_block);

        dec(reg_ki);
        cmp(reg_ki, 0);
        jg(kd_label, T_NEAR);
        L(skip_kd_loop);

        pop(reg_out);
        pop(reg_out_prf);
    }

    store_output(ur_w);
}

void jit_avx512_common_conv_fwd_kernel::compute_loop_fma(int ur_w,
        int pad_l, int pad_r)
{
    bool prf_ker = true;
    bool prf_inp = true;
    int ih = jcp.ih;
    int stride_w = jcp.stride_w;
    int id = jcp.id;
    int iw = jcp.iw;
    int kw = jcp.kw;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int nb_oc_block = jcp.nb_oc_blocking;
    Label kh_label, kd_label, skip_kd_loop;

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
    int inp_mul = !jcp.is_1stconv ? ic_block : 1;

    prepare_output(ur_w);

    if (jcp.ndims == 4) {
        mov(aux_reg_inp, reg_inp);
        mov(aux_reg_ker, reg_ker);
        mov(aux_reg_inp_prf, reg_inp_prf);
        mov(aux_reg_ker_prf, reg_ker_prf);
    }

    size_t max_input_offset = (size_t)jcp.typesize_in * ic_block * iw * ih * id;
    assert(reg_inp_prf == reg_long_offt);
    if (max_input_offset > INT_MAX) push(reg_inp_prf);


    if (jcp.ndims == 5) {
        push(reg_out_prf);
        push(reg_out);

        mov(reg_ki, ptr[param1 + GET_OFF(kd_padding)]);
        mov(aux_reg_ker_d, ptr[param1 + GET_OFF(filt)]);
        mov(aux_reg_inp_d, reg_inp);
        mov(aux_reg_inp_d_prf, reg_inp_prf);
        mov(aux_reg_ker_d_prf, reg_ker_prf);

        if ((jcp.kd - 1) * (jcp.dilate_d + 1) < nstl::max(jcp.f_pad, jcp.back_pad)) {
            cmp(reg_ki, 0);
            je(skip_kd_loop, T_NEAR);
        }
        L(kd_label);
        mov(reg_kj, ptr[param1 + GET_OFF(kh_padding)]);
    } else {
        mov(reg_kj, reg_kh);
    }
    Label skip_kh_loop;
    if ((jcp.kh - 1) * (jcp.dilate_h + 1) < nstl::max(jcp.t_pad, jcp.b_pad)) {
        cmp(reg_kj, 0);
        je(skip_kh_loop, T_NEAR);
    }

    if (jcp.ndims == 5) {
        mov(aux_reg_inp, aux_reg_inp_d);
        mov(aux_reg_ker, aux_reg_ker_d);
        mov(aux_reg_ker_prf, aux_reg_ker_d_prf);
        mov(aux_reg_inp_prf, aux_reg_inp_d_prf);
    }

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
                    aux_kernel_offset
                            = get_kernel_offset(ki, ic, 0, load_offset);
                    vmovups(zmm_ker(ker_load_reg_idx),
                            EVEX_compress_addr(aux_reg_ker, aux_kernel_offset));
                }

                bool ker_prf_inserted = false;
                Zmm zmm_kernel = zmm_ker(step % ker_pipeline_depth);
                int j_start = get_ow_start(ki, pad_l);
                int j_end = get_ow_end(ur_w, ki, pad_r);
                for (int j = j_start; j < j_end; j++) {
                    size_t aux_input_offset = get_input_offset(ki, ic, j, pad_l);
                    auto addr = EVEX_compress_addr_safe(aux_reg_inp,
                            aux_input_offset, reg_long_offt, true);
                    vfmadd231ps(zmm_out(j, 0), zmm_kernel, addr);
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
                                size_t inp_prf_stride = nstl::max(kw, stride_w);
                                size_t inp_prf_offset;
                                if (!jcp.is_1stconv) {
                                    inp_prf_offset
                                            = ic_block * jcp.typesize_in
                                            * ((inp_prf_idx / kw)
                                            * inp_prf_stride
                                            + (inp_prf_idx % kw));
                                } else {
                                    size_t ic_prf_stride =
                                        (size_t)jcp.typesize_in * iw * ih * id;
                                    size_t iw_prf_stride
                                            = jcp.typesize_in * simd_w;
                                    inp_prf_offset = ((inp_prf_idx / ic_block)
                                            * iw_prf_stride
                                            + (inp_prf_idx % ic_block)
                                            * ic_prf_stride);
                                }
                                mic_prefetcht0(EVEX_compress_addr_safe(
                                        aux_reg_inp_prf, inp_prf_offset,
                                        reg_long_offt));
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
        add(aux_reg_inp, jcp.typesize_in * (jcp.dilate_h + 1) * iw * inp_mul);
        if (prf_inp)
            add(aux_reg_inp_prf,
                    jcp.typesize_in * (jcp.dilate_h + 1) * iw * inp_mul);
        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }

    L(skip_kh_loop);

    if (jcp.ndims == 5) {
        add(aux_reg_inp_d,
                typesize * (jcp.dilate_d + 1) * jcp.ih * jcp.iw * inp_mul);
        add(aux_reg_ker_d, typesize * jcp.kw * jcp.kh * jcp.oc_block
                * jcp.ic_block);
        add(aux_reg_inp_d_prf,
                typesize * (jcp.dilate_d + 1) * jcp.ih * jcp.iw * inp_mul);
        add(aux_reg_ker_d_prf, typesize * jcp.kw * jcp.kh * jcp.oc_block
                * jcp.ic_block);

        dec(reg_ki);
        cmp(reg_ki, 0);
        jg(kd_label, T_NEAR);
        L(skip_kd_loop);

        pop(reg_out);
        pop(reg_out_prf);
    }
    if (max_input_offset > INT_MAX) pop(reg_inp_prf);
    store_output(ur_w);
}

void jit_avx512_common_conv_fwd_kernel::compute_loop_fma_core(int ur_w,
    int pad_l, int pad_r)
{
    int kw = jcp.kw;
    int stride_w = jcp.stride_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int nb_oc_block = jcp.nb_oc_blocking;
    Label kh_label, skip_kh_loop, kd_label, skip_kd_loop;
    int shift_kernel_ptr = jcp.typesize_in * jcp.kw * jcp.oc_block
        * jcp.ic_block;
    int inp_mul = !jcp.is_1stconv ? ic_block : 1;
    int shift_input_ptr = jcp.typesize_in * (jcp.dilate_h + 1) * jcp.iw
        * inp_mul;


    auto input_offset = [=](int oi, int ic, int ki) {
        return (size_t)jcp.typesize_in
                * ((size_t)(ki * (jcp.dilate_w + 1) + oi * stride_w - pad_l)
                * inp_mul + (size_t)ic
                * (!jcp.is_1stconv ? 1 : (size_t)jcp.iw * jcp.ih * jcp.id));
    };

    prepare_output(ur_w);

    if (jcp.ndims == 4) {
        mov(aux_reg_inp, reg_inp);
        mov(aux_reg_ker, reg_ker);
    }

    if (jcp.ndims == 5) {
        push(reg_out);

        mov(reg_ki, ptr[param1 + GET_OFF(kd_padding)]);
        mov(aux_reg_ker_d, ptr[param1 + GET_OFF(filt)]);
        mov(aux_reg_inp_d, reg_inp);

        if ((jcp.kd - 1) * (jcp.dilate_d + 1) < nstl::max(jcp.f_pad, jcp.back_pad)) {
            cmp(reg_ki, 0);
            je(skip_kd_loop, T_NEAR);
        }
        L(kd_label);
        mov(reg_kj, ptr[param1 + GET_OFF(kh_padding)]);
    } else {
        mov(reg_kj, reg_kh);
    }
    if ((jcp.kh - 1) * (jcp.dilate_h + 1) < nstl::max(jcp.t_pad, jcp.b_pad)) {
        cmp(reg_kj, 0);
        je(skip_kh_loop, T_NEAR);
    }

    if (jcp.ndims == 5) {
        mov(aux_reg_inp, aux_reg_inp_d);
        mov(aux_reg_ker, aux_reg_ker_d);
    }

    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = get_ow_start(ki, pad_l);
            int jj_end = get_ow_end(ur_w, ki, pad_r);
            for (int ic = 0; ic < ic_block; ic++) {
                if (jcp.kernel_kind == expl_bcast) {
                    for (int jj = jj_start; jj < jj_end; jj++) {
                        size_t aux_input_offset = input_offset(jj, ic, ki);
                        vbroadcastss(zmm_inp(jj, nb_oc_block),
                            EVEX_compress_addr_safe(aux_reg_inp,
                            aux_input_offset, reg_long_offt));
                    }
                }
                for (int ii = 0; ii < nb_oc_block; ii++) {
                    int aux_kernel_offset = jcp.typesize_in
                        * (ii * jcp.nb_ic * jcp.kh * jcp.kw * jcp.kd * ic_block
                        * oc_block + ki * ic_block * oc_block + ic * oc_block);
                    if (jj_end - jj_start > 0)
                        vmovups(zmm_wei, EVEX_compress_addr(aux_reg_ker,
                            aux_kernel_offset));
                    for (int jj = jj_start; jj < jj_end; jj++)
                        if (jcp.kernel_kind == expl_bcast)
                            vfmadd231ps(zmm_out(jj, ii),
                                zmm_inp(jj, nb_oc_block), zmm_wei);
                        else {
                            size_t aux_input_offset = input_offset(jj, ic, ki);
                            vfmadd231ps(zmm_out(jj, ii), zmm_wei,
                                EVEX_compress_addr_safe(aux_reg_inp,
                                aux_input_offset, reg_long_offt, true));
                        }
                }
            }
        }
        add(aux_reg_ker, shift_kernel_ptr);
        add(aux_reg_inp, shift_input_ptr);
        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }
    L(skip_kh_loop);

    if (jcp.ndims == 5) {
        add(aux_reg_inp_d,
                typesize * (jcp.dilate_d + 1) * jcp.ih * jcp.iw * inp_mul);
        add(aux_reg_ker_d, typesize * jcp.kw * jcp.kh * jcp.oc_block
                * jcp.ic_block);

        dec(reg_ki);
        cmp(reg_ki, 0);
        jg(kd_label, T_NEAR);
        L(skip_kd_loop);

        pop(reg_out);
    }

    store_output(ur_w);
}

void jit_avx512_common_conv_fwd_kernel::compute_loop_vnni(
        int ur_w, int pad_l, int pad_r)
{
    Label kh_label, kd_label;
    const int ker_reg_base_idx = 28;
    const int channel_inc = jcp.ver == ver_4vnni ? 4 : 1;
    const int ker_load_number = jcp.ver == ver_4vnni ? 4 : 1;
    const int shift_kernel_ptr = jcp.typesize_in * jcp.kw
                               * jcp.oc_block * jcp.ic_block;
    const int shift_input_ptr
            = jcp.typesize_in * (jcp.dilate_h + 1) * jcp.iw * jcp.ic_block;

    size_t max_input_offset = (size_t)jcp.typesize_in
                * jcp.ic_block * jcp.iw * jcp.ih * jcp.id;
    assert(reg_inp_prf == reg_long_offt);
    if (max_input_offset > INT_MAX) push(reg_inp_prf);

    prepare_output(ur_w);

    if (jcp.ndims == 4) {
        mov(aux_reg_inp, reg_inp);
        mov(aux_reg_ker, reg_ker);
        mov(aux_reg_ker_prf, reg_ker_prf);
        mov(aux_reg_inp_prf, reg_inp_prf);
    }

    Label skip_kh_loop, skip_kd_loop;

    if (jcp.ndims == 5) {
        push(reg_out_prf);
        push(reg_out);

        mov(reg_ki, ptr[param1 + GET_OFF(kd_padding)]);
        mov(aux_reg_ker_d, ptr[param1 + GET_OFF(filt)]);
        mov(aux_reg_inp_d, reg_inp);
        mov(aux_reg_inp_d_prf, reg_inp_prf);
        mov(aux_reg_ker_d_prf, reg_ker_prf);

        if ((jcp.kd - 1) * (jcp.dilate_d + 1) < nstl::max(jcp.f_pad, jcp.back_pad)) {
            cmp(reg_ki, 0);
            je(skip_kd_loop, T_NEAR);
        }
        L(kd_label);
        mov(reg_kj, ptr[param1 + GET_OFF(kh_padding)]);
    } else {
        mov(reg_kj, reg_kh);
    }
    if ((jcp.kh - 1) * (jcp.dilate_h + 1) < nstl::max(jcp.t_pad, jcp.b_pad)) {
        cmp(reg_kj, 0);
        je(skip_kh_loop, T_NEAR);
    }
    if (jcp.ndims == 5) {
        mov(aux_reg_inp, aux_reg_inp_d);
        mov(aux_reg_ker, aux_reg_ker_d);
        mov(aux_reg_ker_prf, aux_reg_ker_d_prf);
        mov(aux_reg_inp_prf, aux_reg_inp_d_prf);
    }

    L(kh_label); {
        for (int ki = 0; ki < jcp.kw; ki++) {
            int ow_start = get_ow_start(ki, pad_l);
            int ow_end = get_ow_end(ur_w, ki, pad_r);
            for (int ic = 0; ic < jcp.ic_block / 2; ic += channel_inc) {
                if (jcp.kernel_kind == expl_bcast) {
                    for (int oi = ow_start; oi < ow_end; oi++) {
                        size_t input_offset = get_input_offset(ki, ic, oi, pad_l);
                        vpbroadcastd(zmm_inp(oi, jcp.nb_oc_blocking),
                            EVEX_compress_addr_safe(aux_reg_inp, input_offset,
                            reg_long_offt));
                    }
                }
                for (int kk = 0; kk < jcp.nb_oc_blocking; kk++) {
                    if (jcp.kernel_kind == expl_bcast) {
                        int kernel_offset = get_kernel_offset(ki, ic, kk, 0);
                        vmovups(zmm_wei,
                            EVEX_compress_addr(aux_reg_ker, kernel_offset));
                    } else {
                        for (int ii = 0; ii < ker_load_number; ii++) {
                            int kernel_offset
                                = get_kernel_offset(ki, ic, kk, ii);
                            vmovups(Zmm(ker_reg_base_idx + ii),
                                    EVEX_compress_addr(
                                            aux_reg_ker, kernel_offset));
                        }
                    }
                    for (int oi = ow_start, prf_count = 0; oi < ow_end; oi++) {
                        size_t input_offset = get_input_offset(ki, ic, oi, pad_l);
                        if (jcp.kernel_kind == expl_bcast) {
                            vpdpwssd(zmm_out(oi, kk), zmm_wei,
                                zmm_inp(oi, jcp.nb_oc_blocking));
                        } else {
                            vpXdpwssd(zmm_out(oi, kk), Zmm(ker_reg_base_idx),
                            EVEX_compress_addr_safe(aux_reg_inp, input_offset,
                            reg_long_offt, jcp.ver != ver_4vnni));
                        }
                        if ((oi % 2) && (prf_count < ker_load_number)) {
                            int kernel_offset = get_kernel_offset(
                                ki, ic, kk, prf_count++);
                            mic_prefetcht0(EVEX_compress_addr(aux_reg_ker_prf,
                                kernel_offset));
                        }
                        if (!(oi % 2) && ki == 0 && ic == 0 && kk == 0) {
                            mic_prefetcht1(EVEX_compress_addr_safe(
                                aux_reg_inp_prf, input_offset, reg_long_offt));
                        }
                        if (!(oi % 2) && ki == 1 && ic == 0 && kk == 0) {
                            mic_prefetcht0(EVEX_compress_addr_safe(aux_reg_inp,
                                input_offset + shift_input_ptr, reg_long_offt));
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

    L(skip_kh_loop);

    if (jcp.ndims == 5) {
        add(aux_reg_inp_d, jcp.typesize_in * jcp.ih * jcp.iw * jcp.ic_block);
        add(aux_reg_ker_d, jcp.typesize_in * jcp.kw * jcp.kh * jcp.oc_block
                * jcp.ic_block);
        add(aux_reg_inp_d_prf, jcp.typesize_in * jcp.ih * jcp.iw * jcp.ic_block);
        add(aux_reg_ker_d_prf, jcp.typesize_in * jcp.kw * jcp.kh * jcp.oc_block
                * jcp.ic_block);

        dec(reg_ki);
        cmp(reg_ki, 0);
        jg(kd_label, T_NEAR);
        L(skip_kd_loop);

        pop(reg_out);
        pop(reg_out_prf);
    }
    if (max_input_offset > INT_MAX) pop(reg_inp_prf);
    store_output(ur_w);
}

void jit_avx512_common_conv_fwd_kernel::compute_loop(int ur_w,
        int pad_l, int pad_r)
{
    if (jcp.ndims == 5) push(reg_oi);
    if (jcp.ver == ver_4vnni || jcp.ver == ver_vnni)
        compute_loop_vnni(ur_w, pad_l, pad_r);
    else if (jcp.ver == ver_4fma)
        if(jcp.is_1stconv)
            compute_loop_4fma_1st(ur_w, pad_l, pad_r);
        else
            compute_loop_4fma(ur_w, pad_l, pad_r);
    else if (jcp.ver == ver_fma)
        if ((jcp.is_1stconv && jcp.kernel_kind != expl_bcast)
                || mayiuse(avx512_mic))
            compute_loop_fma(ur_w, pad_l, pad_r);
        else
            if (jcp.kernel_kind == embd_bcast && jcp.nb_oc_blocking == 1)
                compute_loop_fma(ur_w, pad_l, pad_r);
            else
                compute_loop_fma_core(ur_w, pad_l, pad_r);
    else
        assert(!"unknown convolution version");
    if (jcp.ndims == 5) pop(reg_oi);
}

void jit_avx512_common_conv_fwd_kernel::compute_loop_fma_sparse() {

    Label kh_label, kd_label, skip_kd_loop;
    Label end_label, clear_label;

    /**********************************************

    reg64_t param = abi_param1;
    reg64_t reg_inp = r8;
    reg64_t reg_ker = r9;
    reg64_t reg_out = r10;

    reg64_t reg_inp_prf = r11;
    reg64_t reg_ker_prf = r12;
    reg64_t reg_out_prf = r13;

    reg64_t aux_reg_inp = r14;
    reg64_t aux_reg_ker = r15;

    reg64_t aux_reg_inp_prf = rsi;
    reg64_t aux_reg_ker_prf = rdx;

    reg64_t reg_channel = rsi;
    reg64_t reg_bias = rdx;

    reg64_t aux_reg_ker_d = r9;
    reg64_t aux_reg_inp_d = rbx;
    reg64_t aux_reg_inp_d_prf = r13;
    reg64_t aux_reg_ker_d_prf = abi_not_param1;
    reg64_t reg_ki = r10;

    reg64_t reg_kj = rax;
    reg64_t reg_relu_ns = rax;
    reg64_t reg_oi = rbx;
    reg64_t reg_kh = abi_not_param1;

    reg64_t reg_tmp = rbp;

    reg64_t reg_ic_loop = rdx;
    reg64_t reg_inp_loop = rsi;

    reg64_t reg_init_flag = r13;
    reg64_t reg_bias_ptr = param;

    reg64_t aux_reg_ic = r12;
    reg64_t reg_binp = rax;
    reg64_t reg_bout = r11;
    reg64_t aux1_reg_inp = rbx;
    reg64_t aux_reg_out = abi_not_param1;

    reg64_t reg_long_offt = r11;
    reg64_t reg_out_long_offt = r14;

    ***********************************************/

    int kw = jcp.kw;
    int kh = jcp.kh;

    int ow = jcp.ow;
    int oh = jcp.oh;

    int nb_ic = jcp.nb_ic;
    int nb_oc = jcp.nb_oc;

    int iw = jcp.iw;
    int ih = jcp.ih;

    int mb = jcp.mb;
    int mb_block = jcp.mb_block;

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int l_pad = jcp.l_pad;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;
    int stride_h = jcp.stride_h;

    int ker_pipeline_depth = 4;
    assert(ker_reg_base_idx + ker_pipeline_depth <= 32);
    assert(ic_block >= ker_pipeline_depth);

    int nr = jcp.ur_sparse;
    int oc_buffs = jcp.oc_buffs;

    assert(nr >= kw);
    assert(oc_block == 16); // needed for specific optimization
    assert(typesize == 4);

    int oc_iters = nb_oc / oc_buffs;

    auto zmm_o = Xbyak::Zmm(31);

    auto ymm_zero = Xbyak::Ymm(30);
    auto zmm_zero = Xbyak::Zmm(30);

    vpxord(zmm_zero, zmm_zero, zmm_zero);

    vcmpps(k7, zmm_zero, ptr[reg_inp], 4);
    prefetcht1(ptr[reg_inp_prf]);

    assert(nr % kw == 0);

    Label no_init_label;

    mov(reg_channel, ptr[param + GET_OFF(channel)]);
    cmp(reg_channel, 0);
    jne(no_init_label, T_NEAR);

    if (oc_buffs * ow > 128 || jcp.with_bias) { // threshold may be tweaked later

        Reg64 aux_reg_out = aux_reg_inp;
        Reg64 aux_reg_out_prf = aux_reg_ker;

        if (jcp.with_bias) {
            mov(reg_bias, ptr[param1 + GET_OFF(bias)]);
        }

        mov(aux_reg_out, reg_out);
        mov(aux_reg_out_prf, reg_out_prf);

        mov(reg_channel, oc_buffs);

        Label oc_loop_label;
        L(oc_loop_label);

        if (jcp.with_bias) {
            vmovups(zmm_zero, ptr[reg_bias]);
            add(reg_bias, typesize * oc_block);
        }

        for (int oi = 0; oi < ow; oi++) {
            vmovups(EVEX_compress_addr_safe(aux_reg_out, oi * oc_block * typesize,
                        reg_long_offt), zmm_zero);
            prefetcht1(EVEX_compress_addr_safe(aux_reg_out_prf, oi * oc_block * typesize,
                        reg_long_offt));
        }

        add(aux_reg_out, typesize * oc_block * mb_block * ow);
        add(aux_reg_out_prf, typesize * oc_block * mb_block * ow);

        dec(reg_channel);
        cmp(reg_channel, 0);
        jne(oc_loop_label);

        if (jcp.with_bias) {
            vpxord(zmm_zero, zmm_zero, zmm_zero);
        }

    } else {

        for (int oc = 0; oc < oc_buffs; oc++) {
            for (int oi = 0; oi < ow; oi++) {
                vmovups(EVEX_compress_addr_safe(reg_out, (oc * oc_block * ow 
                            + oi * oc_block) * typesize,
                            reg_long_offt), zmm_zero);
                prefetcht1(EVEX_compress_addr_safe(reg_out_prf, (oc * oc_block * ow 
                            + oi * oc_block) * typesize,
                            reg_long_offt));
            }
        }
    }

    L(no_init_label);

    cmp(reg_kh, 0);
    je(end_label, T_NEAR);

    Reg64 reg_long_offt = reg_kj;

    auto get_reg_idx = [=](int oi, int oc_buff) {
        
        if (oc_buffs * (kw + 1) <= 30) {
            return oc_buff * (kw + 1) + oi % (kw + 1);
        } else {
            return oc_buff * kw + oi % kw;
        }
    };

    auto comp_unrolled = [&](int ii, int step, int cur_oc_buffs) {

        Reg64 mask_reg = reg_oi;
        Reg32 ic_itr_reg = reg_kh.cvt32();
        Reg64 lzcnt_reg = reg_channel;

        kmovw(mask_reg.cvt32(), k7);
        popcnt(ic_itr_reg, mask_reg.cvt32());

        if (ii < iw - step) { // pipelined
            size_t aux_src_offset = typesize * (ii + step) * ic_block;
            vcmpps(k7, zmm_zero, EVEX_compress_addr_safe(reg_inp, aux_src_offset,
                                    reg_long_offt), 4);

            prefetcht1(EVEX_compress_addr_safe(reg_inp_prf, aux_src_offset,
                                    reg_long_offt));
        }

        cout << ii << ":";

        if (ii >= step) {

            cout << " st:";

            for (int ki = kw - 1; ki > -1; ki--) {
                for (int oc_buff = 0; oc_buff < cur_oc_buffs; oc_buff++) {

                    int n = (ii - step) - ki * dilate_w + l_pad;

                    if (n >= 0 && n / stride_w < ow && n % stride_w == 0) {

                        int oi = n / stride_w;

                        int reg_idx = get_reg_idx(oi, oc_buff);

                        Zmm zmm = Xbyak::Zmm(reg_idx);

                        if (stride_w / dilate_w >= kw || dilate_w > stride_w || stride_w % dilate_w != 0
                            || ki == kw - 1) {

                            if (oc_buff == 0) {
                                cout << " " << oi << "-r" << reg_idx;
                            }
                        
                            size_t aux_dst_offset = (size_t)typesize
                                * (oc_buff * oc_block * ow * mb_block
                                + oi * oc_block);
                            vmovups(EVEX_compress_addr_safe(reg_out, aux_dst_offset,
                                        reg_long_offt), zmm);
                        }
                    }
                }
            }
        }

        cout << " ld:";

        if (cur_oc_buffs * (kw + 1) <= 30) {

            if (ii == 0) {
                for (int ki = kw - 1; ki > -1; ki--) {
                    for (int oc_buff = 0; oc_buff < cur_oc_buffs; oc_buff++) {

                        int n = - ki * dilate_w + l_pad;

                        if (n >= 0 && n / stride_w < ow && n % stride_w == 0) {

                            int oi = n / stride_w;

                            int reg_idx = get_reg_idx(oi, oc_buff);

                            Zmm zmm = Xbyak::Zmm(reg_idx);

                            if (oc_buff == 0) {
                                cout << " " << oi << "-r" << reg_idx;
                            }

                            size_t aux_dst_offset = (size_t)typesize
                                * (oc_buff * oc_block * ow * mb_block
                                + oi * oc_block);

                            vmovups(zmm, EVEX_compress_addr_safe(reg_out, aux_dst_offset,
                                        reg_long_offt));
                            //prefetcht1(EVEX_compress_addr_safe(reg_out_prf, aux_dst_offset,
                                        //reg_long_offt));

                        }
                    }
                }

            }
            if (ii < iw - step) {
            
                for (int ki = kw - 1; ki > -1; ki--) {
                    for (int oc_buff = 0; oc_buff < cur_oc_buffs; oc_buff++) {

                        int n = (ii + step) - ki * dilate_w + l_pad;

                        if (n >= 0 && n / stride_w < ow && n % stride_w == 0) {

                            int oi = n / stride_w;

                            int reg_idx = get_reg_idx(oi, oc_buff);

                            Zmm zmm = Xbyak::Zmm(reg_idx);

                            if (stride_w / dilate_w >= kw || dilate_w > stride_w || stride_w % dilate_w != 0
                                || ki == 0) {

                                if (oc_buff == 0) {
                                    cout << " " << oi << "-r" << reg_idx;
                                }

                                size_t aux_dst_offset = (size_t)typesize
                                    * (oc_buff * oc_block * ow * mb_block
                                    + oi * oc_block);

                                vmovups(zmm, EVEX_compress_addr_safe(reg_out, aux_dst_offset,
                                            reg_long_offt));
                                //prefetcht1(EVEX_compress_addr_safe(reg_out_prf, aux_dst_offset,
                                            //reg_long_offt));

                            }
                        }
                    }
                }
            }/* else {
                for (int ki = kw - 1; ki > -1; ki--) {
                    for (int oc_buff = 0; oc_buff < cur_oc_buffs; oc_buff++) {

                        int n = - ki * dilate_w + l_pad;

                        if (n >= 0 && n / stride_w < ow && n % stride_w == 0) {

                            int oi = n / stride_w;

                            size_t aux_dst_offset = (size_t)typesize
                                * (oc_buff * oc_block * ow * mb_block
                                + oi * oc_block);

                            //prefetcht1(EVEX_compress_addr_safe(reg_out_prf, aux_dst_offset,
                            //            reg_long_offt));

                        }
                    }
                }

            }*/

        } else {
            for (int ki = kw - 1; ki > -1; ki--) {
                for (int oc_buff = 0; oc_buff < cur_oc_buffs; oc_buff++) {

                    int n = ii - ki * dilate_w + l_pad;

                    if (n >= 0 && n / stride_w < ow && n % stride_w == 0) {

                        int oi = n / stride_w;

                        int reg_idx = get_reg_idx(oi, oc_buff);

                        Zmm zmm = Xbyak::Zmm(reg_idx);

                        if (stride_w / dilate_w >= kw || dilate_w > stride_w || stride_w % dilate_w != 0
                            || ii == 0 || ki == 0) {

                            if (oc_buff == 0) {
                                cout << " " << oi << "-r" << reg_idx;
                            }

                            size_t aux_dst_offset = (size_t)typesize
                                * (oc_buff * oc_block * ow * mb_block
                                + oi * oc_block);

                            vmovups(zmm, EVEX_compress_addr_safe(reg_out, aux_dst_offset,
                                        reg_long_offt));
                            //prefetcht1(EVEX_compress_addr_safe(reg_out_prf, aux_dst_offset,
                                        //reg_long_offt));

                        }
                    }
                }
            }
        }


        Label ic_loop_end_label;
        jz(ic_loop_end_label, T_NEAR);

        tzcnt(lzcnt_reg.cvt32(), mask_reg.cvt32());
        inc(lzcnt_reg.cvt32());

        shrx(mask_reg.cvt32(), mask_reg.cvt32(), lzcnt_reg.cvt32());

        mov(aux_reg_ker, reg_ker);
        mov(aux_reg_inp, reg_inp);

        Label ic_loop_label;
        L(ic_loop_label); {

            lea(aux_reg_inp, ptr[aux_reg_inp + lzcnt_reg * typesize]);

            int aux_src_offset = typesize * (ii * ic_block - 1);
            vbroadcastss(zmm_o, ptr[aux_reg_inp + aux_src_offset]);

            shl(lzcnt_reg.cvt32(), 6);
            add(aux_reg_ker, lzcnt_reg);

            tzcnt(lzcnt_reg.cvt32(), mask_reg.cvt32()); // pipelined
            inc(lzcnt_reg.cvt32());

            dec(ic_itr_reg);

            shrx(mask_reg.cvt32(), mask_reg.cvt32(), lzcnt_reg.cvt32()); // does not change flags

            cout << " op:";

            for (int oc_buff = 0; oc_buff < cur_oc_buffs; oc_buff++) {                
                for (int ki = 0; ki < kw; ki++) {

                    int n = ii - ki * dilate_w + l_pad;

                    if (n >= 0 && n / stride_w < ow && n % stride_w == 0) {

                        int oi = n / stride_w;

                        int reg_idx = get_reg_idx(oi, oc_buff);

                        Zmm zmm = Xbyak::Zmm(reg_idx);

                        if (oc_buff == 0) {
                            cout << " " << oi << "-r" << reg_idx;
                        }
                    
                        size_t aux_kernel_offset = typesize * (oc_buff
                                * kw * oc_block * ic_block
                                + ki * oc_block * ic_block);
                        vfmadd231ps(zmm, zmm_o,
                                EVEX_compress_addr_safe(aux_reg_ker, aux_kernel_offset,
                                    reg_long_offt)); // probably don't need safe for weight tensor
                    }

                }
            }

            cout << endl;

            jnz(ic_loop_label, T_NEAR);
        }

        L(ic_loop_end_label);


        if (ii >= iw - step) {

            for (int ki = kw - 1; ki > -1; ki--) {
                for (int oc_buff = 0; oc_buff < cur_oc_buffs; oc_buff++) {

                    int n = ii - ki * dilate_w + l_pad;

                    if (n >= 0 && n / stride_w < ow && n % stride_w == 0) {

                        int reg_idx = get_reg_idx(n / stride_w, oc_buff);

                        Zmm zmm = Xbyak::Zmm(reg_idx);
                        
                        size_t aux_dst_offset = (size_t)typesize
                            * (oc_buff * oc_block * ow * mb_block
                            + n / stride_w * oc_block);
                        vmovups(EVEX_compress_addr_safe(reg_out, aux_dst_offset,
                                    reg_long_offt), zmm);
                    }
                }
            }

        }


    };


    auto comp_loop = [&](int idx, int ii, int cur_oc_buffs) {

        Reg64 aux_reg_out = aux_reg_ker;
        Reg64 mask_reg = reg_oi;
        Reg32 ic_itr_reg = reg_kh.cvt32();
        Reg64 lzcnt_reg = reg_channel;

        kmovw(mask_reg.cvt32(), k7);
        popcnt(ic_itr_reg, mask_reg.cvt32());

        size_t aux_src_offset = typesize * (idx + 1) * ic_block;
        vcmpps(k7, zmm_zero, EVEX_compress_addr_safe(aux_reg_inp, aux_src_offset, // pipelined
                                reg_long_offt), 4);

        prefetcht1(EVEX_compress_addr_safe(reg_inp_prf, aux_src_offset, // pipelined
                                reg_long_offt));

        /*int pref_idx = idx + 2;

        aux_src_offset = typesize * pref_idx * ic_block; // may overflow but it's ok
        mic_prefetcht1(EVEX_compress_addr_safe(aux_reg_inp, aux_src_offset,
                                reg_long_offt));*/

        for (int ki = 0; ki < kw; ki++) {
            for (int oc_buff = 0; oc_buff < cur_oc_buffs; oc_buff++) {

                int reg_idx = get_reg_idx(ii, oc_buff);

                Zmm zmm = Xbyak::Zmm(reg_idx);

                if (stride_w / dilate_w >= kw || dilate_w > stride_w
                    || stride_w % dilate_w != 0 || ki < stride_w) {

                    size_t aux_dst_offset = (size_t)typesize
                        * (oc_buff * oc_block * oh * ow 
                        + ((idx - 1) * stride_w - l_pad + ki * dilate_w) * oc_block);
                    vmovups(EVEX_compress_addr_safe(aux_reg_out, aux_dst_offset,
                                reg_long_offt), zmm);
                }
            }
        }

        for (int ki = 0; ki < kw; ki++) {
            for (int oc_buff = 0; oc_buff < cur_oc_buffs; oc_buff++) {

                int reg_idx = get_reg_idx(ii, oc_buff);

                Zmm zmm = Xbyak::Zmm(reg_idx);

                if (stride_w / dilate_w >= kw || dilate_w > stride_w
                    || stride_w % dilate_w != 0 || ki >= kw - stride_w) {

                    size_t aux_dst_offset = (size_t)typesize
                        * (oc_buff * oc_block * oh * ow 
                        + (idx * stride_w  - l_pad + ki * dilate_w) * oc_block);
                    vmovups(zmm, EVEX_compress_addr_safe(aux_reg_out, aux_dst_offset,
                                reg_long_offt));
                    prefetcht1(EVEX_compress_addr_safe(reg_out_prf, aux_dst_offset,
                                reg_long_offt));
                }

            }
        }

        Label ic_loop_end_label;
        jz(ic_loop_end_label, T_NEAR);

        tzcnt(lzcnt_reg.cvt32(), mask_reg.cvt32());
        inc(lzcnt_reg.cvt32());

        shrx(mask_reg.cvt32(), mask_reg.cvt32(), lzcnt_reg.cvt32());

        push(aux_reg_inp);
        push(reg_ker);

        //add(qword[param + GET_OFF(perf_cnt)], ic_itr_reg);

        Label ic_loop_label;
        L(ic_loop_label); {

            lea(aux_reg_inp, ptr[aux_reg_inp + lzcnt_reg * typesize]);

            int aux_src_offset = typesize * (idx * ic_block - 1);
            vbroadcastss(zmm_o, ptr[aux_reg_inp + aux_src_offset]);

            shl(lzcnt_reg.cvt32(), 6);
            add(reg_ker, lzcnt_reg);

            tzcnt(lzcnt_reg.cvt32(), mask_reg.cvt32()); // pipelined
            inc(lzcnt_reg.cvt32());

            dec(ic_itr_reg);

            shrx(mask_reg.cvt32(), mask_reg.cvt32(), lzcnt_reg.cvt32()); // does not change flags

            for (int oc_buff = 0; oc_buff < cur_oc_buffs; oc_buff++) {
                for (int ki = 0; ki < kw; ki++) {

                    int reg_idx = get_reg_idx(ii, oc_buff);

                    Zmm zmm = Xbyak::Zmm(reg_idx);
                    
                    size_t aux_kernel_offset = typesize * (oc_buff
                                    * kw * kh * oc_block * ic_block
                                    - oc_block + ki * oc_block * ic_block);
                    vfmadd231ps(zmm, zmm_o,
                            EVEX_compress_addr_safe(reg_ker, aux_kernel_offset,
                                reg_long_offt)); // probably don't need safe for weight tensor

                }
            }

            jnz(ic_loop_label, T_NEAR);
        }

        pop(reg_ker);
        pop(aux_reg_inp);

        L(ic_loop_end_label);

    };

    auto outer_loop = [&](int cur_oc_buffs) {

        int rotation_unroll_factor = stride_w * kw;

        sub(reg_ker, oc_block * typesize);

        if(0) {
        //if (l_iw <= iw - rotation_unroll_factor * 2) { // threshold needs to be dynamically calculated based on the instruction count per section

            /*int l_iw = kw > l_pad ? kw - l_pad : 1;
            l_iw++; // unroll one more due to pipelined vector write

            int r_iw = iw - 1 - (iw - 1 - l_iw) % rotation_unroll_factor;
            int niter = (r_iw - l_iw) / rotation_unroll_factor;

            cout << "nr:" << nr << " l_iw:" << l_iw << " r_iw:" << r_iw << " oc_iters:" << oc_iters
                << " oc_buffs:" << oc_buffs << endl;

            cout << "leading :" << l_iw << " trailing:" << ow - r_iw
                << " factor:" << rotation_unroll_factor
                << " niter:" << niter << endl;

            for (int ii = 0; ii < l_iw; ii++) {
                comp_unrolled(ii, cur_oc_buffs);
            }

            Reg64 iw_itr_reg = reg_channel;
            Reg64 aux_reg_out = aux_reg_ker;

            mov(iw_itr_reg, niter);

            mov(aux_reg_inp, reg_inp);
            mov(aux_reg_out, reg_out);

            add(aux_reg_inp, l_iw * ic_block * typesize);
            add(aux_reg_out, l_iw * stride_w * oc_block * typesize);

            add(reg_inp_prf, l_iw * ic_block * typesize);
            add(reg_out_prf, l_iw * stride_w * oc_block * typesize);

            Label iw_loop_label;
            L(iw_loop_label); {

                push(iw_itr_reg);

                for (int i = 0; i < rotation_unroll_factor; i++) {
                    comp_loop(i, l_iw + i, cur_oc_buffs);
                }

                pop(iw_itr_reg);

                add(aux_reg_inp, ic_block * typesize * rotation_unroll_factor);
                add(aux_reg_out, stride_w * oc_block * typesize * rotation_unroll_factor);

                add(reg_inp_prf, ic_block * typesize * rotation_unroll_factor);
                add(reg_out_prf, stride_w * oc_block * typesize * rotation_unroll_factor);

                dec(iw_itr_reg);
                jnz(iw_loop_label, T_NEAR);

            }

            mov(reg_inp_prf, ptr[param + GET_OFF(src_prf)]);
            mov(reg_out_prf, ptr[param + GET_OFF(dst_prf)]);

            for (int ii = r_iw; ii < ow; ii++) {
                comp_unrolled(ii, cur_oc_buffs);
            }*/

        } else {

            cout << "fully unrolled oc_buffs:" << oc_buffs << endl;

            int istart = 0;
            int step = 1;

            if (kw == 1 && stride_w > 1) {
                istart = l_pad % 2;
                step = stride_w;
            }

            for (int ii = istart; ii < iw; ii += step) {
                comp_unrolled(ii, step, cur_oc_buffs);
            }
        }
    };

    outer_loop(oc_buffs);

    L(end_label);
}

void jit_avx512_common_conv_fwd_kernel::generate()
{
    preamble();

    mov(reg_inp, ptr[param1 + GET_OFF(src)]);
    mov(reg_out, ptr[param1 + GET_OFF(dst)]);
    mov(reg_ker, ptr[param1 + GET_OFF(filt)]);
    mov(reg_ker_prf, ptr[param1 + GET_OFF(filt_prf)]);
    mov(reg_kh, ptr[param1 + GET_OFF(kh_padding)]);

    mov(reg_inp_prf, ptr[param1 + GET_OFF(src_prf)]);
    mov(reg_out_prf, ptr[param1 + GET_OFF(dst_prf)]);
    compute_loop_fma_sparse();

    postamble();
}

bool jit_avx512_common_conv_fwd_kernel::post_ops_ok(
        jit_conv_conf_t &jcp, const primitive_attr_t &attr) {
    const auto &p = attr.post_ops_;

    auto is_relu = [&](int idx) { return p.entry_[idx].is_relu(); };
    auto is_sum = [&](int idx) { return p.entry_[idx].is_sum(); };

    switch (p.len_) {
    case 0: return true; // no post_ops
    case 1:
        return true // sum OR relu
                && !jcp.with_relu && (is_relu(0) || is_sum(0));
    case 2:
        return true // sum->relu
                && !jcp.with_relu && (is_sum(0) && is_relu(1));
    default: return false;
    }

    return false;
}

status_t jit_avx512_common_conv_fwd_kernel::init_conf(
            jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, cpu_memory_t::pd_t &src_pd,
            cpu_memory_t::pd_t &weights_pd, cpu_memory_t::pd_t &dst_pd,
            cpu_memory_t::pd_t &bias_pd, const primitive_attr_t &attr,
            int nthreads, bool with_relu, float relu_negative_slope)
{
    using namespace prop_kind;

    if (!mayiuse(avx512_common))
        return status::unimplemented;

    const int simd_w = cpu_isa_traits<avx512_common>::vlen / sizeof(float);

    const memory_desc_wrapper src_d(&src_pd);
    const memory_desc_wrapper weights_d(&weights_pd);
    const memory_desc_wrapper dst_d(&dst_pd);
    const memory_desc_wrapper bias_d(&bias_pd);

    const int regs = 28;
    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();

    jcp = zero<decltype(jcp)>();
    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;
    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.ih = src_d.dims()[ndims-2];
    jcp.iw = src_d.dims()[ndims-1];
    jcp.od = (ndims == 5) ? dst_d.dims()[2] : 1;
    jcp.oh = dst_d.dims()[ndims-2];
    jcp.ow = dst_d.dims()[ndims-1];
    jcp.kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = weights_d.dims()[with_groups + ndims-2];
    jcp.kw = weights_d.dims()[with_groups + ndims-1];
    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = cd.padding[0][ndims-4];
    jcp.l_pad = cd.padding[0][ndims-3];
    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = cd.strides[ndims-4];
    jcp.stride_w = cd.strides[ndims-3];
    jcp.src_fmt = src_d.format();
    jcp.with_relu = with_relu;
    jcp.relu_negative_slope = relu_negative_slope;

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = cd.dilates[ndims-4];
    jcp.dilate_w = cd.dilates[ndims-3];

    jcp.b_pad = (jcp.oh - 1) * jcp.stride_h + (jcp.kh - 1) * (jcp.dilate_h + 1)
            - (jcp.ih + jcp.t_pad - 1);
    jcp.back_pad = (jcp.od - 1) * jcp.stride_d
            + (jcp.kd - 1) * (jcp.dilate_d + 1) - (jcp.id + jcp.f_pad - 1);

    jcp.is_1stconv = is_1stconv(jcp);

    jcp.oc_block = simd_w;
    jcp.ic_block = jcp.is_1stconv ? jcp.ic : simd_w;
    jcp.aligned_threads = 0;

    bool ok_to_pad_channels = true
        && jcp.ngroups == 1
        && src_d.data_type() == data_type::f32;

    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, jcp.oc_block);
        jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
    }
    bool args_ok = true
        && jcp.oc % jcp.oc_block == 0
        && jcp.ic % jcp.ic_block == 0;
    if (!args_ok)
        return status::unimplemented;

    if (!post_ops_ok(jcp, attr))
        return status::unimplemented;

    const auto &p = attr.post_ops_;
    jcp.with_sum = p.find(primitive_kind::sum) != -1;
    if (!jcp.with_relu) {
        jcp.with_relu = p.find(primitive_kind::eltwise) != -1;
        jcp.relu_negative_slope = 0;
    }

    auto src_format = (ndims == 5)
        ? (jcp.is_1stconv) ? ncdhw : NhC16nw16c
        : (jcp.is_1stconv) ? nchw : NhC16nw16c;
    auto dst_format = (ndims == 5) ? nCdhw16c : NhC16nw16c;
    auto wei_format = (ndims == 5)
        ? (with_groups) ? hIOw16i16o : hIOw16i16o
        : (with_groups) ? hIOw16i16o : hIOw16i16o;

    if (src_d.format() == any)
        CHECK(src_pd.set_format(src_format));
    if (dst_d.format() == any)
        CHECK(dst_pd.set_format(dst_format));

    switch (src_d.format()) {
        case NhC8nw16c: jcp.mb_block = 8; if (dst_d.format() != NhC8nw16c) return status::unimplemented; break;
        case NhC16nw16c: jcp.mb_block = 16; if (dst_d.format() != NhC16nw16c) return status::unimplemented; break;
        case NhC32nw16c: jcp.mb_block = 32; if (dst_d.format() != NhC32nw16c) return status::unimplemented; break;
        case NhC64nw16c: jcp.mb_block = 64; if (dst_d.format() != NhC64nw16c) return status::unimplemented; break;
        default: return status::unimplemented; break;
    }

    jcp.with_bias = cd.bias_desc.format != memory_format::undef;
    if (jcp.with_bias) {
        if (bias_d.format() == any)
            CHECK(bias_pd.set_format(x));
        if (bias_d.format() != x)
            return status::unimplemented;
    }

    if ((mayiuse(avx512_mic_4ops) || mayiuse(avx512_core_vnni))
         && src_d.data_type() == data_type::s16
         && weights_d.data_type() == data_type::s16
         && dst_d.data_type() == data_type::s32)
    {
        if (jcp.is_1stconv)
            return status::unimplemented;

        if (mayiuse(avx512_mic_4ops)) {
            jcp.ver = ver_4vnni;
        } else {
            jcp.ver = ver_vnni;
        }
        jcp.typesize_in = sizeof(int16_t);
        jcp.typesize_out = sizeof(int32_t);

        const auto w_format = (ndims == 5)
            ? with_groups ? gOIdhw8i16o2i : OIdhw8i16o2i
            : with_groups ? gOIhw8i16o2i : OIhw8i16o2i;
        if (weights_d.format() == any)
            CHECK(weights_pd.set_format(w_format));
        if (weights_d.format() != w_format)
            return status::unimplemented;
    } else if (mayiuse(avx512_common) &&
            src_d.data_type() == data_type::f32
         && weights_d.data_type() == data_type::f32
         && dst_d.data_type() == data_type::f32) {
        jcp.ver = ver_fma;
        jcp.typesize_in = sizeof(float);
        jcp.typesize_out = sizeof(float);
        if (mayiuse(avx512_mic_4ops))
           jcp.ver = ver_4fma;

        if (jcp.is_1stconv) {
            // TODO: fix & remove constraints below
            if (jcp.l_pad != 0 || jcp.r_pad != 0
                || jcp.b_pad != 0 || jcp.t_pad != 0
                || (jcp.kw < 7 && jcp.kh < 7))
                jcp.ver = ver_fma;
            if (jcp.ver == ver_4fma) {
                const auto w_format = (ndims == 5)
                    ? (with_groups) ? gOidhw16o : Oidhw16o
                    : (with_groups) ? gOihw16o : Oihw16o;
                if (weights_d.format() == any)
                    CHECK(weights_pd.set_format(w_format));
                if (weights_d.format() != w_format)
                    return status::unimplemented;
            } else {
                const auto w_format = (ndims == 5)
                    ? (with_groups) ? gOdhwi16o : Odhwi16o
                    : (with_groups) ? gOhwi16o : Ohwi16o;
                if (weights_d.format() == any)
                    CHECK(weights_pd.set_format(w_format));
                if (weights_d.format() != w_format)
                    return status::unimplemented;
            }
        } else {
            if (weights_d.format() == any)
                CHECK(weights_pd.set_format(wei_format));
            switch (weights_d.format()) {
                case hIOw16i16o:
                case IhOw16i16o: break;
                default: return status::unimplemented; break;
            }
        }
    } else {
        return status::unimplemented;
    }

    if (jcp.is_1stconv) {
        jcp.ur_w = nstl::min(jcp.ow, regs);
    } else {
        // avx512_core guard - just to avoid possible regression for other archs
        if (jcp.ver == ver_fma && mayiuse(avx512_core)) {
            jcp.ur_w = nstl::min(jcp.ow, regs);
        } else {
            for (int ur_w = regs; ur_w > 0; --ur_w) {
                if (jcp.ow % ur_w == 0) {
                    jcp.ur_w = ur_w;
                    break;
                }
            }
        }
        if ((ndims == 5 && jcp.ur_w <= 8) || (jcp.ur_w <= 1)) {
            jcp.ur_w = nstl::min(jcp.ow, regs);
        }
    }
    // TODO (Tanya): currently applied to Segnet convolutions only.
    // Need to try for other topologies
    if (jcp.ow > 150 && jcp.ur_w < regs/2)
        jcp.ur_w = regs;

    int n_oi = (jcp.ow / jcp.ur_w);
    int r_pad = (jcp.ur_w * n_oi - 1) * jcp.stride_w
            + (jcp.kw - 1) * (jcp.dilate_w + 1) - (jcp.iw + jcp.l_pad - 1);
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

    jcp.nb_ic = jcp.ic / jcp.ic_block;
    jcp.nb_oc = jcp.oc / jcp.oc_block;
    jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;

    if (jcp.ver == ver_4vnni) {
        jcp.kernel_kind = embd_bcast;
    }
    if (jcp.ver == ver_vnni) {
        // TODO: kernel_kind and nb_oc_blocking selection
        //       should be tuned on real HW
        if (jcp.ow <= 8 && jcp.oh <= 8 && jcp.od <= 8) {
            jcp.kernel_kind = expl_bcast;
            jcp.nb_oc_blocking = 2;
        } else {
            jcp.kernel_kind = embd_bcast;
            jcp.nb_oc_blocking = 2;
        }
        if (jcp.nb_oc_blocking > 1) {
            if (jcp.nb_oc < jcp.nb_oc_blocking) jcp.nb_oc_blocking = jcp.nb_oc;
            if (jcp.nb_oc % jcp.nb_oc_blocking != 0)
                for (int i = jcp.nb_oc_blocking; i > 0; i--)
                    if (jcp.nb_oc % i == 0) {
                        jcp.nb_oc_blocking = i;
                        break;
                    }
            jcp.ur_w = 31 / (jcp.nb_oc_blocking + 1);
            if (jcp.ow < jcp.ur_w)
                jcp.ur_w = jcp.ow;
        }
    }

    if (one_of(jcp.ver, ver_4vnni, ver_4fma) && !jcp.is_1stconv) {
        if (jcp.kw == 3 && jcp.kh == 3 && jcp.ow == 7 && jcp.oh == 7) {
            if (jcp.nb_oc % 2 == 0)
                jcp.nb_oc_blocking = 2;
        } else {
            for (int i = jcp.nb_oc; i > 0; i--)
                if (i * jcp.ur_w <= regs && jcp.nb_oc % i == 0) {
                    jcp.nb_oc_blocking = i;
                    break;
                }
        }
    }

    if (jcp.ver == ver_fma && mayiuse(avx512_core)) {
        int try_nb_oc_blocking = 2;
        unsigned int ker_inp_size = typesize * (jcp.iw / jcp.stride_w)
            * jcp.ic_block * jcp.kh * jcp.kd;
        unsigned int ker_out_size = typesize * jcp.ow * jcp.oc_block
            * try_nb_oc_blocking;
        unsigned int ker_wei_size = typesize * jcp.kh * jcp.kw * jcp.ic_block
            * jcp.oc_block * try_nb_oc_blocking * jcp.kd;
        unsigned int ker_total_size = ker_inp_size + ker_out_size
            + ker_wei_size;

        bool embd_bcast_condition = true
            && (jcp.kw == 3 && jcp.ow <= 28 && ker_total_size < L1_cache_size)
            && !(jcp.kw == 3 && jcp.ow == 13 && jcp.ic >= 192)
            && !(jcp.kw == 3 && jcp.ow == 28 && jcp.ic >= 512);

        if (jcp.mb == 1) {
            jcp.kernel_kind = embd_bcast;
            unsigned int inp_size = jcp.mb * (jcp.ih / jcp.stride_h)
                    * (jcp.iw / jcp.stride_w) * jcp.ic;
            unsigned int wei_size = jcp.ic * jcp.oc * jcp.kh * jcp.kw;

            // Estimate whether we need to limit the number of threads
            // and calculate this number. Includes some heuristic.
            int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
            int work_amount = jcp.mb * jcp.ngroups * oc_chunks * jcp.oh;
            int job_size_min = work_amount / nthreads;
            int job_size_max = div_up(work_amount, nthreads);
            int ch_max = rnd_up(jcp.oh, job_size_max);
            int ch_min = (job_size_min == 0)
                ? jcp.oh
                : rnd_up(jcp.oh, job_size_min);
            bool not_aligned_max = ch_max % jcp.oh != 0 && ch_max / jcp.oh < 2
                    && (jcp.oh != 8 || ch_max / jcp.oh > 1);
            bool not_aligned_min = ch_min % jcp.oh != 0 && ch_min / jcp.oh < 2
                    && (jcp.oh != 8 || ch_min / jcp.oh > 1);
            bool eligible_case = (jcp.stride_h == 1 && jcp.stride_w == 1)
                    || nthreads > oc_chunks;
            if (jcp.loop_order == loop_cgn && oc_chunks > 1 && nthreads > 1
                && wei_size / inp_size > 24
                && (not_aligned_max || not_aligned_min)
                && eligible_case) {
                jcp.aligned_threads = nthreads;
                for (int i = nthreads; i > 0; i--) {
                    if (oc_chunks % i == 0 || i % oc_chunks == 0) {
                        jcp.aligned_threads = i;
                        break;
                    }
                }
            }
        } else if (jcp.kw > 3
            || (jcp.stride_w == 1 && jcp.stride_h == 1
                && embd_bcast_condition)
            || ((jcp.stride_w != 1 || jcp.stride_h != 1)
                && ((jcp.mb <= 16 && (jcp.oc <= 192 || jcp.oh <= 10)
                     && embd_bcast_condition)))
            ) {
            jcp.kernel_kind = embd_bcast;
            jcp.ur_w = nstl::min(jcp.ow, regs);
            jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;
            if (ker_total_size < L1_cache_size && jcp.ow <= 8 && jcp.kh <= 3
                && jcp.kw <= 3) {
                if (jcp.nb_oc % try_nb_oc_blocking == 0 && !jcp.is_1stconv) {
                    jcp.nb_oc_blocking = try_nb_oc_blocking;
                    jcp.ur_w = 31 / (jcp.nb_oc_blocking + 1);
                    if (jcp.ow < jcp.ur_w)
                        jcp.ur_w = jcp.ow;
                }
            }
        } else {
            jcp.kernel_kind = expl_bcast;
            jcp.nb_ic_blocking = 1;
            jcp.nb_oc_blocking = 4;
            if (jcp.nb_oc < jcp.nb_oc_blocking) jcp.nb_oc_blocking = jcp.nb_oc;
            if (jcp.nb_oc % jcp.nb_oc_blocking != 0)
                for (int i = jcp.nb_oc_blocking; i > 0; i--)
                    if (jcp.nb_oc % i == 0) {
                        jcp.nb_oc_blocking = i;
                        break;
                    }
            jcp.ur_w = 31 / (jcp.nb_oc_blocking + 1);
            if (jcp.ow < jcp.ur_w)
                jcp.ur_w = jcp.ow;
        }
    }


    jcp.kernel_kind = embd_bcast;
    jcp.ur_w = nstl::min(jcp.ow, regs);
    jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;

    jcp.ur_w_tail = jcp.ow % jcp.ur_w;

    args_ok = true
        && jcp.l_pad <= jcp.ur_w
        && jcp.ic <= src_d.blocking_desc().padding_dims[1]
        && jcp.oc <= dst_d.blocking_desc().padding_dims[1]
        && jcp.ic <= weights_d.blocking_desc().padding_dims[with_groups + 1]
        && jcp.oc <= weights_d.blocking_desc().padding_dims[with_groups + 0];
    if (!args_ok)
        return status::unimplemented;

    int r_pad_no_tail = nstl::max(0, (jcp.ow - jcp.ur_w_tail - 1) * jcp.stride_w
                    + (jcp.kw - 1) * (jcp.dilate_w + 1)
                    - (jcp.iw + jcp.l_pad - 1));
    if (r_pad_no_tail > jcp.ur_w)
        return status::unimplemented;

    pick_loop_order(jcp);

    jcp.nb_ic_L2 = jcp.nb_ic;

    // TODO check for 4vnni
    if (jcp.ver == ver_4fma) {
        for (int divf = 2, temp_nb = jcp.nb_ic_L2; divf <= jcp.nb_ic;
              divf++) {
            size_t l2_src
                = (size_t)jcp.iw * jcp.ic_block * jcp.ih * temp_nb * jcp.id;
            size_t l2_dst = (size_t)jcp.ow * jcp.oc_block * jcp.nb_oc_blocking
                * jcp.oh * jcp.od;
            size_t l2_filt = (size_t)jcp.kw * jcp.oc_block * jcp.ic_block
                * jcp.kh * jcp.nb_oc_blocking * temp_nb * jcp.kd;
            if (4 * (l2_src + l2_dst + l2_filt) > KNx_L2_EFFECTIVE_CAPACITY) {
                if (jcp.kh == 3 && jcp.oh == 7) {
                    jcp.nb_ic_L2 = 1;
                    break;
                }
                temp_nb = (jcp.nb_ic_L2 % divf == 0 ? jcp.nb_ic_L2 / divf
                                : jcp.nb_ic_L2);
            } else {
                jcp.nb_ic_L2 = temp_nb;
                break;
            }
        }
    }

    int nregs = 30;
    int ur_sparse;
    if (jcp.kw * jcp.nb_oc <= nregs)
        ur_sparse = jcp.kw * jcp.nb_oc;
    else {
        for (int tmp_ur_w = nregs; tmp_ur_w > 0; tmp_ur_w--)
            if (tmp_ur_w % jcp.kw == 0) {
                ur_sparse = tmp_ur_w;
                break;
            }
    }

    jcp.oc_buffs = ur_sparse / jcp.kw;
    for (int i = jcp.oc_buffs; i > 0; i--) {
        if (jcp.nb_oc % i == 0) {
            jcp.oc_buffs = i;
            break;
        }
    }

    // higher than 8 will cause subpar memory performance
    if (jcp.oc_buffs > 8) jcp.oc_buffs = 8;

    jcp.ur_sparse = jcp.oc_buffs * jcp.kw;
    jcp.nb_mb = jcp.mb / jcp.mb_block;

    cout << "stride_w:" << jcp.stride_w << " stride_h:" << jcp.stride_h
        << " l_pad:" << jcp.l_pad << " r_pad:" << jcp.r_pad
        << " t_pad:" << jcp.t_pad << " b_pad:" << jcp.b_pad
        << " iw:" << jcp.iw << " ih:" << jcp.ih << " ic:" << jcp.ic
        << " ow:" << jcp.ow << " oh:" << jcp.oh << " oc:"<< jcp.oc
        << " kw:" << jcp.kw << " kh:"<< jcp.kh << " mb:" << jcp.mb
        << " nb_ic_blocking:" << jcp.nb_ic_blocking
        << " nb_oc_blocking:" << jcp.nb_oc_blocking
        << " ngroups:" << jcp.ngroups << " dilate_w:" << jcp.dilate_w
        << " typesize_in:" << jcp.typesize_in
        << " typesize_out:" << jcp.typesize_out << " with_bias:" << jcp.with_bias
        << " with_sum:" << jcp.with_sum << " with_relu:" << jcp.with_relu << endl;

    return status::success;
}

void jit_avx512_common_conv_bwd_data_kernel_f32::prepare_output(int ur_w)
{
    for (int k = 0; k < jcp.nb_ic_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            Zmm zmm = zmm_out(j, k);
            vpxord(zmm, zmm, zmm);
            size_t aux_src_offset
                = (size_t)typesize * ((size_t)k * jcp.ih * jcp.iw * jcp.id + j)
                * jcp.ic_block;
            mic_prefetcht1(EVEX_compress_addr_safe(reg_src_prf, aux_src_offset,
                        reg_long_offt));
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
            size_t aux_src_offset = (size_t)typesize
                * ((size_t)k * jcp.ih * jcp.iw * jcp.id + j) * jcp.ic_block;
            vadd(zmm, EVEX_compress_addr_safe(reg_src, aux_src_offset,
                        reg_long_offt));
        }
    }

    L(no_update_label);
    for (int k = 0; k < jcp.nb_ic_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            Zmm zmm = zmm_out(j, k);
            size_t aux_src_offset = (size_t)typesize
                * ((size_t)k * jcp.ih * jcp.iw * jcp.id + j) * jcp.ic_block;
            vmovups(EVEX_compress_addr_safe(reg_src, aux_src_offset,
                        reg_long_offt), zmm);
            mic_prefetcht0(EVEX_compress_addr_safe(reg_src_prf, aux_src_offset,
                        reg_long_offt));
        }
    }
}

void jit_avx512_common_conv_bwd_data_kernel_f32::compute_loop_4fma(
        int ur_w, int l_overflow, int r_overflow)
{
    int ow = jcp.ow;
    int kw = jcp.kw;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    Label kh_label, last_iter_label, loop_end_label, kd_label, skip_kd_loop;
    int ker_load_number = 4;
    int shift_ker_ptr = typesize * kw * oc_block * ic_block;
    int shift_dst_ptr = typesize * ow * oc_block;
    int ii_dpref_t0 = get_iw_start(0, l_overflow);
    int iw_end_ipref = get_iw_end(ur_w, 0, r_overflow);

    bool check_last_kh = (jcp.kh > 3);

    auto kernel_offset = [=](int icb, int oc, int ki) {
        int blk_idx = icb * jcp.kh * jcp.kw * jcp.kd + ki;
        int blk_offset = blk_idx * jcp.oc_block * jcp.ic_block;
        int oc_offset = oc * jcp.oc_block;
        return typesize * (blk_offset + oc_offset);
    };
    auto kernel_loads = [=](int ki, int oc, int kk) {
        for (int ii = 0; ii < ker_load_number; ii++) {
            int aux_kernel_offset = kernel_offset(kk, oc + ii, ki);
            vmovups(zmm_ker(ii),
                EVEX_compress_addr(aux_reg_ker, aux_kernel_offset));
        }
    };
    auto prefetch_dst_next_kh = [&](int ki, int ki_start, int cnt0, int cnt1) {
        if (cnt1 >= ker_load_number && cnt0 >= ker_load_number
            && ki >= ki_start && ii_dpref_t0 < iw_end_ipref) {
            int aux_dst_offset = typesize * ((ii_dpref_t0
                + jcp.l_pad) * oc_block + jcp.ow * oc_block);
            prefetcht0(EVEX_compress_addr(aux_reg_dst, aux_dst_offset));
            ii_dpref_t0++;
        }
    };

    prepare_output(ur_w);

    if (jcp.ndims == 4) {
        mov(aux_reg_dst, reg_dst);
        mov(aux_reg_ker, reg_ker);
        mov(aux_reg_dst_prf, reg_dst_prf);
        mov(aux_reg_ker_prf, reg_ker_prf);
    }

    if (jcp.ndims == 5) {
        push(reg_src_prf);
        push(reg_src);

        mov(reg_ki, ptr[param + GET_OFF(kd_padding)]);
        mov(aux_reg_dst_d, reg_dst);
        mov(aux_reg_ker_d, ptr[param + GET_OFF(filt)]);
        mov(aux_reg_dst_d_prf, reg_dst_prf);
        mov(aux_reg_ker_d_prf, reg_ker_prf);

        L(kd_label);
        mov(reg_kj, ptr[param + GET_OFF(kh_padding)]);
    } else {
        mov(reg_kj, reg_kh);
    }

    if (jcp.ndims == 5) {
        mov(aux_reg_dst, aux_reg_dst_d);
        mov(aux_reg_ker, aux_reg_ker_d);
        mov(aux_reg_dst_prf, aux_reg_dst_d_prf);
        mov(aux_reg_ker_prf, aux_reg_ker_d_prf);
    }

    align(16);
    L(kh_label);
    if (check_last_kh) {
        for (int ki = 0; ki < kw; ki++)
        for (int oc = 0; oc < oc_block; oc += 4)
        for (int kk = 0; kk < jcp.nb_ic_blocking; kk++) {
            bool last_kernel_loads = (kk == jcp.nb_ic_blocking - 1
                && ki == kw - 1 && (oc + 4) == oc_block);

            if (last_kernel_loads) {
                cmp(reg_kj, 1);
                je(last_iter_label, T_NEAR);
            }

            kernel_loads(ki, oc, kk);
            for (int ii = get_iw_start(ki, l_overflow),
                    prf_count_t0 = 0, prf_count_t1 = 0;
                    ii < get_iw_end(ur_w, ki, r_overflow); ii++) {
                int aux_dst_offset = typesize
                    * ((ii + jcp.l_pad - ki) * oc_block + oc);
                v4fmaddps(zmm_out(ii, kk), zmm_ker(0),
                    EVEX_compress_addr(aux_reg_dst, aux_dst_offset));

                if (ii % 2) {
                    if (prf_count_t0 < 4) {
                        int aux_kernel_prf;
                        if (last_kernel_loads)
                            aux_kernel_prf= kernel_offset(0, prf_count_t0
                                + oc + 4 - oc_block, 0) + typesize * kw
                                * oc_block * ic_block;
                        else
                            aux_kernel_prf = kernel_offset(kk, oc + 4
                                + prf_count_t0, ki);
                        mic_prefetcht0(EVEX_compress_addr(aux_reg_ker,
                            aux_kernel_prf));
                        prf_count_t0++;
                    } else if (prf_count_t1 < 4) {
                        mic_prefetcht1(EVEX_compress_addr(aux_reg_ker_prf,
                            kernel_offset(kk, oc + prf_count_t1, ki)));
                        prf_count_t1++;
                    }
                } else
                    prefetch_dst_next_kh(ki, 2, prf_count_t0, prf_count_t1);
            }
            if (last_kernel_loads) {
                jmp(loop_end_label, T_NEAR);

                L(last_iter_label);

                kernel_loads(ki, oc, kk);
                for (int ii = get_iw_start(ki, l_overflow),
                        prf_count_t0 = 0, prf_count_t1 = 0;
                        ii < get_iw_end(ur_w, ki, r_overflow); ii++) {
                    int aux_dst_offset = typesize
                        * ((ii + jcp.l_pad - ki) * oc_block + oc);
                    v4fmaddps(zmm_out(ii, kk), zmm_ker(0),
                            EVEX_compress_addr(aux_reg_dst, aux_dst_offset));
                    if (ii % 2) {
                        if (prf_count_t0 < 4) {
                            mic_prefetcht0(EVEX_compress_addr(aux_reg_ker_prf,
                                kernel_offset(0, prf_count_t0, 0)));
                            prf_count_t0++;
                        } else if (prf_count_t1 < 4) {
                            mic_prefetcht1(EVEX_compress_addr(aux_reg_ker_prf,
                                kernel_offset(kk, oc + prf_count_t1, ki)));
                            prf_count_t1++;
                        }
                    }
                }
                L(loop_end_label);
            }
        }
    } else {
        for (int ki = 0; ki < kw; ki++)
        for (int oc = 0; oc < oc_block; oc += 4)
        for (int kk = 0; kk < jcp.nb_ic_blocking; kk++) {
            kernel_loads(ki, oc, kk);

            for (int ii = get_iw_start(ki, l_overflow), prf_count_t1 = 0;
                    ii < get_iw_end(ur_w, ki, r_overflow); ii++) {
                int aux_dst_offset = typesize
                    * ((ii + jcp.l_pad - ki) * oc_block + oc);
                v4fmaddps(zmm_out(ii, kk), zmm_ker(0),
                    EVEX_compress_addr(aux_reg_dst, aux_dst_offset));
                if ((ii % 2) && (prf_count_t1 < 4)) {
                    mic_prefetcht1(EVEX_compress_addr(
                        aux_reg_ker_prf, kernel_offset(kk,
                        oc + prf_count_t1, ki)));
                    prf_count_t1++;
                }
                if ( ki == 1 && oc == 0 && kk == 0)
                    mic_prefetcht1(EVEX_compress_addr(
                        aux_reg_dst_prf, aux_dst_offset));
            }
        }
    }

    add(aux_reg_ker, shift_ker_ptr);
    sub(aux_reg_dst, shift_dst_ptr);
    add(aux_reg_ker_prf, shift_ker_ptr);
    sub(aux_reg_dst_prf, shift_dst_ptr);

    dec(reg_kj);
    cmp(reg_kj, 0);
    jg(kh_label, T_NEAR);

    if (jcp.ndims == 5) {
        sub(aux_reg_dst_d, typesize * (jcp.oh * ow) * ic_block);
        add(aux_reg_ker_d, typesize * jcp.kw * jcp.kh * oc_block * ic_block);
        sub(aux_reg_dst_d_prf, typesize * (jcp.oh * ow) * ic_block);
        add(aux_reg_ker_d_prf, typesize * jcp.kw * jcp.kh *oc_block * ic_block);

        dec(reg_ki);
        cmp(reg_ki, 0);
        jg(kd_label, T_NEAR);
        L(skip_kd_loop);

        pop(reg_src);
        pop(reg_src_prf);
    }

    store_output(ur_w);
}

void jit_avx512_common_conv_bwd_data_kernel_f32::compute_loop_vnni(
        int ur_w, int l_overflow, int r_overflow)
{
    int ow = jcp.ow;
    int kw = jcp.kw;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    const int channel_inc = jcp.ver == ver_4vnni ? 4 : 1;
    const int ker_load_number = jcp.ver == ver_4vnni ? 4 : 1;
    Label kh_label;

    auto kernel_offset = [=](int icb, int oc, int ki) {
        int blk_idx = icb * jcp.kh * jcp.kw + ki;
        int blk_offset = blk_idx * jcp.oc_block * jcp.ic_block;
        int oc_offset = oc * jcp.oc_block;
        return jcp.typesize_in * (blk_offset + oc_offset);
    };

    prepare_output(ur_w);

    mov(aux_reg_dst, reg_dst);
    mov(aux_reg_ker, reg_ker);
    mov(aux_reg_dst_prf, reg_dst_prf);
    mov(aux_reg_ker_prf, reg_ker_prf);

    mov(reg_kj, reg_kh);
    L(kh_label); {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = get_iw_start(ki, l_overflow);
            int jj_end = get_iw_end(ur_w, ki, r_overflow);
            for (int oc = 0; oc < oc_block / 2; oc += channel_inc) {
                if (jcp.kernel_kind == expl_bcast) {
                    for (int jj = jj_start; jj < jj_end; jj++) {
                        int aux_dst_offset = jcp.typesize_in
                            * ((jj + jcp.l_pad - ki) * oc_block + 2 * oc);
                        vpbroadcastd(zmm_inp(jj, jcp.nb_ic_blocking),
                            ptr[aux_reg_dst + aux_dst_offset]);
                    }
                }
                for (int kk = 0; kk < jcp.nb_ic_blocking; kk++) {
                    if (jcp.kernel_kind == expl_bcast) {
                        int aux_kernel_offset = kernel_offset(kk, 2 * oc, ki);
                        vmovups(zmm_wei,
                            EVEX_compress_addr(aux_reg_ker, aux_kernel_offset));
                    } else {
                        for (int ii = 0; ii < ker_load_number; ii++) {
                            int aux_kernel_offset
                                = kernel_offset(kk, 2 * (oc + ii), ki);
                            vmovups(zmm_ker(ii),
                                EVEX_compress_addr(aux_reg_ker,
                                aux_kernel_offset));
                        }
                    }

                    for (int jj = jj_start, prf_count = 0; jj < jj_end; jj++) {
                        int aux_dst_offset = jcp.typesize_in
                            * ((jj + jcp.l_pad - ki) * oc_block + 2 * oc);
                        if (jcp.kernel_kind == expl_bcast) {
                            vpdpwssd(zmm_out(jj, kk), zmm_wei,
                                zmm_inp(jj, jcp.nb_ic_blocking));
                        } else {
                            vpXdpwssd(zmm_out(jj, kk), zmm_ker(0),
                                aux_reg_dst, aux_dst_offset);
                        }

                        if ((jj % 2) && (prf_count < 4)) {
                            int aux_kernel_prf
                                = kernel_offset(kk, oc + prf_count, ki);
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
                                    aux_dst_offset + jcp.typesize_in
                                    * ow * oc_block));
                        }
                    }
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

void jit_avx512_common_conv_bwd_data_kernel_f32::compute_loop_fma(
        int ur_w, int l_overflow, int r_overflow)
{
    Label kh_label, kd_label, skip_kd_loop;
    Label store_output_label;
    int kw = jcp.kw;
    int ow = jcp.ow;

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int l_pad = jcp.l_pad;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;
    int stride_h = jcp.stride_h;

    int ker_pipeline_depth = 4;
    assert(ker_reg_base_idx + ker_pipeline_depth <= 32);
    assert(oc_block >= ker_pipeline_depth);

    int num_ker_loads = oc_block * kw;
    int num_inp_prfs = ur_w * nstl::min(kw, stride_w)
                       + nstl::max(0, kw - stride_w);
    int num_prfs = num_ker_loads + num_inp_prfs;
    int num_fmas = num_ker_loads * ur_w / stride_w;
    int prf_inst_spacing = nstl::max(1, num_fmas / num_prfs);
    int prf_inst_trigger = (num_fmas % prf_inst_spacing) / 2;

    prepare_output(ur_w);

    if (jcp.ndims == 4) {
        mov(aux_reg_dst, reg_dst);
        mov(aux_reg_ker, reg_ker);

        mov(aux_reg_dst_prf, reg_dst_prf);
        mov(aux_reg_ker_prf, reg_ker_prf);
    }

    if (jcp.ndims == 5) {
        push(reg_src_prf);
        push(reg_src);

        mov(reg_ki, ptr[param + GET_OFF(kd_padding)]);
        cmp(reg_ki, 0);
        je(store_output_label, T_NEAR);

        mov(aux_reg_dst_d, reg_dst);
        mov(aux_reg_ker_d, ptr[param + GET_OFF(filt)]);
        mov(aux_reg_dst_d_prf, reg_dst_prf);
        mov(aux_reg_ker_d_prf, reg_ker_prf);

        L(kd_label);
        mov(reg_kj, ptr[param + GET_OFF(kh_padding)]);
    } else {
        mov(reg_kj, reg_kh);
    }
    cmp(reg_kj, 0);
    je(store_output_label, T_NEAR);

    if (jcp.ndims == 5) {
        mov(aux_reg_dst, aux_reg_dst_d);
        mov(aux_reg_ker, aux_reg_ker_d);
        mov(aux_reg_dst_prf, aux_reg_dst_d_prf);
        mov(aux_reg_ker_prf, aux_reg_ker_d_prf);
    }

    L(kh_label); {

#if 0

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

                int jj_start = get_iw_start(ki, l_overflow);
                int jj_end = get_iw_end(ur_w, ki, r_overflow);
                assert(stride_w != 1
                        || jj_start == nstl::max(0,
                            l_overflow - (kw - 1 - ki) * dilate_w));
                assert(stride_w != 1
                        || jj_end == ur_w - nstl::max(0,
                            r_overflow - ki * dilate_w));

                for (int jj = jj_start; jj < jj_end; jj += stride_w) {
                    assert((jj + l_pad - ki * dilate_w) % stride_w == 0);
                    int aux_dst_offset = typesize *
                        (((jj + l_pad - ki * dilate_w)
                                / stride_w) * jcp.oc_block + oc);
                    vfmadd231ps(zmm_out(jj, 0), zmm_kernel,
                        EVEX_compress_addr(aux_reg_dst, aux_dst_offset, true));

                    int fma_idx = (step * ur_w + jj) / stride_w;
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

#elif 1


        int jj_start_min = ur_w;
        int jj_end_max = 0;
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = get_iw_start(ki, l_overflow);
            int jj_end = get_iw_end(ur_w, ki, r_overflow);

            jj_start_min = jj_start < jj_start_min ? jj_start : jj_start_min;
            jj_end_max = jj_end > jj_end_max ? jj_end : jj_end_max;
        }

        for (int ki = 0; ki < kw; ki++) {
            for (int oc = 0; oc < oc_block; oc++) {

                auto zmm_kernel = zmm_ker(0);
                int aux_kernel_offset = typesize * (oc
                        * oc_block + ki * ic_block * oc_block);
                vmovups(zmm_kernel,
                        EVEX_compress_addr(aux_reg_ker, aux_kernel_offset));

                int jj_start = get_iw_start(ki, l_overflow);
                int jj_end = get_iw_end(ur_w, ki, r_overflow);
                assert(stride_w != 1
                        || jj_start == nstl::max(0,
                            l_overflow - (kw - 1 - ki) * dilate_w));
                assert(stride_w != 1
                        || jj_end == ur_w - nstl::max(0,
                            r_overflow - ki * dilate_w));

                for (int jj = jj_start_min; jj < jj_end_max; jj += stride_w) {

                    int jjj = jj + jj_start - jj_start_min;

                    if (jjj >= jj_start && jjj < jj_end) {

                        assert((jjj + l_pad - ki * dilate_w) % stride_w == 0);
                        int aux_dst_offset = typesize *
                            (((jjj + l_pad - ki * dilate_w)
                                    / stride_w) * jcp.oc_block + oc);
                        vfmadd231ps(zmm_out(jjj, 0), zmm_kernel,
                            EVEX_compress_addr(aux_reg_dst, aux_dst_offset, true));
                    }

                }
            }
        }

#else

        if (kw > ker_pipeline_depth) {
            exit(0);
        }

        int jj_start[kw], jj_end[kw];
        for (int ki = 0; ki < kw; ki++) {
            jj_start[ki] = get_iw_start(ki, l_overflow);
            jj_end[ki] = get_iw_end(ur_w, ki, r_overflow);
        }


        for (int oc = 0; oc < oc_block; oc++) {

            assert(kw <= ker_pipeline_depth);

            for (int ki = 0; ki < kw; ki++) {

                int aux_kernel_offset = typesize * (oc
                        * oc_block + ki * ic_block * oc_block);
                vmovups(zmm_ker(ki),
                        EVEX_compress_addr(aux_reg_ker, aux_kernel_offset));
            }


            for (int jj = 1 - kw; jj < ur_w; jj ++) {

                for (int ki = 0; ki < kw; ki++) {

                    int jjj = jj + ki;

                    if (jjj >= jj_start[ki] && jjj < jj_end[ki]
                        && (jjj - jj_start[ki]) % stride_w == 0) {

                        assert((jjj + l_pad - ki * dilate_w) % stride_w == 0);
                        int aux_dst_offset = typesize *
                            (((jjj + l_pad - ki * dilate_w)
                                    / stride_w) * jcp.oc_block + oc);
                        vfmadd231ps(zmm_out(jjj, 0), zmm_ker(ki),
                            EVEX_compress_addr(aux_reg_dst, aux_dst_offset, true));

                    }
                }
            }
        }

#endif


        add(aux_reg_ker, typesize * stride_h * kw * oc_block * ic_block);
        sub(aux_reg_dst, typesize * (jcp.dilate_h + 1) * ow * oc_block);
        add(aux_reg_ker_prf, typesize * stride_h * kw * oc_block * ic_block);
        sub(aux_reg_dst_prf, typesize * (jcp.dilate_h + 1) * ow * oc_block);

        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }
    if (jcp.ndims == 5) {
        sub(aux_reg_dst_d,
                typesize * (jcp.dilate_d + 1) * jcp.oh * ow * ic_block);
        add(aux_reg_ker_d, typesize * jcp.stride_d * jcp.kw * jcp.kh
                * oc_block * ic_block);
        sub(aux_reg_dst_d_prf,
                typesize * (jcp.dilate_d + 1) * jcp.oh * ow * ic_block);
        add(aux_reg_ker_d_prf, typesize * jcp.stride_d * jcp.kw * jcp.kh
                * oc_block * ic_block);

        dec(reg_ki);
        cmp(reg_ki, 0);
        jg(kd_label, T_NEAR);
        L(skip_kd_loop);
    }

    L(store_output_label); {
        if (jcp.ndims == 5)
        {
            pop(reg_src);
            pop(reg_src_prf);
        }
        store_output(ur_w);
    }
}

void jit_avx512_common_conv_bwd_data_kernel_f32::compute_loop_fma_core(
        int ur_w, int l_overflow, int r_overflow)
{
    int kw = jcp.kw;
    int ow = jcp.ow;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int nb_ic_block = jcp.nb_ic_blocking;
    Label kh_label, skip_kh_loop, kd_label, skip_kd_loop;

    int shift_ker_ptr = typesize * kw * oc_block * ic_block;
    int shift_dst_ptr = typesize * (jcp.dilate_h + 1) * ow * oc_block;

    auto output_offset = [=](int oi, int oc, int ki) {
        return typesize *
            (((oi + jcp.l_pad - ki * dilate_w) / stride_w) * oc_block + oc);
    };
    auto kernel_offset = [=](int icb, int oc, int ki) {
        int blk_idx = icb * jcp.kh * jcp.kw * jcp.kd + ki;
        int blk_offset = blk_idx * jcp.oc_block * jcp.ic_block;
        int oc_offset = oc * jcp.oc_block;
        return typesize * (blk_offset + oc_offset);
    };

    prepare_output(ur_w);

    if (jcp.ndims == 4) {
        mov(aux_reg_dst, reg_dst);
        mov(aux_reg_ker, reg_ker);
    }

    if (jcp.ndims == 5) {
        push(reg_src_prf);
        push(reg_src);

        mov(reg_ki, ptr[param + GET_OFF(kd_padding)]);
        mov(aux_reg_dst_d, reg_dst);
        mov(aux_reg_ker_d, ptr[param + GET_OFF(filt)]);

        L(kd_label);
        mov(reg_kj, ptr[param + GET_OFF(kh_padding)]);
    } else {
        mov(reg_kj, reg_kh);
    }
    cmp(reg_kj, 0);
    je(skip_kh_loop, T_NEAR);

    if (jcp.ndims == 5) {
        mov(aux_reg_dst, aux_reg_dst_d);
        mov(aux_reg_ker, aux_reg_ker_d);
    }

    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = get_iw_start(ki, l_overflow);
            int jj_end = get_iw_end(ur_w, ki, r_overflow);
            for (int oc = 0; oc < oc_block; oc++) {
                if (jcp.kernel_kind == expl_bcast) {
                    for (int jj = jj_start; jj < jj_end; jj++) {
                        int aux_output_offset = output_offset(jj, oc, ki);
                        vbroadcastss(zmm_inp(jj, nb_ic_block),
                            ptr[aux_reg_dst + aux_output_offset]);
                    }
                }
                for (int ii = 0; ii < nb_ic_block; ii++) {
                    int aux_kernel_offset = kernel_offset(ii, oc, ki);
                    if (jj_end - jj_start > 0)
                        vmovups(zmm_wei, EVEX_compress_addr(aux_reg_ker,
                            aux_kernel_offset));
                    for (int jj = jj_start; jj < jj_end; jj += stride_w)
                        if (jcp.kernel_kind == expl_bcast)
                            vfmadd231ps(zmm_out(jj, ii),
                                zmm_inp(jj, nb_ic_block), zmm_wei);
                        else
                            vfmadd231ps(zmm_out(jj, ii), zmm_wei,
                                EVEX_compress_addr(aux_reg_dst,
                                output_offset(jj, oc, ki), true));
                }
            }
        }
        add(aux_reg_ker, shift_ker_ptr);
        sub(aux_reg_dst, shift_dst_ptr);
        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }
    L(skip_kh_loop);

    if (jcp.ndims == 5) {
        sub(aux_reg_dst_d,
                typesize * (jcp.dilate_d + 1) * jcp.oh * ow * ic_block);
        add(aux_reg_ker_d, typesize * jcp.kw * jcp.kh * oc_block * ic_block);

        dec(reg_ki);
        cmp(reg_ki, 0);
        jg(kd_label, T_NEAR);
        L(skip_kd_loop);

        pop(reg_src);
        pop(reg_src_prf);
    }

    store_output(ur_w);
}

inline void jit_avx512_common_conv_bwd_data_kernel_f32::compute_loop(
        int ur_w, int l_overflow, int r_overflow)
{
    if (jcp.ndims == 5) push(reg_oi);
    if (jcp.ver == ver_4vnni || jcp.ver == ver_vnni)
        compute_loop_vnni(ur_w, l_overflow, r_overflow);
    else if (jcp.ver == ver_4fma)
        compute_loop_4fma(ur_w, l_overflow, r_overflow);
    else if (jcp.ver == ver_fma)
        if (mayiuse(avx512_mic))
            compute_loop_fma(ur_w, l_overflow, r_overflow);
        else
          if (jcp.kernel_kind == embd_bcast && jcp.nb_ic_blocking == 1)
              compute_loop_fma(ur_w, l_overflow, r_overflow);
          else
              compute_loop_fma_core(ur_w, l_overflow, r_overflow);
    else
        assert("!unknown convolution version");
    if (jcp.ndims == 5) pop(reg_oi);
}

#if 0

void jit_avx512_common_conv_bwd_data_kernel_f32::generate()
{
    int iw = jcp.iw;
    int kw = jcp.kw;
    int ur_w = jcp.ur_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int ur_w_tail = jcp.ur_w_tail;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;

    int dst_shift = jcp.typesize_in * (ur_w / stride_w) * ic_block;
    int src_shift = jcp.typesize_out * ur_w * oc_block;

    preamble();

    mov(reg_src, ptr[param + GET_OFF(src)]);
    mov(reg_dst, ptr[param + GET_OFF(dst)]);
    mov(reg_ker, ptr[param + GET_OFF(filt)]);

    mov(reg_kh, ptr[param + GET_OFF(kh_padding)]);
    mov(reg_src_prf, ptr[param + GET_OFF(src_prf)]);
    mov(reg_dst_prf, ptr[param + GET_OFF(dst_prf)]);
    mov(reg_ker_prf, ptr[param + GET_OFF(filt_prf)]);

    int l_overflow = nstl::max(0, ((kw - 1) * dilate_w - jcp.l_pad) / stride_w);
    int r_overflow = nstl::max(0, ((kw - 1) * dilate_w
                    - nstl::max(0, jcp.r_pad)) / stride_w);
    int r_overflow1 = nstl::max(0, ((kw - 1) * dilate_w
                    - nstl::max(0, jcp.r_pad) - ur_w_tail) / stride_w);

    int n_oi = iw / ur_w;
    if (r_overflow1 > 0) n_oi--;

    cout << " stride_w:" << stride_w << " l_overflow:"
            << l_overflow << " r_overflow:" << r_overflow
            << " l_pad:" << jcp.l_pad << " r_pad:" << jcp.r_pad
            << " iw:" << iw << " ih:" << jcp.ih << " ic:" << jcp.ic
            << " ow:" << jcp.ow << " oh:" << jcp.oh << " oc:"<< jcp.oc
            << " kw:" << jcp.kw << " kh:"<< jcp.kh << " mb:" << jcp.mb
            << " nb_ic_blocking:" << jcp.nb_ic_blocking
            << " ngroups:" << jcp.ngroups << " ur_w:" << ur_w << endl;

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
            || (l_overflow > 0 && n_oi > 1)) {
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

#else

#if 0

void jit_avx512_common_conv_bwd_data_kernel_f32::compute_loop_fma_sparse(
        int ur_w, int l_overflow, int r_overflow)
{
    Label kh_label, kd_label, skip_kd_loop;
    Label end_label, clear_label;

    int kw = jcp.kw;
    int kh = jcp.kh;

    int ow = jcp.ow;
    int oh = jcp.oh;

    int nb_ic = jcp.nb_ic;

    int iw = jcp.iw;
    int ih = jcp.ih;

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int l_pad = jcp.l_pad;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;
    int stride_h = jcp.stride_h;

    int ker_pipeline_depth = 4;
    assert(ker_reg_base_idx + ker_pipeline_depth <= 32);
    assert(oc_block >= ker_pipeline_depth);

    int num_ker_loads = oc_block * kw;
    int num_inp_prfs = ur_w * nstl::min(kw, stride_w)
                       + nstl::max(0, kw - stride_w);
    int num_prfs = num_ker_loads + num_inp_prfs;
    int num_fmas = num_ker_loads * ur_w / stride_w;
    int prf_inst_spacing = nstl::max(1, num_fmas / num_prfs);
    int prf_inst_trigger = (num_fmas % prf_inst_spacing) / 2;

    int regs = 31;
    int nr;
    if (kw * nb_ic <= regs)
        nr = kw * nb_ic;
    else {
        for (int tmp_ur_w = regs; tmp_ur_w > 0; --tmp_ur_w)
            if (tmp_ur_w % kw == 0) {
                nr = tmp_ur_w;
                break;
            }
    }

    assert(nr >= kw);

    int ic_buffs = nr / kw;
    int rem_ic_buffs = nb_ic % ic_buffs;
    int ic_iters = nb_ic / ic_buffs; 

    auto zmm_o = Xbyak::Zmm(31);

    assert(nr % kw == 0);

    Label no_init_label;

    mov(reg_channel, ptr[param + GET_OFF(channel)]);
    cmp(reg_channel, 0);
    jne(no_init_label, T_NEAR);

    vpxord(zmm_o, zmm_o, zmm_o);

    if (nb_ic * iw > 128) { // threshold may be tweaked later

        Reg64 aux_reg_src = aux_reg_dst;

        mov(aux_reg_src, reg_src);
        mov(reg_channel, nb_ic);

        Label ic_loop_label;
        L(ic_loop_label);

        for (int ii = 0; ii < iw; ii++) {
            vmovups(EVEX_compress_addr_safe(aux_reg_src, ii * ic_block * typesize,
                        reg_long_offt), zmm_o);
        }

        add(aux_reg_src, typesize * ic_block * ih * iw);

        dec(reg_channel);
        cmp(reg_channel, 0);
        jne(ic_loop_label);

    } else {

        for (int ic = 0; ic < nb_ic; ic++) {
            for (int ii = 0; ii < iw; ii++) {
                vmovups(EVEX_compress_addr_safe(reg_src, (ic * ic_block * ih * iw 
                            + ii * ic_block) * typesize,
                            reg_long_offt), zmm_o);
            }
        }
    }

    L(no_init_label);


    cmp(reg_kh, 0);
    je(end_label, T_NEAR);

    Reg64 reg_long_offt = reg_kj;

    L(kh_label); {

        assert(iw > (jcp.dilates + 1) * kw);

        push(reg_src);
        push(reg_ker);

        if (ic_iters > 1) {
            push(reg_kh);
        }

        int l_ow = 1;
        while (l_ow * stride_w - l_pad < 0) { // left unroll factor, first iteration always peeled
            l_ow++;
        }

        int r_ow = ow - 2;
        while (r_ow * stride_w - l_pad + (kw - 1) * dilate_w >= iw) { // right unroll factor, last iteration always peeled
            r_ow--;
        }

        int rotation_unroll_factor;
        if (stride_w / dilate_w < kw && dilate_w <= stride_w && stride_w % dilate_w == 0) {
            rotation_unroll_factor = kw % (stride_w / dilate_w) == 0 ? kw / (stride_w / dilate_w) : kw;
        } else {
            rotation_unroll_factor = 1;
        }

        cout << "nr:" << nr << " l_ow:" << l_ow << " r_ow:" << r_ow << " ic_iters:" << ic_iters
            << " ic_buffs:" << ic_buffs << " rem_ic_buffs:" << rem_ic_buffs << endl;


        auto get_reg_idx = [=](int oi, int ic_buff, int ki) {

            int rotation_idx = oi % rotation_unroll_factor;
            
            if (stride_w / dilate_w < kw && dilate_w <= stride_w && stride_w % dilate_w == 0) {
                return ic_buff * kw + ((stride_w / dilate_w) * rotation_idx + ki) % kw;
            } else {
                return ic_buff * kw + ki;
            }
        };

        auto comp_unrolled = [&](int oi, int cur_ic_buffs) {

            for (int ic_buff = 0; ic_buff < cur_ic_buffs; ic_buff++) {
                for (int ki = 0; ki < kw; ki++) {

                    if (oi * stride_w - l_pad + ki * dilate_w < 0
                        || oi * stride_w - l_pad + ki * dilate_w >= iw) {
                        continue;
                    }

                    int reg_idx = get_reg_idx(oi, ic_buff, ki);

                    Zmm zmm = Xbyak::Zmm(reg_idx);

                    if (stride_w / dilate_w >= kw || dilate_w > stride_w || stride_w % dilate_w != 0
                        || oi == 0 || ki >= kw - stride_w) {

                        size_t aux_src_offset = (size_t)typesize
                            * (ic_buff * ic_block * ih * iw 
                            + (oi * stride_w - l_pad + ki * dilate_w) * ic_block);

                        vmovups(zmm, EVEX_compress_addr_safe(reg_src, aux_src_offset,
                                    reg_long_offt));

                    }
                }
            }

            for (int oc = 0; oc < oc_block; oc++) {

                Label skip_zero_oc_label;

                size_t aux_dst_offset = typesize * (oi * oc_block + oc);

                cmp(dword[reg_dst + aux_dst_offset], 0);
                je(skip_zero_oc_label, T_NEAR);

                vbroadcastss(zmm_o, ptr[reg_dst + aux_dst_offset]);

                for (int ic_buff = 0; ic_buff < cur_ic_buffs; ic_buff++) {
                    for (int ki = 0; ki < kw; ki++) {

                        if (oi * stride_w - l_pad + ki * dilate_w < 0
                            || oi * stride_w - l_pad + ki * dilate_w >= iw) {
                            continue;
                        }

                        int reg_idx = get_reg_idx(oi, ic_buff, ki);

                        Zmm zmm = Xbyak::Zmm(reg_idx);
                        
                        size_t aux_kernel_offset = typesize * (ic_buff
                                * kw * kh * ic_block * oc_block
                                + oc * ic_block + ki * ic_block * oc_block);
                        vfmadd231ps(zmm, zmm_o,
                                EVEX_compress_addr_safe(reg_ker, aux_kernel_offset,
                                    reg_long_offt)); // probably don't need safe for weight tensor

                    }
                }

                L(skip_zero_oc_label);

            }

            for (int ic_buff = 0; ic_buff < cur_ic_buffs; ic_buff++) {
                for (int ki = 0; ki < kw; ki++) {

                    if (oi * stride_w - l_pad + ki * dilate_w < 0
                        || oi * stride_w - l_pad + ki * dilate_w >= iw) {
                        continue;
                    }

                    int reg_idx = get_reg_idx(oi, ic_buff, ki);

                    Zmm zmm = Xbyak::Zmm(reg_idx);

                    Label no_update_label;

                    if (stride_w / dilate_w >= kw || dilate_w > stride_w || stride_w % dilate_w != 0
                        || oi == ow - 1 || ki < stride_w) {
                    
                        size_t aux_src_offset = (size_t)typesize
                            * (ic_buff * ic_block * ih * iw 
                            + (oi * stride_w - l_pad + ki * dilate_w) * ic_block);
                        vmovups(EVEX_compress_addr_safe(reg_src, aux_src_offset,
                                    reg_long_offt), zmm);
                    }
                }
            }

        };


        auto comp_loop = [&](int idx, int oi, int cur_ic_buffs) {

            Reg64 aux_reg_src = aux_reg_ker;

            for (int ic_buff = 0; ic_buff < cur_ic_buffs; ic_buff++) {
                for (int ki = 0; ki < kw; ki++) {

                    int reg_idx = get_reg_idx(oi, ic_buff, ki);

                    Zmm zmm = Xbyak::Zmm(reg_idx);

                    if (stride_w / dilate_w >= kw || dilate_w > stride_w
                        || stride_w % dilate_w != 0 || ki >= kw - stride_w) {

                        size_t aux_src_offset = (size_t)typesize
                            * (ic_buff * ic_block * ih * iw 
                            + (idx * stride_w  - l_pad + ki * dilate_w) * ic_block);
                        vmovups(zmm, EVEX_compress_addr_safe(aux_reg_src, aux_src_offset,
                                    reg_long_offt));
                    }
                }
            }

            for (int oc = 0; oc < oc_block; oc++) {

                Label skip_zero_oc_label;

                int aux_dst_offset = typesize * (idx * oc_block + oc);

                cmp(dword[aux_reg_dst + aux_dst_offset], 0);
                je(skip_zero_oc_label, T_NEAR);

                vbroadcastss(zmm_o, ptr[aux_reg_dst + aux_dst_offset]);

                for (int ic_buff = 0; ic_buff < cur_ic_buffs; ic_buff++) {
                    for (int ki = 0; ki < kw; ki++) {

                        int reg_idx = get_reg_idx(oi, ic_buff, ki);

                        Zmm zmm = Xbyak::Zmm(reg_idx);
                        
                        size_t aux_kernel_offset = typesize * (ic_buff
                                        * kw * kh * ic_block * oc_block
                                        + oc * ic_block + ki * ic_block * oc_block);
                        vfmadd231ps(zmm, zmm_o,
                                EVEX_compress_addr_safe(reg_ker, aux_kernel_offset,
                                    reg_long_offt)); // probably don't need safe for weight tensor

                    }
                }

                L(skip_zero_oc_label);

            }

            for (int ic_buff = 0; ic_buff < cur_ic_buffs; ic_buff++) {
                for (int ki = 0; ki < kw; ki++) {

                    int reg_idx = get_reg_idx(oi, ic_buff, ki);

                    Zmm zmm = Xbyak::Zmm(reg_idx);

                    if (stride_w / dilate_w >= kw || dilate_w > stride_w
                        || stride_w % dilate_w != 0 || ki < stride_w) {

                        size_t aux_src_offset = (size_t)typesize
                            * (ic_buff * ic_block * ih * iw 
                            + (idx * stride_w - l_pad + ki * dilate_w) * ic_block);
                        vmovups(EVEX_compress_addr_safe(aux_reg_src, aux_src_offset,
                                    reg_long_offt), zmm);
                    }
                }
            }
        };

        auto outer_loop = [&](int cur_ic_buffs) {

            if (l_ow <= r_ow - rotation_unroll_factor * 2) { // threshold needs to be dynamically calculated based on the instruction count per section

                int rr_ow = r_ow - (r_ow - l_ow) % rotation_unroll_factor;
                int niter = (rr_ow - l_ow) / rotation_unroll_factor;

                cout << "leading :" << l_ow << " trailing:" << ow - rr_ow
                    << " factor:" << rotation_unroll_factor
                    << " niter:" << niter << endl;

                for (int oi = 0; oi < l_ow; oi++) {
                    comp_unrolled(oi, cur_ic_buffs);
                }

                Reg64 ow_itr_reg = reg_channel;
                Reg64 aux_reg_src = aux_reg_ker;

                mov(ow_itr_reg, niter);

                mov(aux_reg_dst, reg_dst);
                mov(aux_reg_src, reg_src);

                add(aux_reg_dst, l_ow * oc_block * typesize);
                add(aux_reg_src, l_ow * stride_w * ic_block * typesize);

                Label ow_loop_label;
                L(ow_loop_label);

                for (int i = 0; i < rotation_unroll_factor; i++) {
                    comp_loop(i, l_ow + i, cur_ic_buffs);
                }

                add(aux_reg_dst, oc_block * typesize * rotation_unroll_factor);
                add(aux_reg_src, stride_w * ic_block * typesize * rotation_unroll_factor);

                dec(ow_itr_reg);
                cmp(ow_itr_reg, 0);
                jne(ow_loop_label, T_NEAR);


                for (int oi = rr_ow; oi < ow; oi++) {
                    comp_unrolled(oi, cur_ic_buffs);
                }

            } else {

                cout << "fully unrolled" << endl;

                for (int oi = 0; oi < ow; oi++) {
                    comp_unrolled(oi, cur_ic_buffs);
                }
            }
        };

        if (ic_iters > 1) {

            mov(reg_kh, ic_iters);

            Label ic_iters_label;
            L(ic_iters_label); {

                outer_loop(ic_buffs);

                add(reg_src, typesize * ic_buffs * ic_block * ih * iw);
                add(reg_ker, typesize * ic_buffs * kw * kh * ic_block * oc_block);

            }

            dec(reg_kh);
            cmp(reg_kh, 0);
            jne(ic_iters_label);

        } else if (ic_iters == 1) {

            outer_loop(ic_buffs);

            if (rem_ic_buffs) {

                add(reg_src, typesize * ic_buffs * ic_block * ih * iw);
                add(reg_ker, typesize * ic_buffs * kw * kh * ic_block * oc_block);

            }

        }

        if (rem_ic_buffs) {
            outer_loop(rem_ic_buffs);
        }

        if (ic_iters > 1) {
            pop(reg_kh);
        }

        pop(reg_ker);
        pop(reg_src);

        add(reg_ker, typesize * stride_h * kw * oc_block * ic_block);
        sub(reg_dst, typesize * (jcp.dilate_h + 1) * ow * oc_block);

        dec(reg_kh);
        cmp(reg_kh, 0);
        jg(kh_label, T_NEAR);
    }

    L(end_label);
}

#elif 0

void jit_avx512_common_conv_bwd_data_kernel_f32::compute_loop_fma_sparse(
        int ur_w, int l_overflow, int r_overflow)
{
    Label kh_label, kd_label, skip_kd_loop;
    Label end_label, clear_label;

    int kw = jcp.kw;
    int kh = jcp.kh;

    int ow = jcp.ow;
    int oh = jcp.oh;

    int nb_ic = jcp.nb_ic;

    int iw = jcp.iw;
    int ih = jcp.ih;

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int l_pad = jcp.l_pad;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;
    int stride_h = jcp.stride_h;

    int ker_pipeline_depth = 4;
    assert(ker_reg_base_idx + ker_pipeline_depth <= 32);
    assert(oc_block >= ker_pipeline_depth);

    int num_ker_loads = oc_block * kw;
    int num_inp_prfs = ur_w * nstl::min(kw, stride_w)
                       + nstl::max(0, kw - stride_w);
    int num_prfs = num_ker_loads + num_inp_prfs;
    int num_fmas = num_ker_loads * ur_w / stride_w;
    int prf_inst_spacing = nstl::max(1, num_fmas / num_prfs);
    int prf_inst_trigger = (num_fmas % prf_inst_spacing) / 2;

    int regs = 31;
    int nr;
    if (kw * nb_ic <= regs)
        nr = kw * nb_ic;
    else {
        for (int tmp_ur_w = regs; tmp_ur_w > 0; --tmp_ur_w)
            if (tmp_ur_w % kw == 0) {
                nr = tmp_ur_w;
                break;
            }
    }

    assert(nr >= kw);

    int ic_buffs = nr / kw;
    int rem_ic_buffs = nb_ic % ic_buffs;
    int ic_iters = nb_ic / ic_buffs; 

    auto zmm_o = Xbyak::Zmm(31);

    assert(nr % kw == 0);

    Label no_init_label;

    mov(reg_channel, ptr[param + GET_OFF(channel)]);
    cmp(reg_channel, 0);
    jne(no_init_label, T_NEAR);

    vpxord(zmm_o, zmm_o, zmm_o);

    if (nb_ic * iw > 128) { // threshold may be tweaked later

        Reg64 aux_reg_src = aux_reg_dst;

        mov(aux_reg_src, reg_src);
        mov(reg_channel, nb_ic);

        Label ic_loop_label;
        L(ic_loop_label);

        for (int ii = 0; ii < iw; ii++) {
            vmovups(EVEX_compress_addr_safe(aux_reg_src, ii * ic_block * typesize,
                        reg_long_offt), zmm_o);
        }

        add(aux_reg_src, typesize * ic_block * ih * iw);

        dec(reg_channel);
        cmp(reg_channel, 0);
        jne(ic_loop_label);

    } else {

        for (int ic = 0; ic < nb_ic; ic++) {
            for (int ii = 0; ii < iw; ii++) {
                vmovups(EVEX_compress_addr_safe(reg_src, (ic * ic_block * ih * iw 
                            + ii * ic_block) * typesize,
                            reg_long_offt), zmm_o);
            }
        }
    }

    L(no_init_label);


    cmp(reg_kh, 0);
    je(end_label, T_NEAR);

    Reg64 reg_long_offt = reg_kj;

    L(kh_label); {

        assert(iw > (jcp.dilates + 1) * kw);

        push(reg_src);
        push(reg_ker);

        if (ic_iters > 1) {
            push(reg_kh);
        }

        int l_ow = 1;
        while (l_ow * stride_w - l_pad < 0) { // left unroll factor, first iteration always peeled
            l_ow++;
        }

        int r_ow = ow - 2;
        while (r_ow * stride_w - l_pad + (kw - 1) * dilate_w >= iw) { // right unroll factor, last iteration always peeled
            r_ow--;
        }

        int rotation_unroll_factor;
        if (stride_w / dilate_w < kw && dilate_w <= stride_w && stride_w % dilate_w == 0) {
            rotation_unroll_factor = kw % (stride_w / dilate_w) == 0 ? kw / (stride_w / dilate_w) : kw;
        } else {
            rotation_unroll_factor = 1;
        }

        cout << "nr:" << nr << " l_ow:" << l_ow << " r_ow:" << r_ow << " ic_iters:" << ic_iters
            << " ic_buffs:" << ic_buffs << " rem_ic_buffs:" << rem_ic_buffs << endl;


        auto get_reg_idx = [=](int oi, int ic_buff, int ki) {

            int rotation_idx = oi % rotation_unroll_factor;
            
            if (stride_w / dilate_w < kw && dilate_w <= stride_w && stride_w % dilate_w == 0) {
                return ic_buff * kw + ((stride_w / dilate_w) * rotation_idx + ki) % kw;
            } else {
                return ic_buff * kw + ki;
            }
        };

        auto comp_unrolled = [&](int oi, int cur_ic_buffs) {

            for (int ic_buff = 0; ic_buff < cur_ic_buffs; ic_buff++) {
                for (int ki = 0; ki < kw; ki++) {

                    if (oi * stride_w - l_pad + ki * dilate_w < 0
                        || oi * stride_w - l_pad + ki * dilate_w >= iw) {
                        continue;
                    }

                    int reg_idx = get_reg_idx(oi, ic_buff, ki);

                    Zmm zmm = Xbyak::Zmm(reg_idx);

                    if (stride_w / dilate_w >= kw || dilate_w > stride_w || stride_w % dilate_w != 0
                        || oi == 0 || ki >= kw - stride_w) {

                        size_t aux_src_offset = (size_t)typesize
                            * (ic_buff * ic_block * ih * iw 
                            + (oi * stride_w - l_pad + ki * dilate_w) * ic_block);

                        vmovups(zmm, EVEX_compress_addr_safe(reg_src, aux_src_offset,
                                    reg_long_offt));

                    }
                }
            }


            int oc_unroll_factor = 4;
            assert(oc_block % oc_unroll_factor == 0);

            Reg64 oc_itr_reg = reg_channel;
            mov(oc_itr_reg, oc_block / oc_unroll_factor);

            Label oc_loop_label; 
            L(oc_loop_label); {

                for (int oc = 0; oc < oc_unroll_factor; oc++) {

                    Label skip_zero_oc_label;

                    size_t aux_dst_offset = typesize * (oi * oc_block + oc);

                    cmp(dword[reg_dst + aux_dst_offset], 0);
                    je(skip_zero_oc_label, T_NEAR);

                    vbroadcastss(zmm_o, ptr[reg_dst + aux_dst_offset]);

                    for (int ic_buff = 0; ic_buff < cur_ic_buffs; ic_buff++) {
                        for (int ki = 0; ki < kw; ki++) {

                            if (oi * stride_w - l_pad + ki * dilate_w < 0
                                || oi * stride_w - l_pad + ki * dilate_w >= iw) {
                                continue;
                            }

                            int reg_idx = get_reg_idx(oi, ic_buff, ki);

                            Zmm zmm = Xbyak::Zmm(reg_idx);
                            
                            size_t aux_kernel_offset = typesize * (ic_buff
                                    * kw * kh * ic_block * oc_block
                                    + oc * ic_block + ki * ic_block * oc_block);
                            vfmadd231ps(zmm, zmm_o,
                                    EVEX_compress_addr_safe(reg_ker, aux_kernel_offset,
                                        reg_long_offt)); // probably don't need safe for weight tensor

                        }
                    }

                    L(skip_zero_oc_label);
                }

            }

            add(reg_ker, oc_unroll_factor * typesize * ic_block);
            add(reg_dst, oc_unroll_factor * typesize);

            dec(oc_itr_reg);
            cmp(oc_itr_reg, 0);
            jne(oc_loop_label);

            sub(reg_ker, typesize * oc_block * ic_block);
            sub(reg_dst, typesize * oc_block);

            for (int ic_buff = 0; ic_buff < cur_ic_buffs; ic_buff++) {
                for (int ki = 0; ki < kw; ki++) {

                    if (oi * stride_w - l_pad + ki * dilate_w < 0
                        || oi * stride_w - l_pad + ki * dilate_w >= iw) {
                        continue;
                    }

                    int reg_idx = get_reg_idx(oi, ic_buff, ki);

                    Zmm zmm = Xbyak::Zmm(reg_idx);

                    Label no_update_label;

                    if (stride_w / dilate_w >= kw || dilate_w > stride_w || stride_w % dilate_w != 0
                        || oi == ow - 1 || ki < stride_w) {
                    
                        size_t aux_src_offset = (size_t)typesize
                            * (ic_buff * ic_block * ih * iw 
                            + (oi * stride_w - l_pad + ki * dilate_w) * ic_block);
                        vmovups(EVEX_compress_addr_safe(reg_src, aux_src_offset,
                                    reg_long_offt), zmm);
                    }
                }
            }

        };


        auto comp_loop = [&](int idx, int oi, int cur_ic_buffs) {

            Reg64 aux_reg_src = aux_reg_ker;

            for (int ic_buff = 0; ic_buff < cur_ic_buffs; ic_buff++) {
                for (int ki = 0; ki < kw; ki++) {

                    int reg_idx = get_reg_idx(oi, ic_buff, ki);

                    Zmm zmm = Xbyak::Zmm(reg_idx);

                    if (stride_w / dilate_w >= kw || dilate_w > stride_w
                        || stride_w % dilate_w != 0 || ki >= kw - stride_w) {

                        size_t aux_src_offset = (size_t)typesize
                            * (ic_buff * ic_block * ih * iw 
                            + (idx * stride_w  - l_pad + ki * dilate_w) * ic_block);
                        vmovups(zmm, EVEX_compress_addr_safe(aux_reg_src, aux_src_offset,
                                    reg_long_offt));
                    }
                }
            }

            int oc_unroll_factor = 4;
            assert(oc_block % oc_unroll_factor == 0);

            Reg64 oc_itr_reg = reg_channel;
            mov(oc_itr_reg, oc_block / oc_unroll_factor);

            Label oc_loop_label; 
            L(oc_loop_label); {

                for (int oc = 0; oc < oc_unroll_factor; oc++) {

                    Label skip_zero_oc_label;

                    int aux_dst_offset = typesize * (idx * oc_block + oc);

                    cmp(dword[aux_reg_dst + aux_dst_offset], 0);
                    je(skip_zero_oc_label, T_NEAR);

                    vbroadcastss(zmm_o, ptr[aux_reg_dst + aux_dst_offset]);

                    for (int ic_buff = 0; ic_buff < cur_ic_buffs; ic_buff++) {
                        for (int ki = 0; ki < kw; ki++) {

                            int reg_idx = get_reg_idx(oi, ic_buff, ki);

                            Zmm zmm = Xbyak::Zmm(reg_idx);
                            
                            size_t aux_kernel_offset = typesize * (ic_buff
                                            * kw * kh * ic_block * oc_block
                                            + oc * ic_block + ki * ic_block * oc_block);
                            vfmadd231ps(zmm, zmm_o,
                                    EVEX_compress_addr_safe(reg_ker, aux_kernel_offset,
                                        reg_long_offt)); // probably don't need safe for weight tensor

                        }
                    }

                    L(skip_zero_oc_label);
                }

            }

            add(reg_ker, oc_unroll_factor * typesize * ic_block);
            add(aux_reg_dst, oc_unroll_factor * typesize);

            dec(oc_itr_reg);
            cmp(oc_itr_reg, 0);
            jne(oc_loop_label);

            sub(reg_ker, typesize * oc_block * ic_block);
            sub(aux_reg_dst, typesize * oc_block);

            for (int ic_buff = 0; ic_buff < cur_ic_buffs; ic_buff++) {
                for (int ki = 0; ki < kw; ki++) {

                    int reg_idx = get_reg_idx(oi, ic_buff, ki);

                    Zmm zmm = Xbyak::Zmm(reg_idx);

                    if (stride_w / dilate_w >= kw || dilate_w > stride_w
                        || stride_w % dilate_w != 0 || ki < stride_w) {

                        size_t aux_src_offset = (size_t)typesize
                            * (ic_buff * ic_block * ih * iw 
                            + (idx * stride_w - l_pad + ki * dilate_w) * ic_block);
                        vmovups(EVEX_compress_addr_safe(aux_reg_src, aux_src_offset,
                                    reg_long_offt), zmm);
                    }
                }
            }
        };

        auto outer_loop = [&](int cur_ic_buffs) {

            if (l_ow <= r_ow - rotation_unroll_factor * 2) { // threshold needs to be dynamically calculated based on the instruction count per section

                int rr_ow = r_ow - (r_ow - l_ow) % rotation_unroll_factor;
                int niter = (rr_ow - l_ow) / rotation_unroll_factor;

                cout << "leading :" << l_ow << " trailing:" << ow - rr_ow
                    << " factor:" << rotation_unroll_factor
                    << " niter:" << niter << endl;

                for (int oi = 0; oi < l_ow; oi++) {
                    comp_unrolled(oi, cur_ic_buffs);
                }

                Reg64 ow_itr_reg = reg_channel;
                Reg64 aux_reg_src = aux_reg_ker;

                mov(ow_itr_reg, niter);

                mov(aux_reg_dst, reg_dst);
                mov(aux_reg_src, reg_src);

                add(aux_reg_dst, l_ow * oc_block * typesize);
                add(aux_reg_src, l_ow * stride_w * ic_block * typesize);

                Label ow_loop_label;
                L(ow_loop_label); {

                    push(ow_itr_reg);

                    for (int i = 0; i < rotation_unroll_factor; i++) {
                        comp_loop(i, l_ow + i, cur_ic_buffs);
                    }

                    pop(ow_itr_reg);

                    add(aux_reg_dst, oc_block * typesize * rotation_unroll_factor);
                    add(aux_reg_src, stride_w * ic_block * typesize * rotation_unroll_factor);

                }

                dec(ow_itr_reg);
                cmp(ow_itr_reg, 0);
                jne(ow_loop_label, T_NEAR);


                for (int oi = rr_ow; oi < ow; oi++) {
                    comp_unrolled(oi, cur_ic_buffs);
                }

            } else {

                cout << "fully unrolled" << endl;

                for (int oi = 0; oi < ow; oi++) {
                    comp_unrolled(oi, cur_ic_buffs);
                }
            }
        };

        if (ic_iters > 1) {

            mov(reg_kh, ic_iters);

            Label ic_iters_label;
            L(ic_iters_label); {

                outer_loop(ic_buffs);

                add(reg_src, typesize * ic_buffs * ic_block * ih * iw);
                add(reg_ker, typesize * ic_buffs * kw * kh * ic_block * oc_block);

            }

            dec(reg_kh);
            cmp(reg_kh, 0);
            jne(ic_iters_label);

        } else if (ic_iters == 1) {

            outer_loop(ic_buffs);

            if (rem_ic_buffs) {

                add(reg_src, typesize * ic_buffs * ic_block * ih * iw);
                add(reg_ker, typesize * ic_buffs * kw * kh * ic_block * oc_block);

            }

        }

        if (rem_ic_buffs) {
            outer_loop(rem_ic_buffs);
        }

        if (ic_iters > 1) {
            pop(reg_kh);
        }

        pop(reg_ker);
        pop(reg_src);

        add(reg_ker, typesize * stride_h * kw * oc_block * ic_block);
        sub(reg_dst, typesize * (jcp.dilate_h + 1) * ow * oc_block);

        dec(reg_kh);
        cmp(reg_kh, 0);
        jg(kh_label, T_NEAR);
    }

    L(end_label);
}

#else

void jit_avx512_common_conv_bwd_data_kernel_f32::compute_loop_fma_sparse()
{
    Label kh_label, kd_label, skip_kd_loop;
    Label end_label, clear_label;

    int kw = jcp.kw;
    int kh = jcp.kh;

    int ow = jcp.ow;
    int oh = jcp.oh;

    int nb_ic = jcp.nb_ic;

    int iw = jcp.iw;
    int ih = jcp.ih;

    int mb_block = jcp.mb_block;

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int l_pad = jcp.l_pad;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;
    int stride_h = jcp.stride_h;

    int ker_pipeline_depth = 4;
    assert(ker_reg_base_idx + ker_pipeline_depth <= 32);
    assert(oc_block >= ker_pipeline_depth);

    int nr = jcp.ur_sparse;
    int ic_buffs = jcp.ic_buffs;

    assert(nr >= kw);
    assert(ic_block == 16); // needed for specific optimization
    assert(typesize == 4);

    int ic_iters = nb_ic / ic_buffs;

    int l_ow = 1;
    while (l_ow * stride_w - l_pad < 0) { // left unroll factor, first iteration always peeled
        l_ow++;
    }
    l_ow++; // unroll one more due to pipelined vector write

    int r_ow = ow - 2;
    while (r_ow * stride_w - l_pad + (kw - 1) * dilate_w >= iw) { // right unroll factor, last iteration always peeled
        r_ow--;
    }

    int rotation_unroll_factor;
    int rw = ic_buffs * (kw + 1) <= 30 && stride_w == 1 && kw > 1 ? kw + 1 : kw;

    if (stride_w / dilate_w < kw && dilate_w <= stride_w && stride_w % dilate_w == 0) {
        rotation_unroll_factor = rw % (stride_w / dilate_w) == 0 ? rw / (stride_w / dilate_w) : rw;
    } else {
        rotation_unroll_factor = 1;
    }

    cout << "nr:" << nr << " l_ow:" << l_ow << " r_ow:" << r_ow << " ic_iters:" << ic_iters
        << " ic_buffs:" << ic_buffs << endl;

    auto zmm_o = Xbyak::Zmm(31);

    auto ymm_zero = Xbyak::Ymm(30);
    auto zmm_zero = Xbyak::Zmm(30);

    vpxord(zmm_zero, zmm_zero, zmm_zero);

    vcmpps(k7, zmm_zero, ptr[reg_dst], 4);
    prefetcht1(ptr[reg_dst_prf]);

    assert(nr % kw == 0);

    Label no_init_label;

    mov(reg_channel, ptr[param + GET_OFF(channel)]);
    cmp(reg_channel, 0);
    jne(no_init_label, T_NEAR);

    if (ic_buffs * iw > 128) { // threshold may be tweaked later

        Reg64 aux_reg_src = aux_reg_dst;
        Reg64 aux_reg_src_prf = aux_reg_ker;

        mov(aux_reg_src, reg_src);
        mov(aux_reg_src_prf, reg_src_prf);

        mov(reg_channel, ic_buffs);

        Label ic_loop_label;
        L(ic_loop_label);

        for (int ii = 0; ii < iw; ii++) {
            vmovups(EVEX_compress_addr_safe(aux_reg_src, ii * ic_block * typesize,
                        reg_long_offt), zmm_zero);
            prefetcht1(EVEX_compress_addr_safe(aux_reg_src_prf, ii * ic_block * typesize,
                        reg_long_offt));
        }

        add(aux_reg_src, typesize * ic_block * mb_block * iw);
        add(aux_reg_src_prf, typesize * ic_block * mb_block * iw);

        dec(reg_channel);
        cmp(reg_channel, 0);
        jne(ic_loop_label);

    } else {

        for (int ic = 0; ic < ic_buffs; ic++) {
            for (int ii = 0; ii < iw; ii++) {
                vmovups(EVEX_compress_addr_safe(reg_src, (ic * ic_block * mb_block * iw 
                            + ii * ic_block) * typesize,
                            reg_long_offt), zmm_zero);
                prefetcht1(EVEX_compress_addr_safe(reg_src_prf, (ic * ic_block * mb_block * iw 
                            + ii * ic_block) * typesize,
                            reg_long_offt));
            }
        }
    }

    L(no_init_label);

    cmp(reg_kh, 0);
    je(end_label, T_NEAR);

    Reg64 reg_long_offt = reg_kj;

    auto get_reg_idx = [=](int oi, int ic_buff, int ki) {

        int rotation_idx = oi % rotation_unroll_factor;
        
        if (stride_w / dilate_w < kw && dilate_w <= stride_w && stride_w % dilate_w == 0) {
            return ic_buff * rw + ((stride_w / dilate_w) * rotation_idx + ki) % rw;
        } else {
            return ic_buff * rw + ki;
        }
    };

    auto comp_unrolled = [&](int oi, int cur_ic_buffs) {

        Reg64 mask_reg = reg_oi;
        Reg32 oc_itr_reg = reg_kh.cvt32();
        Reg64 tzcnt_reg = reg_channel;

        kmovw(mask_reg.cvt32(), k7);
        popcnt(oc_itr_reg, mask_reg.cvt32());

        if (oi < ow - 1) { // pipelined
            size_t aux_dst_offset = typesize * (oi + 1) * oc_block;
            vcmpps(k7, zmm_zero, EVEX_compress_addr_safe(reg_dst, aux_dst_offset,
                                    reg_long_offt), 4);

            prefetcht1(EVEX_compress_addr_safe(reg_dst_prf, aux_dst_offset,
                                    reg_long_offt));
        }


        if (oi > 0) {
                
            for (int ki = 0; ki < kw; ki++) {
                for (int ic_buff = 0; ic_buff < cur_ic_buffs; ic_buff++) {

                    if ((oi - 1) * stride_w - l_pad + ki * dilate_w >= 0
                        && (oi - 1) * stride_w - l_pad + ki * dilate_w < iw) {

                        int reg_idx = get_reg_idx(oi - 1, ic_buff, ki);

                        Zmm zmm = Xbyak::Zmm(reg_idx);

                        Label no_update_label;

                        if (stride_w / dilate_w >= kw || dilate_w > stride_w || stride_w % dilate_w != 0
                            || ki < stride_w) {
                        
                            size_t aux_src_offset = (size_t)typesize
                                * (ic_buff * ic_block* mb_block * iw 
                                + ((oi - 1) * stride_w - l_pad + ki * dilate_w) * ic_block);
                            vmovups(EVEX_compress_addr_safe(reg_src, aux_src_offset,
                                        reg_long_offt), zmm);
                        }
                    }
                }
            }
        }

        if (cur_ic_buffs * (kw + 1) <= 30 && stride_w == 1 && kw > 1) {

            if (oi == 0) {
                for (int ki = 0; ki < kw; ki++) {
                    for (int ic_buff = 0; ic_buff < cur_ic_buffs; ic_buff++) {

                        if (oi * stride_w - l_pad + ki * dilate_w >= 0
                            && oi * stride_w - l_pad + ki * dilate_w < iw) {

                            int reg_idx = get_reg_idx(oi, ic_buff, ki);

                            Zmm zmm = Xbyak::Zmm(reg_idx);


                            size_t aux_src_offset = (size_t)typesize
                                * (ic_buff * ic_block * mb_block * iw 
                                + (oi * stride_w - l_pad + ki * dilate_w) * ic_block);

                            vmovups(zmm, EVEX_compress_addr_safe(reg_src, aux_src_offset,
                                        reg_long_offt));
                            //prefetcht1(EVEX_compress_addr_safe(reg_src_prf, aux_src_offset,
                            //            reg_long_offt));

                        }
                    }
                }
            }
            if (oi < ow - 1) {
                for (int ki = 0; ki < kw; ki++) {
                    for (int ic_buff = 0; ic_buff < cur_ic_buffs; ic_buff++) {

                        if ((oi + 1) * stride_w - l_pad + ki * dilate_w >= 0
                            && (oi + 1) * stride_w - l_pad + ki * dilate_w < iw) {

                            int reg_idx = get_reg_idx(oi + 1, ic_buff, ki);

                            Zmm zmm = Xbyak::Zmm(reg_idx);

                            if (stride_w / dilate_w >= kw || dilate_w > stride_w || stride_w % dilate_w != 0
                                || ki >= kw - stride_w) {

                                size_t aux_src_offset = (size_t)typesize
                                    * (ic_buff * ic_block * mb_block * iw 
                                    + ((oi + 1) * stride_w - l_pad + ki * dilate_w) * ic_block);

                                vmovups(zmm, EVEX_compress_addr_safe(reg_src, aux_src_offset,
                                            reg_long_offt));
                                //prefetcht1(EVEX_compress_addr_safe(reg_src_prf, aux_src_offset,
                                //            reg_long_offt));

                            }
                        }
                    }
                }
            }

        } else {
        
            for (int ki = 0; ki < kw; ki++) {
                for (int ic_buff = 0; ic_buff < cur_ic_buffs; ic_buff++) {

                    if (oi * stride_w - l_pad + ki * dilate_w >= 0
                        && oi * stride_w - l_pad + ki * dilate_w < iw) {

                        int reg_idx = get_reg_idx(oi, ic_buff, ki);

                        Zmm zmm = Xbyak::Zmm(reg_idx);

                        if (stride_w / dilate_w >= kw || dilate_w > stride_w || stride_w % dilate_w != 0
                            || oi == 0 || ki >= kw - stride_w) {

                            size_t aux_src_offset = (size_t)typesize
                                * (ic_buff * ic_block * mb_block * iw 
                                + (oi * stride_w - l_pad + ki * dilate_w) * ic_block);

                            vmovups(zmm, EVEX_compress_addr_safe(reg_src, aux_src_offset,
                                        reg_long_offt));
                            //prefetcht1(EVEX_compress_addr_safe(reg_src_prf, aux_src_offset,
                            //            reg_long_offt));

                        }
                    }
                }
            }
        }


        Label oc_loop_end_label;
        jz(oc_loop_end_label, T_NEAR);

        tzcnt(tzcnt_reg.cvt32(), mask_reg.cvt32());
        inc(tzcnt_reg.cvt32());

        shrx(mask_reg.cvt32(), mask_reg.cvt32(), tzcnt_reg.cvt32());

        mov(aux_reg_ker, reg_ker);
        mov(aux_reg_dst, reg_dst);

        Label oc_loop_label;
        L(oc_loop_label); {

            lea(aux_reg_dst, ptr[aux_reg_dst + tzcnt_reg * typesize]);

            int aux_dst_offset = typesize * (oi * oc_block - 1);
            vbroadcastss(zmm_o, ptr[aux_reg_dst + aux_dst_offset]);

            shl(tzcnt_reg.cvt32(), 6);
            add(aux_reg_ker, tzcnt_reg);

            tzcnt(tzcnt_reg.cvt32(), mask_reg.cvt32()); // pipelined
            inc(tzcnt_reg.cvt32());

            dec(oc_itr_reg);

            shrx(mask_reg.cvt32(), mask_reg.cvt32(), tzcnt_reg.cvt32()); // does not change flags

            for (int ic_buff = 0; ic_buff < cur_ic_buffs; ic_buff++) {
                for (int ki = 0; ki < kw; ki++) {

                    if (oi * stride_w - l_pad + ki * dilate_w < 0
                        || oi * stride_w - l_pad + ki * dilate_w >= iw) {
                        continue;
                    }

                    int reg_idx = get_reg_idx(oi, ic_buff, ki);

                    Zmm zmm = Xbyak::Zmm(reg_idx);
                    
                    size_t aux_kernel_offset = typesize * (ic_buff
                            * kw * ic_block * oc_block
                            + ki * ic_block * oc_block);
                    vfmadd231ps(zmm, zmm_o,
                            EVEX_compress_addr_safe(aux_reg_ker, aux_kernel_offset,
                                reg_long_offt)); // probably don't need safe for weight tensor

                }
            }

            jnz(oc_loop_label, T_NEAR);
        }

        L(oc_loop_end_label);

    };


    auto comp_loop = [&](int idx, int oi, int cur_ic_buffs) {

        Reg64 aux_reg_src = aux_reg_ker;
        Reg64 mask_reg = reg_oi;
        Reg32 oc_itr_reg = reg_kh.cvt32();
        Reg64 tzcnt_reg = reg_channel;

        kmovw(mask_reg.cvt32(), k7);
        popcnt(oc_itr_reg, mask_reg.cvt32());

        size_t aux_dst_offset = typesize * (idx + 1) * oc_block;
        vcmpps(k7, zmm_zero, EVEX_compress_addr_safe(aux_reg_dst, aux_dst_offset, // pipelined
                                reg_long_offt), 4);

        prefetcht1(EVEX_compress_addr_safe(reg_dst_prf, aux_dst_offset, // pipelined
                                reg_long_offt));

        for (int ki = 0; ki < kw; ki++) {
            for (int ic_buff = 0; ic_buff < cur_ic_buffs; ic_buff++) {

                int reg_idx = get_reg_idx(oi - 1, ic_buff, ki);

                Zmm zmm = Xbyak::Zmm(reg_idx);

                if (stride_w / dilate_w >= kw || dilate_w > stride_w
                    || stride_w % dilate_w != 0 || ki < stride_w) {

                    size_t aux_src_offset = (size_t)typesize
                        * (ic_buff * ic_block * mb_block * iw 
                        + ((idx - 1) * stride_w - l_pad + ki * dilate_w) * ic_block);
                    vmovups(EVEX_compress_addr_safe(aux_reg_src, aux_src_offset,
                                reg_long_offt), zmm);
                }
            }
        }

        for (int ki = 0; ki < kw; ki++) {
            for (int ic_buff = 0; ic_buff < cur_ic_buffs; ic_buff++) {

                int reg_idx = get_reg_idx(oi, ic_buff, ki);

                Zmm zmm = Xbyak::Zmm(reg_idx);

                if (stride_w / dilate_w >= kw || dilate_w > stride_w
                    || stride_w % dilate_w != 0 || ki >= kw - stride_w) {

                    size_t aux_src_offset = (size_t)typesize
                        * (ic_buff * ic_block * mb_block * iw 
                        + (idx * stride_w  - l_pad + ki * dilate_w) * ic_block);
                    vmovups(zmm, EVEX_compress_addr_safe(aux_reg_src, aux_src_offset,
                                reg_long_offt));
                    prefetcht1(EVEX_compress_addr_safe(reg_src_prf, aux_src_offset,
                                reg_long_offt));
                }

                /*int pref_idx = idx + 2;

                if (stride_w / dilate_w >= kw || dilate_w > stride_w
                    || stride_w % dilate_w != 0 || ki >= kw - stride_w) {

                    size_t aux_src_offset = (size_t)typesize
                        * (ic_buff * ic_block * ih * iw 
                        + (pref_idx * stride_w  - l_pad + ki * dilate_w) * ic_block);
                    mic_prefetcht0(EVEX_compress_addr_safe(aux_reg_src, aux_src_offset,
                                reg_long_offt));
                }*/
            }
        }

        Label oc_loop_end_label;
        jz(oc_loop_end_label, T_NEAR);

        tzcnt(tzcnt_reg.cvt32(), mask_reg.cvt32());
        inc(tzcnt_reg.cvt32());

        shrx(mask_reg.cvt32(), mask_reg.cvt32(), tzcnt_reg.cvt32());

        push(aux_reg_dst);
        push(reg_ker);

        //add(qword[param + GET_OFF(perf_cnt)], oc_itr_reg);

        Label oc_loop_label;
        L(oc_loop_label); {

            lea(aux_reg_dst, ptr[aux_reg_dst + tzcnt_reg * typesize]);

            int aux_dst_offset = typesize * (idx * oc_block - 1);
            vbroadcastss(zmm_o, ptr[aux_reg_dst + aux_dst_offset]);

            shl(tzcnt_reg.cvt32(), 6);
            add(reg_ker, tzcnt_reg);

            tzcnt(tzcnt_reg.cvt32(), mask_reg.cvt32()); // pipelined
            inc(tzcnt_reg.cvt32());

            dec(oc_itr_reg);

            shrx(mask_reg.cvt32(), mask_reg.cvt32(), tzcnt_reg.cvt32()); // does not change flags

            for (int ic_buff = 0; ic_buff < cur_ic_buffs; ic_buff++) {
                for (int ki = 0; ki < kw; ki++) {

                    int reg_idx = get_reg_idx(oi, ic_buff, ki);

                    Zmm zmm = Xbyak::Zmm(reg_idx);
                    
                    size_t aux_kernel_offset = typesize * (ic_buff
                                    * kw * ic_block * oc_block
                                    + ki * ic_block * oc_block);
                    vfmadd231ps(zmm, zmm_o,
                            EVEX_compress_addr_safe(reg_ker, aux_kernel_offset,
                                reg_long_offt)); // probably don't need safe for weight tensor

                }
            }

            jnz(oc_loop_label, T_NEAR);
        }

        pop(reg_ker);
        pop(aux_reg_dst);

        L(oc_loop_end_label);

    };

    auto epilogue = [&](int cur_ic_buffs) {


        for (int ki = 0; ki < kw; ki++) {
            for (int ic_buff = 0; ic_buff < cur_ic_buffs; ic_buff++) {

                if ((ow - 1) * stride_w - l_pad + ki * dilate_w >= 0
                    && (ow - 1) * stride_w - l_pad + ki * dilate_w < iw) {

                    int reg_idx = get_reg_idx((ow - 1), ic_buff, ki);

                    Zmm zmm = Xbyak::Zmm(reg_idx);

                    Label no_update_label;
                    
                    size_t aux_src_offset = (size_t)typesize
                        * (ic_buff * ic_block * mb_block * iw 
                        + ((ow - 1) * stride_w - l_pad + ki * dilate_w) * ic_block);
                    vmovups(EVEX_compress_addr_safe(reg_src, aux_src_offset,
                                reg_long_offt), zmm);
                }
            }
        }
    };

    auto outer_loop = [&](int cur_ic_buffs) {

        sub(reg_ker, ic_block * typesize);

        if (0) {
        //if (l_ow <= r_ow - rotation_unroll_factor * 2) { // threshold needs to be dynamically calculated based on the instruction count per section

            int rr_ow = r_ow - (r_ow - l_ow) % rotation_unroll_factor;
            int niter = (rr_ow - l_ow) / rotation_unroll_factor;

            cout << "leading :" << l_ow << " trailing:" << ow - rr_ow
                << " factor:" << rotation_unroll_factor
                << " niter:" << niter << endl;

            for (int oi = 0; oi < l_ow; oi++) {
                comp_unrolled(oi, cur_ic_buffs);
            }

            Reg64 ow_itr_reg = reg_channel;
            Reg64 aux_reg_src = aux_reg_ker;

            mov(ow_itr_reg, niter);

            mov(aux_reg_dst, reg_dst);
            mov(aux_reg_src, reg_src);

            add(aux_reg_dst, l_ow * oc_block * typesize);
            add(aux_reg_src, l_ow * stride_w * ic_block * typesize);

            add(reg_dst_prf, l_ow * oc_block * typesize);
            add(reg_src_prf, l_ow * stride_w * ic_block * typesize);

            Label ow_loop_label;
            L(ow_loop_label); {

                push(ow_itr_reg);

                for (int i = 0; i < rotation_unroll_factor; i++) {
                    comp_loop(i, l_ow + i, cur_ic_buffs);
                }

                pop(ow_itr_reg);

                add(aux_reg_dst, oc_block * typesize * rotation_unroll_factor);
                add(aux_reg_src, stride_w * ic_block * typesize * rotation_unroll_factor);

                add(reg_dst_prf, oc_block * typesize * rotation_unroll_factor);
                add(reg_src_prf, stride_w * ic_block * typesize * rotation_unroll_factor);

                dec(ow_itr_reg);
                jnz(ow_loop_label, T_NEAR);

            }

            mov(reg_src_prf, ptr[param + GET_OFF(src_prf)]);
            mov(reg_dst_prf, ptr[param + GET_OFF(dst_prf)]);

            for (int oi = rr_ow; oi < ow; oi++) {
                comp_unrolled(oi, cur_ic_buffs);
            }

        } else {

            cout << "fully unrolled" << endl;

            for (int oi = 0; oi < ow; oi++) {
                comp_unrolled(oi, cur_ic_buffs);
            }
        }

        epilogue(cur_ic_buffs);
    };

    outer_loop(ic_buffs);

    L(end_label);
}

#endif

void jit_avx512_common_conv_bwd_data_kernel_f32::generate()
{
    int iw = jcp.iw;
    int kw = jcp.kw;
    int ur_w = jcp.ur_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int ur_w_tail = jcp.ur_w_tail;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;

    int dst_shift = jcp.typesize_in * (ur_w / stride_w) * ic_block;
    int src_shift = jcp.typesize_out * ur_w * oc_block;

    preamble();

    mov(reg_src, ptr[param + GET_OFF(src)]);
    mov(reg_dst, ptr[param + GET_OFF(dst)]);
    mov(reg_ker, ptr[param + GET_OFF(filt)]);

    mov(reg_kh, ptr[param + GET_OFF(kh_padding)]);
    mov(reg_src_prf, ptr[param + GET_OFF(src_prf)]);
    mov(reg_dst_prf, ptr[param + GET_OFF(dst_prf)]);
    mov(reg_ker_prf, ptr[param + GET_OFF(filt_prf)]);

    int l_overflow = nstl::max(0, ((kw - 1) * dilate_w - jcp.l_pad) / stride_w);
    int r_overflow = nstl::max(0, ((kw - 1) * dilate_w
                    - nstl::max(0, jcp.r_pad)) / stride_w);
    int r_overflow1 = nstl::max(0, ((kw - 1) * dilate_w
                    - nstl::max(0, jcp.r_pad) - ur_w_tail) / stride_w);


    cout << "stride_w:" << stride_w << " stride_h:" << jcp.stride_h
            << " l_overflow:" << l_overflow << " r_overflow:" << r_overflow
            << " l_pad:" << jcp.l_pad << " r_pad:" << jcp.r_pad
            << " iw:" << iw << " ih:" << jcp.ih << " ic:" << jcp.ic
            << " ow:" << jcp.ow << " oh:" << jcp.oh << " oc:"<< jcp.oc
            << " kw:" << jcp.kw << " kh:"<< jcp.kh << " mb:" << jcp.mb
            << " nb_ic_blocking:" << jcp.nb_ic_blocking
            << " ngroups:" << jcp.ngroups << " dilate_w:" << dilate_w
            << " typesize_in:" << jcp.typesize_in
            << " typesize_out:" << jcp.typesize_out << endl;

    if (jcp.ndims == 5) {

        int n_oi = iw / ur_w;
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
                || (l_overflow > 0 && n_oi > 1)) {
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
    } else {
        compute_loop_fma_sparse();
    }

    postamble();
}

#endif

status_t jit_avx512_common_conv_bwd_data_kernel_f32::init_conf(
        jit_conv_conf_t &jcp,
        const convolution_desc_t &cd,
        const memory_desc_wrapper &diff_src_d,
        const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &diff_dst_d)
{
    if (!mayiuse(avx512_common)) return status::unimplemented;

    const int simd_w = cpu_isa_traits<avx512_common>::vlen / sizeof(float);
    const bool with_groups = weights_d.ndims() == diff_src_d.ndims() + 1;
    int ndims = diff_src_d.ndims();

    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = diff_src_d.dims()[0];

    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = diff_src_d.dims()[1] / jcp.ngroups;

    jcp.id = (ndims == 5) ? diff_src_d.dims()[2] : 1;
    jcp.ih = diff_src_d.dims()[ndims-2];
    jcp.iw = diff_src_d.dims()[ndims-1];
    jcp.od = (ndims == 5) ? diff_dst_d.dims()[2] : 1;
    jcp.oh = diff_dst_d.dims()[ndims-2];
    jcp.ow = diff_dst_d.dims()[ndims-1];

    jcp.kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];

    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = cd.padding[0][ndims-4];
    jcp.l_pad = cd.padding[0][ndims-3];

    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = cd.strides[ndims-4];
    jcp.stride_w = cd.strides[ndims-3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = cd.dilates[ndims-4];
    jcp.dilate_w = cd.dilates[ndims-3];
    if ((jcp.dilate_w != 0 && jcp.stride_w != 1)
            || (jcp.dilate_d != 0 && jcp.stride_d != 1)
            || (jcp.dilate_h != 0 && jcp.stride_h != 1))
        return status::unimplemented;

    jcp.r_pad = (jcp.ow - 1) * jcp.stride_w + (jcp.kw - 1) * (jcp.dilate_w + 1)
            - (jcp.iw + jcp.l_pad - 1);
    jcp.b_pad = (jcp.oh - 1) * jcp.stride_h + (jcp.kh - 1) * (jcp.dilate_h + 1)
            - (jcp.ih + jcp.t_pad - 1);
    jcp.back_pad = (jcp.od - 1) * jcp.stride_d
            + (jcp.kd - 1) * (jcp.dilate_d + 1) - (jcp.id + jcp.f_pad - 1);

    jcp.aligned_threads = 0;

    jcp.is_1stconv = false;

    jcp.oc_block = simd_w;
    jcp.ic_block = jcp.is_1stconv ? jcp.ic : simd_w;

    bool ok_to_pad_channels = true
        && jcp.ngroups == 1
        && diff_src_d.data_type() == data_type::f32;

    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, jcp.oc_block);
        jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
    }

    auto src_format = (ndims == 5) ? nCdhw16c : nChw16c;
    auto wei_format = (ndims == 5)
        ? (with_groups) ? OhIw16o16i : OhIw16o16i
        : (with_groups) ? OhIw16o16i : OhIw16o16i;

    bool args_ok = true
        && jcp.oc % jcp.oc_block == 0
        && jcp.ic % jcp.ic_block == 0;
    if (!args_ok)
        return status::unimplemented;

    switch (diff_src_d.format()) {
        case NhC8nw16c: jcp.mb_block = 8; if (diff_dst_d.format() != NhC8nw16c) return status::unimplemented; break;
        case NhC16nw16c: jcp.mb_block = 16; if (diff_dst_d.format() != NhC16nw16c) return status::unimplemented; break;
        case NhC32nw16c: jcp.mb_block = 32; if (diff_dst_d.format() != NhC32nw16c) return status::unimplemented; break;
        case NhC64nw16c: jcp.mb_block = 64; if (diff_dst_d.format() != NhC64nw16c) return status::unimplemented; break;
        default: return status::unimplemented; break;
    }

    jcp.nb_ic = jcp.ic / jcp.ic_block;
    jcp.nb_oc = jcp.oc / jcp.oc_block;

    jcp.ur_w = jcp.stride_w;

    int regs = 28;
    if (jcp.iw <= regs)
        jcp.ur_w = jcp.iw;
    else {
        for (int ur_w = regs; ur_w > 0; --ur_w)
            if (ur_w % jcp.stride_w == 0) {
                jcp.ur_w = ur_w;
                break;
            }
    }
    int l_overflow = nstl::max(0, ((jcp.kw - 1) * (jcp.dilate_w + 1)
                    - jcp.l_pad) / jcp.stride_w);
    int r_overflow1 = nstl::max(0, ((jcp.kw - 1) * (jcp.dilate_w + 1)
                    - nstl::max(0, jcp.r_pad) - jcp.iw % jcp.ur_w) / jcp.stride_w);
    int n_oi = jcp.iw / jcp.ur_w;
    if (r_overflow1 > 0) n_oi--;

    if ((mayiuse(avx512_mic_4ops) || mayiuse(avx512_core_vnni))
           && jcp.stride_w == 1 && jcp.stride_h == 1
           && diff_dst_d.data_type() == data_type::s16
           && weights_d.data_type() == data_type::s16
           && diff_src_d.data_type() == data_type::s32) {
        if (weights_d.format() != (with_groups ? gOIhw8o16i2o : OIhw8o16i2o))
            return status::unimplemented;
        if (mayiuse(avx512_mic_4ops)) {
            jcp.ver = ver_4vnni;
        } else {
            jcp.ver = ver_vnni;
        }
        jcp.typesize_in = sizeof(int16_t);
        jcp.typesize_out = sizeof(int32_t);
    } else if (mayiuse(avx512_common)
         && diff_dst_d.data_type() == data_type::f32
         && weights_d.data_type() == data_type::f32
         && diff_src_d.data_type() == data_type::f32) {
        switch (weights_d.format()) {
            case hOIw16o16i:
            case OhIw16o16i: break;
            default: return status::unimplemented; break;
        }
        jcp.ver = ver_fma;
        jcp.typesize_in = sizeof(float);
        jcp.typesize_out = sizeof(float);
        if (mayiuse(avx512_mic_4ops)
            && jcp.stride_w == 1 && jcp.stride_h == 1 && jcp.stride_d == 1) {
                jcp.ver = ver_4fma;
            }
    } else {
        return status::unimplemented;
    }
    if (!utils::everyone_is(0, jcp.dilate_d, jcp.dilate_h, jcp.dilate_w)
            && jcp.ver != ver_fma)
        return status::unimplemented;

    jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;
    if (jcp.ver == ver_4vnni) {
        jcp.kernel_kind = embd_bcast;
    }
    if (jcp.ver == ver_vnni) {
        // TODO: kernel_kind and nb_oc_blocking selection
        //       should be tuned on real HW
        if ((jcp.iw <= 56 && jcp.ih <= 56 && jcp.kh < 5)
            || (jcp.iw <= 17 && jcp.ih <= 17 && jcp.kh >= 5) ) {
            jcp.kernel_kind = expl_bcast;
            jcp.nb_ic_blocking = 4;
        } else {
            jcp.kernel_kind = embd_bcast;
            jcp.nb_ic_blocking = 2;
        }
        if (jcp.nb_ic_blocking > 1) {
            if (jcp.nb_ic < jcp.nb_ic_blocking) jcp.nb_ic_blocking = jcp.nb_ic;
            if (jcp.nb_ic % jcp.nb_ic_blocking != 0)
                for (int i = jcp.nb_ic_blocking; i > 0; i--)
                    if (jcp.nb_ic % i == 0) {
                        jcp.nb_ic_blocking = i;
                        break;
                    }
            jcp.ur_w = 31 / (jcp.nb_ic_blocking + 1);
            if (jcp.iw < jcp.ur_w) jcp.ur_w = jcp.iw;
        }
    }
    if (jcp.ver == ver_4fma) {
        if (jcp.kw == 3 && jcp.kh == 3 && jcp.iw == 7 && jcp.ih == 7) {
            jcp.nb_ic_blocking = 2;
        } else {
            for (int i = jcp.nb_ic; i > 0; i--)
                if (i * jcp.ur_w <= regs && jcp.nb_ic % i == 0) {
                    jcp.nb_ic_blocking = i;
                    break;
                }
        }
    }

    jcp.loop_order = loop_gnc;

    bool large_code_size = (jcp.ur_w != jcp.ow)
         && ((l_overflow <= 0 && n_oi > 0) ||(l_overflow > 0 && n_oi > 1))
         && (r_overflow1 > 0) && (l_overflow > 0);
    if (large_code_size) {
        const int max_code_size = 24 * 1024;
        const int num_ops_per_reg = 6 + jcp.oc_block * jcp.kw;
        int mult = 1;
        if (l_overflow > 0) mult += 1;
        if (r_overflow1 > 0) mult += 1;
        for (int ur_w = jcp.ur_w; ur_w > regs/2; --ur_w) {
            if ((ur_w / jcp.stride_w) * mult * num_ops_per_reg * 9.2
                    < max_code_size) {
                if (ur_w % jcp.stride_w == 0) {
                    jcp.ur_w = ur_w;
                    break;
                }
            }
        }
    }

    if (jcp.ver == ver_fma && mayiuse(avx512_core)) {
        int try_nb_ic_blocking = 2;
        unsigned int ker_inp_size = typesize * jcp.iw * jcp.ic_block
            * try_nb_ic_blocking * jcp.kh;
        unsigned int ker_out_size = typesize * jcp.ow * jcp.oc_block;
        unsigned int ker_wei_size = typesize * jcp.kh * jcp.kw * jcp.ic_block
            * jcp.oc_block * try_nb_ic_blocking;
        unsigned int ker_total_size = ker_inp_size + ker_out_size
            + ker_wei_size;
        if (!(jcp.kw == 1 || (jcp.kw == 5 && jcp.iw < 8)
            || (jcp.kw < 5 && ((jcp.iw <= 5 || (jcp.iw > 8 && jcp.iw <= 13))
            || ker_total_size > L1_cache_size )))
                || jcp.stride_h > 1 || jcp.stride_d > 1) {
            jcp.kernel_kind = embd_bcast;
            jcp.ur_w = nstl::min(jcp.iw, regs);
            jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;
            if (!(jcp.kw > 3 || (jcp.kw == 3 && ker_total_size < L1_cache_size
                && jcp.ow > 8)) && jcp.stride_h == 1)
                if (jcp.nb_ic % try_nb_ic_blocking == 0) {
                    jcp.nb_ic_blocking = try_nb_ic_blocking;
                    jcp.ur_w = 31 / (jcp.nb_ic_blocking + 1);
                    if (jcp.iw < jcp.ur_w) jcp.ur_w = jcp.iw;
                }
         } else {
            jcp.kernel_kind = expl_bcast;
            jcp.nb_oc_blocking = 1;
            jcp.nb_ic_blocking = 4;
            if (jcp.nb_ic < jcp.nb_ic_blocking) jcp.nb_ic_blocking = jcp.nb_ic;
            if (jcp.nb_ic % jcp.nb_ic_blocking != 0)
                for (int i = jcp.nb_ic_blocking; i > 0; i--)
                    if (jcp.nb_ic % i == 0) {
                        jcp.nb_ic_blocking = i;
                        break;
                    }
            jcp.ur_w = 31 / (jcp.nb_ic_blocking + 1);
            if (jcp.iw < jcp.ur_w) jcp.ur_w = jcp.iw;
        }
    }
    jcp.ur_w_tail = jcp.iw % jcp.ur_w;

    if (l_overflow * jcp.stride_w > jcp.ur_w)
        return status::unimplemented;
    int r_overflow_no_tail = nstl::max(0, ((jcp.kw - 1) * (jcp.dilate_w + 1)
                    - nstl::max(0, jcp.r_pad) - jcp.ur_w_tail) / jcp.stride_w);
    if (r_overflow_no_tail * jcp.stride_w > jcp.ur_w)
        return status::unimplemented;
    if ((jcp.iw > jcp.ur_w) && (jcp.ur_w % jcp.stride_w != 0))
        return status::unimplemented;

    pick_loop_order(jcp);

    jcp.nb_oc_L2 = jcp.nb_oc;
    // TODO check for 4vnni
    if (jcp.ver == ver_4fma && (jcp.kh < 5 && jcp.kw < 5)) {
        for (int divf = 2, temp_nb = jcp.nb_oc_L2; divf <= jcp.nb_oc;
              divf++) {
            size_t l2_src = jcp.iw * jcp.ic_block * jcp.nb_ic_blocking * jcp.ih
                * jcp.id;
            size_t l2_dst = jcp.ow * jcp.oc_block * temp_nb * jcp.oh * jcp.od;
            size_t l2_filt = jcp.kw * jcp.oc_block * jcp.ic_block * jcp.kh
                * jcp.kd * jcp.nb_ic_blocking * temp_nb;
            if (4 * (l2_src + l2_dst + l2_filt) > KNx_L2_EFFECTIVE_CAPACITY) {
                if (jcp.kh == 3 && jcp.ih == 7) {
                    jcp.nb_oc_L2 = 1;
                    break;
                }
                temp_nb = (jcp.nb_oc_L2 % divf == 0 ? jcp.nb_oc_L2 / divf
                                : jcp.nb_oc_L2);
            } else {
                jcp.nb_oc_L2 = temp_nb;
                break;
            }
        }
    }

    regs = 30;
    int ur_sparse;
    if (jcp.kw * jcp.nb_ic <= regs)
        ur_sparse = jcp.kw * jcp.nb_ic;
    else {
        for (int tmp_ur_w = regs; tmp_ur_w > 0; tmp_ur_w--)
            if (tmp_ur_w % jcp.kw == 0) {
                ur_sparse = tmp_ur_w;
                break;
            }
    }

    jcp.ic_buffs = ur_sparse / jcp.kw;
    for (int i = jcp.ic_buffs; i > 0; i--) {
        if (jcp.nb_ic % i == 0) {
            jcp.ic_buffs = i;
            break;
        }
    }


    // higher than 8 will cause subpar memory performance
    if (jcp.ic_buffs > 8) jcp.ic_buffs = 8;

    jcp.ur_sparse = jcp.ic_buffs * jcp.kw;
    jcp.nb_mb = jcp.mb / jcp.mb_block;

    args_ok = true
        && jcp.ic <= diff_src_d.blocking_desc().padding_dims[1]
        && jcp.oc <= diff_dst_d.blocking_desc().padding_dims[1]
        && jcp.ic <= weights_d.blocking_desc().padding_dims[with_groups + 1]
        && jcp.oc <= weights_d.blocking_desc().padding_dims[with_groups + 0];

    return args_ok ? status::success : status::unimplemented;
}

const int jit_avx512_common_conv_bwd_weights_kernel_f32::max_ur_w = 28;


void jit_avx512_common_conv_bwd_weights_kernel_f32::bias_kernel()
{
    /*Label skip_bias, bias_loop, skip_load_bias;

    mov(reg_tmp, ptr[param + GET_OFF(flags)]);
    test(reg_tmp,reg_tmp);
    jne(skip_bias, T_NEAR);

    mov(reg_bias, ptr[param + GET_OFF(bias)]);
    mov(reg_output, ptr[param + GET_OFF(dst)]);
    vpxord(Zmm(1), Zmm(1), Zmm(1));

    mov(reg_tmp, ptr[param + GET_OFF(channel)]);
    cmp(reg_tmp, 0);
    jne(skip_load_bias, T_NEAR);
    vmovups(Zmm(1), ptr[reg_bias]);

    L(skip_load_bias);

    mov(reg_oi, ptr[param + GET_OFF(kh_padding)]);
    mov(reg_tmp, jcp.oc_block * jcp.ow * jcp.oh * jcp.typesize_out);
    imul(reg_oi, reg_tmp);

    xor_(reg_tmp, reg_tmp);
    L(bias_loop); {
        vmovups(Zmm(0), ptr[reg_output + reg_tmp]);
        vaddps(Zmm(1), Zmm(1), Zmm(0));
        add(reg_tmp, jcp.oc_block * jcp.typesize_out);
        cmp(reg_tmp, reg_oi);
        jl(bias_loop);
    }
    vmovups(EVEX_compress_addr(reg_bias,0), Zmm(1));

    L(skip_bias);*/
}

void jit_avx512_common_conv_bwd_weights_kernel_f32::compute_loop_fma_sparse() {

    Label kh_label, kd_label, skip_kd_loop;
    Label end_label, clear_label;

    /**********************************************

    reg64_t param = abi_param1;
    reg64_t reg_inp = r8;
    reg64_t reg_ker = r9;
    reg64_t reg_out = r10;

    reg64_t reg_inp_prf = r11;
    reg64_t reg_ker_prf = r12;
    reg64_t reg_out_prf = r13;

    reg64_t aux_reg_inp = r14;
    reg64_t aux_reg_ker = r15;

    reg64_t aux_reg_inp_prf = rsi;
    reg64_t aux_reg_ker_prf = rdx;

    reg64_t reg_channel = rsi;
    reg64_t reg_bias = rdx;

    reg64_t aux_reg_ker_d = r9;
    reg64_t aux_reg_inp_d = rbx;
    reg64_t aux_reg_inp_d_prf = r13;
    reg64_t aux_reg_ker_d_prf = abi_not_param1;
    reg64_t reg_ki = r10;

    reg64_t reg_kj = rax;
    reg64_t reg_relu_ns = rax;
    reg64_t reg_oi = rbx;
    reg64_t reg_kh = abi_not_param1;

    reg64_t reg_tmp = rbp;

    reg64_t reg_ic_loop = rdx;
    reg64_t reg_inp_loop = rsi;

    reg64_t reg_init_flag = r13;
    reg64_t reg_bias_ptr = param;

    reg64_t aux_reg_ic = r12;
    reg64_t reg_binp = rax;
    reg64_t reg_bout = r11;
    reg64_t aux1_reg_inp = rbx;
    reg64_t aux_reg_out = abi_not_param1;

    reg64_t reg_long_offt = r11;
    reg64_t reg_out_long_offt = r14;

    ***********************************************/

    int kw = jcp.kw;
    int kh = jcp.kh;

    int ow = jcp.ow;
    int oh = jcp.oh;

    int nb_ic = jcp.nb_ic;
    int nb_oc = jcp.nb_oc;

    int iw = jcp.iw;
    int ih = jcp.ih;

    int mb = jcp.mb;
    int mb_block = jcp.mb_block;

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int l_pad = jcp.l_pad;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;
    int stride_h = jcp.stride_h;

    int nr = jcp.ur_sparse;
    int oc_buffs = jcp.oc_buffs;

    assert(nr >= kw);
    assert(oc_block == 16); // needed for specific optimization
    assert(typesize == 4);

    int oc_iters = nb_oc / oc_buffs;

    auto zmm_o = Xbyak::Zmm(31);

    auto zmm_zero = Xbyak::Zmm(30);

    auto zmm_gather = Xbyak::Zmm(0);

    auto zmm_cmp = Xbyak::Zmm(28);

    assert(nr % kw == 0);
    
    const int vsize = 16;

    /*Label no_init_label;

    mov(reg_channel, ptr[param + GET_OFF(channel)]);
    cmp(reg_channel, 0);
    jne(no_init_label, T_NEAR);

    if (oc_buffs * ow > 128 || jcp.with_bias) { // threshold may be tweaked later

        Reg64 aux_reg_out = aux_reg_inp;
        Reg64 aux_reg_out_prf = aux_reg_ker;

        if (jcp.with_bias) {
            mov(reg_bias, ptr[param1 + GET_OFF(bias)]);
        }

        mov(aux_reg_out, reg_out);
        mov(aux_reg_out_prf, reg_out_prf);

        mov(reg_channel, oc_buffs);

        Label oc_loop_label;
        L(oc_loop_label);

        if (jcp.with_bias) {
            vmovups(zmm_zero, ptr[reg_bias]);
            add(reg_bias, typesize * oc_block);
        }

        for (int oi = 0; oi < ow; oi++) {
            vmovups(EVEX_compress_addr_safe(aux_reg_out, oi * oc_block * typesize,
                        reg_long_offt), zmm_zero);
            prefetcht1(EVEX_compress_addr_safe(aux_reg_out_prf, oi * oc_block * typesize,
                        reg_long_offt));
        }

        add(aux_reg_out, typesize * oc_block * mb_block * ow);
        add(aux_reg_out_prf, typesize * oc_block * mb_block * ow);

        dec(reg_channel);
        cmp(reg_channel, 0);
        jne(oc_loop_label);

        if (jcp.with_bias) {
            vpxord(zmm_zero, zmm_zero, zmm_zero);
        }

    } else {

        for (int oc = 0; oc < oc_buffs; oc++) {
            for (int oi = 0; oi < ow; oi++) {
                vmovups(EVEX_compress_addr_safe(reg_out, (oc * oc_block * ow 
                            + oi * oc_block) * typesize,
                            reg_long_offt), zmm_zero);
                prefetcht1(EVEX_compress_addr_safe(reg_out_prf, (oc * oc_block * ow 
                            + oi * oc_block) * typesize,
                            reg_long_offt));
            }
        }
    }

    L(no_init_label);*/

    auto get_reg_idx = [=](int ki, int oc_buff) {
        
        return oc_buff * kw + ki + 1;

    };

    Reg64 reg_long_offt = reg_kj;
    Reg64 reg_mul_src = rdx;

    int disp = oc_block * oc_buffs * typesize;
    int disp_shift = log2(disp);
    bool use_shift = ceil(log2(disp)) == floor(log2(disp));

    cout << "use_shift:" << use_shift << " disp_shift:" << disp_shift << endl;

    auto comp_unrolled = [&](int ii, int cur_oc_buffs) {

        Reg64 mask_reg = reg_oi;
        Reg32 ic_itr_reg = reg_out_prf.cvt32();
        Reg64 lzcnt_reg = aux_reg_ker;

        kmovw(mask_reg.cvt32(), k7);
        popcnt(ic_itr_reg, mask_reg.cvt32());

        if (ii < iw - 1) { // pipelined
            size_t aux_src_offset = typesize * (ii + 1) * mb_block;
            vcmpps(k7, zmm_zero, EVEX_compress_addr_safe(reg_inp, aux_src_offset,
                                    reg_long_offt), 4);

            prefetcht1(EVEX_compress_addr_safe(reg_inp_prf, aux_src_offset,
                                    reg_long_offt));

            /*for (int oc_buff = 0; oc_buff < cur_oc_buffs; oc_buff++) {
                for (int ki = kw - 1; ki > -1; ki--) {

                    int n = (ii + 1) - ki * dilate_w + l_pad;

                    if (n >= 0 && n / stride_w < ow && n % stride_w == 0) {

                        int oi = n / stride_w;

                        size_t aux_dst_offset = (size_t)typesize
                                * (oc_buff * oc_block * ow * mb_block
                                + oi * oc_block + ic_block * ow);

                        prefetcht0(EVEX_compress_addr_safe(reg_out, aux_dst_offset,
                                reg_long_offt));
                    }

                }
            }*/
        }

        Label ic_loop_end_label;
        jz(ic_loop_end_label, T_NEAR);

        tzcnt(lzcnt_reg.cvt32(), mask_reg.cvt32());
        inc(lzcnt_reg.cvt32());

        if (!use_shift) {
            mulx(reg_kj, reg_ker_prf, lzcnt_reg);
        }
        shrx(mask_reg.cvt32(), mask_reg.cvt32(), lzcnt_reg.cvt32());

        mov(aux_reg_out, reg_out);
        mov(aux_reg_inp, reg_inp);

        Label ic_loop_label;
        L(ic_loop_label); {

            lea(aux_reg_inp, ptr[aux_reg_inp + lzcnt_reg * typesize]);

            int aux_src_offset = typesize * (ii * mb_block - 1);
            vbroadcastss(zmm_o, ptr[aux_reg_inp + aux_src_offset]);

            if (use_shift) {
                shl(lzcnt_reg.cvt32(), disp_shift);
                add(aux_reg_out, lzcnt_reg);
            } else {
                add(aux_reg_out, reg_ker_prf);
            }

            tzcnt(lzcnt_reg.cvt32(), mask_reg.cvt32()); // pipelined
            inc(lzcnt_reg.cvt32());

            dec(ic_itr_reg);

            if (!use_shift) {
                mulx(reg_kj, reg_ker_prf, lzcnt_reg);
            }
            shrx(mask_reg.cvt32(), mask_reg.cvt32(), lzcnt_reg.cvt32()); // does not change flags

            cout << "op:";

            for (int ki = kw - 1; ki > -1; ki--) {
                for (int oc_buff = 0; oc_buff < cur_oc_buffs; oc_buff++) {

                    int n = ii - ki * dilate_w + l_pad;

                    if (n >= 0 && n / stride_w < ow && n % stride_w == 0) {

                        int oi = n / stride_w;

                        int reg_idx = get_reg_idx(ki, oc_buff);

                        Zmm zmm = Xbyak::Zmm(reg_idx);

                        if (oc_buff == 0) {
                            cout << " " << oi << "-r" << reg_idx;
                        }

                        size_t aux_dst_offset = (size_t)typesize
                                * (oc_buff * oc_block
                                + oi * oc_block * cur_oc_buffs * mb_block);
                    
                        vfmadd231ps(zmm, zmm_o,
                                EVEX_compress_addr_safe(aux_reg_out, aux_dst_offset,
                                    reg_long_offt));
                    }

                }
            }

            cout << endl;

            jnz(ic_loop_label, T_NEAR);
        }

        L(ic_loop_end_label);

    };

    auto outer_loop = [&](int cur_oc_buffs) {

        vpxord(zmm_zero, zmm_zero, zmm_zero);

        vcmpps(k7, zmm_zero, ptr[reg_inp], 4);
        prefetcht1(ptr[reg_inp_prf]);

        if (!use_shift) {
            mov(reg_mul_src, disp);
        }
        sub(reg_out, disp);

        for (int oc_buff = 0; oc_buff < oc_buffs; oc_buff++) {
            for (int ki = 0; ki < kw; ki++) {
                int idx = get_reg_idx(ki, oc_buff);
                auto zmm = Xbyak::Zmm(idx);
                vpxord(zmm, zmm, zmm);

                size_t kernel_offset = typesize * (oc_buff
                        * kw * oc_block * ic_block
                        + ki * oc_block * ic_block);
                prefetcht1(EVEX_compress_addr_safe(reg_ker, kernel_offset,
                                        reg_long_offt));

            }
        }

        for (int ii = 0; ii < iw; ii++) {
            comp_unrolled(ii, cur_oc_buffs);
        }
    };

    outer_loop(oc_buffs);

    Label no_load_label;

    cmp(reg_channel, 0);
    je(no_load_label, T_NEAR);

    for (int oc_buff = 0; oc_buff < oc_buffs; oc_buff++) {
        for (int ki = 0; ki < kw; ki++) {
            int idx = get_reg_idx(ki, oc_buff);
            auto zmm = Xbyak::Zmm(idx);

            size_t kernel_offset = typesize * (oc_buff
                    * kw * oc_block * ic_block
                    + ki * oc_block * ic_block);
            vaddps(zmm, zmm, EVEX_compress_addr_safe(reg_ker, kernel_offset,
                                    reg_long_offt));

        }
    }

    L(no_load_label);

    for (int oc_buff = 0; oc_buff < oc_buffs; oc_buff++) {
        for (int ki = 0; ki < kw; ki++) {
            int idx = get_reg_idx(ki, oc_buff);
            auto zmm = Xbyak::Zmm(idx);

            size_t kernel_offset = typesize * (oc_buff
                    * kw * oc_block * ic_block
                    + ki * oc_block * ic_block);
            vmovups(EVEX_compress_addr_safe(reg_ker, kernel_offset,
                                    reg_long_offt), zmm);
            //prefetcht1(EVEX_compress_addr_safe(reg_ker_prf, kernel_offset,
                                    //reg_long_offt));

        }
    }

}

void jit_avx512_common_conv_bwd_weights_kernel_f32::generate()
{
    preamble();

    mov(reg_inp, ptr[param1 + GET_OFF(src)]);
    mov(reg_out, ptr[param1 + GET_OFF(dst)]);
    mov(reg_ker, ptr[param1 + GET_OFF(filt)]);

    mov(reg_inp_prf, ptr[param1 + GET_OFF(src_prf)]);
    mov(reg_channel, ptr[param + GET_OFF(channel)]);
    //mov(reg_out_prf, ptr[param1 + GET_OFF(dst_prf)]);
    //mov(reg_ker_prf, ptr[param + GET_OFF(filt_prf)]);

    compute_loop_fma_sparse();

    postamble();
}

status_t jit_avx512_common_conv_bwd_weights_kernel_f32::init_conf(
    jit_conv_conf_t &jcp, const convolution_desc_t &cd,
    cpu_memory_t::pd_t &src_pd, cpu_memory_t::pd_t &diff_weights_pd,
    cpu_memory_t::pd_t &diff_bias_pd, cpu_memory_t::pd_t &diff_dst_pd)
{
    if (!mayiuse(avx512_common))
        return status::unimplemented;

    const int simd_w = cpu_isa_traits<avx512_common>::vlen / sizeof(float);

    const memory_desc_wrapper src_d(&src_pd);
    const memory_desc_wrapper diff_weights_d(&diff_weights_pd);
    const memory_desc_wrapper diff_bias_d(&diff_bias_pd);
    const memory_desc_wrapper diff_dst_d(&diff_dst_pd);

    const bool with_groups = diff_weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();

    jcp = zero<decltype(jcp)>();
    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;

    jcp.ngroups = with_groups ? diff_weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];

    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;

    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.ih = src_d.dims()[ndims-2];
    jcp.iw = src_d.dims()[ndims-1];
    jcp.od = (ndims == 5) ? diff_dst_d.dims()[2] : 1;
    jcp.oh = diff_dst_d.dims()[ndims-2];
    jcp.ow = diff_dst_d.dims()[ndims-1];

    jcp.kd = (ndims == 5) ? diff_weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = diff_weights_d.dims()[with_groups + ndims-2];
    jcp.kw = diff_weights_d.dims()[with_groups + ndims-1];

    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = cd.padding[0][ndims-4];
    jcp.l_pad = cd.padding[0][ndims-3];

    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = cd.strides[ndims-4];
    jcp.stride_w = cd.strides[ndims-3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = cd.dilates[ndims-4];
    jcp.dilate_w = cd.dilates[ndims-3];

    const int kh_range = 1 + (jcp.kh - 1) * (jcp.dilate_h + 1);
    bool ok = true
        // general condition to simplify dilations
        && implication(jcp.dilate_d != 0, jcp.stride_d == 1)
        && implication(jcp.dilate_h != 0, jcp.stride_h == 1)
        && implication(jcp.dilate_w != 0, jcp.stride_w == 1)
        // special condition to simplify dilations in compute_oh_loop_common
        && implication(jcp.dilate_h != 0, kh_range <= jcp.ih);
    if (!ok)
        return status::unimplemented;

    jcp.r_pad = nstl::max(0, (jcp.ow - 1) * jcp.stride_w
            + (jcp.kw - 1) * (jcp.dilate_w + 1) - (jcp.iw + jcp.l_pad - 1));
    jcp.b_pad = nstl::max(0, (jcp.oh - 1) * jcp.stride_h
            + (jcp.kh - 1) * (jcp.dilate_h + 1) - (jcp.ih + jcp.t_pad - 1));
    jcp.back_pad = nstl::max(0, (jcp.od - 1) * jcp.stride_d
            + (jcp.kd - 1) * (jcp.dilate_d + 1) - (jcp.id + jcp.f_pad - 1));

    if ( ndims == 5 )
        if (jcp.f_pad != 0 || jcp.back_pad != 0)
            return status::unimplemented;

    jcp.ihp = jcp.ih + jcp.t_pad + jcp.b_pad;
    jcp.iwp = jcp.iw + jcp.l_pad + jcp.r_pad;
    jcp.ohp = jcp.oh;
    jcp.owp = jcp.ow;
    jcp.aligned_threads = 0;

    /* check for the 1st convolution */
    jcp.is_1stconv = is_1stconv(jcp);

    jcp.oc_block = simd_w;

    bool ok_to_pad_channels = true
        && jcp.ngroups == 1
        && src_d.data_type() == data_type::f32;

    if (ok_to_pad_channels)
        jcp.oc = rnd_up(jcp.oc, simd_w);

    if (jcp.oc % jcp.oc_block)
        return status::unimplemented;

    auto src_format = (ndims == 5) ? nCdhw16c : Nhcw16n;
    auto dst_format = (ndims == 5) ? nCdhw16c : NhCw16n128c;
    auto wei_format = (ndims == 5)
        ? (with_groups) ? hIOw16i16o : hIOw16i16o
        : (with_groups) ? hIOw16i16o : hIOw16i16o;

    /* conditions on bias memory */
    jcp.with_bias = cd.diff_bias_desc.format != memory_format::undef;
    if (jcp.with_bias) {
        if (diff_bias_d.format() == any)
            CHECK(diff_bias_pd.set_format(x));
        if (diff_bias_d.format() != x)
            return status::unimplemented;
    }

    jcp.nb_oc = jcp.oc / jcp.oc_block;

    if (src_d.format() == any)
        CHECK(src_pd.set_format(src_format));
    if (diff_dst_d.format() == any)
        CHECK(diff_dst_pd.set_format(dst_format));

    switch (src_d.format()) {
        case Nhcw16n:
            jcp.mb_block = 16;
            if (diff_dst_d.format() != NhCw16n128c && diff_dst_d.format() != NhCw16n64c) {
                return status::unimplemented;
            }
            break;
        default: return status::unimplemented; break;
    }

    /* kernel applicability check wrt boundaries
     * the conditions are quite general across the kernels we have,
     * but ideally the check should belong to a specific kernel... */
    const int max_pad = ((jcp.kh - 1) * (jcp.dilate_h + 1) + 1) / 2;
    const bool boundaries_ok = true
        && jcp.t_pad <= max_pad
        && jcp.b_pad <= max_pad;
    if (!boundaries_ok)
        return status::unimplemented;

    /* yet another common check */
    if (jcp.kw > 14)
        return status::unimplemented;

    /* setting register strategy */
    for (int ur_w = nstl::min(max_ur_w, jcp.ow); ur_w > 0; --ur_w) {
        if (jcp.ow % ur_w == 0) { jcp.ur_w = ur_w; break; }
    }

    if (jcp.is_1stconv) {
        const auto want_src_format = (ndims == 5) ? ncdhw : nchw;
        if (src_d.format() == any)
            CHECK(src_pd.set_format(want_src_format));

        const bool src_ok = true
            && utils::everyone_is(data_type::f32,
                src_d.data_type(), diff_weights_d.data_type(),
                diff_dst_d.data_type())
            && one_of(jcp.ic, 1, 3)
            && implication(jcp.ic == 1, one_of(src_d.format(), want_src_format,
                (ndims == 5) ? ndhwc : nhwc))
            && implication(jcp.ic != 1, src_d.format() == want_src_format)
            && jcp.ngroups == 1;
        if (!src_ok)
            return status::unimplemented;

        const int tr_ld = rnd_up(div_up(jcp.iw + jcp.l_pad + jcp.r_pad,
                    jcp.stride_w), 16);
        const int kh_step = nstl::max((28 - jcp.with_bias) / jcp.kw, 1);
        const int kh_step_rem = jcp.kh % kh_step;
        const auto want_4fma_wfmt = (ndims == 5)
            ? with_groups ? gOidhw16o : Oidhw16o
            : with_groups ? gOihw16o : Oihw16o;
        const bool use_4fma = true
            && ndims == 4
            && mayiuse(avx512_mic_4ops)
            && mkldnn_thr_syncable()
            && everyone_is(0, jcp.dilate_d, jcp.dilate_h, jcp.dilate_w)
            && everyone_is(0, jcp.l_pad, jcp.r_pad, jcp.t_pad, jcp.b_pad)
            && jcp.kw <= 28 - jcp.with_bias
            && jcp.stride_w == 4
            && tr_ld / simd_w <= 4 /* [bwd_w:tr_src:r1] */
            && implication(jcp.with_bias, kh_step_rem == 1) /* [bwd_w:b:r1] */
            && implication(diff_weights_d.format() != any,
                    diff_weights_d.format() == want_4fma_wfmt);

        if (use_4fma) {
            jcp.ver = ver_4fma;
            jcp.kh_step = kh_step;
            jcp.tr_ld = tr_ld;
            jcp.ic_block = 1;
            if (diff_weights_d.format() == any)
                CHECK(diff_weights_pd.set_format(want_4fma_wfmt));
        } else {
            jcp.ver = ver_fma;
            jcp.ic_block = jcp.ic;

            const auto want_wfmt = (ndims == 5)
                ? with_groups ? gOdhwi16o : Odhwi16o
                : with_groups ? gOhwi16o : Ohwi16o;
            if (diff_weights_d.format() == any)
                CHECK(diff_weights_pd.set_format(want_wfmt));
            if (diff_weights_d.format() != want_wfmt)
                return status::unimplemented;
        }

        jcp.nb_ic = jcp.ic / jcp.ic_block;
        jcp.src_fmt = src_d.format();
    } else {

        if (diff_weights_d.format() == any)
            CHECK(diff_weights_pd.set_format(wei_format));
        switch (diff_weights_d.format()) {
            case hIOw16i16o:
            case IhOw16i16o: break;
            default: return status::unimplemented; break;
        }

        jcp.ic_block = simd_w;
        if (ok_to_pad_channels)
            jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
        jcp.nb_ic = jcp.ic / jcp.ic_block;
        jcp.src_fmt = src_d.format();
        if ((mayiuse(avx512_mic_4ops) || mayiuse(avx512_core_vnni))
            && mkldnn_thr_syncable()
            && ndims == 4
            && jcp.stride_w == 1
            && everyone_is(0, jcp.dilate_d, jcp.dilate_h, jcp.dilate_w)
            && ((src_d.data_type() == data_type::s16
            && diff_weights_d.data_type() == data_type::s32
            && diff_dst_d.data_type() == data_type::s16))) {
            if (mayiuse(avx512_core_vnni)) jcp.ver = ver_vnni;
            else jcp.ver = ver_4vnni;
        } else if ((mayiuse(avx512_mic) || mayiuse(avx512_core))
                && utils::everyone_is(data_type::f32,
                    src_d.data_type(), diff_weights_d.data_type(),
                    diff_dst_d.data_type())) {
            jcp.ver = ver_fma;
            if (ndims == 4 && mayiuse(avx512_mic_4ops) && jcp.stride_w == 1 &&
                    everyone_is(0, jcp.dilate_d, jcp.dilate_h, jcp.dilate_w) &&
                    mkldnn_thr_syncable()) {
                jcp.ver = ver_4fma;
            }
        } else {
            return status::unimplemented;
        }
        if (utils::one_of(jcp.ver, ver_4fma, ver_4vnni, ver_vnni)) {
            jcp.ur_w = jcp.ow;
            // XXX, BUGBUGBUG, but not a FIXME: this assumes that it's OK to
            // cross the right boundary. The only requirement is not to have
            // NaNs there because another multiplicand is always guaranteed to
            // be zero. This also may require the top-level driver to allocate
            // four extra guarding elements at the very end of the buffer.
            // I'm not proud of this hack, but it improves performance by
            // about 5-10% depending on the dimensions (Roma)

            // for vnni, that's results of performance tuning
            const int tr_round = (utils::one_of(jcp.ver, ver_4fma, ver_vnni))
                ? 4 : 8;

            jcp.tr_iw = rnd_up(jcp.iw + jcp.kw - 1, tr_round);
            jcp.tr_src_num_guard_elems = tr_round; // upper bound

            if (utils::one_of(jcp.ver, ver_4vnni, ver_vnni)) {
                jcp.tr_ow = rnd_up(jcp.ow, 2);
                jcp.ur_w = jcp.tr_ow;
            }
        }
    }

    if (utils::one_of(jcp.ver, ver_4vnni, ver_vnni)) {
        jcp.typesize_in = sizeof(int16_t);
        jcp.typesize_out = sizeof(int32_t);
    } else if (utils::one_of(jcp.ver, ver_4fma, ver_fma)) {
        jcp.typesize_in = sizeof(float);
        jcp.typesize_out = sizeof(float);
    } else
        return status::unimplemented;


    int nregs = 30;
    int ur_sparse;
    if (jcp.kw * jcp.nb_oc <= nregs)
        ur_sparse = jcp.kw * jcp.nb_oc;
    else {
        for (int tmp_ur_w = nregs; tmp_ur_w > 0; tmp_ur_w--)
            if (tmp_ur_w % jcp.kw == 0) {
                ur_sparse = tmp_ur_w;
                break;
            }
    }

    jcp.oc_buffs = ur_sparse / jcp.kw;
    for (int i = jcp.oc_buffs; i > 0; i--) {
        if (jcp.nb_oc % i == 0) {
            jcp.oc_buffs = i;
            break;
        }
    }

    //jcp.oc_buffs = 1;

    jcp.ur_sparse = jcp.oc_buffs * jcp.kw;
    jcp.nb_mb = jcp.mb / jcp.mb_block;


    bool args_ok = true
        && jcp.ic % jcp.ic_block == 0
        && jcp.oc % jcp.oc_block == 0
        && jcp.ic <= src_d.blocking_desc().padding_dims[1]
        && jcp.oc <= diff_dst_d.blocking_desc().padding_dims[1]
        && jcp.ic <= diff_weights_d.blocking_desc().padding_dims[with_groups + 1]
        && jcp.oc <= diff_weights_d.blocking_desc().padding_dims[with_groups + 0];


    return args_ok ? status::success : status::unimplemented;
}

}
}
}

