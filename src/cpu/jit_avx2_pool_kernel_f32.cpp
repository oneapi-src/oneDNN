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
#include "utils.hpp"

#include "jit_avx2_pool_kernel_f32.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace Xbyak;

#define ymm_k_offset Ymm(15)
#define ymm_one Ymm(14)
#define xmm_one Xmm(14)
#define ymm_tmp Ymm(13)
#define xmm_tmp Xmm(13)
#define ymm_input Ymm(12)

status_t jit_avx2_pool_kernel_f32::init_conf(jit_pool_conf_t &jpp,
            const pooling_desc_t &pd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &dst_d) {
    if (!mayiuse(avx2)) return status::unimplemented;

    bool args_ok = true
        && utils::one_of(pd.alg_kind, alg_kind::pooling_max,
                alg_kind::pooling_avg)
        && src_d.format() == memory_format::nChw8c
        && dst_d.format() == src_d.format()
        && pd.kernel[0] == pd.kernel[1]
        && pd.padding[0][0] == pd.padding[1][0] /* top = bottom */
        && pd.padding[0][1] == pd.padding[1][1] /* left = right */;
    if (!args_ok) return status::unimplemented;

    jpp.mb = src_d.dims()[0];
    jpp.c = src_d.dims()[1];
    jpp.ih = src_d.dims()[2];
    jpp.iw = src_d.dims()[3];
    jpp.oh = dst_d.dims()[2];
    jpp.ow = dst_d.dims()[3];

    jpp.stride_h = pd.strides[0];
    jpp.stride_w = pd.strides[1];
    jpp.kh = pd.kernel[0];
    jpp.kw = pd.kernel[1];

    jpp.t_pad = pd.padding[0][0];
    jpp.l_pad = pd.padding[0][1];

    jpp.is_max = pd.alg_kind == alg_kind::pooling_max;
    jpp.is_training = pd.prop_kind == prop_kind::forward_training;
    jpp.is_backward = pd.prop_kind == prop_kind::backward_data;

    jpp.c_block = simd_w;
    jpp.nb_c = jpp.c / jpp.c_block;
    if (jpp.is_max) {
        jpp.ur_w = 4;
        if (jpp.is_training)
            jpp.ur_w = 3;
        else if (jpp.is_backward)
            jpp.ur_w = 3;
    } else {
        if (jpp.is_backward)
            jpp.ur_w = 6;
        else
            jpp.ur_w = 12;
    }
    if (jpp.ow < jpp.ur_w) jpp.ur_w = jpp.ow;
    jpp.ur_w_tail = jpp.ow % jpp.ur_w;

    return status::success;
}

inline void jit_avx2_pool_kernel_f32::avg_oh_step(int ur_w, int pad_l,
        int pad_r, const char* kh_label) {
    using Xbyak::Ymm;
    using Xbyak::Xmm;

    int iw = jpp.iw;
    int kw = jpp.kw;
    int kh = jpp.kh;
    int stride_w = jpp.stride_w;
    int c_block = jpp.c_block;

    mov(tmp_gpr, float2int(kw * kh));
    movq(xmm_tmp, tmp_gpr);
    vbroadcastss(ymm_tmp, xmm_tmp);

    for (int jj = 0; jj < ur_w; jj++) {
        if (jpp.is_backward) {
            vmovups(Ymm(jj), ptr[reg_output + sizeof(float)*jj*c_block]);
            vdivps(Ymm(jj), Ymm(jj), ymm_tmp);
        } else {
            vpxor(Ymm(jj), Ymm(jj), Ymm(jj));
        }
    }

    mov(aux_reg_input, reg_input);
    xor_(kj, kj);
    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = nstl::max(0, pad_l - ki);
            int jj_end = ur_w - nstl::max(0, ki + pad_r - (kw-1));
            for (int jj = jj_start; jj  < jj_end; jj++) {
                int aux_input_offset = (ki+jj*stride_w-pad_l)* c_block;
                if (aux_input_offset > iw * c_block)
                    continue;
                int input_offset = sizeof(float)*aux_input_offset;
                if (jpp.is_backward) {
                    vmovups(Ymm(ur_w+jj), ptr[aux_reg_input + input_offset]);
                    vaddps(Ymm(ur_w+jj), Ymm(ur_w+jj), Ymm(jj));
                    vmovups(yword[aux_reg_input + input_offset], Ymm(ur_w+jj));
                } else {
                    vaddps(Ymm(jj), Ymm(jj), ptr[aux_reg_input + input_offset]);
                }
            }
        }
        add(aux_reg_input,  sizeof(float) * iw * c_block);
        inc(kj);
        cmp(kj, reg_kh);
        jl(kh_label, T_NEAR);
    }

    if (!jpp.is_backward) {
        for (int jj = 0; jj < ur_w; jj++) {
            vdivps(Ymm(jj), Ymm(jj), ymm_tmp);
            vmovups(yword[reg_output + sizeof(float)*jj*c_block], Ymm(jj));
        }
    }
}

inline void jit_avx2_pool_kernel_f32::max_oh_step_fwd(int ur_w, int pad_l,
        int pad_r, const char *kh_label) {
    unsigned char _cmp_lt_os = 1;

    int iw = jpp.iw;
    int kw = jpp.kw;
    int stride_w = jpp.stride_w;
    int c_block = jpp.c_block;


    mov(tmp_gpr, float2int(-FLT_MAX));
    movq(xmm_tmp, tmp_gpr);
    vbroadcastss(ymm_tmp, xmm_tmp);
    for (int jj = 0; jj < ur_w; jj++) {
        vmovups(Ymm(jj), ymm_tmp);
        if (jpp.is_training)
            vpxor(Ymm(3*ur_w+jj), Ymm(3*ur_w+jj), Ymm(3*ur_w+jj));
    }
    if (jpp.is_training)
    {
        movq(xmm_tmp, reg_k_shift);
        vpbroadcastd(ymm_k_offset, xmm_tmp);
    }

    mov(aux_reg_input, reg_input);
    xor_(kj, kj);
    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = nstl::max(0, pad_l - ki);
            int jj_end = ur_w - nstl::max(0, ki + pad_r - (kw-1));
            for (int jj = jj_start; jj  < jj_end; jj++) {
                int aux_input_offset = (ki+jj*stride_w-pad_l)* c_block;
                if (aux_input_offset > iw * c_block)
                    continue;
                int input_offset = sizeof(float)*aux_input_offset;
                vmovups(Ymm(ur_w+jj), ptr[aux_reg_input + input_offset]);
                vcmpps(Ymm(2*ur_w+jj), Ymm(jj), Ymm(ur_w+jj), _cmp_lt_os);
                vblendvps(Ymm(jj), Ymm(jj), Ymm(ur_w+jj), Ymm(2*ur_w+jj));
                if (jpp.is_training)
                    vblendvps(Ymm(3*ur_w+jj), Ymm(3*ur_w+jj), ymm_k_offset,
                            Ymm(2*ur_w+jj));
            }
            if (jpp.is_training)
                vpaddd(ymm_k_offset, ymm_k_offset, ymm_one);
        }
        add(aux_reg_input,  sizeof(float) * iw * c_block);
        inc(kj);
        cmp(kj, reg_kh);
        jl(kh_label, T_NEAR);
    }

    for (int jj = 0; jj < ur_w; jj++) {
        vmovups(yword[reg_output + sizeof(float)*jj*c_block], Ymm(jj));
        if (jpp.is_training)
            vmovdqu(yword[reg_index + sizeof(int)*jj*c_block], Ymm(3*ur_w+jj));
    }
}

inline void jit_avx2_pool_kernel_f32::max_oh_step_bwd(int ur_w, int pad_l,
        int pad_r, const char *kh_label) {

    int iw = jpp.iw;
    int kw = jpp.kw;
    int stride_w = jpp.stride_w;
    int c_block = jpp.c_block;

    for (int jj = 0; jj < ur_w; jj++) {
        vmovups(Ymm(jj), ptr[reg_output + sizeof(float)*jj*c_block]);
        vmovdqu(Ymm(ur_w+jj), ptr[reg_index + sizeof(int)*jj*c_block]);
    }

    mov(aux_reg_input, reg_input);
    movq(xmm_tmp, reg_k_shift);
    vpbroadcastd(ymm_k_offset, xmm_tmp);
    xor_(kj, kj);
    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = nstl::max(0, pad_l - ki);
            int jj_end = ur_w - nstl::max(0, ki + pad_r - (kw-1));
            for (int jj = jj_start; jj  < jj_end; jj++) {
                int aux_input_offset = (ki+jj*stride_w-pad_l)* c_block;
                if (aux_input_offset > iw * c_block)
                    continue;
                int input_offset = sizeof(float)*aux_input_offset;
                vmovups(Ymm(2*ur_w+jj), ptr[aux_reg_input + input_offset]);
                vpcmpeqd(Ymm(3*ur_w+jj), Ymm(ur_w+jj), ymm_k_offset);
                vaddps(Ymm(2*ur_w+jj), Ymm(2*ur_w+jj), Ymm(jj));
                vmaskmovps(yword[aux_reg_input + input_offset],
                    Ymm(3*ur_w+jj), Ymm(2*ur_w+jj));
            }
            vpaddd(ymm_k_offset, ymm_k_offset, ymm_one);
        }
        add(aux_reg_input,  sizeof(float) * iw * c_block);
        inc(kj);
        cmp(kj, reg_kh);
        jl(kh_label, T_NEAR);
    }
}


void jit_avx2_pool_kernel_f32::generate() {
    using Xbyak::Ymm;
    this->preamble();

    int ow = jpp.ow;
    int iw = jpp.iw;
    int kw = jpp.kw;
    int ur_w = jpp.ur_w;
    int c_block = jpp.c_block;
    int stride_w = jpp.stride_w;
    int l_pad = jpp.l_pad;
    int ur_w_tail = jpp.ur_w_tail;

    int n_oi = ow / ur_w;

#   define GET_OFF(field) offsetof(jit_pool_call_s, field)
    mov(reg_input, ptr[this->param1 + GET_OFF(src)]);
    mov(reg_output, ptr[this->param1 + GET_OFF(dst)]);
    if (jpp.is_max && (jpp.is_training || jpp.is_backward))
        mov(reg_index, ptr[this->param1 + GET_OFF(indices)]);
    mov(reg_kh, ptr[this->param1 + GET_OFF(kh_padding)]);
    mov(reg_k_shift, ptr[this->param1 + GET_OFF(kh_padding_shift)]);

#   undef GET_OFF

    if (jpp.is_max && (jpp.is_training || jpp.is_backward)) {
        mov(tmp_gpr, 1);
        movq(xmm_one, tmp_gpr);
        vpbroadcastd(ymm_one, xmm_one);
    }

    int r_pad  = nstl::max(0, ((ow-1)*stride_w) + kw - 1 - (iw + l_pad - 1 ));
    int r_pad1 = (ur_w*n_oi - 1)*stride_w + kw - 1 - (iw + l_pad - 1);
    if (r_pad1 > 0) n_oi--;

    if (l_pad > 0) {
        n_oi--;
        if (n_oi < 0 && r_pad1 > 0) {
            oh_step(ur_w, l_pad, r_pad1, ".kh_loop_oimain_padwl");
        } else  {
            oh_step(ur_w, l_pad, 0, ".kh_loop_oimain_padwl");
        }

        add(reg_input,  sizeof(float)*(ur_w*stride_w - l_pad)*c_block);
        add(reg_output,  sizeof(float)*ur_w*c_block);
        if (jpp.is_max && (jpp.is_training || jpp.is_backward))
            add(reg_index, sizeof(int)*ur_w*c_block);
    }

    xor_(oi_iter, oi_iter);
    if (n_oi > 0) {
        L(".ow_loop"); {
            oh_step( ur_w, 0, 0, ".kh_loop_oimain");
            add(reg_input, sizeof(float)*ur_w*stride_w*c_block);
            add(reg_output, sizeof(float)*ur_w*c_block);
            if (jpp.is_max && (jpp.is_training || jpp.is_backward))
                add(reg_index, sizeof(int)*ur_w*c_block);

            inc(oi_iter);
            cmp(oi_iter, n_oi); jl(".ow_loop", T_NEAR);
        } L(".ow_loop_end");
    }

    if (r_pad1 > 0 && n_oi >= 0) {
        oh_step( ur_w, 0, r_pad1, ".kh_loop_oimain_padwr");
        add(reg_input, sizeof(float)*ur_w*stride_w*c_block);
        add(reg_output, sizeof(float)*ur_w*c_block);
        if (jpp.is_max && (jpp.is_training || jpp.is_backward))
            add(reg_index, sizeof(int) * ur_w * c_block);
    }

    if (ur_w_tail != 0)
        oh_step(ur_w_tail, 0, r_pad, ".kh_loop_oitail");

    this->postamble();
}

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
