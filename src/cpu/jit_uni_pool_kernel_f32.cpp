/*******************************************************************************
* Copyright 2017 Intel Corporation
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

#include "jit_uni_pool_kernel_f32.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace Xbyak;
using namespace alg_kind;

template <cpu_isa_t isa>
status_t jit_uni_pool_kernel_f32<isa>::init_conf(jit_pool_conf_t &jpp,
            const pooling_desc_t &pd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &dst_d) {

    bool args_ok = true
        && utils::one_of(pd.alg_kind, pooling_max,
                pooling_avg_include_padding,
                pooling_avg_exclude_padding)
        && pd.kernel[0] == pd.kernel[1];
    if (!args_ok) return status::unimplemented;

    const int simd_w = isa == avx2 ? 8 : 16;

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

    jpp.alg = pd.alg_kind;

    jpp.is_training = pd.prop_kind == prop_kind::forward_training;
    jpp.is_backward = pd.prop_kind == prop_kind::backward_data;

    jpp.c_block = simd_w;

    jpp.nb_c = jpp.c / jpp.c_block;
    if (jpp.alg == pooling_max) {
        jpp.ur_w = isa == avx2 ? 4 : 16;
        if (jpp.is_training)
            jpp.ur_w = isa == avx2 ? 3 : 10;
        else if (jpp.is_backward)
            jpp.ur_w = isa == avx2 ? 3 : 6;
    } else {
        if (jpp.is_backward)
            jpp.ur_w = isa == avx2 ? 6 : 12;
        else
            jpp.ur_w = isa == avx2 ? 12 : 24;
    }
    if (jpp.ow < jpp.ur_w) jpp.ur_w = jpp.ow;
    if (jpp.l_pad > jpp.ur_w) return status::unimplemented;

    jpp.ur_w_tail = jpp.ow % jpp.ur_w;

    return status::success;
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel_f32<isa>::maybe_recalculate_divisor(int jj,
        int ur_w, int pad_l, int pad_r) {
    if (jpp.alg == pooling_avg_exclude_padding) {
        int kw = jpp.kw;
        int stride_w = jpp.stride_w;

        int non_zero_kw = kw;
        non_zero_kw -= nstl::max(0, pad_l - jj*stride_w);
        non_zero_kw -= nstl::max(0, pad_r - (ur_w - 1 - jj)*stride_w);

        if (non_zero_kw != prev_kw) {
            mov(tmp_gpr, float2int(non_zero_kw));
            movq(xmm_tmp, tmp_gpr);
            vbroadcastss(vmm_tmp, xmm_tmp);
            vmulps(vmm_tmp, vmm_tmp, vmm_ker_area_h);
            prev_kw = non_zero_kw;
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel_f32<isa>::avg_step(int ur_w, int pad_l,
        int pad_r, const char* kh_label) {

    int iw = jpp.iw;
    int kw = jpp.kw;
    int stride_w = jpp.stride_w;
    int c_block = jpp.c_block;

    for (int jj = 0; jj < ur_w; jj++) {
        if (jpp.is_backward) {
            vmovups(vreg(jj), ptr[reg_output + sizeof(float)*jj*c_block]);
            maybe_recalculate_divisor(jj, ur_w, pad_l, pad_r);
            vdivps(vreg(jj), vreg(jj), vmm_tmp);
        } else {
            uni_vpxor(vreg(jj), vreg(jj), vreg(jj));
        }
    }

    mov(aux_reg_input, reg_input);
    xor_(kj, kj);
    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = nstl::max(0, pad_l - ki);
            int jj_end = ur_w
                - utils::div_up(nstl::max(0, ki + pad_r - (kw-1)), stride_w);
            for (int jj = jj_start; jj  < jj_end; jj++) {
                int aux_input_offset = (ki+jj*stride_w-pad_l)* c_block;
                if (aux_input_offset > iw * c_block)
                    continue;
                int input_offset = sizeof(float)*aux_input_offset;
                if (jpp.is_backward) {
                    vmovups(vreg(ur_w+jj), ptr[aux_reg_input + input_offset]);
                    vaddps(vreg(ur_w+jj), vreg(ur_w+jj), vreg(jj));
                    vmovups(vmmword[aux_reg_input + input_offset], vreg(ur_w+jj));
                } else {
                    vaddps(vreg(jj), vreg(jj), ptr[aux_reg_input + input_offset]);
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
            maybe_recalculate_divisor(jj, ur_w, pad_l, pad_r);
            vdivps(vreg(jj), vreg(jj), vmm_tmp);
            vmovups(vmmword[reg_output + sizeof(float)*jj*c_block], vreg(jj));
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel_f32<isa>::max_step_fwd(int ur_w, int pad_l,
        int pad_r, const char *kh_label) {
    unsigned char _cmp_lt_os = 1;

    int iw = jpp.iw;
    int kw = jpp.kw;
    int stride_w = jpp.stride_w;
    int c_block = jpp.c_block;

    mov(tmp_gpr, float2int(-FLT_MAX));
    movq(xmm_tmp, tmp_gpr);
    vbroadcastss(vmm_tmp, xmm_tmp);
    for (int jj = 0; jj < ur_w; jj++) {
        vmovups(vreg(jj), vmm_tmp);
        if (jpp.is_training)
            uni_vpxor(vreg(2*ur_w+jj), vreg(2*ur_w+jj), vreg(2*ur_w+jj));
    }
    if (jpp.is_training)
    {
        movq(xmm_tmp, reg_k_shift);
        vpbroadcastd(vmm_k_offset, xmm_tmp);
    }
    mov(aux_reg_input, reg_input);
    xor_(kj, kj);
    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = nstl::max(0, pad_l - ki);
            int jj_end = ur_w
                - utils::div_up(nstl::max(0, ki + pad_r - (kw-1)), stride_w);
            for (int jj = jj_start; jj  < jj_end; jj++) {
                int aux_input_offset = (ki+jj*stride_w-pad_l)* c_block;
                if (aux_input_offset > iw * c_block)
                    continue;
                int input_offset = sizeof(float)*aux_input_offset;
                vmovups(vreg(ur_w+jj), ptr[aux_reg_input + input_offset]);
                if (isa == avx2) {
                    vcmpps(vreg(3*ur_w+jj), vreg(jj), vreg(ur_w+jj), _cmp_lt_os);
                    vblendvps(vreg(jj), vreg(jj), vreg(ur_w+jj), vreg(3*ur_w+jj));
                    if (jpp.is_training)
                        vblendvps(vreg(2*ur_w+jj), vreg(2*ur_w+jj), vmm_k_offset,
                                                    vreg(3*ur_w+jj));
                } else {
                    vcmpps(k_store_mask, vreg(jj), vreg(ur_w+jj), _cmp_lt_os);
                    vblendmps(vreg(jj) | k_store_mask, vreg(jj), vreg(ur_w+jj));
                    if (jpp.is_training)
                        vblendmps(vreg(2*ur_w+jj) | k_store_mask, vreg(2*ur_w+jj),
                                                    vmm_k_offset);
                }
            }
            if (jpp.is_training)
                vpaddd(vmm_k_offset, vmm_k_offset, vmm_one);
        }
        add(aux_reg_input,  sizeof(float) * iw * c_block);
        inc(kj);
        cmp(kj, reg_kh);
        jl(kh_label, T_NEAR);
    }

    for (int jj = 0; jj < ur_w; jj++) {
        vmovups(vmmword[reg_output + sizeof(float)*jj*c_block], vreg(jj));
        if (jpp.is_training)
            uni_vmovdqu(vmmword[reg_index + sizeof(int)*jj*c_block], vreg(2*ur_w+jj));
    }
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel_f32<isa>::max_step_bwd(int ur_w, int pad_l,
        int pad_r, const char *kh_label) {

    int iw = jpp.iw;
    int kw = jpp.kw;
    int stride_w = jpp.stride_w;
    int c_block = jpp.c_block;

    for (int jj = 0; jj < ur_w; jj++) {
        vmovups(vreg(jj), ptr[reg_output + sizeof(float)*jj*c_block]);
        uni_vmovdqu(vreg(ur_w+jj), ptr[reg_index + sizeof(int)*jj*c_block]);
    }

    mov(aux_reg_input, reg_input);
    movq(xmm_tmp, reg_k_shift);
    vpbroadcastd(vmm_k_offset, xmm_tmp);
    xor_(kj, kj);
    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = nstl::max(0, pad_l - ki);
            int jj_end = ur_w
                - utils::div_up(nstl::max(0, ki + pad_r - (kw-1)), stride_w);
            for (int jj = jj_start; jj  < jj_end; jj++) {
                int aux_input_offset = (ki+jj*stride_w-pad_l)* c_block;
                if (aux_input_offset > iw * c_block)
                    continue;
                int input_offset = sizeof(float)*aux_input_offset;
                vmovups(vreg(2*ur_w+jj), ptr[aux_reg_input + input_offset]);
                if (isa == avx2) {
                    vpcmpeqd(vreg(3*ur_w+jj), vreg(ur_w+jj), vmm_k_offset);
                    vaddps(vreg(2*ur_w+jj), vreg(2*ur_w+jj), vreg(jj));
                    vmaskmovps(vmmword[aux_reg_input + input_offset],
                            vreg(3*ur_w+jj), vreg(2*ur_w+jj));
                } else {
                    vpcmpeqd(k_store_mask, vreg(ur_w+jj), vmm_k_offset);
                    vblendmps(vmm_tmp | k_store_mask | T_z, vreg(jj), vreg(jj));
                    vaddps(vreg(2*ur_w+jj), vreg(2*ur_w+jj), vmm_tmp);
                    vmovups(vmmword[aux_reg_input + sizeof(float)*aux_input_offset],
                            vreg(2*ur_w+jj));
                }
            }
            vpaddd(vmm_k_offset, vmm_k_offset, vmm_one);
        }
        add(aux_reg_input,  sizeof(float) * iw * c_block);
        inc(kj);
        cmp(kj, reg_kh);
        jl(kh_label, T_NEAR);
    }
}

template <cpu_isa_t isa>
void jit_uni_pool_kernel_f32<isa>::generate() {

    this->preamble();

    int ow = jpp.ow;
    int iw = jpp.iw;
    int kw = jpp.kw;
    int kh = jpp.kh;
    int ur_w = jpp.ur_w;
    int c_block = jpp.c_block;
    int stride_w = jpp.stride_w;
    int l_pad = jpp.l_pad;
    int ur_w_tail = jpp.ur_w_tail;

    int n_oi = ow / ur_w;

    prev_kw = 0;

#   define GET_OFF(field) offsetof(jit_pool_call_s, field)
    mov(reg_input, ptr[this->param1 + GET_OFF(src)]);
    mov(reg_output, ptr[this->param1 + GET_OFF(dst)]);
    if (jpp.alg == pooling_max && (jpp.is_training || jpp.is_backward))
        mov(reg_index, ptr[this->param1 + GET_OFF(indices)]);
    mov(reg_kh, ptr[this->param1 + GET_OFF(kh_padding)]);
    mov(reg_k_shift, ptr[this->param1 + GET_OFF(kh_padding_shift)]);
    mov(reg_ker_area_h, ptr[this->param1 + GET_OFF(ker_area_h)]);

#   undef GET_OFF

    if (jpp.alg == pooling_max && (jpp.is_training || jpp.is_backward)) {
        mov(tmp_gpr, 1);
        movq(xmm_one, tmp_gpr);
        vpbroadcastd(vmm_one, xmm_one);
    }

    int r_pad  = nstl::max(0, ((ow-1)*stride_w) + kw - 1 - (iw + l_pad - 1 ));
    int r_pad1 = (ur_w*n_oi - 1)*stride_w + kw - 1 - (iw + l_pad - 1);
    if (r_pad1 > 0) n_oi--;

    if (jpp.alg == pooling_avg_exclude_padding) {
        movq(xmm_ker_area_h, reg_ker_area_h);
        vpbroadcastd(vmm_ker_area_h, xmm_ker_area_h);
    }

    if (jpp.alg == pooling_avg_include_padding) {
        mov(tmp_gpr, float2int(kw * kh));
        movq(xmm_tmp, tmp_gpr);
        vbroadcastss(vmm_tmp, xmm_tmp);
    }

    if (l_pad > 0) {
        n_oi--;
        if (n_oi < 0 && r_pad1 > 0) {
            step(ur_w, l_pad, r_pad1, ".kh_loop_oimain_padwl");
        } else  {
            step(ur_w, l_pad, 0, ".kh_loop_oimain_padwl");
        }

        add(reg_input,  sizeof(float)*(ur_w*stride_w - l_pad)*c_block);
        add(reg_output,  sizeof(float)*ur_w*c_block);
        if (jpp.alg == pooling_max && (jpp.is_training || jpp.is_backward))
            add(reg_index, sizeof(int)*ur_w*c_block);
    }

    xor_(oi_iter, oi_iter);
    if (n_oi > 0) {
        L(".ow_loop"); {
            step( ur_w, 0, 0, ".kh_loop_oimain");
            add(reg_input, sizeof(float)*ur_w*stride_w*c_block);
            add(reg_output, sizeof(float)*ur_w*c_block);
            if (jpp.alg == pooling_max && (jpp.is_training || jpp.is_backward))
                add(reg_index, sizeof(int)*ur_w*c_block);

            inc(oi_iter);
            cmp(oi_iter, n_oi); jl(".ow_loop", T_NEAR);
        } L(".ow_loop_end");
    }

    if (r_pad1 > 0 && n_oi >= 0) {
        step( ur_w, 0, r_pad1, ".kh_loop_oimain_padwr");
        add(reg_input, sizeof(float)*ur_w*stride_w*c_block);
        add(reg_output, sizeof(float)*ur_w*c_block);
        if (jpp.alg == pooling_max && (jpp.is_training || jpp.is_backward))
            add(reg_index, sizeof(int) * ur_w * c_block);
    }

    if (ur_w_tail != 0)
        step(ur_w_tail, 0, r_pad, ".kh_loop_oitail");

    this->postamble();
}

template struct jit_uni_pool_kernel_f32<avx2>;
template struct jit_uni_pool_kernel_f32<avx512_mic>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
