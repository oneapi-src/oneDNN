/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "jit_avx512_core_fork_bf16_dw_conv_kernel.hpp"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace Xbyak;

void jit_avx512_fork_dw_conv_fwd_kernel_bf16::load_src(int ur_ch_blocks, int ur_w) {
    for (int ch = 0; ch < ur_ch_blocks; ch++) {
        for (int ow = 0; ow < ur_w; ow++) {
            Zmm zmm_acc = get_acc_reg(ch * ur_w + ow);

            if (this->jcp.with_bias) {
                int b_off = ch * jcp.ch_block;
                uni_vmovups(zmm_acc, vmmword[reg_bias + b_off * sizeof(float)]);
            } else {
                uni_vpxor(zmm_acc, zmm_acc, zmm_acc);
            }
            if (this->jcp.with_sum) {
                int o_off = ch * jcp.oh * jcp.ow * jcp.ch_block
                        + ow * jcp.ch_block;
                if (jcp.dst_dt == data_type::bf16) {
                    vpmovzxwd(zmm_prev_dst,
                            vmmword[reg_output + o_off * jcp.typesize_out]);
                    vpslld(zmm_prev_dst, zmm_prev_dst, 16);
                    vaddps(zmm_acc, zmm_prev_dst);
                } else {
                    uni_vaddps(zmm_acc, zmm_acc,
                            vmmword[reg_output + o_off * jcp.typesize_out]);
                }
            }
        }
    }
}

void jit_avx512_fork_dw_conv_fwd_kernel_bf16::apply_filter(
        int ur_ch_blocks, int ur_w) {
    int ch_block = jcp.ch_block;
    int dilate_h = jcp.dilate_h + 1;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;

    Label iter_exit_label;

    cmp(reg_kh, 0);
    je(iter_exit_label, T_NEAR);
    cmp(reg_kw, 0);
    je(iter_exit_label, T_NEAR);

    mov(iter_kh, reg_kh);
    Label kh_label;
    L(kh_label); {
        mov(iter_kw, reg_kw);
        mov(aux1_reg_input, aux_reg_input);
        mov(aux1_reg_kernel, aux_reg_kernel);

        Label kw_label;
        L(kw_label); {
            for (int ch = 0; ch < ur_ch_blocks; ch++) {
                int ker_off = ch * jcp.kh * jcp.kw * ch_block;
                vpmovzxwd(zmm_ker_reg,
                        ptr[aux1_reg_kernel + ker_off * jcp.typesize_in]);
                for (int ow = 0; ow < ur_w; ow++) {
                    Zmm zmm_acc = get_acc_reg(ch * ur_w + ow);
                    int inp_off = ch * jcp.ih * jcp.iw * ch_block
                            + ow * stride_w * ch_block;
                    /* zero-extend bf16 to packed 32-bit int */
                    vpmovzxwd(zmm_src_reg,
                            ptr[aux1_reg_input + inp_off * jcp.typesize_in]);
                    if (!isa_has_bf16(jcp.isa)) {
                        bf16_emu_->vdpbf16ps(zmm_acc, zmm_ker_reg, zmm_src_reg);
                    } else {
                        vdpbf16ps(zmm_acc, zmm_ker_reg, zmm_src_reg);
                    }
                }
            }
            add(aux1_reg_kernel, jcp.ch_block * jcp.typesize_in);
            add(aux1_reg_input, jcp.ch_block * dilate_w * jcp.typesize_in);

            dec(iter_kw);
            cmp(iter_kw, 0);
            jg(kw_label, T_NEAR);
        }
        add(aux_reg_kernel, jcp.kw * jcp.ch_block * jcp.typesize_in);
        add(aux_reg_input, jcp.iw * jcp.ch_block * dilate_h * jcp.typesize_in);

        dec(iter_kh);
        cmp(iter_kh, 0);
        jg(kh_label, T_NEAR);
    }

    L(iter_exit_label);
}

void jit_avx512_fork_dw_conv_fwd_kernel_bf16::apply_filter_unrolled(
        int ur_ch_blocks, int ur_w) {
    int ch_blk = jcp.ch_block;
    int dilate_h = jcp.dilate_h + 1;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;

    Label iter_exit_label;

    cmp(reg_kh, 0);
    je(iter_exit_label, T_NEAR);

    mov(iter_kh, reg_kh);
    Label kh_label;
    L(kh_label); {
        for (int ch = 0; ch < ur_ch_blocks; ch++) {
            for (int kw = 0; kw < jcp.kw; kw++) {
                int ker_off = ch * jcp.kh * jcp.kw * ch_blk + kw * ch_blk;

                vpmovzxwd(zmm_ker_reg,
                        ptr[aux_reg_kernel + ker_off * jcp.typesize_in]);
                for (int ow = 0; ow < ur_w; ow++) {
                    Zmm zmm_acc = get_acc_reg(ch * ur_w + ow);
                    int inp_off = ch * jcp.ih * jcp.iw * ch_blk
                            + ow * stride_w * ch_blk + kw * ch_blk * dilate_w;
                    /* zero-extend bf16 to packed 32-bit int */
                    vpmovzxwd(zmm_src_reg,
                            ptr[aux_reg_input + inp_off * jcp.typesize_in]);
                    if (!isa_has_bf16(jcp.isa)) {
                        bf16_emu_->vdpbf16ps(zmm_acc, zmm_ker_reg, zmm_src_reg);
                    } else {
                        vdpbf16ps(zmm_acc, zmm_ker_reg, zmm_src_reg);
                    }
                }
            }
        }

        add(aux_reg_kernel, jcp.kw * ch_blk * jcp.typesize_in);
        add(aux_reg_input, jcp.iw * ch_blk * dilate_h * jcp.typesize_in);

        dec(iter_kh);
        cmp(iter_kh, 0);
        jg(kh_label, T_NEAR);
    }

    L(iter_exit_label);
}

void jit_avx512_fork_dw_conv_fwd_kernel_bf16::apply_postprocess(
        int ur_ch_blocks, int ur_w) {
    int eltwise_inj_idx = 0;
    int depthwise_inj_idx = 0;
    const auto& p = attr_.post_ops_;

    for (int i = 0; i < p.len(); i++) {
        auto& post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
            int start_idx = get_acc_reg(0).getIdx();
            int end_idx = get_acc_reg(ur_w * ur_ch_blocks).getIdx();

            eltwise_injectors[eltwise_inj_idx]->compute_vector_range(start_idx, end_idx);
            eltwise_inj_idx++;
        } else if (post_op.is_depthwise()) {
            mov(reg_d_weights, reinterpret_cast<size_t>(post_op.depthwise.weights_data));
            mov(reg_d_bias, reinterpret_cast<size_t>(post_op.depthwise.biases_data));

            add(reg_d_weights, ptr[this->param1 + GET_OFF(oc_off)]);
            add(reg_d_bias, ptr[this->param1 + GET_OFF(oc_off)]);

            for (int ch = 0; ch < ur_ch_blocks; ch++) {
                int start_idx = get_acc_reg(ur_w * ch).getIdx();
                int end_idx = get_acc_reg(ur_w * ch + ur_w).getIdx();

                depthwise_injectors[depthwise_inj_idx]->compute_vector_range(
                    start_idx, end_idx, reg_d_weights, reg_d_bias);

                add(reg_d_weights, jcp.ch_block * sizeof(float));
                add(reg_d_bias, jcp.ch_block * sizeof(float));
            }

            depthwise_inj_idx++;
        }
    }
}

void jit_avx512_fork_dw_conv_fwd_kernel_bf16::store_dst(int ur_ch_blocks, int ur_w) {
    int ch_blk = jcp.ch_block;

    if (jcp.dst_dt == data_type::bf16 && (!isa_has_bf16(jcp.isa)))
        bf16_emu_->init_vcvtneps2bf16();

    for (int ch = 0; ch < ur_ch_blocks; ch++) {
        if (jcp.dst_dt == data_type::f32) {
            for (int ow = 0; ow < ur_w; ow++) {
                int o_off = ch * jcp.oh * jcp.ow * ch_blk + ow * ch_blk;
                Zmm zmm_dst = get_acc_reg(ch * ur_w + ow);
                uni_vmovups(vmmword[reg_output + o_off * jcp.typesize_out],
                        zmm_dst);
            }
        } else if (jcp.dst_dt == data_type::bf16) {
            if (isa_has_bf16(jcp.isa)) {
                int n_2bf2ps = (ur_w / 2) * 2;
                int j = 0;
                for (; j < n_2bf2ps; j += 2) {
                    size_t aux_output_offset
                            = ((size_t)ch * jcp.oh * jcp.ow + j) * jcp.ch_block;
                    auto addr = ptr[reg_output
                            + aux_output_offset * jcp.typesize_out];
                    auto zmm_dst = get_acc_reg(ch * ur_w + j);
                    vcvtne2ps2bf16(zmm_dst, get_acc_reg(ch * ur_w + j + 1),
                            get_acc_reg(ch * ur_w + j));
                    vmovups(addr, zmm_dst);
                }
                /* Perform tail write for odd ur_w sizes */
                if (j < ur_w) {
                    size_t aux_output_offset
                            = ((size_t)ch * jcp.oh * jcp.ow + j) * jcp.ch_block;
                    auto addr = ptr[reg_output
                            + aux_output_offset * jcp.typesize_out];
                    auto zmm_dst = get_acc_reg(ch * ur_w + j);
                    auto ymm_dst = Ymm(zmm_dst.getIdx());
                    vcvtneps2bf16(ymm_dst, zmm_dst);
                    vmovups(addr, ymm_dst);
                }
            } else {
                for (int ow = 0; ow < ur_w; ow++) {
                    int o_off = ch * jcp.oh * jcp.ow * ch_blk + ow * ch_blk;
                    Zmm zmm_dst = get_acc_reg(ch * ur_w + ow);

                    /* down-convert f32 output to bf16 */
                    auto ymm_dst = Ymm(zmm_dst.getIdx());
                    bf16_emu_->vcvtneps2bf16(ymm_dst, zmm_dst);

                    uni_vmovups(ptr[reg_output + o_off * jcp.typesize_out],
                            ymm_dst);
                }
            }
        } else
            assert(!"unsupported destination type");
    }
}

void jit_avx512_fork_dw_conv_fwd_kernel_bf16::loop_ow(int ur_ch_blocks) {

    Label unrolled_w_label;
    Label tail_w_label;
    Label exit_label;

    L(unrolled_w_label); {
        int ur_w = jcp.ur_w;

        cmp(reg_ur_w, ur_w);
        jl(tail_w_label, T_NEAR);

        mov(aux_reg_input, reg_input);
        mov(aux_reg_kernel, reg_kernel);

        load_src(ur_ch_blocks, ur_w);
        apply_filter_unrolled(ur_ch_blocks, ur_w);
        apply_postprocess(ur_ch_blocks, ur_w);
        store_dst(ur_ch_blocks, ur_w);

        add(reg_input, jcp.typesize_in * ur_w * jcp.ch_block * jcp.stride_w);
        add(reg_output, jcp.typesize_out * ur_w * jcp.ch_block);

        sub(reg_ur_w, ur_w);
        jmp(unrolled_w_label);
    }

    L(tail_w_label); {
        int ur_w = 1;

        cmp(reg_ur_w, ur_w);
        jl(exit_label, T_NEAR);

        mov(aux_reg_input, reg_input);
        mov(aux_reg_kernel, reg_kernel);

        load_src(ur_ch_blocks, ur_w);
        apply_filter(ur_ch_blocks, ur_w);
        apply_postprocess(ur_ch_blocks, ur_w);
        store_dst(ur_ch_blocks, ur_w);

        add(reg_input, jcp.typesize_in * ur_w * jcp.ch_block * jcp.stride_w);
        add(reg_output, jcp.typesize_out * ur_w * jcp.ch_block);

        sub(reg_ur_w, ur_w);
        jmp(tail_w_label);
    }

    L(exit_label);
}

void jit_avx512_fork_dw_conv_fwd_kernel_bf16::generate() {
    const auto& p = attr_.post_ops_;
    for (int i = 0; i < p.len(); i++) {
        auto& post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
            eltwise_injectors.push_back(new jit_uni_eltwise_injector_f32<avx512_core>(
                this,
                post_op.eltwise
                ));
        } else if (post_op.is_depthwise()) {
            depthwise_injectors.push_back(new jit_uni_depthwise_injector_f32<avx512_core>(
                this,
                post_op.depthwise.alg
                ));
        }
    }

    this->preamble();

    mov(reg_input, ptr[this->param1 + GET_OFF(src)]);
    mov(reg_output, ptr[this->param1 + GET_OFF(dst)]);
    mov(reg_kernel, ptr[this->param1 + GET_OFF(filt)]);
    if (jcp.with_bias)
        mov(reg_bias, ptr[this->param1 + GET_OFF(bias)]);
    mov(reg_kh, ptr[this->param1 + GET_OFF(kh_padding)]);
    mov(reg_kw, ptr[this->param1 + GET_OFF(kw_padding)]);
    mov(reg_ch_blocks, ptr[this->param1 + GET_OFF(ch_blocks)]);
    mov(reg_ur_w, ptr[this->param1 + GET_OFF(ur_w)]);

    Label ch_blocks_tail_label;
    Label exit_label;

    int ch_blocks_tail = jcp.nb_ch % jcp.nb_ch_blocking;

    cmp(reg_ch_blocks, jcp.nb_ch_blocking);
    jne(ch_blocks_tail ? ch_blocks_tail_label : exit_label, T_NEAR);

    loop_ow(jcp.nb_ch_blocking); // channel main loop

    if (ch_blocks_tail) {
        L(ch_blocks_tail_label);

        cmp(reg_ch_blocks, ch_blocks_tail);
        jne(exit_label, T_NEAR);

        loop_ow(ch_blocks_tail); // channel tail loop
    }

    L(exit_label);

    this->postamble();

    for (auto& inj : eltwise_injectors)
        inj->prepare_table();
}

inline void jit_avx512_fork_dw_conv_bwd_data_kernel_bf16::load_ddst(
        int ur_ch_blocks, int ur_str_w) {
    for (int ch = 0; ch < ur_ch_blocks; ch++) {
        for (int w = 0; w < ur_str_w; w++) {
            Zmm zmm_acc = get_acc_reg(ch * ur_str_w + w);
            uni_vpxor(zmm_acc, zmm_acc, zmm_acc);
        }
    }
}

inline void jit_avx512_fork_dw_conv_bwd_data_kernel_bf16::apply_filter(
        int ur_ch_blocks, int ur_str_w) {
    int kw = jcp.kw;
    int kh = jcp.kh;
    int ow = jcp.ow;
    int oh = jcp.oh;

    int ch_blk = jcp.ch_block;
    int stride_h = jcp.stride_h;
    int stride_w = jcp.stride_w;

    Label iter_exit_label;

    cmp(reg_kh, 0);
    je(iter_exit_label, T_NEAR);

    cmp(reg_kw, 0);
    je(iter_exit_label, T_NEAR);

    mov(iter_kh, reg_kh);
    Label kh_label;
    L(kh_label); {
        mov(aux1_reg_ddst, aux_reg_ddst);
        mov(aux1_reg_kernel, aux_reg_kernel);

        mov(iter_kw, reg_kw);
        Label kw_label;
        L(kw_label); {
            for (int ch = 0; ch < ur_ch_blocks; ch++) {
                int ker_off = ch * kh * kw * ch_blk;
                vpmovzxwd(zmm_ker_reg,
                        ptr[aux1_reg_kernel + ker_off * jcp.typesize_in]);

                for (int w = 0; w < ur_str_w; w++) {
                    Zmm zmm_acc = get_acc_reg(ch * ur_str_w + w);
                    int ddst_off = (ch * oh * ow + w) * ch_blk;
                    vpmovzxwd(zmm_dst_reg,
                            ptr[aux1_reg_ddst + ddst_off * jcp.typesize_in]);

                    if (!isa_has_bf16(jcp.isa)) {
                        bf16_emu_->vdpbf16ps(
                                zmm_acc, zmm_dst_reg, zmm_ker_reg);
                    } else {
                        vdpbf16ps(zmm_acc, zmm_ker_reg, zmm_dst_reg);
                    }
                }
            }

            add(aux1_reg_kernel, ch_blk * stride_w * jcp.typesize_in);
            sub(aux1_reg_ddst, ch_blk * jcp.typesize_in);

            sub(iter_kw, stride_w);
            cmp(iter_kw, 0);
            jg(kw_label, T_NEAR);
        }

        add(aux_reg_kernel, kw * ch_blk * stride_h * jcp.typesize_in);
        sub(aux_reg_ddst, ow * ch_blk * jcp.typesize_in);

        sub(iter_kh, stride_h);
        cmp(iter_kh, 0);
        jg(kh_label, T_NEAR);
    }

    L(iter_exit_label);
}

inline void jit_avx512_fork_dw_conv_bwd_data_kernel_bf16::store_dsrc(
        int ur_ch_blocks, int ur_str_w) {
    int ch_blk = jcp.ch_block;
    int iw = jcp.iw;
    int ih = jcp.ih;
    int stride_w = jcp.stride_w;

    if (jcp.dsrc_dt == data_type::bf16 && (!isa_has_bf16(jcp.isa)))
        bf16_emu_->init_vcvtneps2bf16();

    for (int ch = 0; ch < ur_ch_blocks; ch++) {
        for (int w = 0; w < ur_str_w; w++) {
            int dsrc_off = (ch * ih * iw + w * stride_w) * ch_blk;
            auto zmm_dsrc = get_acc_reg(ch * ur_str_w + w);

            if (jcp.dsrc_dt == data_type::f32) {
                uni_vmovups(
                        ptr[reg_dsrc + dsrc_off * jcp.typesize_out], zmm_dsrc);
            } else if (jcp.dsrc_dt == data_type::bf16) {
                auto ymm_dsrc = Ymm(zmm_dsrc.getIdx());
                if (isa_has_bf16(jcp.isa)) {
                    vcvtneps2bf16(ymm_dsrc, zmm_dsrc);
                } else {
                    bf16_emu_->vcvtneps2bf16(ymm_dsrc, zmm_dsrc);
                }
                vmovups(ptr[reg_dsrc + dsrc_off * jcp.typesize_out], ymm_dsrc);
            }
        }
    }
    /* Note: current 'store_dsrc' is limited to storing 'ymm' output. This is
     * because of the current implementation approach that calculates convolution as
     * a strided backward-pass. To increase store throughput by writing 'zmm'
     * registers, changes are needed in both JIT-kernel and Driver code. */
}

inline void jit_avx512_fork_dw_conv_bwd_data_kernel_bf16::loop_body(
        int ur_ch_blocks) {
    Label unrolled_w_label;
    Label tail_w_label;
    Label exit_label;

    L(unrolled_w_label); {
        int ur_w = jcp.ur_w;

        cmp(reg_ur_str_w, ur_w);
        jl(tail_w_label, T_NEAR);

        mov(aux_reg_ddst, reg_ddst);
        mov(aux_reg_kernel, reg_kernel);

        load_ddst(ur_ch_blocks, ur_w);
        apply_filter(ur_ch_blocks, ur_w);
        store_dsrc(ur_ch_blocks, ur_w);

        add(reg_dsrc, jcp.typesize_out * ur_w * jcp.ch_block * jcp.stride_w);
        add(reg_ddst, jcp.typesize_in * ur_w * jcp.ch_block);

        sub(reg_ur_str_w, ur_w);
        jmp(unrolled_w_label);
    }

    L(tail_w_label); {
        int ur_w = 1;

        cmp(reg_ur_str_w, ur_w);
        jl(exit_label, T_NEAR);

        mov(aux_reg_ddst, reg_ddst);
        mov(aux_reg_kernel, reg_kernel);

        load_ddst(ur_ch_blocks, ur_w);
        apply_filter(ur_ch_blocks, ur_w);
        store_dsrc(ur_ch_blocks, ur_w);

        add(reg_dsrc, jcp.typesize_out * ur_w * jcp.ch_block * jcp.stride_w);
        add(reg_ddst, jcp.typesize_in * ur_w * jcp.ch_block);

        sub(reg_ur_str_w, ur_w);
        jmp(tail_w_label);
    }

    L(exit_label);
}

void jit_avx512_fork_dw_conv_bwd_data_kernel_bf16::generate() {
    preamble();
    mov(reg_dsrc, ptr[this->param1 + GET_OFF(src)]);
    mov(reg_ddst, ptr[this->param1 + GET_OFF(dst)]);
    mov(reg_kernel, ptr[this->param1 + GET_OFF(filt)]);
    mov(reg_kh, ptr[this->param1 + GET_OFF(kh_padding)]);
    mov(reg_kw, ptr[this->param1 + GET_OFF(kw_padding)]);
    mov(reg_ch_blocks, ptr[this->param1 + GET_OFF(ch_blocks)]);
    mov(reg_ur_str_w, ptr[this->param1 + GET_OFF(ur_str_w)]);

    Label ch_blocks_tail_label;
    Label exit_label;

    int ch_blocks_tail = jcp.nb_ch % jcp.nb_ch_blocking;

    cmp(reg_ch_blocks, jcp.nb_ch_blocking);
    jne(ch_blocks_tail ? ch_blocks_tail_label : exit_label, T_NEAR);

    loop_body(jcp.nb_ch_blocking); // channel main loop

    if (ch_blocks_tail) {
        L(ch_blocks_tail_label);

        cmp(reg_ch_blocks, ch_blocks_tail);
        jne(exit_label, T_NEAR);

        loop_body(ch_blocks_tail); // channel tail loop
    }

    L(exit_label);
    this->postamble();
}

}
}
}
}
