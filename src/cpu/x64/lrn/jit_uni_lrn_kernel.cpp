/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
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

#include <array>
#include <cmath>
#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"
#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/lrn/jit_uni_lrn_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::format_tag;

#define IRB_LOOP(statement) \
    if (1 == reg_block) { \
        const int irb_off = 0; \
        const int irb = this->reg_block_idx_ % vsum.size(); \
        statement; \
        MAYBE_UNUSED(irb_off); \
    } else { \
        for (int irb = 0; irb < reg_block; irb++) { \
            const int irb_off = irb * single_pixel_offset_; \
            statement; \
            MAYBE_UNUSED(irb_off); \
        } \
    }

using namespace Xbyak;

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_fwd_kernel<isa, d_type>::load_data(
        Vmm reg, const Xbyak::Address p) {
    this->uni_vmovups(reg, p);
}

template <>
void jit_uni_lrn_fwd_kernel<avx512_common,
        dnnl::impl::data_type::bf16>::load_data(Vmm reg,
        const Xbyak::Address p) {
    this->vpmovzxwd(reg, p);
    this->vpslld(reg, reg, 0x10);
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_fwd_kernel<isa, d_type>::store_data(
        const Xbyak::Address addr, Vmm reg) {
    this->uni_vmovups(addr, reg);
}

template <>
void jit_uni_lrn_fwd_kernel<avx512_common,
        dnnl::impl::data_type::bf16>::store_data(const Xbyak::Address addr,
        Zmm zr) {
    const Ymm yr = Ymm(zr.getIdx());
    if (mayiuse(avx512_core_bf16))
        vcvtneps2bf16(yr, zr);
    else
        bf16_emu_->vcvtneps2bf16(yr, zr);
    vmovdqu16(addr, yr);
}

//////////////////////////////////////////////////////////////////////////////
// forward kernel
template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_fwd_kernel<isa, d_type>::within_body(int hoff, int Hoff,
        int woff, int Woff, int stride, prop_kind_t pk, const int reg_block,
        int pixel_offset) {

    static const bool emulateBfloat
            = (isa == avx512_common && d_type == dnnl::impl::data_type::bf16
                    && !mayiuse(avx512_core_bf16));
    static const std::array<Vmm, 3> vsum {{Vmm(2), Vmm(11), Vmm(20)}};
    static const std::array<Vmm, 3> vsum2 {{Vmm(3), Vmm(12), Vmm(21)}};
    static const std::array<Vmm, 3> vdst {{Vmm(4), Vmm(13), Vmm(22)}};
    static const std::array<std::array<Vmm, 6u>, 3u> vtmp {
            {{{Vmm(5), Vmm(6), Vmm(7), Vmm(8), Vmm(9), Vmm(14)}},
                    {{Vmm(18), Vmm(15), Vmm(16), Vmm(17), Vmm(29), Vmm(30)}},
                    {{Vmm(23), Vmm(24), Vmm(25), Vmm(26), Vmm(28), Vmm(31)}}}};
    static const std::array<Vmm, 3> vscratch = {{Vmm(10), Vmm(19), Vmm(27)}};
    MAYBE_UNUSED(
            emulateBfloat); // workaround for gcc8.1 compiler unused variable
    static const std::size_t used_tmp_regs
            = emulateBfloat ? vtmp[0].size() - 2 : vtmp[0].size();

    IRB_LOOP(uni_vxorps(vsum[irb], vsum[irb], vsum[irb]));
    for (int i = hoff; i <= Hoff; ++i) {
        for (int j = woff; j <= Woff; ++j) {
            if (i == 0 && j == 0) {
                IRB_LOOP(load_data(
                        vdst[irb], ptr[src_ + pixel_offset + irb_off]));
                IRB_LOOP(vfmadd231ps(vsum[irb], vdst[irb], vdst[irb]));
            } else {
                const auto idx = tempIdx_ % used_tmp_regs;
                IRB_LOOP(load_data(vtmp[irb][idx],
                        ptr[(src_ + pixel_offset + irb_off)
                                + (i * stride + j) * single_pixel_offset_]));
                IRB_LOOP(
                        vfmadd231ps(vsum[irb], vtmp[irb][idx], vtmp[irb][idx]));
                ++tempIdx_;
            }
        }
    }

    tempIdx_ = tempIdx_ % used_tmp_regs;

    IRB_LOOP(vfmadd132ps(vsum[irb], vk_, valpha_)); // ysum <- ysum*valpha_+yk_
    IRB_LOOP(vmovaps(vscratch[irb], vsum[irb]));
    if (pk != prop_kind::forward_inference) {
        IRB_LOOP(store_data(ptr[scratch_], vscratch[irb]));
    }

    IRB_LOOP(vmulps(vsum2[irb], vsum[irb], vsum[irb]));
    IRB_LOOP(vmulps(
            vsum[irb], vsum[irb], vsum2[irb])); // ysum = (ysum*valpha_+yk_)^3;
    IRB_LOOP(vsqrtps(vsum[irb], vsum[irb]));
    IRB_LOOP(vsqrtps(vsum[irb], vsum[irb])); // ysum = (ysum*valpha_+yk_)^0.75
    IRB_LOOP(vdivps(vdst[irb], vdst[irb], vsum[irb])); // ydst <- ydst / ysum
    IRB_LOOP(store_data(ptr[dst_ + pixel_offset + irb_off], vdst[irb]));
    if (isa == avx512_common)
        this->reg_block_idx_ = (this->reg_block_idx_ % vsum.size()) + 1;
}

template <>
void jit_uni_lrn_fwd_kernel<sse41, dnnl::impl::data_type::f32>::within_body(
        int hoff, int Hoff, int woff, int Woff, int stride, prop_kind_t pk,
        int reg_block, int pixel_offset) {

    static const Xbyak::Xmm xtmp_lo = Xmm(2);
    static const Xbyak::Xmm xtmp_hi = Xmm(3);
    static const Xbyak::Xmm xsum_lo = Xmm(4);
    static const Xbyak::Xmm xsum_hi = Xmm(5);
    static const Xbyak::Xmm xdst_lo = Xmm(6);
    static const Xbyak::Xmm xdst_hi = Xmm(7);
    static const Xbyak::Xmm xsum2_lo = Xmm(8);
    static const Xbyak::Xmm xsum2_hi = Xmm(9);

    xorps(xsum_lo, xsum_lo);
    xorps(xsum_hi, xsum_hi);
    for (int i = hoff; i <= Hoff; ++i) {
        for (int j = woff; j <= Woff; ++j) {
            if (i == 0 && j == 0) {
                movups(xdst_lo, ptr[src_]);
                movups(xdst_hi, ptr[src_ + 4 * sizeof(float)]);
                mulps(xdst_lo, xdst_lo);
                mulps(xdst_hi, xdst_hi);
                addps(xsum_lo, xdst_lo);
                addps(xsum_hi, xdst_hi);
            } else {
                movups(xtmp_lo,
                        ptr[src_ + pixel_offset
                                + (i * stride + j) * VECTOR_LENGTH * 4]);
                movups(xtmp_hi,
                        ptr[src_ + pixel_offset
                                + (i * stride + j) * VECTOR_LENGTH * 4
                                + 4 * sizeof(float)]);
                mulps(xtmp_lo, xtmp_lo);
                mulps(xtmp_hi, xtmp_hi);
                addps(xsum_lo, xtmp_lo);
                addps(xsum_hi, xtmp_hi);
            }
        }
    }
    mulps(xsum_lo, xalpha_);
    mulps(xsum_hi, xalpha_);
    addps(xsum_lo, xk_);
    addps(xsum_hi, xk_); // xsum <- xsum*xalpha_+xk_
    movaps(xtmp_lo, xsum_lo);
    movaps(xtmp_hi, xsum_hi);
    if (pk != prop_kind::forward_inference) {
        movups(ptr[scratch_ + pixel_offset], xtmp_lo);
        movups(ptr[scratch_ + pixel_offset + 4 * sizeof(float)], xtmp_hi);
    }
    movaps(xsum2_lo, xsum_lo);
    movaps(xsum2_hi, xsum_hi);
    mulps(xsum2_lo, xsum_lo);
    mulps(xsum2_hi, xsum_hi);
    mulps(xsum_lo, xsum2_lo);
    mulps(xsum_hi, xsum2_hi); // xsum = (xsum*xalpha_+xk_)^3;

    sqrtps(xsum_lo, xsum_lo);
    sqrtps(xsum_hi, xsum_hi);
    sqrtps(xsum_lo, xsum_lo);
    sqrtps(xsum_hi, xsum_hi); // xsum = (xsum*xalpha_+xk_)^0.75

    movups(xdst_lo, ptr[src_ + pixel_offset]);
    movups(xdst_hi, ptr[src_ + pixel_offset + 4 * sizeof(float)]);
    divps(xdst_lo, xsum_lo);
    divps(xdst_hi, xsum_hi); // xdst <- xdst / xsum

    movups(ptr[dst_ + pixel_offset], xdst_lo);
    movups(ptr[dst_ + pixel_offset + 4 * sizeof(float)], xdst_hi);
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_fwd_kernel<isa, d_type>::move_data_pointers(
        int pixel_count, prop_kind_t pk) {

    const int pixel_offset = single_pixel_offset_ * pixel_count;
    add(src_, pixel_offset);
    add(dst_, pixel_offset);
    if (pk != prop_kind::forward_inference) add(scratch_, pixel_offset);
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_fwd_kernel<isa, d_type>::within_body_reg_blocked(
        int loop_count, int max_reg_blocks, int hoff, int Hoff, int woff,
        int Woff, int stride, prop_kind_t pk) {

    Label label_t;

    const auto res = std::div(loop_count, max_reg_blocks);
    if (res.quot) {
        mov(w_, res.quot);
        L(label_t);
        within_body(hoff, Hoff, woff, Woff, stride, pk, max_reg_blocks, 0);
        move_data_pointers(max_reg_blocks, pk);
        dec(w_);
        cmp(w_, 0);
        jne(label_t, T_NEAR);
    }
    if (res.rem) {
        within_body(hoff, Hoff, woff, Woff, stride, pk, res.rem, 0);
        move_data_pointers(res.rem, pk);
    }
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_fwd_kernel<isa, d_type>::load_constant(
        float constant, Vmm v_constant, Xbyak::Xmm x_constant) {
    mov(imm_addr64_, float2int(constant));
    uni_vmovq(x_constant, imm_addr64_);
    vbroadcastss(v_constant, x_constant);
}

template <>
void jit_uni_lrn_fwd_kernel<sse41, dnnl::impl::data_type::f32>::load_constant(
        float constant, Vmm v_constant, Xbyak::Xmm x_constant) {
    mov(imm_addr64_, float2int(constant));
    uni_vmovq(x_constant, imm_addr64_);
    shufps(x_constant, x_constant, 0);
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_lrn_fwd_kernel<isa, d_type>::jit_uni_lrn_fwd_kernel(
        const within_config &J, float A, float K, prop_kind_t pk,
        void *code_ptr, size_t code_size)
    : jit_generator(code_ptr, code_size)
    , alpha_(A)
    , k_(K)
    , single_pixel_offset_(J.dat_tag == nhwc
                      ? J.C * sizeof(typename prec_traits<d_type>::type)
                      : VECTOR_LENGTH
                              * sizeof(typename prec_traits<d_type>::type)) {

    if (isa == avx512_common && d_type == dnnl::impl::data_type::bf16
            && !mayiuse(avx512_core_bf16)) {
        bf16_emu_ = utils::make_unique<bf16_emulation_t>(this,
                bf16_emu_reserv_1_, bf16_emu_reserv_2_, bf16_emu_reserv_3_,
                bf16_emu_scratch_, bf16_emu_reserv_4_);
        bf16_emu_->init_vcvtneps2bf16();
    }

    static const int max_reg_blocks = isa == avx512_common ? 3 : 1;

    this->preamble();

#define GET_OFF(field) offsetof(jit_args_fwd_t, field)
    mov(src_, ptr[this->param1 + GET_OFF(src)]);
    mov(dst_, ptr[this->param1 + GET_OFF(dst)]);
    if (pk != prop_kind::forward_inference)
        mov(scratch_, ptr[this->param1 + GET_OFF(scratch)]);
#undef GET_OFF

    load_constant(alpha_, valpha_, xalpha_);
    load_constant(k_, vk_, xk_);

    const int s2 = (J.size - 1) / 2, S2 = J.size - s2 - 1;

    int pixel_count = 0;

    for (int i = 0; i < s2; ++i) {
        pixel_count = 0;
        for (int j = 0; j < s2; ++j)
            within_body(-i, S2, -j, S2, J.W, pk, 1,
                    pixel_count++ * single_pixel_offset_);
        move_data_pointers(pixel_count, pk);

        within_body_reg_blocked(
                J.W - J.size + 1, max_reg_blocks, -i, S2, -s2, S2, J.W, pk);

        pixel_count = 0;
        for (int j = J.W - S2; j < J.W; ++j)
            within_body(-i, S2, -s2, J.W - 1 - j, J.W, pk, 1,
                    pixel_count++ * single_pixel_offset_);
        move_data_pointers(pixel_count, pk);
    }

    mov(h_, J.H - J.size + 1);
    Label lrn_loop_h;
    L(lrn_loop_h);
    pixel_count = 0;
    for (int j = 0; j < s2; ++j)
        within_body(-s2, S2, -j, S2, J.W, pk, 1,
                pixel_count++ * single_pixel_offset_);
    move_data_pointers(pixel_count, pk);

    within_body_reg_blocked(
            J.W - J.size + 1, max_reg_blocks, -s2, S2, -s2, S2, J.W, pk);

    pixel_count = 0;
    for (int j = J.W - S2; j < J.W; ++j)
        within_body(-s2, S2, -s2, J.W - 1 - j, J.W, pk, 1,
                pixel_count++ * single_pixel_offset_);
    move_data_pointers(pixel_count, pk);

    dec(h_);
    cmp(h_, 0);
    jne(lrn_loop_h, T_NEAR);

    for (int i = J.H - S2; i < J.H; ++i) {
        pixel_count = 0;
        for (int j = 0; j < s2; ++j)
            within_body(-s2, J.H - 1 - i, -j, S2, J.W, pk, 1,
                    pixel_count++ * single_pixel_offset_);
        move_data_pointers(pixel_count, pk);

        within_body_reg_blocked(J.W - J.size + 1, max_reg_blocks, -s2,
                J.H - 1 - i, -s2, S2, J.W, pk);

        pixel_count = 0;
        for (int j = J.W - S2; j < J.W; ++j)
            within_body(-s2, J.H - 1 - i, -s2, J.W - 1 - j, J.W, pk, 1,
                    pixel_count++ * single_pixel_offset_);
        move_data_pointers(pixel_count, pk);
    }

    this->postamble();

    ker = reinterpret_cast<decltype(ker)>(
            const_cast<uint8_t *>(this->getCode()));
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_lrn_fwd_kernel<isa, d_type>::jit_uni_lrn_fwd_kernel(
        const struct nchw8c_across &J, float A, float K, prop_kind_t pk,
        void *code_ptr, size_t code_size)
    : jit_generator(code_ptr, code_size), bf16_emu_(nullptr), alpha_(A), k_(K) {
    Xbyak::Reg64 t = rsp;
    Xbyak::Reg64 hw = r9;
    Xbyak::Xmm xsrc_prev = xmm2;
    Xbyak::Ymm ysrc = ymm3;
    Xbyak::Ymm yc = ymm3;
    Xbyak::Xmm xsrc_next = xmm4;
    Xbyak::Ymm ya = ymm5;
    Xbyak::Ymm yb = ymm6;
    Xbyak::Ymm yd = ymm7;
    Xbyak::Ymm ye = ymm8;
    Xbyak::Ymm ysum = ymm9;
    Xbyak::Ymm ysum2 = ymm10;
    Xbyak::Ymm ydst = ymm11;
    Xbyak::Ymm ybase = ymm12;

    this->preamble();

    mov(src_, ptr[this->param1 + 0]);
    mov(dst_, ptr[this->param1 + 8]);
    if (pk != prop_kind::forward_inference)
        mov(scratch_, ptr[this->param1 + 16]);
    sub(t, 64);
    mov(imm_addr64_, float2int(this->alpha_));
    movq(xalpha_, imm_addr64_);
    vbroadcastss(valpha_, xalpha_);

    mov(imm_addr64_, float2int(this->k_));
    movq(xk_, imm_addr64_);
    vbroadcastss(yk_, xk_);

    if (J.version == -1) {
        vxorps(xsrc_prev, xsrc_prev, xsrc_prev);
        vmovups(ptr[t + 0], xsrc_prev);
    }
    if (J.version == +1) {
        vxorps(xsrc_next, xsrc_next, xsrc_next);
        vmovups(ptr[t + 48], xsrc_next);
    }

    mov(hw, J.H * J.W);

    Label lrn_loop;
    L(lrn_loop);

    if (J.version != -1) vmovups(xsrc_prev, ptr[src_ - J.H * J.W * 32 + 16]);
    vmovups(ysrc, ptr[src_]);
    if (J.version != +1) vmovups(xsrc_next, ptr[src_ + J.H * J.W * 32]);

    if (J.version != -1) vmovups(ptr[t + 0], xsrc_prev);
    vmovups(ptr[t + 16], ysrc);
    if (J.version != +1) vmovups(ptr[t + 48], xsrc_next);

    vmovups(ya, ptr[t + 16 - 8]);
    vmovups(yb, ptr[t + 16 - 4]);
    vmovups(yd, ptr[t + 16 + 4]);
    vmovups(ye, ptr[t + 16 + 8]);
    vmulps(ysum, yc, yc);
    vfmadd231ps(ysum, ya, ya); // ysum <- ysum + ya*ya
    vfmadd231ps(ysum, yb, yb);
    vfmadd231ps(ysum, yd, yd);
    vfmadd231ps(ysum, ye, ye);
    vfmadd132ps(ysum, yk_, valpha_); // ysum <- ysum*valpha_+yk_

    vmovaps(ybase, ysum);
    if (pk != prop_kind::forward_inference) vmovups(ptr[scratch_], ybase);
    vmulps(ysum2, ysum, ysum);
    vmulps(ysum, ysum, ysum2); // ysum = ybase^3;
    vsqrtps(ysum, ysum);
    vsqrtps(ysum, ysum); // ysum = ybase^0.75
    vdivps(ydst, ysrc, ysum); // ydst = ysrc / ysum
    vmovups(ptr[dst_], ydst);

    add(src_, 32);
    add(dst_, 32);
    if (pk != prop_kind::forward_inference) add(scratch_, 32);
    dec(hw);
    cmp(hw, 0);
    jne(lrn_loop, T_NEAR);

    add(t, 64);
    this->postamble();

    ker = reinterpret_cast<decltype(ker)>(
            const_cast<uint8_t *>(this->getCode()));
}

template <>
jit_uni_lrn_fwd_kernel<sse41, dnnl::impl::data_type::f32>::
        jit_uni_lrn_fwd_kernel(const struct nchw8c_across &J, float A, float K,
                prop_kind_t pk, void *code_ptr, size_t code_size)
    : jit_generator(code_ptr, code_size), bf16_emu_(nullptr), alpha_(A), k_(K) {
    Xbyak::Reg64 t = rsp;
    Xbyak::Reg64 hw = r9;

    Xbyak::Xmm xsrc_lo = xmm2;
    Xbyak::Xmm xsrc_hi = xmm3;
    Xbyak::Xmm xc_lo = xmm4;
    Xbyak::Xmm xc_hi = xmm5;
    Xbyak::Xmm xsum_lo = xc_lo;
    Xbyak::Xmm xsum_hi = xc_hi;
    Xbyak::Xmm xsrc_prev = xmm6;
    Xbyak::Xmm xsrc_next = xmm7;
    Xbyak::Xmm xa_lo = xmm8;
    Xbyak::Xmm xa_hi = xmm9;
    Xbyak::Xmm xb_lo = xmm10;
    Xbyak::Xmm xb_hi = xmm11;
    Xbyak::Xmm xd_lo = xmm12;
    Xbyak::Xmm xd_hi = xmm13;
    Xbyak::Xmm xe_lo = xmm14;
    Xbyak::Xmm xe_hi = xmm15;
    Xbyak::Xmm xbase_lo = xmm14;
    Xbyak::Xmm xbase_hi = xmm15;

    this->preamble();

    mov(src_, ptr[this->param1 + 0]);
    mov(dst_, ptr[this->param1 + 8]);
    if (pk != prop_kind::forward_inference)
        mov(scratch_, ptr[this->param1 + 16]);
    sub(t, 64);
    mov(imm_addr64_, float2int(this->alpha_));
    movq(xalpha_, imm_addr64_);
    shufps(xalpha_, xalpha_, 0);

    mov(imm_addr64_, float2int(this->k_));
    movq(xk_, imm_addr64_);
    shufps(xk_, xk_, 0);

    if (J.version == -1) {
        xorps(xsrc_prev, xsrc_prev);
        movups(ptr[t + 0], xsrc_prev);
    }
    if (J.version == +1) {
        xorps(xsrc_next, xsrc_next);
        movups(ptr[t + 48], xsrc_next);
    }

    mov(hw, J.H * J.W);
    Label lrn_loop;
    L(lrn_loop);

    if (J.version != -1) movups(xsrc_prev, ptr[src_ - J.H * J.W * 32 + 16]);
    movups(xsrc_lo, ptr[src_]);
    movups(xsrc_hi, ptr[src_ + 4 * sizeof(float)]);
    if (J.version != +1) movups(xsrc_next, ptr[src_ + J.H * J.W * 32]);

    if (J.version != -1) movups(ptr[t + 0], xsrc_prev);
    movups(ptr[t + 16], xsrc_lo);
    movups(ptr[t + 16 + 4 * sizeof(float)], xsrc_hi);
    if (J.version != +1) movups(ptr[t + 48], xsrc_next);

    movups(xa_lo, ptr[t + 16 - 8]);
    movups(xa_hi, ptr[t + 16 - 8 + 4 * sizeof(float)]);
    movups(xb_lo, ptr[t + 16 - 4]);
    movups(xb_hi, ptr[t + 16 - 4 + 4 * sizeof(float)]);
    movups(xd_lo, ptr[t + 16 + 4]);
    movups(xd_hi, ptr[t + 16 + 4 + 4 * sizeof(float)]);
    movups(xe_lo, ptr[t + 16 + 8]);
    movups(xe_hi, ptr[t + 16 + 8 + 4 * sizeof(float)]);
    movaps(xc_lo, xsrc_lo);
    movaps(xc_hi, xsrc_hi);
    mulps(xsum_lo, xc_lo);
    mulps(xsum_hi, xc_hi);
    mulps(xa_lo, xa_lo);
    mulps(xa_hi, xa_hi);
    addps(xsum_lo, xa_lo);
    addps(xsum_hi, xa_hi); // xsum <- xsum + xa*xa
    mulps(xb_lo, xb_lo);
    mulps(xb_hi, xb_hi);
    addps(xsum_lo, xb_lo);
    addps(xsum_hi, xb_hi);
    mulps(xd_lo, xd_lo);
    mulps(xd_hi, xd_hi);
    addps(xsum_lo, xd_lo);
    addps(xsum_hi, xd_hi);
    mulps(xe_lo, xe_lo);
    mulps(xe_hi, xe_hi);
    addps(xsum_lo, xe_lo);
    addps(xsum_hi, xe_hi);

    mulps(xsum_lo, xalpha_);
    mulps(xsum_hi, xalpha_);
    addps(xsum_lo, xk_);
    addps(xsum_hi, xk_); // xsum <- xsum*xalpha_+xk_

    movaps(xbase_lo, xsum_lo);
    movaps(xbase_hi, xsum_hi);
    if (pk != prop_kind::forward_inference) {
        movups(ptr[scratch_], xbase_lo);
        movups(ptr[scratch_ + 4 * sizeof(float)], xbase_hi);
    }
    mulps(xsum_lo, xsum_lo);
    mulps(xsum_hi, xsum_hi);
    mulps(xsum_lo, xbase_lo);
    mulps(xsum_hi, xbase_hi); // xsum = xbase^3;
    sqrtps(xsum_lo, xsum_lo);
    sqrtps(xsum_hi, xsum_hi);
    sqrtps(xsum_lo, xsum_lo);
    sqrtps(xsum_hi, xsum_hi); // xsum = xbase^0.75
    divps(xsrc_lo, xsum_lo);
    divps(xsrc_hi, xsum_hi); // xdst = xsrc / xsum
    movups(ptr[dst_], xsrc_lo);
    movups(ptr[dst_ + 4 * sizeof(float)], xsrc_hi);

    add(src_, 32);
    add(dst_, 32);
    if (pk != prop_kind::forward_inference) add(scratch_, 32);
    dec(hw);
    cmp(hw, 0);
    jne(lrn_loop, T_NEAR);

    add(t, 64);
    this->postamble();

    ker = reinterpret_cast<decltype(ker)>(
            const_cast<uint8_t *>(this->getCode()));
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_lrn_fwd_kernel<isa, d_type>::jit_uni_lrn_fwd_kernel(
        const struct nhwc_across &J, float A, float K, prop_kind_t pk,
        void *code_ptr, size_t code_size)
    : jit_generator(code_ptr, code_size), bf16_emu_(nullptr), alpha_(A), k_(K) {
    static const uint32_t mask[] = {0, 0, 0x80000000, 0x80000000, 0x80000000,
            0x80000000, 0x80000000, 0x80000000, 0x80000000, 0, 0};

    Xbyak::Reg64 c = r9;
    Xbyak::Ymm ya = ymm2;
    Xbyak::Ymm yb = ymm3;
    Xbyak::Ymm yc = ymm4;
    Xbyak::Ymm yd = ymm5;
    Xbyak::Ymm ye = ymm6;
    Xbyak::Ymm ysum = ymm7;
    Xbyak::Ymm ydst = ymm8;
    Xbyak::Ymm ybase = ymm9;
    Xbyak::Ymm ymask = ymm10;

    this->preamble();

    mov(src_, ptr[this->param1 + 0]);
    mov(dst_, ptr[this->param1 + 8]);
    if (pk != prop_kind::forward_inference)
        mov(scratch_, ptr[this->param1 + 16]);
    mov(imm_addr64_, float2int(this->alpha_));
    movq(xalpha_, imm_addr64_);
    vbroadcastss(valpha_, xalpha_);

    mov(imm_addr64_, float2int(this->k_));
    movq(xk_, imm_addr64_);
    vbroadcastss(yk_, xk_);

    vxorps(ysum, ysum, ysum);

    mov(imm_addr64_, reinterpret_cast<size_t>(&mask[0]));
    vmovups(ymask, ptr[imm_addr64_]);
    vmaskmovps(ya, ymask, ptr[src_ - 8]);
    vfmadd231ps(ysum, ya, ya); // ysum <- ysum + ya^2+yb^2+yc^2+yd^2+ye^2

    mov(imm_addr64_, reinterpret_cast<size_t>(&mask[1]));
    vmovups(ymask, ptr[imm_addr64_]);
    vmaskmovps(yb, ymask, ptr[src_ - 4]);
    vfmadd231ps(ysum, yb, yb);

    mov(c, J.C / 8 - 1);
    Label lrn_loop;
    L(lrn_loop);

    vmovups(yc, ptr[src_]);
    vmovups(yd, ptr[src_ + 4]);
    vmovups(ye, ptr[src_ + 8]);
    vfmadd231ps(ysum, yc, yc);
    vfmadd231ps(ysum, yd, yd);
    vfmadd231ps(ysum, ye, ye);

    vmovups(ydst, ysum);
    vfmadd132ps(ydst, yk_, valpha_); // ydst <- ysum*valpha_+yk_

    vmovaps(ybase, ydst);
    if (pk != prop_kind::forward_inference) vmovups(ptr[scratch_], ybase);
    vmulps(ydst, ydst, ydst);
    vmulps(ydst, ydst, ybase); // ydst = (ysum*valpha_+yk_)^3;
    vsqrtps(ydst, ydst);
    vsqrtps(ydst, ydst); // ydst = (ysum*valpha_+yk_)^0.75

    vdivps(ydst, yc, ydst); // ydst = ysrc / (ysum*valpha_+yk_)^0.75
    vmovups(ptr[dst_], ydst);

    vxorps(ysum, ysum, ysum);

    add(src_, 32);
    add(dst_, 32);
    if (pk != prop_kind::forward_inference) add(scratch_, 32);

    vmovups(ya, ptr[src_ - 8]);
    vfmadd231ps(ysum, ya, ya);
    vmovups(yb, ptr[src_ - 4]);
    vfmadd231ps(ysum, yb, yb);

    dec(c);
    cmp(c, 0);
    jne(lrn_loop, T_NEAR);

    vmovups(yc, ptr[src_]);
    vfmadd231ps(ysum, yc, yc);

    mov(imm_addr64_, reinterpret_cast<size_t>(&mask[2]));
    vmovups(ymask, ptr[imm_addr64_]);
    vmaskmovps(yd, ymask, ptr[src_ + 4]);
    vfmadd231ps(ysum, yd, yd); // ysum <- ysum + ya^2+yb^2+yc^2+yd^2+ye^2

    mov(imm_addr64_, reinterpret_cast<size_t>(&mask[3]));
    vmovups(ymask, ptr[imm_addr64_]);
    vmaskmovps(ye, ymask, ptr[src_ + 8]);
    vfmadd231ps(ysum, ye, ye);

    vmovups(ydst, ysum);
    vfmadd132ps(ydst, yk_, valpha_); // ydst <- ysum*valpha_+yk_

    vmovaps(ybase, ydst);
    if (pk != prop_kind::forward_inference) vmovups(ptr[scratch_], ybase);
    vmulps(ydst, ydst, ydst);
    vmulps(ydst, ydst, ybase); // ydst = (ysum*valpha_+yk_)^3;
    vsqrtps(ydst, ydst);
    vsqrtps(ydst, ydst); // ydst = (ysum*valpha_+yk_)^0.75
    vdivps(ydst, yc, ydst); // ydst = ysrc / (ysum*valpha_+yk_)^0.75

    vmovups(ptr[dst_], ydst);

    this->postamble();

    ker = reinterpret_cast<decltype(ker)>(
            const_cast<uint8_t *>(this->getCode()));
}

template <>
jit_uni_lrn_fwd_kernel<sse41, dnnl::impl::data_type::f32>::
        jit_uni_lrn_fwd_kernel(const struct nhwc_across &J, float A, float K,
                prop_kind_t pk, void *code_ptr, size_t code_size)
    : jit_generator(code_ptr, code_size), bf16_emu_(nullptr), alpha_(A), k_(K) {

    static uint32_t store[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    Xbyak::Reg64 c = r9;

    Xbyak::Xmm xdst_lo = xmm0;
    Xbyak::Xmm xdst_hi = xmm1;
    Xbyak::Xmm xa_lo = xmm2;
    Xbyak::Xmm xa_hi = xmm3;
    Xbyak::Xmm xb_lo = xmm2;
    Xbyak::Xmm xb_hi = xmm3;
    Xbyak::Xmm xc_lo = xmm4;
    Xbyak::Xmm xc_hi = xmm5;
    Xbyak::Xmm xd_lo = xmm6;
    Xbyak::Xmm xd_hi = xmm7;
    Xbyak::Xmm xe_lo = xmm8;
    Xbyak::Xmm xe_hi = xmm9;
    Xbyak::Xmm xsum_lo = xmm10;
    Xbyak::Xmm xsum_hi = xmm11;
    // unused: xmm12, xmm13;
    Xbyak::Xmm xbase_lo = xmm14;
    Xbyak::Xmm xbase_hi = xmm15;

    this->preamble();

    mov(src_, ptr[this->param1 + 0]);
    mov(dst_, ptr[this->param1 + 8]);
    if (pk != prop_kind::forward_inference)
        mov(scratch_, ptr[this->param1 + 16]);
    mov(imm_addr64_, float2int(this->alpha_));
    movq(xalpha_, imm_addr64_);
    shufps(xalpha_, xalpha_, 0);

    mov(imm_addr64_, float2int(this->k_));
    movq(xk_, imm_addr64_);
    shufps(xk_, xk_, 0);

    mov(store_addr_, reinterpret_cast<size_t>(&store[0]));
    and_(store_addr_, -15);
    movups(ptr[store_addr_], xalpha_);
    movups(ptr[store_addr_ + 4 * sizeof(float)], xk_);

    xorps(xsum_lo, xsum_lo);
    xorps(xsum_hi, xsum_hi);

    /* load the 2 first blocks of channels
     * block:         | -- low -- | -- hi --  |
     * C:             [c1,c2,c3,c4,c5,c6,c7,c8]
     * xa_lo << 2 [0,0,c1,c2]
     * xa_hi                [c3,c4,c5,c6]
     * xb_lo << 1   [0,c1,c2,c3]
     * xb_hi                   [c4,c5,c6,c7]
     *                | --  data  --     (...)
     *                ^ memory boundary
     */
    movups(xa_lo, ptr[src_]);
    movups(xa_hi, ptr[src_ + 2 * sizeof(float)]);
    pslldq(xa_lo, 2 * sizeof(float));
    mulps(xa_lo, xa_lo);
    mulps(xa_hi, xa_hi);
    addps(xsum_lo, xa_lo);
    addps(xsum_hi, xa_hi); // xsum <- xsum + xa^2+xb^2+xc^2+xd^2+xe^2

    movups(xb_lo, ptr[src_]);
    movups(xb_hi, ptr[src_ + 3 * sizeof(float)]);
    pslldq(xb_lo, 1 * sizeof(float));
    mulps(xb_lo, xb_lo);
    mulps(xb_hi, xb_hi);
    addps(xsum_lo, xb_lo);
    addps(xsum_hi, xb_hi);

    mov(c, J.C / 8 - 1);
    Label lrn_loop;
    L(lrn_loop);

    movups(xc_lo, ptr[src_]);
    movups(xc_hi, ptr[src_ + 4 * sizeof(float)]);
    movups(xd_lo, ptr[src_ + 4]);
    movups(xd_hi, ptr[src_ + 4 + 4 * sizeof(float)]);
    movups(xe_lo, ptr[src_ + 8]);
    movups(xe_hi, ptr[src_ + 8 + 4 * sizeof(float)]);
    mulps(xc_lo, xc_lo);
    mulps(xc_hi, xc_hi);
    addps(xsum_lo, xc_lo);
    addps(xsum_hi, xc_hi);
    mulps(xd_lo, xd_lo);
    mulps(xd_hi, xd_hi);
    addps(xsum_lo, xd_lo);
    addps(xsum_hi, xd_hi);
    mulps(xe_lo, xe_lo);
    mulps(xe_hi, xe_hi);
    addps(xsum_lo, xe_lo);
    addps(xsum_hi, xe_hi);

    movaps(xdst_lo, xsum_lo);
    movaps(xdst_hi, xsum_hi);
    // xdst <- xsum*xalpha_+xk_
    mulps(xdst_lo, ptr[store_addr_]);
    mulps(xdst_hi, ptr[store_addr_]);
    addps(xdst_lo, ptr[store_addr_ + 4 * sizeof(float)]);
    addps(xdst_hi, ptr[store_addr_ + 4 * sizeof(float)]);

    movaps(xbase_lo, xdst_lo);
    movaps(xbase_hi, xdst_hi);
    if (pk != prop_kind::forward_inference) {
        movups(ptr[scratch_], xbase_lo);
        movups(ptr[scratch_ + 4 * sizeof(float)], xbase_hi);
    }
    mulps(xdst_lo, xdst_lo);
    mulps(xdst_hi, xdst_hi);
    mulps(xdst_lo, xbase_lo);
    mulps(xdst_hi, xbase_hi); // xdst = (xsum*xalpha_+xk_)^3;
    sqrtps(xdst_lo, xdst_lo);
    sqrtps(xdst_hi, xdst_hi);
    sqrtps(xdst_lo, xdst_lo);
    sqrtps(xdst_hi, xdst_hi); // xdst = (xsum*xalpha_+xk_)^0.75

    movups(xc_lo, ptr[src_]);
    movups(xc_hi, ptr[src_ + 4 * sizeof(float)]);
    divps(xc_lo, xdst_lo);
    divps(xc_hi, xdst_hi); // xdst = xsrc / (xsum*xalpha_+xk_)^0.75
    movups(ptr[dst_], xc_lo);
    movups(ptr[dst_ + 4 * sizeof(float)], xc_hi);

    xorps(xsum_lo, xsum_lo);
    xorps(xsum_hi, xsum_hi);

    add(src_, 32);
    add(dst_, 32);
    if (pk != prop_kind::forward_inference) add(scratch_, 32);

    movups(xa_lo, ptr[src_ - 8]);
    movups(xa_hi, ptr[src_ - 8 + 4 * sizeof(float)]);
    mulps(xa_lo, xa_lo);
    mulps(xa_hi, xa_hi);
    addps(xsum_lo, xa_lo);
    addps(xsum_hi, xa_hi);
    movups(xb_lo, ptr[src_ - 4]);
    movups(xb_hi, ptr[src_ - 4 + 4 * sizeof(float)]);
    mulps(xb_lo, xb_lo);
    mulps(xb_hi, xb_hi);
    addps(xsum_lo, xb_lo);
    addps(xsum_hi, xb_hi);

    dec(c);
    cmp(c, 0);
    jne(lrn_loop, T_NEAR);

    /* compute last 3 blocks of channels:
     * block:       | -- low -- | -- hi --  |
     * C:           [c1,c2,c3,c4,c5,c6,c7,c8]
     * xc_lo|xc_hi  [c1,c2,c3,c4|c5,c6,c7,c8]
     * xd_lo           [c2,c3,c4,c5]
     * xd_hi >> 1                  [c6,c7,c8, 0]
     * xe_lo              [c3,c4,c5,c6]
     * xe_hi >> 2                     [c7,c8, 0, 0]
     *                  (...) --  data  --  | -- illegal reading -- (...)
     *                                      ^ memory boundary
     */
    movups(xc_lo, ptr[src_]);
    movups(xc_hi, ptr[src_ + 4 * sizeof(float)]);
    mulps(xc_lo, xc_lo);
    mulps(xc_hi, xc_hi);
    addps(xsum_lo, xc_lo);
    addps(xsum_hi, xc_hi);

    movups(xd_lo, ptr[src_ + 1 * sizeof(float)]);
    movups(xd_hi, ptr[src_ + 4 * sizeof(float)]);
    psrldq(xd_hi, 1 * sizeof(float));
    mulps(xd_lo, xd_lo);
    mulps(xd_hi, xd_hi);
    addps(xsum_lo, xd_lo);
    addps(xsum_hi, xd_hi); // xsum <- xsum + xa^2+xb^2+xc^2+xd^2+xe^2

    movups(xe_lo, ptr[src_ + 2 * sizeof(float)]);
    movups(xe_hi, ptr[src_ + 4 * sizeof(float)]);
    psrldq(xe_hi, 2 * sizeof(float));
    mulps(xe_lo, xe_lo);
    mulps(xe_hi, xe_hi);
    addps(xsum_lo, xe_lo);
    addps(xsum_hi, xe_hi);

    movups(xdst_lo, xsum_lo);
    movups(xdst_hi, xsum_hi);
    // xdst <- xsum*xalpha_+xk_
    mulps(xdst_lo, ptr[store_addr_]);
    mulps(xdst_hi, ptr[store_addr_]);
    addps(xdst_lo, ptr[store_addr_ + 4 * sizeof(float)]);
    addps(xdst_hi, ptr[store_addr_ + 4 * sizeof(float)]);

    movaps(xbase_lo, xdst_lo);
    movaps(xbase_hi, xdst_hi);
    if (pk != prop_kind::forward_inference) {
        movups(ptr[scratch_], xbase_lo);
        movups(ptr[scratch_ + 4 * sizeof(float)], xbase_hi);
    }
    mulps(xdst_lo, xdst_lo);
    mulps(xdst_hi, xdst_hi);
    mulps(xdst_lo, xbase_lo);
    mulps(xdst_hi, xbase_hi); // xdst = (xsum*xalpha_+xk_)^3;
    sqrtps(xdst_lo, xdst_lo);
    sqrtps(xdst_hi, xdst_hi);
    sqrtps(xdst_lo, xdst_lo);
    sqrtps(xdst_hi, xdst_hi); // xdst = (xsum*xalpha_+xk_)^0.75
    movups(xc_lo, ptr[src_]);
    movups(xc_hi, ptr[src_ + 4 * sizeof(float)]);
    divps(xc_lo, xdst_lo);
    divps(xc_hi, xdst_hi); // xdst = xsrc / (xsum*xalpha_+xk_)^0.75

    movups(ptr[dst_], xc_lo);
    movups(ptr[dst_ + 4 * sizeof(float)], xc_hi);

    this->postamble();

    ker = reinterpret_cast<decltype(ker)>(
            const_cast<uint8_t *>(this->getCode()));
}

template <>
void jit_uni_lrn_fwd_kernel<sse41, dnnl::impl::data_type::f32>::nchw_body(
        int tail, int HW, prop_kind_t pk, Xbyak::Ymm ymask, Xbyak::Ymm ya,
        Xbyak::Ymm yb, Xbyak::Ymm yc, Xbyak::Ymm yd, Xbyak::Ymm ye,
        Xbyak::Ymm ysum) {}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_fwd_kernel<isa, d_type>::nchw_body(int tail, int HW,
        prop_kind_t pk, Xbyak::Ymm ymask, Xbyak::Ymm ya, Xbyak::Ymm yb,
        Xbyak::Ymm yc, Xbyak::Ymm yd, Xbyak::Ymm ye, Xbyak::Ymm ysum) {
    Xbyak::Ymm ydst = ymm14;
    Xbyak::Ymm ybase = ymm15;

    vfmadd231ps(ysum, ye, ye);

    vmovups(ydst, ysum);
    vfmadd132ps(ydst, yk_, valpha_); // ydst <- ysum*valpha_+yk_

    vmovaps(ybase, ydst);
    if (pk != prop_kind::forward_inference) {
        if (tail != 0)
            vmaskmovps(ptr[scratch_], ymask, ybase);
        else
            vmovups(ptr[scratch_], ybase);
    }
    vmulps(ydst, ydst, ydst);
    vmulps(ydst, ydst, ybase); // ydst = (ysum*valpha_+yk_)^3;
    vsqrtps(ydst, ydst);
    vsqrtps(ydst, ydst); // ydst = (ysum*valpha_+yk_)^0.75
    vdivps(ydst, yc, ydst); // ydst = ysrc / (ysum*valpha_+yk_)^0.75

    if (tail != 0)
        vmaskmovps(ptr[dst_], ymask, ydst);
    else
        vmovups(ptr[dst_], ydst);

    vfnmadd231ps(ysum, ya, ya);
    vmovups(ya, yb);
    vmovups(yb, yc);
    vmovups(yc, yd);
    vmovups(yd, ye);
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_fwd_kernel<isa, d_type>::nchw_tail_sse41(int tail,
        Xbyak::Reg64 reg_dst, Xbyak::Xmm xtail_lo, Xbyak::Xmm xtail_hi) {}

template <>
void jit_uni_lrn_fwd_kernel<sse41, dnnl::impl::data_type::f32>::nchw_tail_sse41(
        int tail, Xbyak::Reg64 reg_dst, Xbyak::Xmm xtail_lo,
        Xbyak::Xmm xtail_hi) {
    Xbyak::Xmm xmm_tmp = xmm10;
    movaps(xmm_tmp, xtail_hi);

    if (tail > 3) {
        /* Store upper-half directly */
        movups(ptr[reg_dst + (tail - 4) * sizeof(float)], xtail_hi);
        movaps(xmm_tmp, xtail_lo);
        tail -= 4;
    }
    if (tail > 0) {
        /* Store on a single-element basis when 'tail' overlaps
         * with 'src_' */
        psrldq(xmm_tmp, (4 - tail) * sizeof(float));
        movss(ptr[reg_dst], xmm_tmp);

        for (int i = 1; i < tail; i++) {
            psrldq(xmm_tmp, sizeof(float));
            movss(ptr[reg_dst + i * sizeof(float)], xmm_tmp);
        }
    }
}

template <>
void jit_uni_lrn_fwd_kernel<sse41, dnnl::impl::data_type::f32>::nchw_body_sse41(
        int tail, int HW, prop_kind_t pk, Xbyak::Xmm xe_lo, Xbyak::Xmm xe_hi,
        Xbyak::Xmm xsum_lo, Xbyak::Xmm xsum_hi) {
    Xbyak::Xmm xdst_lo = xmm0;
    Xbyak::Xmm xdst_hi = xmm1;
    Xbyak::Xmm xbase_lo = xmm6;
    Xbyak::Xmm xbase_hi = xmm7;
    Xbyak::Xmm xtmp_lo = xmm8;
    Xbyak::Xmm xtmp_hi = xmm9;
    Xbyak::Xmm xa_lo = xmm6;
    Xbyak::Xmm xa_hi = xmm7;
    Xbyak::Xmm xb_lo = xmm8;
    Xbyak::Xmm xb_hi = xmm9;
    Xbyak::Xmm xc_lo = xmm10;
    Xbyak::Xmm xc_hi = xmm11;
    Xbyak::Xmm xd_lo = xmm12;
    Xbyak::Xmm xd_hi = xmm13;

    // store xe
    movaps(ptr[store_addr_ + 10 * 4 * sizeof(float)], xe_lo);
    movaps(ptr[store_addr_ + 11 * 4 * sizeof(float)], xe_hi);

    mulps(xe_lo, xe_lo);
    mulps(xe_hi, xe_hi);
    addps(xsum_lo, xe_lo);
    addps(xsum_hi, xe_hi);

    // xdst <- xsum*xalpha_+xk_
    movaps(xdst_lo, xsum_lo);
    movaps(xdst_hi, xsum_hi);
    mulps(xdst_lo, ptr[store_addr_ + 0 * 4 * sizeof(float)]);
    mulps(xdst_hi, ptr[store_addr_ + 0 * 4 * sizeof(float)]);
    addps(xdst_lo, ptr[store_addr_ + 1 * 4 * sizeof(float)]);
    addps(xdst_hi, ptr[store_addr_ + 1 * 4 * sizeof(float)]);

    movaps(xbase_lo, xdst_lo);
    movaps(xbase_hi, xdst_hi);
    if (pk != prop_kind::forward_inference) {
        if (tail != 0) {
            nchw_tail_sse41(tail, scratch_, xbase_lo, xbase_hi);
        } else {
            movups(ptr[scratch_], xbase_lo);
            movups(ptr[scratch_ + 4 * sizeof(float)], xbase_hi);
        }
    }
    mulps(xdst_lo, xdst_lo);
    mulps(xdst_hi, xdst_hi);
    mulps(xdst_lo, xbase_lo);
    mulps(xdst_hi, xbase_hi); // xdst = (xsum*xalpha_+xk_)^3;
    sqrtps(xdst_lo, xdst_lo);
    sqrtps(xdst_hi, xdst_hi);
    sqrtps(xdst_lo, xdst_lo);
    sqrtps(xdst_hi, xdst_hi); // xdst = (xsum*xalpha_+xk_)^0.75
    movaps(xtmp_lo, ptr[store_addr_ + 6 * 4 * sizeof(float)]);
    movaps(xtmp_hi, ptr[store_addr_ + 7 * 4 * sizeof(float)]);
    divps(xtmp_lo, xdst_lo);
    divps(xtmp_hi, xdst_hi); // xdst = xsrc / (xsum*xalpha_+xk_)^0.75
    movaps(xdst_lo, xtmp_lo);
    movaps(xdst_hi, xtmp_hi);

    if (tail != 0) {
        nchw_tail_sse41(tail, dst_, xdst_lo, xdst_hi);
    } else {
        movups(ptr[dst_], xdst_lo);
        movups(ptr[dst_ + 4 * sizeof(float)], xdst_hi);
    }

    movaps(xa_lo, ptr[store_addr_ + 2 * 4 * sizeof(float)]);
    movaps(xa_hi, ptr[store_addr_ + 3 * 4 * sizeof(float)]);
    mulps(xa_lo, xa_lo);
    mulps(xa_hi, xa_hi);
    subps(xsum_lo, xa_lo);
    subps(xsum_hi, xa_hi);

    // xa <- xb
    movaps(xb_lo, ptr[store_addr_ + 4 * 4 * sizeof(float)]);
    movaps(xb_hi, ptr[store_addr_ + 5 * 4 * sizeof(float)]);
    movaps(ptr[store_addr_ + 2 * 4 * sizeof(float)], xb_lo);
    movaps(ptr[store_addr_ + 3 * 4 * sizeof(float)], xb_hi);

    // xb <- xc
    movaps(xc_lo, ptr[store_addr_ + 6 * 4 * sizeof(float)]);
    movaps(xc_hi, ptr[store_addr_ + 7 * 4 * sizeof(float)]);
    movaps(ptr[store_addr_ + 4 * 4 * sizeof(float)], xc_lo);
    movaps(ptr[store_addr_ + 5 * 4 * sizeof(float)], xc_hi);

    // xc <- xd
    movaps(xd_lo, ptr[store_addr_ + 8 * 4 * sizeof(float)]);
    movaps(xd_hi, ptr[store_addr_ + 9 * 4 * sizeof(float)]);
    movaps(ptr[store_addr_ + 6 * 4 * sizeof(float)], xd_lo);
    movaps(ptr[store_addr_ + 7 * 4 * sizeof(float)], xd_hi);

    // xd <- xe
    movaps(xe_lo, ptr[store_addr_ + 10 * 4 * sizeof(float)]);
    movaps(xe_hi, ptr[store_addr_ + 11 * 4 * sizeof(float)]);
    movaps(ptr[store_addr_ + 8 * 4 * sizeof(float)], xe_lo);
    movaps(ptr[store_addr_ + 9 * 4 * sizeof(float)], xe_hi);
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_fwd_kernel<isa, d_type>::nchw_body_sse41(int tail, int HW,
        prop_kind_t pk, Xbyak::Xmm xe_lo, Xbyak::Xmm xe_hi, Xbyak::Xmm xsum_lo,
        Xbyak::Xmm xsum_hi) {}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_lrn_fwd_kernel<isa, d_type>::jit_uni_lrn_fwd_kernel(
        const nchw_across &J, float A, float K, prop_kind_t pk, void *code_ptr,
        size_t code_size)
    : jit_generator(code_ptr, code_size), alpha_(A), k_(K) {
    static const uint32_t mask[]
            = {0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000,
                    0x80000000, 0x80000000, 0, 0, 0, 0, 0, 0, 0};
    Xbyak::Reg64 c = r10;
    Xbyak::Ymm ymask = ymm2;
    Xbyak::Ymm ye = ymm3;
    Xbyak::Ymm ya = ymm4;
    Xbyak::Ymm yb = ymm5;
    Xbyak::Ymm yc = ymm6;
    Xbyak::Ymm yd = ymm7;
    Xbyak::Ymm ysum = ymm8;

    this->preamble();

    if (J.tail != 0) {
        mov(imm_addr64_, reinterpret_cast<size_t>(&mask[7 - J.tail]));
        vmovups(ymask, ptr[imm_addr64_]);
    }
    mov(imm_addr64_, float2int(this->alpha_));
    movq(xalpha_, imm_addr64_);
    vbroadcastss(valpha_, xalpha_);

    mov(imm_addr64_, float2int(this->k_));
    movq(xk_, imm_addr64_);
    vbroadcastss(yk_, xk_);

    mov(src_, ptr[this->param1 + 0]);
    mov(dst_, ptr[this->param1 + 8]);
    if (pk != prop_kind::forward_inference)
        mov(scratch_, ptr[this->param1 + 16]);

    vxorps(ya, ya, ya);
    vxorps(yb, yb, yb);
    if (J.tail != 0)
        vmaskmovps(yc, ymask, ptr[src_ + J.HW * 0]);
    else
        vmovups(yc, ptr[src_ + J.HW * 0]);
    if (J.tail != 0)
        vmaskmovps(yd, ymask, ptr[src_ + J.HW * 4]);
    else
        vmovups(yd, ptr[src_ + J.HW * 4]);

    vxorps(ysum, ysum, ysum);
    vfmadd231ps(ysum, yc, yc); // ysum <- ysum + ya^2+yb^2+yc^2+yd^2+ye^2
    vfmadd231ps(ysum, yd, yd);

    mov(c, J.C - 2);
    Label lrn_loop;
    L(lrn_loop);

    if (J.tail != 0)
        vmaskmovps(ye, ymask, ptr[src_ + J.HW * 8]);
    else
        vmovups(ye, ptr[src_ + J.HW * 8]);

    nchw_body(J.tail, J.HW, pk, ymask, ya, yb, yc, yd, ye, ysum);

    add(src_, J.HW * 4);
    add(dst_, J.HW * 4);
    if (pk != prop_kind::forward_inference) add(scratch_, J.HW * 4);
    dec(c);
    cmp(c, 0);
    jne(lrn_loop, T_NEAR);

    vxorps(ye, ye, ye);

    nchw_body(J.tail, J.HW, pk, ymask, ya, yb, yc, yd, ye, ysum);
    add(src_, J.HW * 4);
    add(dst_, J.HW * 4);
    if (pk != prop_kind::forward_inference) add(scratch_, J.HW * 4);

    nchw_body(J.tail, J.HW, pk, ymask, ya, yb, yc, yd, ye, ysum);

    this->postamble();

    ker = reinterpret_cast<decltype(ker)>(
            const_cast<uint8_t *>(this->getCode()));
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_lrn_fwd_kernel<isa, d_type>::~jit_uni_lrn_fwd_kernel() = default;

template <>
jit_uni_lrn_fwd_kernel<sse41,
        dnnl::impl::data_type::f32>::jit_uni_lrn_fwd_kernel(const nchw_across
                                                                    &J,
        float A, float K, prop_kind_t pk, void *code_ptr, size_t code_size)
    : jit_generator(code_ptr, code_size), alpha_(A), k_(K) {

    /* Load from within the memory boundary of 'src_' and apply a zero-mask to
     * the 'x_hi' register:
     *  block:       src_  |tail = 3
     *  src_:      [x,x,x,x|a,b,c]
     *  x_hi:           [x,a,b,c]
     *  mask:           [0,1,1,1]
     *      (...) --  data  --  | -- illegal reading -- (...)
     *                          ^ memory boundary
     *
     * 'x_lo' is loaded with the elements between 'src_' and 'x_hi' when
     * tail.size is between [5:7]. The register is then left-shifted to
     * clear the overlapping elements with 'x_hi'.
     *  block: - src_ - |  tail = 7
     *  src_:  (...) [x,|a,b,c,d,e,f,g]
     *  x_hi                 [d,e,f,g]
     *  x_lo           [a,b,c,d]
     *    x_lo >> 1: [0,a,b,c]
     *           (...) --  data  --  | -- illegal reading -- (...)
     *                               ^ memory boundary
     *
     *  - seg-fault happens if read occurs anywhere outside the
     *  memory boundary.
     * */
    static const uint32_t mask[]
            = {0, 0, 0, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff};
    assert(J.HW > 3);

    Xbyak::Reg64 c = r10;

    // unused: xmm2
    Xbyak::Xmm xmask_hi = xmm3;
    Xbyak::Xmm xsum_lo = xmm4;
    Xbyak::Xmm xsum_hi = xmm5;
    Xbyak::Xmm xa_lo = xmm6;
    Xbyak::Xmm xa_hi = xmm7;
    Xbyak::Xmm xb_lo = xmm8;
    Xbyak::Xmm xb_hi = xmm9;
    Xbyak::Xmm xc_lo = xmm10;
    Xbyak::Xmm xc_hi = xmm11;
    Xbyak::Xmm xd_lo = xmm12;
    Xbyak::Xmm xd_hi = xmm13;
    Xbyak::Xmm xe_lo = xmm14;
    Xbyak::Xmm xe_hi = xmm15;

    const int vlen = cpu_isa_traits<sse41>::vlen / sizeof(float);

    bool compute_tail = J.tail != 0;
    bool load_lo = J.tail == 0 || J.tail > 4;

    size_t h_offset = vlen;
    size_t l_shift = 0;

    this->preamble();

    mov(src_, ptr[this->param1 + 0]);
    mov(dst_, ptr[this->param1 + 8]);
    if (pk != prop_kind::forward_inference)
        mov(scratch_, ptr[this->param1 + 16]);

    sub(rsp, stack_space_needed_);
    mov(store_addr_, rsp);
    and_(store_addr_, -15);

    mov(imm_addr64_, float2int(this->alpha_));
    movq(xalpha_, imm_addr64_);
    shufps(xalpha_, xalpha_, 0);

    mov(imm_addr64_, float2int(this->k_));
    movq(xk_, imm_addr64_);
    shufps(xk_, xk_, 0);

    // put alpha_ and k_ into store (free up regs)
    movaps(ptr[store_addr_ + 0 * 4 * sizeof(float)], xalpha_);
    movaps(ptr[store_addr_ + 1 * 4 * sizeof(float)], xk_);

    if (compute_tail) {
        assert(J.tail > 0 && J.tail < 2 * vlen);
        h_offset = J.tail - vlen;
        l_shift = nstl::min(2 * vlen - J.tail, vlen);

        /* if 'tail' is between [1:3], need to zero-mask for underflow */
        size_t m_off = nstl::min(J.tail - 1, 3);
        mov(imm_addr64_, reinterpret_cast<size_t>(&mask[m_off]));
        movups(xmask_hi, ptr[imm_addr64_]);
    }
    // init xa, xb
    xorps(xa_lo, xa_lo);
    xorps(xa_hi, xa_hi);
    xorps(xb_lo, xb_lo);
    xorps(xb_hi, xb_hi);

    // read xc, xd
    if (load_lo) movups(xc_lo, ptr[src_ + J.HW * 0]);
    movups(xc_hi, ptr[src_ + J.HW * 0 + h_offset * sizeof(float)]);
    if (compute_tail) {
        pslldq(xc_lo, l_shift * sizeof(float));
        andps(xc_hi, xmask_hi);
    }

    if (load_lo) movups(xd_lo, ptr[src_ + J.HW * 4]);
    movups(xd_hi, ptr[src_ + J.HW * 4 + h_offset * sizeof(float)]);
    if (compute_tail) {
        pslldq(xd_lo, l_shift * sizeof(float));
        andps(xd_hi, xmask_hi);
    }

    // put xa, xb, xc, xd into store to free-up regs
    movaps(ptr[store_addr_ + 2 * 4 * sizeof(float)], xa_lo);
    movaps(ptr[store_addr_ + 3 * 4 * sizeof(float)], xa_hi);
    movaps(ptr[store_addr_ + 4 * 4 * sizeof(float)], xb_lo);
    movaps(ptr[store_addr_ + 5 * 4 * sizeof(float)], xb_hi);
    movaps(ptr[store_addr_ + 6 * 4 * sizeof(float)], xc_lo);
    movaps(ptr[store_addr_ + 7 * 4 * sizeof(float)], xc_hi);
    movaps(ptr[store_addr_ + 8 * 4 * sizeof(float)], xd_lo);
    movaps(ptr[store_addr_ + 9 * 4 * sizeof(float)], xd_hi);

    xorps(xsum_lo, xsum_lo);
    xorps(xsum_hi, xsum_hi);
    mulps(xc_lo, xc_lo);
    mulps(xc_hi, xc_hi);
    addps(xsum_lo, xc_lo);
    addps(xsum_hi, xc_hi);
    mulps(xd_lo, xd_lo);
    mulps(xd_hi, xd_hi);
    addps(xsum_lo, xd_lo);
    addps(xsum_hi, xd_hi); // xsum <- xsum + xa^2+xb^2+xc^2+xd^2+xe^2

    mov(c, J.C - 2);
    Label lrn_loop;
    L(lrn_loop);

    if (load_lo) movups(xe_lo, ptr[src_ + J.HW * 8]);
    movups(xe_hi, ptr[src_ + J.HW * 8 + h_offset * sizeof(float)]);
    if (compute_tail) {
        pslldq(xe_lo, l_shift * sizeof(float));
        andps(xe_hi, xmask_hi);
    }

    nchw_body_sse41(J.tail, J.HW, pk, xe_lo, xe_hi, xsum_lo, xsum_hi);

    add(src_, J.HW * 4);
    add(dst_, J.HW * 4);
    if (pk != prop_kind::forward_inference) add(scratch_, J.HW * 4);
    dec(c);
    cmp(c, 0);
    jne(lrn_loop, T_NEAR);

    xorps(xe_lo, xe_lo);
    xorps(xe_hi, xe_hi);

    nchw_body_sse41(J.tail, J.HW, pk, xe_lo, xe_hi, xsum_lo, xsum_hi);
    add(src_, J.HW * 4);
    add(dst_, J.HW * 4);
    if (pk != prop_kind::forward_inference) add(scratch_, J.HW * 4);

    nchw_body_sse41(J.tail, J.HW, pk, xe_lo, xe_hi, xsum_lo, xsum_hi);

    add(rsp, stack_space_needed_);

    this->postamble();

    ker = reinterpret_cast<decltype(ker)>(
            const_cast<uint8_t *>(this->getCode()));
}

//////////////////////////////////////////////////////////////////////////////
// backward kernel
template <cpu_isa_t isa>
jit_uni_lrn_bwd_kernel_f32<isa>::jit_uni_lrn_bwd_kernel_f32(
        const struct nchw8c_across &J, float A, float B, int use_h_parallel,
        void *code_ptr, size_t code_size)
    : jit_generator(code_ptr, code_size)
    , nalphabeta(-2 * A * B)
    , use_h_parallelizm(use_h_parallel) {
    Xbyak::Reg64 t = rsp;
    Xbyak::Reg64 hw = r10;

    Xbyak::Xmm xsrc_prev = xmm1;
    Xbyak::Xmm xws_prev = xmm2;
    Xbyak::Xmm xdiffdst_prev = xmm3;
    Xbyak::Ymm ysrc = ymm4;
    Xbyak::Ymm yws = ymm5;
    Xbyak::Ymm ydiffdst = ymm6;
    Xbyak::Xmm xsrc_next = xmm7;
    Xbyak::Xmm xws_next = xmm8;
    Xbyak::Xmm xdiffdst_next = xmm9;
    Xbyak::Ymm ya = ymm10;
    Xbyak::Xmm xa = xmm10;
    Xbyak::Ymm yb = ymm11;
    Xbyak::Ymm yd = ymm12;
    Xbyak::Ymm ye = ymm13;
    Xbyak::Ymm ysum = ymm14;
    Xbyak::Ymm ydiffsrc = ymm15;

    this->preamble();

    mov(src_, ptr[this->param1 + 0]);
    mov(diffdst, ptr[this->param1 + 8]);
    mov(workspace, ptr[this->param1 + 16]);
    mov(diffsrc, ptr[this->param1 + 24]);

    sub(t, 64);
    mov(imm_addr64_, float2int(this->nalphabeta));
    movq(xnalphabeta, imm_addr64_);
    vbroadcastss(ynalphabeta, xnalphabeta);

    bool is_single = J.version == 3;
    bool is_first = J.version == -1 || J.version == -2;
    bool is_last = J.version == +1 || J.version == -2;

    if (is_first || is_single) {
        vxorps(xsrc_prev, xsrc_prev, xsrc_prev);
        vmovups(ptr[t + 0], xsrc_prev);
    }
    if (is_last || is_single) {
        vxorps(xsrc_next, xsrc_next, xsrc_next);
        vmovups(ptr[t + 48], xsrc_next);
    }
    mov(hw, this->use_h_parallelizm ? J.W : J.H * J.W);
    Label lrn_loop;
    L(lrn_loop);
    {
        if (!is_first && !is_single) {
            vmovups(xws_prev, ptr[workspace - J.H * J.W * 32 + 16]);
            vmovups(xsrc_prev, ptr[src_ - J.H * J.W * 32 + 16]);
            vmovups(xdiffdst_prev, ptr[diffdst - J.H * J.W * 32 + 16]);
            vmulps(xa, xws_prev, xws_prev);
            vmulps(xa, xa, xws_prev);
            vsqrtps(xa, xa);
            vsqrtps(xa, xa);
            vmulps(xa, xa, xws_prev);
            vdivps(xsrc_prev, xsrc_prev, xa);
            vmulps(xdiffdst_prev, xdiffdst_prev, xsrc_prev);
        }

        vmovups(ysrc, ptr[src_]);
        vmovups(yws, ptr[workspace]);
        vmovups(ydiffdst, ptr[diffdst]);
        vmulps(ya, yws, yws);
        vmulps(ya, ya, yws);
        vsqrtps(ya, ya);
        vsqrtps(ya, ya);
        vdivps(ydiffsrc, ydiffdst, ya);
        vdivps(ysum, ydiffsrc, yws);
        vmulps(ysum, ysum, ysrc);

        if (!is_last && !is_single) {
            vmovups(xws_next, ptr[workspace + J.H * J.W * 32]);
            vmovups(xsrc_next, ptr[src_ + J.H * J.W * 32]);
            vmovups(xdiffdst_next, ptr[diffdst + J.H * J.W * 32]);
            vmulps(xa, xws_next, xws_next);
            vmulps(xa, xa, xws_next);
            vsqrtps(xa, xa);
            vsqrtps(xa, xa);
            vmulps(xa, xa, xws_next);
            vdivps(xsrc_next, xsrc_next, xa);
            vmulps(xdiffdst_next, xdiffdst_next, xsrc_next);
        }

        if (!is_first && !is_single) vmovups(ptr[t + 0], xdiffdst_prev);
        vmovups(ptr[t + 16], ysum);
        if (!is_last && !is_single) vmovups(ptr[t + 48], xdiffdst_next);

        vmovups(ya, ptr[t + 16 - 8]);
        vmovups(yb, ptr[t + 16 - 4]);
        vaddps(ysum, ysum, ya);
        vmulps(ysrc, ysrc, ynalphabeta);
        vaddps(ysum, ysum, yb);

        vmovups(yd, ptr[t + 16 + 4]);
        vmovups(ye, ptr[t + 16 + 8]);
        vaddps(ysum, ysum, yd);
        vaddps(ysum, ysum, ye);

        vfmadd231ps(ydiffsrc, ysum, ysrc);

        vmovups(ptr[diffsrc], ydiffsrc);

        add(src_, 32);
        add(diffsrc, 32);
        add(diffdst, 32);
        add(workspace, 32);

        dec(hw);
        cmp(hw, 0);
        jne(lrn_loop, T_NEAR);
    }

    add(t, 64);
    this->postamble();

    ker = reinterpret_cast<decltype(ker)>(
            const_cast<uint8_t *>(this->getCode()));
}

template struct jit_uni_lrn_fwd_kernel<sse41, dnnl::impl::data_type::f32>;
template struct jit_uni_lrn_fwd_kernel<avx2, dnnl::impl::data_type::f32>;
template struct jit_uni_lrn_fwd_kernel<avx512_common,
        dnnl::impl::data_type::f32>;
template struct jit_uni_lrn_fwd_kernel<avx512_common,
        dnnl::impl::data_type::bf16>;
template struct jit_uni_lrn_bwd_kernel_f32<avx2>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
