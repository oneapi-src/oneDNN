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

#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/gemm/s8x8s32/common_u8.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

jit_avx2_vnni_u8_copy_sum_an_kern::jit_avx2_vnni_u8_copy_sum_an_kern()
    : jit_generator(nullptr, U8_COPY_KERNEL_CODE_SIZE) {}

void jit_avx2_vnni_u8_copy_sum_an_kern::generate() {

#ifndef _WIN32
#define M rdi
#define N rsi
#define A rdx
#define LDA rcx
#define ALPHA r8
#define B r9

#define I rax
#define A1 r10
#define A2 r8
#define LDA3 r11

#define ARG_BIAS (24 + stacksize + rsp)

#else

#define M rcx
#define N rdx
#define A r8
#define LDA r9
#define ALPHA rax
#define B rdi

#define I rax
#define A1 rsi
#define A2 r10
#define LDA3 r11

#define ARG_ALPHA (40 + stacksize + rsp)
#define ARG_B (48 + stacksize + rsp)
#define ARG_BIAS (72 + stacksize + rsp)

#endif

    inLocalLabel();
    {

        Xbyak::Label l1cc;
        Xbyak::Label l20;
        Xbyak::Label l2a0;
        Xbyak::Label l318;
        Xbyak::Label l354;
        Xbyak::Label l360;
        Xbyak::Label l388;
        Xbyak::Label l488;
        Xbyak::Label l51c;
        Xbyak::Label l54;
        Xbyak::Label l570;
        Xbyak::Label l5a0;
        Xbyak::Label l5ac;
        Xbyak::Label l5cc;
        Xbyak::Label l6e8;
        Xbyak::Label l780;
        Xbyak::Label l7d4;
        Xbyak::Label l80c;
        Xbyak::Label l830;
        Xbyak::Label l83c;
        Xbyak::Label l854;
        Xbyak::Label l910;
        Xbyak::Label l974;
        Xbyak::Label l9b0;
        Xbyak::Label l9d4;
        Xbyak::Label l9f4;
        Xbyak::Label la00;
        Xbyak::Label la18;
        Xbyak::Label ladc;
        Xbyak::Label lb48;
        Xbyak::Label lb8c;
        Xbyak::Label lbb4;
        Xbyak::Label lbd2;
        Xbyak::Label lbdc;
        Xbyak::Label lbf4;
        Xbyak::Label lc8c;
        Xbyak::Label lcec;
        Xbyak::Label ld30;
        Xbyak::Label ld54;
        Xbyak::Label ld74;

        preamble();
        auto stacksize = get_size_of_abi_save_regs();
#ifdef _WIN32
        mov(ALPHA, ptr[ARG_ALPHA]);
        mov(B, ptr[ARG_B]);
#endif

        mov(M, qword[M]);
        mov(N, qword[N]);
        mov(LDA, qword[LDA]);
        lea(LDA3, ptr[LDA + LDA * 2]);
        sub(A, -128);
        sub(B, -128);
        cmp(N, 0x18);
        jl(l354, T_NEAR);
        align(4);

        L(l20);
        mov(A1, A);
        add(A, 0x18);
        pxor(xmm8, xmm8);
        pxor(xmm9, xmm9);
        pxor(xmm10, xmm10);
        pxor(xmm11, xmm11);
        pxor(xmm12, xmm12);
        pxor(xmm13, xmm13);
        mov(I, M);
        sar(I, 0x2);
        jle(l1cc, T_NEAR);
        align(4);

        L(l54);
        movdqu(xmm0, xword[A1 - 0x80]);
        movq(xmm4, qword[A1 - 0x70]);
        add(A1, LDA);
        movq(xmm6, qword[A1 - 0x70]);
        punpcklbw(xmm4, xmm6);
        movdqu(xmm6, xword[A1 - 0x80]);
        add(A1, LDA);
        movdqa(xmm2, xmm0);
        punpcklbw(xmm0, xmm6);
        punpckhbw(xmm2, xmm6);
        movdqu(xmm1, xword[A1 - 0x80]);
        movq(xmm5, qword[A1 - 0x70]);
        add(A1, LDA);
        movq(xmm6, qword[A1 - 0x70]);
        punpcklbw(xmm5, xmm6);
        movdqu(xmm6, xword[A1 - 0x80]);
        add(A1, LDA);
        movdqa(xmm3, xmm1);
        punpcklbw(xmm1, xmm6);
        punpckhbw(xmm3, xmm6);
        movdqa(xmm6, xmm0);
        punpcklwd(xmm0, xmm1);
        punpckhwd(xmm6, xmm1);
        pmovsxbw(xmm1, xmm0);
        movhlps(xmm7, xmm0);
        pmovsxbw(xmm7, xmm7);
        phaddw(xmm1, xmm7);
        phaddw(xmm1, xmm1);
        pmovsxwd(xmm1, xmm1);
        paddd(xmm8, xmm1);
        pmovsxbw(xmm1, xmm6);
        movhlps(xmm7, xmm6);
        pmovsxbw(xmm7, xmm7);
        phaddw(xmm1, xmm7);
        phaddw(xmm1, xmm1);
        pmovsxwd(xmm1, xmm1);
        paddd(xmm9, xmm1);
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x70], xmm6);
        movdqa(xmm6, xmm2);
        punpcklwd(xmm2, xmm3);
        punpckhwd(xmm6, xmm3);
        pmovsxbw(xmm1, xmm2);
        movhlps(xmm7, xmm2);
        pmovsxbw(xmm7, xmm7);
        phaddw(xmm1, xmm7);
        phaddw(xmm1, xmm1);
        pmovsxwd(xmm1, xmm1);
        paddd(xmm10, xmm1);
        pmovsxbw(xmm1, xmm6);
        movhlps(xmm7, xmm6);
        pmovsxbw(xmm7, xmm7);
        phaddw(xmm1, xmm7);
        phaddw(xmm1, xmm1);
        pmovsxwd(xmm1, xmm1);
        paddd(xmm11, xmm1);
        movdqu(xword[B - 0x60], xmm2);
        movdqu(xword[B - 0x50], xmm6);
        movdqa(xmm6, xmm4);
        punpcklwd(xmm4, xmm5);
        punpckhwd(xmm6, xmm5);
        pmovsxbw(xmm1, xmm4);
        movhlps(xmm7, xmm4);
        pmovsxbw(xmm7, xmm7);
        phaddw(xmm1, xmm7);
        phaddw(xmm1, xmm1);
        pmovsxwd(xmm1, xmm1);
        paddd(xmm12, xmm1);
        pmovsxbw(xmm1, xmm6);
        movhlps(xmm7, xmm6);
        pmovsxbw(xmm7, xmm7);
        phaddw(xmm1, xmm7);
        phaddw(xmm1, xmm1);
        pmovsxwd(xmm1, xmm1);
        paddd(xmm13, xmm1);
        movdqu(xword[B - 0x40], xmm4);
        movdqu(xword[B - 0x30], xmm6);
        sub(B, -96);
        dec(I);
        jg(l54, T_NEAR);
        align(4);

        L(l1cc);
        test(M, 0x2);
        jle(l2a0, T_NEAR);
        movdqu(xmm0, xword[A1 - 0x80]);
        movq(xmm1, qword[A1 - 0x70]);
        add(A1, LDA);
        movdqu(xmm2, xword[A1 - 0x80]);
        movq(xmm3, qword[A1 - 0x70]);
        add(A1, LDA);
        movdqa(xmm4, xmm0);
        punpcklbw(xmm0, xmm2);
        punpckhbw(xmm4, xmm2);
        punpcklbw(xmm1, xmm3);
        pmovsxbw(xmm5, xmm0);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm8, xmm5);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm6, xmm6);
        pmovsxwd(xmm6, xmm6);
        paddd(xmm9, xmm6);
        movdqu(xword[B - 0x80], xmm0);
        pmovsxbw(xmm5, xmm4);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm10, xmm5);
        movhlps(xmm6, xmm4);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm6, xmm6);
        pmovsxwd(xmm6, xmm6);
        paddd(xmm11, xmm6);
        movdqu(xword[B - 0x70], xmm4);
        pmovsxbw(xmm5, xmm1);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm12, xmm5);
        movhlps(xmm6, xmm1);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm6, xmm6);
        pmovsxwd(xmm6, xmm6);
        paddd(xmm13, xmm6);
        movdqu(xword[B - 0x60], xmm1);
        sub(B, -48);
        align(4);

        L(l2a0);
        test(M, 0x1);
        jle(l318, T_NEAR);
        movdqu(xmm0, xword[A1 - 0x80]);
        movq(xmm1, qword[A1 - 0x70]);
        add(A1, LDA);
        pmovsxbd(xmm5, xmm0);
        paddd(xmm8, xmm5);
        pshufd(xmm6, xmm0, 0x55);
        pmovsxbd(xmm6, xmm6);
        paddd(xmm9, xmm6);
        pshufd(xmm5, xmm0, 0xaa);
        pmovsxbd(xmm5, xmm5);
        paddd(xmm10, xmm5);
        pshufd(xmm6, xmm0, 0xff);
        pmovsxbd(xmm6, xmm6);
        paddd(xmm11, xmm6);
        movdqu(xword[B - 0x80], xmm0);
        pmovsxbd(xmm5, xmm1);
        pshufd(xmm6, xmm1, 0x55);
        pmovsxbd(xmm6, xmm6);
        paddd(xmm12, xmm5);
        paddd(xmm13, xmm6);
        movq(qword[B - 0x70], xmm1);
        sub(B, -24);
        align(4);

        L(l318);
        mov(A1, qword[ARG_BIAS]);
        movdqu(xword[A1], xmm8);
        movdqu(xword[A1 + 0x10], xmm9);
        movdqu(xword[A1 + 0x20], xmm10);
        movdqu(xword[A1 + 0x30], xmm11);
        movdqu(xword[A1 + 0x40], xmm12);
        movdqu(xword[A1 + 0x50], xmm13);
        add(qword[ARG_BIAS], 0x60);
        sub(N, 0x18);
        cmp(N, 0x18);
        jge(l20, T_NEAR);
        align(4);

        L(l354);
        cmp(N, 0x10);
        jl(l5a0, T_NEAR);
        align(4);

        L(l360);
        mov(A1, A);
        add(A, 0x10);
        pxor(xmm8, xmm8);
        pxor(xmm9, xmm9);
        pxor(xmm10, xmm10);
        pxor(xmm11, xmm11);
        mov(I, M);
        sar(I, 0x2);
        jle(l488, T_NEAR);
        align(4);

        L(l388);
        movdqu(xmm0, xword[A1 - 0x80]);
        add(A1, LDA);
        movdqu(xmm1, xword[A1 - 0x80]);
        add(A1, LDA);
        movdqu(xmm2, xword[A1 - 0x80]);
        add(A1, LDA);
        movdqu(xmm3, xword[A1 - 0x80]);
        add(A1, LDA);
        movdqa(xmm4, xmm0);
        punpcklbw(xmm0, xmm1);
        punpckhbw(xmm4, xmm1);
        movdqa(xmm1, xmm2);
        punpcklbw(xmm2, xmm3);
        punpckhbw(xmm1, xmm3);
        movdqa(xmm3, xmm0);
        punpcklwd(xmm0, xmm2);
        punpckhwd(xmm3, xmm2);
        movdqa(xmm2, xmm4);
        punpcklwd(xmm4, xmm1);
        punpckhwd(xmm2, xmm1);
        pmovsxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm8, xmm5);
        pmovsxbw(xmm5, xmm3);
        movhlps(xmm6, xmm3);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm9, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x70], xmm3);
        pmovsxbw(xmm5, xmm4);
        movhlps(xmm6, xmm4);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm10, xmm5);
        pmovsxbw(xmm5, xmm2);
        movhlps(xmm6, xmm2);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm11, xmm5);
        movdqu(xword[B - 0x60], xmm4);
        movdqu(xword[B - 0x50], xmm2);
        sub(B, -64);
        dec(I);
        jg(l388, T_NEAR);
        align(4);

        L(l488);
        test(M, 0x2);
        jle(l51c, T_NEAR);
        movdqu(xmm0, xword[A1 - 0x80]);
        add(A1, LDA);
        movdqu(xmm1, xword[A1 - 0x80]);
        add(A1, LDA);
        movdqa(xmm2, xmm0);
        punpcklbw(xmm0, xmm1);
        punpckhbw(xmm2, xmm1);
        pmovsxbw(xmm5, xmm0);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm8, xmm5);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm6, xmm6);
        pmovsxwd(xmm6, xmm6);
        paddd(xmm9, xmm6);
        pmovsxbw(xmm5, xmm2);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm10, xmm5);
        movhlps(xmm6, xmm2);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm6, xmm6);
        pmovsxwd(xmm6, xmm6);
        paddd(xmm11, xmm6);
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x70], xmm2);
        sub(B, -32);
        align(4);

        L(l51c);
        test(M, 0x1);
        jle(l570, T_NEAR);
        movdqu(xmm0, xword[A1 - 0x80]);
        add(A1, LDA);
        pmovsxbd(xmm5, xmm0);
        paddd(xmm8, xmm5);
        pshufd(xmm6, xmm0, 0x55);
        pmovsxbd(xmm6, xmm6);
        paddd(xmm9, xmm6);
        pshufd(xmm5, xmm0, 0xaa);
        pmovsxbd(xmm5, xmm5);
        paddd(xmm10, xmm5);
        pshufd(xmm6, xmm0, 0xff);
        pmovsxbd(xmm6, xmm6);
        paddd(xmm11, xmm6);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l570);
        mov(A1, qword[ARG_BIAS]);
        movdqu(xword[A1], xmm8);
        movdqu(xword[A1 + 0x10], xmm9);
        movdqu(xword[A1 + 0x20], xmm10);
        movdqu(xword[A1 + 0x30], xmm11);
        add(qword[ARG_BIAS], 0x40);
        sub(N, 0x10);
        cmp(N, 0x10);
        jge(l360, T_NEAR);
        align(4);

        L(l5a0);
        cmp(N, 0x8);
        jl(l830, T_NEAR);
        align(4);

        L(l5ac);
        mov(A1, A);
        add(A, 0x8);
        pxor(xmm8, xmm8);
        pxor(xmm9, xmm9);
        mov(I, M);
        sar(I, 0x3);
        jle(l6e8, T_NEAR);
        align(4);

        L(l5cc);
        movq(xmm0, qword[A1 - 0x80]);
        add(A1, LDA);
        movq(xmm1, qword[A1 - 0x80]);
        add(A1, LDA);
        movq(xmm2, qword[A1 - 0x80]);
        add(A1, LDA);
        movq(xmm3, qword[A1 - 0x80]);
        add(A1, LDA);
        punpcklbw(xmm0, xmm1);
        punpcklbw(xmm2, xmm3);
        movdqa(xmm1, xmm0);
        punpcklwd(xmm0, xmm2);
        punpckhwd(xmm1, xmm2);
        pmovsxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm8, xmm5);
        pmovsxbw(xmm5, xmm1);
        movhlps(xmm6, xmm1);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm9, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x70], xmm1);
        movq(xmm0, qword[A1 - 0x80]);
        add(A1, LDA);
        movq(xmm1, qword[A1 - 0x80]);
        add(A1, LDA);
        movq(xmm2, qword[A1 - 0x80]);
        add(A1, LDA);
        movq(xmm3, qword[A1 - 0x80]);
        add(A1, LDA);
        punpcklbw(xmm0, xmm1);
        punpcklbw(xmm2, xmm3);
        movdqa(xmm1, xmm0);
        punpcklwd(xmm0, xmm2);
        punpckhwd(xmm1, xmm2);
        pmovsxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm8, xmm5);
        pmovsxbw(xmm5, xmm1);
        movhlps(xmm6, xmm1);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm9, xmm5);
        movdqu(xword[B - 0x60], xmm0);
        movdqu(xword[B - 0x50], xmm1);
        sub(B, -64);
        dec(I);
        jg(l5cc, T_NEAR);
        align(4);

        L(l6e8);
        test(M, 0x4);
        jle(l780, T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        add(A1, LDA);
        movq(xmm1, qword[A1 - 0x80]);
        add(A1, LDA);
        movq(xmm2, qword[A1 - 0x80]);
        add(A1, LDA);
        movq(xmm3, qword[A1 - 0x80]);
        add(A1, LDA);
        punpcklbw(xmm0, xmm1);
        punpcklbw(xmm2, xmm3);
        movdqa(xmm1, xmm0);
        punpcklwd(xmm0, xmm2);
        punpckhwd(xmm1, xmm2);
        pmovsxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm8, xmm5);
        pmovsxbw(xmm5, xmm1);
        movhlps(xmm6, xmm1);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm9, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x70], xmm1);
        sub(B, -32);
        align(4);

        L(l780);
        test(M, 0x2);
        jle(l7d4, T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        add(A1, LDA);
        movq(xmm1, qword[A1 - 0x80]);
        add(A1, LDA);
        punpcklbw(xmm0, xmm1);
        pmovsxbw(xmm5, xmm0);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm8, xmm5);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm6, xmm6);
        pmovsxwd(xmm6, xmm6);
        paddd(xmm9, xmm6);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l7d4);
        test(M, 0x1);
        jle(l80c, T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        add(A1, LDA);
        pmovsxbd(xmm5, xmm0);
        pshufd(xmm6, xmm0, 0x55);
        pmovsxbd(xmm6, xmm6);
        paddd(xmm8, xmm5);
        paddd(xmm9, xmm6);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l80c);
        mov(A1, qword[ARG_BIAS]);
        movdqu(xword[A1], xmm8);
        movdqu(xword[A1 + 0x10], xmm9);
        add(qword[ARG_BIAS], 0x20);
        sub(N, 0x8);
        cmp(N, 0x8);
        jge(l5ac, T_NEAR);
        align(4);

        L(l830);
        cmp(N, 0x4);
        jl(l9f4, T_NEAR);
        align(4);

        L(l83c);
        mov(A1, A);
        add(A, 0x4);
        pxor(xmm7, xmm7);
        mov(I, M);
        sar(I, 0x3);
        jle(l910, T_NEAR);
        align(4);

        L(l854);
        movd(xmm0, dword[A1 - 0x80]);
        add(A1, LDA);
        movd(xmm1, dword[A1 - 0x80]);
        add(A1, LDA);
        movd(xmm2, dword[A1 - 0x80]);
        add(A1, LDA);
        movd(xmm3, dword[A1 - 0x80]);
        add(A1, LDA);
        punpcklbw(xmm0, xmm1);
        punpcklbw(xmm2, xmm3);
        punpcklwd(xmm0, xmm2);
        pmovsxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        movd(xmm0, dword[A1 - 0x80]);
        add(A1, LDA);
        movd(xmm1, dword[A1 - 0x80]);
        add(A1, LDA);
        movd(xmm2, dword[A1 - 0x80]);
        add(A1, LDA);
        movd(xmm3, dword[A1 - 0x80]);
        add(A1, LDA);
        punpcklbw(xmm0, xmm1);
        punpcklbw(xmm2, xmm3);
        punpcklwd(xmm0, xmm2);
        pmovsxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movdqu(xword[B - 0x70], xmm0);
        sub(B, -32);
        dec(I);
        jg(l854, T_NEAR);
        align(4);

        L(l910);
        test(M, 0x4);
        jle(l974, T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        add(A1, LDA);
        movd(xmm1, dword[A1 - 0x80]);
        add(A1, LDA);
        movd(xmm2, dword[A1 - 0x80]);
        add(A1, LDA);
        movd(xmm3, dword[A1 - 0x80]);
        add(A1, LDA);
        punpcklbw(xmm0, xmm1);
        punpcklbw(xmm2, xmm3);
        punpcklwd(xmm0, xmm2);
        pmovsxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l974);
        test(M, 0x2);
        jle(l9b0, T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        add(A1, LDA);
        movd(xmm1, dword[A1 - 0x80]);
        add(A1, LDA);
        punpcklbw(xmm0, xmm1);
        pmovsxbw(xmm5, xmm0);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l9b0);
        test(M, 0x1);
        jle(l9d4, T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        pmovsxbd(xmm5, xmm0);
        paddd(xmm7, xmm5);
        movd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(l9d4);
        mov(A1, qword[ARG_BIAS]);
        movdqu(xword[A1], xmm7);
        add(qword[ARG_BIAS], 0x10);
        sub(N, 0x4);
        cmp(N, 0x4);
        jge(l83c, T_NEAR);
        align(4);

        L(l9f4);
        cmp(N, 0x2);
        jl(lbd2, T_NEAR);
        align(4);

        L(la00);
        mov(A1, A);
        add(A, 0x2);
        pxor(xmm7, xmm7);
        mov(LDA3, M);
        sar(LDA3, 0x3);
        jle(ladc, T_NEAR);
        align(4);

        L(la18);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm0, eax, 0x0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm1, eax, 0x0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm2, eax, 0x0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm3, eax, 0x0);
        punpcklbw(xmm0, xmm1);
        punpcklbw(xmm2, xmm3);
        punpcklwd(xmm0, xmm2);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm1, eax, 0x0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm2, eax, 0x0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm3, eax, 0x0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm4, eax, 0x0);
        punpcklbw(xmm1, xmm2);
        punpcklbw(xmm3, xmm4);
        punpcklwd(xmm1, xmm3);
        punpcklqdq(xmm0, xmm1);
        pshufd(xmm6, xmm0, 0xd8);
        pmovsxbw(xmm5, xmm6);
        movhlps(xmm6, xmm6);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        dec(LDA3);
        jg(la18, T_NEAR);
        align(4);

        L(ladc);
        test(M, 0x4);
        jle(lb48, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm0, eax, 0x0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm1, eax, 0x0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm2, eax, 0x0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm3, eax, 0x0);
        punpcklbw(xmm0, xmm1);
        punpcklbw(xmm2, xmm3);
        punpcklwd(xmm0, xmm2);
        pmovsxbw(xmm5, xmm0);
        phaddw(xmm5, xmm5);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(lb48);
        test(M, 0x2);
        jle(lb8c, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm0, eax, 0x0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm1, eax, 0x0);
        punpcklbw(xmm0, xmm1);
        pmovsxbw(xmm5, xmm0);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(lb8c);
        test(M, 0x1);
        jle(lbb4, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        pinsrw(xmm0, eax, 0x0);
        pmovsxbd(xmm5, xmm0);
        paddd(xmm7, xmm5);
        mov(word[B - 0x80], ax);
        sub(B, -2);
        align(4);

        L(lbb4);
        mov(A1, qword[ARG_BIAS]);
        movq(qword[A1], xmm7);
        add(qword[ARG_BIAS], 0x8);
        sub(N, 0x2);
        cmp(N, 0x2);
        jge(la00, T_NEAR);
        align(4);

        L(lbd2);
        cmp(N, 0x1);
        jl(ld74, T_NEAR);
        align(4);

        L(lbdc);
        mov(A1, A);
        add(A, 0x1);
        pxor(xmm7, xmm7);
        mov(LDA3, M);
        sar(LDA3, 0x3);
        jle(lc8c, T_NEAR);
        align(4);

        L(lbf4);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x0);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x1);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x2);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x3);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x4);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x5);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x6);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x7);
        pmovsxbw(xmm5, xmm0);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        dec(LDA3);
        jg(lbf4, T_NEAR);
        align(4);

        L(lc8c);
        test(M, 0x4);
        jle(lcec, T_NEAR);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x0);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x1);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x2);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x3);
        pmovsxbw(xmm5, xmm0);
        phaddw(xmm5, xmm5);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(lcec);
        test(M, 0x2);
        jle(ld30, T_NEAR);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x0);
        mov(byte[B - 0x80], al);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        pinsrb(xmm0, eax, 0x1);
        pmovsxbw(xmm5, xmm0);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        mov(byte[B - 0x7f], al);
        sub(B, -2);
        align(4);

        L(ld30);
        test(M, 0x1);
        jle(ld54, T_NEAR);
        mov(al, byte[A1 - 0x80]);
        pinsrw(xmm0, eax, 0x0);
        pmovsxbd(xmm5, xmm0);
        paddd(xmm7, xmm5);
        mov(byte[B - 0x80], al);
        sub(B, -1);
        align(4);

        L(ld54);
        mov(A1, qword[ARG_BIAS]);
        movd(dword[A1], xmm7);
        add(qword[ARG_BIAS], 0x4);
        sub(N, 0x1);
        cmp(N, 0x1);
        jge(lbdc, T_NEAR);
        align(4);

        L(ld74);
        postamble();
    }
    outLocalLabel();

#undef M
#undef N
#undef A
#undef LDA
#undef ALPHA
#undef B
#undef I
#undef A1
#undef A2
#undef LDA3
#ifdef _WIN32
#undef ARG_ALPHA
#undef ARG_B
#endif
#undef ARG_BIAS
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
