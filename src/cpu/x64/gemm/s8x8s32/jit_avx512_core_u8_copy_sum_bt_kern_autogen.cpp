/*******************************************************************************
* Copyright 2018-2024 Intel Corporation
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

jit_avx512_core_u8_copy_sum_bt_kern::jit_avx512_core_u8_copy_sum_bt_kern(
        bool s8_case)
    : jit_generator(jit_name()), s8_case(s8_case) {}

void jit_avx512_core_u8_copy_sum_bt_kern::generate() {

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

#define ARG_ALPHA 40 + stacksize + rsp
#define ARG_B 48 + stacksize + rsp
#define ARG_BIAS 72 + stacksize + rsp

#endif

    inLocalLabel();
    {
        std::vector<Xbyak::Label> labels(28);

        preamble();
        auto stacksize = get_size_of_abi_save_regs();
#ifdef _WIN32
        mov(ALPHA, ptr[ARG_ALPHA]);
        mov(B, ptr[ARG_B]);
#endif

        alignas(16) static unsigned int hbit[]
                = {0x80808080u, 0x80808080u, 0x80808080u, 0x80808080u};
        mov(A1, (size_t)&hbit);
        movdqu(xmm15, xword[A1]);

        auto maybe_perform_s8_shift_xmm = [this](Xbyak::Xmm x) {
            if (s8_case) xorps(x, xmm15);
        };
        auto maybe_perform_s8_shift_r8 = [this](const Xbyak::Reg8 &r) {
            if (s8_case) xor_(r, (int8_t)0x80);
        };
        auto maybe_perform_s8_shift_r16 = [this](const Xbyak::Reg16 &r) {
            if (s8_case) xor_(r, (int16_t)0x8080);
        };

        mov(M, qword[M]);
        mov(N, qword[N]);
        mov(LDA, qword[LDA]);
        lea(LDA3, ptr[LDA + LDA * 2]);
        sub(A, -128);
        sub(B, -128);
        cmp(N, 0x8);
        jl(labels[5], T_NEAR);
        align(4);

        L(labels[2]);
        mov(A1, A);
        add(A, 0x8);
        pxor(xmm8, xmm8);
        pxor(xmm9, xmm9);
        mov(I, M);
        sar(I, 0x3);
        jle(labels[0], T_NEAR);
        align(4);

        L(labels[10]);
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
        maybe_perform_s8_shift_xmm(xmm0);
        maybe_perform_s8_shift_xmm(xmm1);
        pmovzxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovzxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovzxwd(xmm5, xmm5);
        paddd(xmm8, xmm5);
        pmovzxbw(xmm5, xmm1);
        movhlps(xmm6, xmm1);
        pmovzxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovzxwd(xmm5, xmm5);
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
        maybe_perform_s8_shift_xmm(xmm0);
        maybe_perform_s8_shift_xmm(xmm1);
        pmovzxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovzxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovzxwd(xmm5, xmm5);
        paddd(xmm8, xmm5);
        pmovzxbw(xmm5, xmm1);
        movhlps(xmm6, xmm1);
        pmovzxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovzxwd(xmm5, xmm5);
        paddd(xmm9, xmm5);
        movdqu(xword[B - 0x60], xmm0);
        movdqu(xword[B - 0x50], xmm1);
        sub(B, -64);
        dec(I);
        jg(labels[10], T_NEAR);
        align(4);

        L(labels[0]);
        test(M, 0x4);
        jle(labels[1], T_NEAR);
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
        maybe_perform_s8_shift_xmm(xmm0);
        maybe_perform_s8_shift_xmm(xmm1);
        pmovzxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovzxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovzxwd(xmm5, xmm5);
        paddd(xmm8, xmm5);
        pmovzxbw(xmm5, xmm1);
        movhlps(xmm6, xmm1);
        pmovzxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovzxwd(xmm5, xmm5);
        paddd(xmm9, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x70], xmm1);
        sub(B, -32);
        align(4);

        L(labels[1]);
        test(M, 0x2);
        jle(labels[3], T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        add(A1, LDA);
        movq(xmm1, qword[A1 - 0x80]);
        add(A1, LDA);
        punpcklbw(xmm0, xmm1);
        maybe_perform_s8_shift_xmm(xmm0);
        pmovzxbw(xmm5, xmm0);
        phaddw(xmm5, xmm5);
        pmovzxwd(xmm5, xmm5);
        paddd(xmm8, xmm5);
        movhlps(xmm6, xmm0);
        pmovzxbw(xmm6, xmm6);
        phaddw(xmm6, xmm6);
        pmovzxwd(xmm6, xmm6);
        paddd(xmm9, xmm6);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(labels[3]);
        test(M, 0x1);
        jle(labels[4], T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        add(A1, LDA);
        maybe_perform_s8_shift_xmm(xmm0);
        pmovzxbd(xmm5, xmm0);
        pshufd(xmm6, xmm0, 0x55);
        pmovzxbd(xmm6, xmm6);
        paddd(xmm8, xmm5);
        paddd(xmm9, xmm6);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(labels[4]);
        mov(A1, qword[ARG_BIAS]);
        movdqu(xword[A1], xmm8);
        movdqu(xword[A1 + 0x10], xmm9);
        add(qword[ARG_BIAS], 0x20);
        sub(N, 0x8);
        cmp(N, 0x8);
        jge(labels[2], T_NEAR);
        align(4);

        L(labels[5]);
        cmp(N, 0x4);
        jl(labels[13], T_NEAR);
        align(4);

        L(labels[6]);
        mov(A1, A);
        add(A, 0x4);
        pxor(xmm7, xmm7);
        mov(I, M);
        sar(I, 0x3);
        jle(labels[8], T_NEAR);
        align(4);

        L(labels[7]);
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
        maybe_perform_s8_shift_xmm(xmm0);
        pmovzxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovzxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovzxwd(xmm5, xmm5);
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
        maybe_perform_s8_shift_xmm(xmm0);
        pmovzxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovzxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovzxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movdqu(xword[B - 0x70], xmm0);
        sub(B, -32);
        dec(I);
        jg(labels[7], T_NEAR);
        align(4);

        L(labels[8]);
        test(M, 0x4);
        jle(labels[9], T_NEAR);
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
        maybe_perform_s8_shift_xmm(xmm0);
        pmovzxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovzxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovzxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(labels[9]);
        test(M, 0x2);
        jle(labels[11], T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        add(A1, LDA);
        movd(xmm1, dword[A1 - 0x80]);
        add(A1, LDA);
        punpcklbw(xmm0, xmm1);
        maybe_perform_s8_shift_xmm(xmm0);
        pmovzxbw(xmm5, xmm0);
        phaddw(xmm5, xmm5);
        pmovzxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(labels[11]);
        test(M, 0x1);
        jle(labels[12], T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        maybe_perform_s8_shift_xmm(xmm0);
        pmovzxbd(xmm5, xmm0);
        paddd(xmm7, xmm5);
        movd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(labels[12]);
        mov(A1, qword[ARG_BIAS]);
        movdqu(xword[A1], xmm7);
        add(qword[ARG_BIAS], 0x10);
        sub(N, 0x4);
        cmp(N, 0x4);
        jge(labels[6], T_NEAR);
        align(4);

        L(labels[13]);
        cmp(N, 0x2);
        jl(labels[20], T_NEAR);
        align(4);

        L(labels[14]);
        mov(A1, A);
        add(A, 0x2);
        pxor(xmm7, xmm7);
        mov(LDA3, M);
        sar(LDA3, 0x3);
        jle(labels[16], T_NEAR);
        align(4);

        L(labels[15]);
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
        maybe_perform_s8_shift_xmm(xmm0);
        pshufd(xmm6, xmm0, 0xd8);
        pmovzxbw(xmm5, xmm6);
        movhlps(xmm6, xmm6);
        pmovzxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        phaddw(xmm5, xmm5);
        pmovzxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        dec(LDA3);
        jg(labels[15], T_NEAR);
        align(4);

        L(labels[16]);
        test(M, 0x4);
        jle(labels[17], T_NEAR);
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
        maybe_perform_s8_shift_xmm(xmm0);
        pmovzxbw(xmm5, xmm0);
        phaddw(xmm5, xmm5);
        phaddw(xmm5, xmm5);
        pmovzxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(labels[17]);
        test(M, 0x2);
        jle(labels[18], T_NEAR);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm0, eax, 0x0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm1, eax, 0x0);
        punpcklbw(xmm0, xmm1);
        maybe_perform_s8_shift_xmm(xmm0);
        pmovzxbw(xmm5, xmm0);
        phaddw(xmm5, xmm5);
        pmovzxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(labels[18]);
        test(M, 0x1);
        jle(labels[19], T_NEAR);
        mov(ax, word[A1 - 0x80]);
        maybe_perform_s8_shift_r16(ax);
        pinsrw(xmm0, eax, 0x0);
        pmovzxbd(xmm5, xmm0);
        paddd(xmm7, xmm5);
        mov(word[B - 0x80], ax);
        sub(B, -2);
        align(4);

        L(labels[19]);
        mov(A1, qword[ARG_BIAS]);
        movq(qword[A1], xmm7);
        add(qword[ARG_BIAS], 0x8);
        sub(N, 0x2);
        cmp(N, 0x2);
        jge(labels[14], T_NEAR);
        align(4);

        L(labels[20]);
        cmp(N, 0x1);
        jl(labels[27], T_NEAR);
        align(4);

        L(labels[21]);
        mov(A1, A);
        add(A, 0x1);
        pxor(xmm7, xmm7);
        mov(LDA3, M);
        sar(LDA3, 0x3);
        jle(labels[23], T_NEAR);
        align(4);

        L(labels[22]);
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
        maybe_perform_s8_shift_xmm(xmm0);
        pmovzxbw(xmm5, xmm0);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        phaddw(xmm5, xmm5);
        pmovzxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        dec(LDA3);
        jg(labels[22], T_NEAR);
        align(4);

        L(labels[23]);
        test(M, 0x4);
        jle(labels[24], T_NEAR);
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
        maybe_perform_s8_shift_xmm(xmm0);
        pmovzxbw(xmm5, xmm0);
        phaddw(xmm5, xmm5);
        phaddw(xmm5, xmm5);
        pmovzxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(labels[24]);
        test(M, 0x2);
        jle(labels[25], T_NEAR);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        maybe_perform_s8_shift_r8(al);
        pinsrb(xmm0, eax, 0x0);
        mov(byte[B - 0x80], al);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        maybe_perform_s8_shift_r8(al);
        pinsrb(xmm0, eax, 0x1);
        pmovzxbw(xmm5, xmm0);
        phaddw(xmm5, xmm5);
        pmovzxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        mov(byte[B - 0x7f], al);
        sub(B, -2);
        align(4);

        L(labels[25]);
        test(M, 0x1);
        jle(labels[26], T_NEAR);
        mov(al, byte[A1 - 0x80]);
        maybe_perform_s8_shift_r8(al);
        pinsrw(xmm0, eax, 0x0);
        pmovzxbd(xmm5, xmm0);
        paddd(xmm7, xmm5);
        mov(byte[B - 0x80], al);
        sub(B, -1);
        align(4);

        L(labels[26]);
        mov(A1, qword[ARG_BIAS]);
        movd(dword[A1], xmm7);
        add(qword[ARG_BIAS], 0x4);
        sub(N, 0x1);
        cmp(N, 0x1);
        jge(labels[21], T_NEAR);
        align(4);

        L(labels[27]);

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
