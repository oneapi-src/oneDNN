/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

jit_avx_u8_copy_an_kern::jit_avx_u8_copy_an_kern()
    : jit_generator(jit_name()) {}

void jit_avx_u8_copy_an_kern::generate() {

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

#endif

    inLocalLabel();
    {
        std::vector<Xbyak::Label> labels(35);

        preamble();
#ifdef _WIN32
        auto stacksize = get_size_of_abi_save_regs();
        mov(ALPHA, ptr[ARG_ALPHA]);
        mov(B, ptr[ARG_B]);
#endif

        mov(M, qword[M]);
        mov(N, qword[N]);
        mov(LDA, qword[LDA]);
        lea(LDA3, ptr[LDA + LDA * 2]);
        sub(A, -128);
        sub(B, -128);
        cmp(N, 0x10);
        jl(labels[0], T_NEAR);
        align(4);

        L(labels[4]);
        mov(A1, A);
        add(A, 0x10);
        mov(I, M);
        sar(I, 0x2);
        jle(labels[31], T_NEAR);
        align(4);

        L(labels[12]);
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
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x70], xmm3);
        movdqu(xword[B - 0x60], xmm4);
        movdqu(xword[B - 0x50], xmm2);
        sub(B, -64);
        dec(I);
        jg(labels[12], T_NEAR);
        align(4);

        L(labels[31]);
        test(M, 0x2);
        jle(labels[32], T_NEAR);
        movdqu(xmm0, xword[A1 - 0x80]);
        add(A1, LDA);
        movdqu(xmm1, xword[A1 - 0x80]);
        add(A1, LDA);
        movdqa(xmm2, xmm0);
        punpcklbw(xmm0, xmm1);
        punpckhbw(xmm2, xmm1);
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x70], xmm2);
        sub(B, -32);
        align(4);

        L(labels[32]);
        test(M, 0x1);
        jle(labels[33], T_NEAR);
        movdqu(xmm0, xword[A1 - 0x80]);
        add(A1, LDA);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(labels[33]);
        sub(N, 0x10);
        cmp(N, 0x10);
        jge(labels[4], T_NEAR);
        align(4);

        L(labels[0]);
        cmp(N, 0x8);
        jl(labels[8], T_NEAR);
        align(4);

        L(labels[1]);
        mov(A1, A);
        add(A, 0x8);
        mov(I, M);
        sar(I, 0x3);
        jle(labels[3], T_NEAR);
        align(4);

        L(labels[2]);
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
        movdqu(xword[B - 0x60], xmm0);
        movdqu(xword[B - 0x50], xmm1);
        sub(B, -64);
        dec(I);
        jg(labels[2], T_NEAR);
        align(4);

        L(labels[3]);
        test(M, 0x4);
        jle(labels[5], T_NEAR);
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
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x70], xmm1);
        sub(B, -32);
        align(4);

        L(labels[5]);
        test(M, 0x2);
        jle(labels[6], T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        add(A1, LDA);
        movq(xmm1, qword[A1 - 0x80]);
        add(A1, LDA);
        punpcklbw(xmm0, xmm1);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(labels[6]);
        test(M, 0x1);
        jle(labels[7], T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        add(A1, LDA);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(labels[7]);
        sub(N, 0x8);
        cmp(N, 0x8);
        jge(labels[1], T_NEAR);
        align(4);

        L(labels[8]);
        cmp(N, 0x4);
        jl(labels[16], T_NEAR);
        align(4);

        L(labels[9]);
        mov(A1, A);
        add(A, 0x4);
        mov(I, M);
        sar(I, 0x3);
        jle(labels[11], T_NEAR);
        align(4);

        L(labels[10]);
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
        movdqu(xword[B - 0x70], xmm0);
        sub(B, -32);
        dec(I);
        jg(labels[10], T_NEAR);
        align(4);

        L(labels[11]);
        test(M, 0x4);
        jle(labels[13], T_NEAR);
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
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(labels[13]);
        test(M, 0x2);
        jle(labels[14], T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        add(A1, LDA);
        movd(xmm1, dword[A1 - 0x80]);
        add(A1, LDA);
        punpcklbw(xmm0, xmm1);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(labels[14]);
        test(M, 0x1);
        jle(labels[15], T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        movd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(labels[15]);
        sub(N, 0x4);
        cmp(N, 0x4);
        jge(labels[9], T_NEAR);
        align(4);

        L(labels[16]);
        cmp(N, 0x2);
        jl(labels[23], T_NEAR);
        align(4);

        L(labels[17]);
        mov(A1, A);
        add(A, 0x2);
        mov(LDA3, M);
        sar(LDA3, 0x3);
        jle(labels[19], T_NEAR);
        align(4);

        L(labels[18]);
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
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        dec(LDA3);
        jg(labels[18], T_NEAR);
        align(4);

        L(labels[19]);
        test(M, 0x4);
        jle(labels[20], T_NEAR);
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
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(labels[20]);
        test(M, 0x2);
        jle(labels[21], T_NEAR);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm0, eax, 0x0);
        mov(ax, word[A1 - 0x80]);
        add(A1, LDA);
        pinsrw(xmm1, eax, 0x0);
        punpcklbw(xmm0, xmm1);
        movd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(labels[21]);
        test(M, 0x1);
        jle(labels[22], T_NEAR);
        mov(ax, word[A1 - 0x80]);
        mov(word[B - 0x80], ax);
        sub(B, -2);
        align(4);

        L(labels[22]);
        sub(N, 0x2);
        cmp(N, 0x2);
        jge(labels[17], T_NEAR);
        align(4);

        L(labels[23]);
        cmp(N, 0x1);
        jl(labels[30], T_NEAR);
        align(4);

        L(labels[24]);
        mov(A1, A);
        add(A, 0x1);
        mov(LDA3, M);
        sar(LDA3, 0x3);
        jle(labels[26], T_NEAR);
        align(4);

        L(labels[25]);
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
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        dec(LDA3);
        jg(labels[25], T_NEAR);
        align(4);

        L(labels[26]);
        test(M, 0x4);
        jle(labels[27], T_NEAR);
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
        movd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(labels[27]);
        test(M, 0x2);
        jle(labels[28], T_NEAR);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        mov(byte[B - 0x80], al);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        mov(byte[B - 0x7f], al);
        sub(B, -2);
        align(4);

        L(labels[28]);
        test(M, 0x1);
        jle(labels[29], T_NEAR);
        mov(al, byte[A1 - 0x80]);
        mov(byte[B - 0x80], al);
        sub(B, -1);
        align(4);

        L(labels[29]);
        sub(N, 0x1);
        cmp(N, 0x1);
        jge(labels[24], T_NEAR);
        align(4);

        L(labels[30]);
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
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
