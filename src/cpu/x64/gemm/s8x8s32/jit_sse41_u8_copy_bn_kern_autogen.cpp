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

jit_sse41_u8_copy_bn_kern::jit_sse41_u8_copy_bn_kern()
    : jit_generator(jit_name()) {}

void jit_sse41_u8_copy_bn_kern::generate() {

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
        std::vector<Xbyak::Label> labels(16);

        preamble();
#ifdef _WIN32
        auto stacksize = get_size_of_abi_save_regs();
        mov(ALPHA, ptr[ARG_ALPHA]);
        mov(B, ptr[ARG_B]);
#endif

        mov(N, qword[N]);
        mov(M, qword[M]);
        mov(LDA, qword[LDA]);
        sub(A, -128);
        sub(B, -128);
        lea(LDA3, ptr[LDA + LDA * 2]);
        cmp(N, 0x2);
        jl(labels[1], T_NEAR);
        align(4);

        L(labels[10]);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 1]);
        lea(I, ptr[A1 + LDA * 2]);
        mov(A, I);
        mov(I, M);
        sar(I, 0x4);
        jle(labels[12], T_NEAR);
        align(4);

        L(labels[11]);
        movdqu(xmm0, xword[A1 - 0x80]);
        sub(A1, -16);
        movdqu(xmm1, xword[A2 - 0x80]);
        sub(A2, -16);
        movdqa(xmm2, xmm0);
        punpckldq(xmm0, xmm1);
        punpckhdq(xmm2, xmm1);
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x70], xmm2);
        sub(B, -32);
        dec(I);
        jg(labels[11], T_NEAR);
        align(4);

        L(labels[12]);
        test(M, 0x8);
        jle(labels[13], T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        sub(A1, -8);
        movq(xmm1, qword[A2 - 0x80]);
        sub(A2, -8);
        punpckldq(xmm0, xmm1);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(labels[13]);
        test(M, 0x4);
        jle(labels[14], T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        sub(A1, -4);
        movd(xmm1, dword[A2 - 0x80]);
        sub(A2, -4);
        punpckldq(xmm0, xmm1);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(labels[14]);
        test(M, 0x2);
        jle(labels[15], T_NEAR);
        mov(ax, word[A1 - 0x80]);
        sub(A1, -2);
        pinsrw(xmm0, eax, 0x0);
        mov(ax, word[A2 - 0x80]);
        sub(A2, -2);
        pinsrw(xmm0, eax, 0x1);
        movd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(labels[15]);
        test(M, 0x1);
        jle(labels[0], T_NEAR);
        mov(al, byte[A1 - 0x80]);
        mov(byte[B - 0x80], al);
        mov(al, byte[A2 - 0x80]);
        mov(byte[B - 0x7f], al);
        sub(B, -2);
        align(4);

        L(labels[0]);
        sub(N, 0x2);
        cmp(N, 0x2);
        jge(labels[10], T_NEAR);
        align(4);

        L(labels[1]);
        cmp(N, 0x1);
        jl(labels[9], T_NEAR);
        align(4);

        L(labels[2]);
        mov(A1, A);
        add(A, LDA);
        mov(I, M);
        sar(I, 0x4);
        jle(labels[4], T_NEAR);
        align(4);

        L(labels[3]);
        movdqu(xmm0, xword[A1 - 0x80]);
        sub(A1, -16);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        dec(I);
        jg(labels[3], T_NEAR);
        align(4);

        L(labels[4]);
        test(M, 0x8);
        jle(labels[5], T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        sub(A1, -8);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(labels[5]);
        test(M, 0x4);
        jle(labels[6], T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        sub(A1, -4);
        movd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(labels[6]);
        test(M, 0x2);
        jle(labels[7], T_NEAR);
        mov(ax, word[A1 - 0x80]);
        mov(word[B - 0x80], ax);
        sub(A1, -2);
        sub(B, -2);
        align(4);

        L(labels[7]);
        test(M, 0x1);
        jle(labels[8], T_NEAR);
        mov(al, byte[A1 - 0x80]);
        mov(byte[B - 0x80], al);
        sub(B, -1);
        align(4);

        L(labels[8]);
        sub(N, 0x1);
        cmp(N, 0x1);
        jge(labels[2], T_NEAR);
        align(4);

        L(labels[9]);
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
