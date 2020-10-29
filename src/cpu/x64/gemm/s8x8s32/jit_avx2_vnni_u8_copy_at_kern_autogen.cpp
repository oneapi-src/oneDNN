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

jit_avx2_vnni_u8_copy_at_kern::jit_avx2_vnni_u8_copy_at_kern()
    : jit_generator(nullptr, U8_COPY_KERNEL_CODE_SIZE) {}

void jit_avx2_vnni_u8_copy_at_kern::generate() {

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

        Xbyak::Label l1024;
        Xbyak::Label l1090;
        Xbyak::Label l10a0;
        Xbyak::Label l10ac;
        Xbyak::Label l10c4;
        Xbyak::Label l1138;
        Xbyak::Label l1188;
        Xbyak::Label l11cc;
        Xbyak::Label l1214;
        Xbyak::Label l1254;
        Xbyak::Label l1264;
        Xbyak::Label l1270;
        Xbyak::Label l1288;
        Xbyak::Label l12c0;
        Xbyak::Label l12ec;
        Xbyak::Label l1318;
        Xbyak::Label l1348;
        Xbyak::Label l1368;
        Xbyak::Label l1376;
        Xbyak::Label l1380;
        Xbyak::Label l1390;
        Xbyak::Label l13ac;
        Xbyak::Label l13cc;
        Xbyak::Label l13ec;
        Xbyak::Label l1408;
        Xbyak::Label l1420;
        Xbyak::Label l1430;
        Xbyak::Label l20;
        Xbyak::Label l2d8;
        Xbyak::Label l40;
        Xbyak::Label l468;
        Xbyak::Label l5a4;
        Xbyak::Label l6e8;
        Xbyak::Label l81c;
        Xbyak::Label l82c;
        Xbyak::Label l838;
        Xbyak::Label l854;
        Xbyak::Label la04;
        Xbyak::Label lb14;
        Xbyak::Label lbf0;
        Xbyak::Label lccc;
        Xbyak::Label ld9c;
        Xbyak::Label ldac;
        Xbyak::Label ldb8;
        Xbyak::Label ldd4;
        Xbyak::Label leb0;
        Xbyak::Label lf40;
        Xbyak::Label lfb0;

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
        cmp(N, 0x18);
        jl(l82c, T_NEAR);
        align(4);

        L(l20);
        mov(A1, A);
        mov(I, LDA);
        shl(I, 0x4);
        lea(I, ptr[I + LDA * 8]);
        add(A, I);
        mov(I, M);
        sar(I, 0x4);
        jle(l2d8, T_NEAR);
        align(4);

        L(l40);
        movdqu(xmm0, xword[A1 - 0x80]);
        movdqu(xmm1, xword[A1 + LDA * 1 - 0x80]);
        movdqu(xmm2, xword[A1 + LDA * 2 - 0x80]);
        movdqu(xmm3, xword[A1 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A1 + LDA * 4]);
        movdqa(xmm4, xmm0);
        punpckldq(xmm0, xmm1);
        punpckhdq(xmm4, xmm1);
        movdqa(xmm5, xmm2);
        punpckldq(xmm2, xmm3);
        punpckhdq(xmm5, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqa(xmm3, xmm4);
        punpcklqdq(xmm4, xmm5);
        punpckhqdq(xmm3, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x20], xmm1);
        movdqu(xword[B + 0x40], xmm4);
        movdqu(xword[B + 0xa0], xmm3);
        movdqu(xmm0, xword[A2 - 0x80]);
        movdqu(xmm1, xword[A2 + LDA * 1 - 0x80]);
        movdqu(xmm2, xword[A2 + LDA * 2 - 0x80]);
        movdqu(xmm3, xword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        movdqa(xmm4, xmm0);
        punpckldq(xmm0, xmm1);
        punpckhdq(xmm4, xmm1);
        movdqa(xmm5, xmm2);
        punpckldq(xmm2, xmm3);
        punpckhdq(xmm5, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqa(xmm3, xmm4);
        punpcklqdq(xmm4, xmm5);
        punpckhqdq(xmm3, xmm5);
        movdqu(xword[B - 0x70], xmm0);
        movdqu(xword[B - 0x10], xmm1);
        movdqu(xword[B + 0x50], xmm4);
        movdqu(xword[B + 0xb0], xmm3);
        movdqu(xmm0, xword[A2 - 0x80]);
        movdqu(xmm1, xword[A2 + LDA * 1 - 0x80]);
        movdqu(xmm2, xword[A2 + LDA * 2 - 0x80]);
        movdqu(xmm3, xword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        movdqa(xmm4, xmm0);
        punpckldq(xmm0, xmm1);
        punpckhdq(xmm4, xmm1);
        movdqa(xmm5, xmm2);
        punpckldq(xmm2, xmm3);
        punpckhdq(xmm5, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqa(xmm3, xmm4);
        punpcklqdq(xmm4, xmm5);
        punpckhqdq(xmm3, xmm5);
        movdqu(xword[B - 0x60], xmm0);
        movdqu(xword[B], xmm1);
        movdqu(xword[B + 0x60], xmm4);
        movdqu(xword[B + 0xc0], xmm3);
        movdqu(xmm0, xword[A2 - 0x80]);
        movdqu(xmm1, xword[A2 + LDA * 1 - 0x80]);
        movdqu(xmm2, xword[A2 + LDA * 2 - 0x80]);
        movdqu(xmm3, xword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        movdqa(xmm4, xmm0);
        punpckldq(xmm0, xmm1);
        punpckhdq(xmm4, xmm1);
        movdqa(xmm5, xmm2);
        punpckldq(xmm2, xmm3);
        punpckhdq(xmm5, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqa(xmm3, xmm4);
        punpcklqdq(xmm4, xmm5);
        punpckhqdq(xmm3, xmm5);
        movdqu(xword[B - 0x50], xmm0);
        movdqu(xword[B + 0x10], xmm1);
        movdqu(xword[B + 0x70], xmm4);
        movdqu(xword[B + 0xd0], xmm3);
        movdqu(xmm0, xword[A2 - 0x80]);
        movdqu(xmm1, xword[A2 + LDA * 1 - 0x80]);
        movdqu(xmm2, xword[A2 + LDA * 2 - 0x80]);
        movdqu(xmm3, xword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        movdqa(xmm4, xmm0);
        punpckldq(xmm0, xmm1);
        punpckhdq(xmm4, xmm1);
        movdqa(xmm5, xmm2);
        punpckldq(xmm2, xmm3);
        punpckhdq(xmm5, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqa(xmm3, xmm4);
        punpcklqdq(xmm4, xmm5);
        punpckhqdq(xmm3, xmm5);
        movdqu(xword[B - 0x40], xmm0);
        movdqu(xword[B + 0x20], xmm1);
        movdqu(xword[B + 0x80], xmm4);
        movdqu(xword[B + 0xe0], xmm3);
        movdqu(xmm0, xword[A2 - 0x80]);
        movdqu(xmm1, xword[A2 + LDA * 1 - 0x80]);
        movdqu(xmm2, xword[A2 + LDA * 2 - 0x80]);
        movdqu(xmm3, xword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        movdqa(xmm4, xmm0);
        punpckldq(xmm0, xmm1);
        punpckhdq(xmm4, xmm1);
        movdqa(xmm5, xmm2);
        punpckldq(xmm2, xmm3);
        punpckhdq(xmm5, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqa(xmm3, xmm4);
        punpcklqdq(xmm4, xmm5);
        punpckhqdq(xmm3, xmm5);
        movdqu(xword[B - 0x30], xmm0);
        movdqu(xword[B + 0x30], xmm1);
        movdqu(xword[B + 0x90], xmm4);
        movdqu(xword[B + 0xf0], xmm3);
        sub(A1, -16);
        sub(B, -384);
        dec(I);
        jg(l40, T_NEAR);
        align(4);

        L(l2d8);
        test(M, 0x8);
        jle(l468, T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        movq(xmm1, qword[A1 + LDA * 1 - 0x80]);
        movq(xmm2, qword[A1 + LDA * 2 - 0x80]);
        movq(xmm3, qword[A1 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A1 + LDA * 4]);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x20], xmm1);
        movq(xmm0, qword[A2 - 0x80]);
        movq(xmm1, qword[A2 + LDA * 1 - 0x80]);
        movq(xmm2, qword[A2 + LDA * 2 - 0x80]);
        movq(xmm3, qword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqu(xword[B - 0x70], xmm0);
        movdqu(xword[B - 0x10], xmm1);
        movq(xmm0, qword[A2 - 0x80]);
        movq(xmm1, qword[A2 + LDA * 1 - 0x80]);
        movq(xmm2, qword[A2 + LDA * 2 - 0x80]);
        movq(xmm3, qword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqu(xword[B - 0x60], xmm0);
        movdqu(xword[B], xmm1);
        movq(xmm0, qword[A2 - 0x80]);
        movq(xmm1, qword[A2 + LDA * 1 - 0x80]);
        movq(xmm2, qword[A2 + LDA * 2 - 0x80]);
        movq(xmm3, qword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqu(xword[B - 0x50], xmm0);
        movdqu(xword[B + 0x10], xmm1);
        movq(xmm0, qword[A2 - 0x80]);
        movq(xmm1, qword[A2 + LDA * 1 - 0x80]);
        movq(xmm2, qword[A2 + LDA * 2 - 0x80]);
        movq(xmm3, qword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqu(xword[B - 0x40], xmm0);
        movdqu(xword[B + 0x20], xmm1);
        movq(xmm0, qword[A2 - 0x80]);
        movq(xmm1, qword[A2 + LDA * 1 - 0x80]);
        movq(xmm2, qword[A2 + LDA * 2 - 0x80]);
        movq(xmm3, qword[A2 + LDA3 * 1 - 0x80]);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqu(xword[B - 0x30], xmm0);
        movdqu(xword[B + 0x30], xmm1);
        sub(A1, -8);
        sub(B, -192);
        align(4);

        L(l468);
        test(M, 0x4);
        jle(l5a4, T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        movd(xmm1, dword[A1 + LDA * 1 - 0x80]);
        movd(xmm2, dword[A1 + LDA * 2 - 0x80]);
        movd(xmm3, dword[A1 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A1 + LDA * 4]);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        punpcklqdq(xmm0, xmm2);
        movdqu(xword[B - 0x80], xmm0);
        movd(xmm0, dword[A2 - 0x80]);
        movd(xmm1, dword[A2 + LDA * 1 - 0x80]);
        movd(xmm2, dword[A2 + LDA * 2 - 0x80]);
        movd(xmm3, dword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        punpcklqdq(xmm0, xmm2);
        movdqu(xword[B - 0x70], xmm0);
        movd(xmm0, dword[A2 - 0x80]);
        movd(xmm1, dword[A2 + LDA * 1 - 0x80]);
        movd(xmm2, dword[A2 + LDA * 2 - 0x80]);
        movd(xmm3, dword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        punpcklqdq(xmm0, xmm2);
        movdqu(xword[B - 0x60], xmm0);
        movd(xmm0, dword[A2 - 0x80]);
        movd(xmm1, dword[A2 + LDA * 1 - 0x80]);
        movd(xmm2, dword[A2 + LDA * 2 - 0x80]);
        movd(xmm3, dword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        punpcklqdq(xmm0, xmm2);
        movdqu(xword[B - 0x50], xmm0);
        movd(xmm0, dword[A2 - 0x80]);
        movd(xmm1, dword[A2 + LDA * 1 - 0x80]);
        movd(xmm2, dword[A2 + LDA * 2 - 0x80]);
        movd(xmm3, dword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        punpcklqdq(xmm0, xmm2);
        movdqu(xword[B - 0x40], xmm0);
        movd(xmm0, dword[A2 - 0x80]);
        movd(xmm1, dword[A2 + LDA * 1 - 0x80]);
        movd(xmm2, dword[A2 + LDA * 2 - 0x80]);
        movd(xmm3, dword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        punpcklqdq(xmm0, xmm2);
        movdqu(xword[B - 0x30], xmm0);
        sub(A1, -4);
        sub(B, -96);
        align(4);

        L(l5a4);
        test(M, 0x2);
        jle(l6e8, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        pinsrw(xmm0, eax, 0x0);
        mov(ax, word[A1 + LDA * 1 - 0x80]);
        pinsrw(xmm0, eax, 0x1);
        mov(ax, word[A1 + LDA * 2 - 0x80]);
        pinsrw(xmm0, eax, 0x2);
        mov(ax, word[A1 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A1 + LDA * 4]);
        pinsrw(xmm0, eax, 0x3);
        mov(ax, word[A2 - 0x80]);
        pinsrw(xmm0, eax, 0x4);
        mov(ax, word[A2 + LDA * 1 - 0x80]);
        pinsrw(xmm0, eax, 0x5);
        mov(ax, word[A2 + LDA * 2 - 0x80]);
        pinsrw(xmm0, eax, 0x6);
        mov(ax, word[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        pinsrw(xmm0, eax, 0x7);
        movdqu(xword[B - 0x80], xmm0);
        mov(ax, word[A2 - 0x80]);
        pinsrw(xmm0, eax, 0x0);
        mov(ax, word[A2 + LDA * 1 - 0x80]);
        pinsrw(xmm0, eax, 0x1);
        mov(ax, word[A2 + LDA * 2 - 0x80]);
        pinsrw(xmm0, eax, 0x2);
        mov(ax, word[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        pinsrw(xmm0, eax, 0x3);
        mov(ax, word[A2 - 0x80]);
        pinsrw(xmm0, eax, 0x4);
        mov(ax, word[A2 + LDA * 1 - 0x80]);
        pinsrw(xmm0, eax, 0x5);
        mov(ax, word[A2 + LDA * 2 - 0x80]);
        pinsrw(xmm0, eax, 0x6);
        mov(ax, word[A2 + LDA3 * 1 - 0x80]);
        pinsrw(xmm0, eax, 0x7);
        lea(A2, ptr[A2 + LDA * 4]);
        movdqu(xword[B - 0x70], xmm0);
        mov(ax, word[A2 - 0x80]);
        pinsrw(xmm0, eax, 0x0);
        mov(ax, word[A2 + LDA * 1 - 0x80]);
        pinsrw(xmm0, eax, 0x1);
        mov(ax, word[A2 + LDA * 2 - 0x80]);
        pinsrw(xmm0, eax, 0x2);
        mov(ax, word[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        pinsrw(xmm0, eax, 0x3);
        mov(ax, word[A2 - 0x80]);
        pinsrw(xmm0, eax, 0x4);
        mov(ax, word[A2 + LDA * 1 - 0x80]);
        pinsrw(xmm0, eax, 0x5);
        mov(ax, word[A2 + LDA * 2 - 0x80]);
        pinsrw(xmm0, eax, 0x6);
        mov(ax, word[A2 + LDA3 * 1 - 0x80]);
        pinsrw(xmm0, eax, 0x7);
        lea(A2, ptr[A2 + LDA * 4]);
        movdqu(xword[B - 0x60], xmm0);
        sub(A1, -2);
        sub(B, -48);
        align(4);

        L(l6e8);
        test(M, 0x1);
        jle(l81c, T_NEAR);
        mov(al, byte[A1 - 0x80]);
        pinsrb(xmm0, eax, 0x0);
        mov(al, byte[A1 + LDA * 1 - 0x80]);
        pinsrb(xmm0, eax, 0x1);
        mov(al, byte[A1 + LDA * 2 - 0x80]);
        pinsrb(xmm0, eax, 0x2);
        mov(al, byte[A1 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A1 + LDA * 4]);
        pinsrb(xmm0, eax, 0x3);
        mov(al, byte[A2 - 0x80]);
        pinsrb(xmm0, eax, 0x4);
        mov(al, byte[A2 + LDA * 1 - 0x80]);
        pinsrb(xmm0, eax, 0x5);
        mov(al, byte[A2 + LDA * 2 - 0x80]);
        pinsrb(xmm0, eax, 0x6);
        mov(al, byte[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        pinsrb(xmm0, eax, 0x7);
        mov(al, byte[A2 - 0x80]);
        pinsrb(xmm0, eax, 0x8);
        mov(al, byte[A2 + LDA * 1 - 0x80]);
        pinsrb(xmm0, eax, 0x9);
        mov(al, byte[A2 + LDA * 2 - 0x80]);
        pinsrb(xmm0, eax, 0xa);
        mov(al, byte[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        pinsrb(xmm0, eax, 0xb);
        mov(al, byte[A2 - 0x80]);
        pinsrb(xmm0, eax, 0xc);
        mov(al, byte[A2 + LDA * 1 - 0x80]);
        pinsrb(xmm0, eax, 0xd);
        mov(al, byte[A2 + LDA * 2 - 0x80]);
        pinsrb(xmm0, eax, 0xe);
        mov(al, byte[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        pinsrb(xmm0, eax, 0xf);
        movdqu(xword[B - 0x80], xmm0);
        mov(al, byte[A2 - 0x80]);
        pinsrb(xmm0, eax, 0x0);
        mov(al, byte[A2 + LDA * 1 - 0x80]);
        pinsrb(xmm0, eax, 0x1);
        mov(al, byte[A2 + LDA * 2 - 0x80]);
        pinsrb(xmm0, eax, 0x2);
        mov(al, byte[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        pinsrb(xmm0, eax, 0x3);
        mov(al, byte[A2 - 0x80]);
        pinsrb(xmm0, eax, 0x4);
        mov(al, byte[A2 + LDA * 1 - 0x80]);
        pinsrb(xmm0, eax, 0x5);
        mov(al, byte[A2 + LDA * 2 - 0x80]);
        pinsrb(xmm0, eax, 0x6);
        mov(al, byte[A2 + LDA3 * 1 - 0x80]);
        pinsrb(xmm0, eax, 0x7);
        movq(qword[B - 0x70], xmm0);
        sub(B, -24);
        align(4);

        L(l81c);
        sub(N, 0x18);
        cmp(N, 0x18);
        jge(l20, T_NEAR);
        align(4);

        L(l82c);
        cmp(N, 0x10);
        jl(ldac, T_NEAR);
        align(4);

        L(l838);
        mov(A1, A);
        mov(I, LDA);
        shl(I, 0x4);
        add(A, I);
        mov(I, M);
        sar(I, 0x4);
        jle(la04, T_NEAR);
        align(4);

        L(l854);
        movdqu(xmm0, xword[A1 - 0x80]);
        movdqu(xmm1, xword[A1 + LDA * 1 - 0x80]);
        movdqu(xmm2, xword[A1 + LDA * 2 - 0x80]);
        movdqu(xmm3, xword[A1 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A1 + LDA * 4]);
        movdqa(xmm4, xmm0);
        punpckldq(xmm0, xmm1);
        punpckhdq(xmm4, xmm1);
        movdqa(xmm5, xmm2);
        punpckldq(xmm2, xmm3);
        punpckhdq(xmm5, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqa(xmm3, xmm4);
        punpcklqdq(xmm4, xmm5);
        punpckhqdq(xmm3, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x40], xmm1);
        movdqu(xword[B], xmm4);
        movdqu(xword[B + 0x40], xmm3);
        movdqu(xmm0, xword[A2 - 0x80]);
        movdqu(xmm1, xword[A2 + LDA * 1 - 0x80]);
        movdqu(xmm2, xword[A2 + LDA * 2 - 0x80]);
        movdqu(xmm3, xword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        movdqa(xmm4, xmm0);
        punpckldq(xmm0, xmm1);
        punpckhdq(xmm4, xmm1);
        movdqa(xmm5, xmm2);
        punpckldq(xmm2, xmm3);
        punpckhdq(xmm5, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqa(xmm3, xmm4);
        punpcklqdq(xmm4, xmm5);
        punpckhqdq(xmm3, xmm5);
        movdqu(xword[B - 0x70], xmm0);
        movdqu(xword[B - 0x30], xmm1);
        movdqu(xword[B + 0x10], xmm4);
        movdqu(xword[B + 0x50], xmm3);
        movdqu(xmm0, xword[A2 - 0x80]);
        movdqu(xmm1, xword[A2 + LDA * 1 - 0x80]);
        movdqu(xmm2, xword[A2 + LDA * 2 - 0x80]);
        movdqu(xmm3, xword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        movdqa(xmm4, xmm0);
        punpckldq(xmm0, xmm1);
        punpckhdq(xmm4, xmm1);
        movdqa(xmm5, xmm2);
        punpckldq(xmm2, xmm3);
        punpckhdq(xmm5, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqa(xmm3, xmm4);
        punpcklqdq(xmm4, xmm5);
        punpckhqdq(xmm3, xmm5);
        movdqu(xword[B - 0x60], xmm0);
        movdqu(xword[B - 0x20], xmm1);
        movdqu(xword[B + 0x20], xmm4);
        movdqu(xword[B + 0x60], xmm3);
        movdqu(xmm0, xword[A2 - 0x80]);
        movdqu(xmm1, xword[A2 + LDA * 1 - 0x80]);
        movdqu(xmm2, xword[A2 + LDA * 2 - 0x80]);
        movdqu(xmm3, xword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        movdqa(xmm4, xmm0);
        punpckldq(xmm0, xmm1);
        punpckhdq(xmm4, xmm1);
        movdqa(xmm5, xmm2);
        punpckldq(xmm2, xmm3);
        punpckhdq(xmm5, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqa(xmm3, xmm4);
        punpcklqdq(xmm4, xmm5);
        punpckhqdq(xmm3, xmm5);
        movdqu(xword[B - 0x50], xmm0);
        movdqu(xword[B - 0x10], xmm1);
        movdqu(xword[B + 0x30], xmm4);
        movdqu(xword[B + 0x70], xmm3);
        sub(A1, -16);
        sub(B, -256);
        dec(I);
        jg(l854, T_NEAR);
        align(4);

        L(la04);
        test(M, 0x8);
        jle(lb14, T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        movq(xmm1, qword[A1 + LDA * 1 - 0x80]);
        movq(xmm2, qword[A1 + LDA * 2 - 0x80]);
        movq(xmm3, qword[A1 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A1 + LDA * 4]);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x40], xmm1);
        movq(xmm0, qword[A2 - 0x80]);
        movq(xmm1, qword[A2 + LDA * 1 - 0x80]);
        movq(xmm2, qword[A2 + LDA * 2 - 0x80]);
        movq(xmm3, qword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqu(xword[B - 0x70], xmm0);
        movdqu(xword[B - 0x30], xmm1);
        movq(xmm0, qword[A2 - 0x80]);
        movq(xmm1, qword[A2 + LDA * 1 - 0x80]);
        movq(xmm2, qword[A2 + LDA * 2 - 0x80]);
        movq(xmm3, qword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqu(xword[B - 0x60], xmm0);
        movdqu(xword[B - 0x20], xmm1);
        movq(xmm0, qword[A2 - 0x80]);
        movq(xmm1, qword[A2 + LDA * 1 - 0x80]);
        movq(xmm2, qword[A2 + LDA * 2 - 0x80]);
        movq(xmm3, qword[A2 + LDA3 * 1 - 0x80]);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqu(xword[B - 0x50], xmm0);
        movdqu(xword[B - 0x10], xmm1);
        sub(A1, -8);
        sub(B, -128);
        align(4);

        L(lb14);
        test(M, 0x4);
        jle(lbf0, T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        movd(xmm1, dword[A1 + LDA * 1 - 0x80]);
        movd(xmm2, dword[A1 + LDA * 2 - 0x80]);
        movd(xmm3, dword[A1 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A1 + LDA * 4]);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        punpcklqdq(xmm0, xmm2);
        movdqu(xword[B - 0x80], xmm0);
        movd(xmm0, dword[A2 - 0x80]);
        movd(xmm1, dword[A2 + LDA * 1 - 0x80]);
        movd(xmm2, dword[A2 + LDA * 2 - 0x80]);
        movd(xmm3, dword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        punpcklqdq(xmm0, xmm2);
        movdqu(xword[B - 0x70], xmm0);
        movd(xmm0, dword[A2 - 0x80]);
        movd(xmm1, dword[A2 + LDA * 1 - 0x80]);
        movd(xmm2, dword[A2 + LDA * 2 - 0x80]);
        movd(xmm3, dword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        punpcklqdq(xmm0, xmm2);
        movdqu(xword[B - 0x60], xmm0);
        movd(xmm0, dword[A2 - 0x80]);
        movd(xmm1, dword[A2 + LDA * 1 - 0x80]);
        movd(xmm2, dword[A2 + LDA * 2 - 0x80]);
        movd(xmm3, dword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        punpcklqdq(xmm0, xmm2);
        movdqu(xword[B - 0x50], xmm0);
        sub(A1, -4);
        sub(B, -64);
        align(4);

        L(lbf0);
        test(M, 0x2);
        jle(lccc, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        pinsrw(xmm0, eax, 0x0);
        mov(ax, word[A1 + LDA * 1 - 0x80]);
        pinsrw(xmm0, eax, 0x1);
        mov(ax, word[A1 + LDA * 2 - 0x80]);
        pinsrw(xmm0, eax, 0x2);
        mov(ax, word[A1 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A1 + LDA * 4]);
        pinsrw(xmm0, eax, 0x3);
        mov(ax, word[A2 - 0x80]);
        pinsrw(xmm0, eax, 0x4);
        mov(ax, word[A2 + LDA * 1 - 0x80]);
        pinsrw(xmm0, eax, 0x5);
        mov(ax, word[A2 + LDA * 2 - 0x80]);
        pinsrw(xmm0, eax, 0x6);
        mov(ax, word[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        pinsrw(xmm0, eax, 0x7);
        movdqu(xword[B - 0x80], xmm0);
        mov(ax, word[A2 - 0x80]);
        pinsrw(xmm0, eax, 0x0);
        mov(ax, word[A2 + LDA * 1 - 0x80]);
        pinsrw(xmm0, eax, 0x1);
        mov(ax, word[A2 + LDA * 2 - 0x80]);
        pinsrw(xmm0, eax, 0x2);
        mov(ax, word[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        pinsrw(xmm0, eax, 0x3);
        mov(ax, word[A2 - 0x80]);
        pinsrw(xmm0, eax, 0x4);
        mov(ax, word[A2 + LDA * 1 - 0x80]);
        pinsrw(xmm0, eax, 0x5);
        mov(ax, word[A2 + LDA * 2 - 0x80]);
        pinsrw(xmm0, eax, 0x6);
        mov(ax, word[A2 + LDA3 * 1 - 0x80]);
        pinsrw(xmm0, eax, 0x7);
        movdqu(xword[B - 0x70], xmm0);
        sub(A1, -2);
        sub(B, -32);
        align(4);

        L(lccc);
        test(M, 0x1);
        jle(ld9c, T_NEAR);
        mov(al, byte[A1 - 0x80]);
        pinsrb(xmm0, eax, 0x0);
        mov(al, byte[A1 + LDA * 1 - 0x80]);
        pinsrb(xmm0, eax, 0x1);
        mov(al, byte[A1 + LDA * 2 - 0x80]);
        pinsrb(xmm0, eax, 0x2);
        mov(al, byte[A1 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A1 + LDA * 4]);
        pinsrb(xmm0, eax, 0x3);
        mov(al, byte[A2 - 0x80]);
        pinsrb(xmm0, eax, 0x4);
        mov(al, byte[A2 + LDA * 1 - 0x80]);
        pinsrb(xmm0, eax, 0x5);
        mov(al, byte[A2 + LDA * 2 - 0x80]);
        pinsrb(xmm0, eax, 0x6);
        mov(al, byte[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        pinsrb(xmm0, eax, 0x7);
        mov(al, byte[A2 - 0x80]);
        pinsrb(xmm0, eax, 0x8);
        mov(al, byte[A2 + LDA * 1 - 0x80]);
        pinsrb(xmm0, eax, 0x9);
        mov(al, byte[A2 + LDA * 2 - 0x80]);
        pinsrb(xmm0, eax, 0xa);
        mov(al, byte[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        pinsrb(xmm0, eax, 0xb);
        mov(al, byte[A2 - 0x80]);
        pinsrb(xmm0, eax, 0xc);
        mov(al, byte[A2 + LDA * 1 - 0x80]);
        pinsrb(xmm0, eax, 0xd);
        mov(al, byte[A2 + LDA * 2 - 0x80]);
        pinsrb(xmm0, eax, 0xe);
        mov(al, byte[A2 + LDA3 * 1 - 0x80]);
        pinsrb(xmm0, eax, 0xf);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(ld9c);
        sub(N, 0x10);
        cmp(N, 0x10);
        jge(l838, T_NEAR);
        align(4);

        L(ldac);
        cmp(N, 0x8);
        jl(l10a0, T_NEAR);
        align(4);

        L(ldb8);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        lea(I, ptr[A1 + LDA * 8]);
        mov(A, I);
        mov(I, M);
        sar(I, 0x4);
        jle(leb0, T_NEAR);
        align(4);

        L(ldd4);
        movdqu(xmm0, xword[A1 - 0x80]);
        movdqu(xmm1, xword[A1 + LDA * 1 - 0x80]);
        movdqu(xmm2, xword[A1 + LDA * 2 - 0x80]);
        movdqu(xmm3, xword[A1 + LDA3 * 1 - 0x80]);
        sub(A1, -16);
        movdqa(xmm4, xmm0);
        punpckldq(xmm0, xmm1);
        punpckhdq(xmm4, xmm1);
        movdqa(xmm5, xmm2);
        punpckldq(xmm2, xmm3);
        punpckhdq(xmm5, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqa(xmm3, xmm4);
        punpcklqdq(xmm4, xmm5);
        punpckhqdq(xmm3, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x60], xmm1);
        movdqu(xword[B - 0x40], xmm4);
        movdqu(xword[B - 0x20], xmm3);
        movdqu(xmm0, xword[A2 - 0x80]);
        movdqu(xmm1, xword[A2 + LDA * 1 - 0x80]);
        movdqu(xmm2, xword[A2 + LDA * 2 - 0x80]);
        movdqu(xmm3, xword[A2 + LDA3 * 1 - 0x80]);
        sub(A2, -16);
        movdqa(xmm4, xmm0);
        punpckldq(xmm0, xmm1);
        punpckhdq(xmm4, xmm1);
        movdqa(xmm5, xmm2);
        punpckldq(xmm2, xmm3);
        punpckhdq(xmm5, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqa(xmm3, xmm4);
        punpcklqdq(xmm4, xmm5);
        punpckhqdq(xmm3, xmm5);
        movdqu(xword[B - 0x70], xmm0);
        movdqu(xword[B - 0x50], xmm1);
        movdqu(xword[B - 0x30], xmm4);
        movdqu(xword[B - 0x10], xmm3);
        sub(B, -128);
        dec(I);
        jg(ldd4, T_NEAR);
        align(4);

        L(leb0);
        test(M, 0x8);
        jle(lf40, T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        movq(xmm1, qword[A1 + LDA * 1 - 0x80]);
        movq(xmm2, qword[A1 + LDA * 2 - 0x80]);
        movq(xmm3, qword[A1 + LDA3 * 1 - 0x80]);
        sub(A1, -8);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x60], xmm1);
        movq(xmm0, qword[A2 - 0x80]);
        movq(xmm1, qword[A2 + LDA * 1 - 0x80]);
        movq(xmm2, qword[A2 + LDA * 2 - 0x80]);
        movq(xmm3, qword[A2 + LDA3 * 1 - 0x80]);
        sub(A2, -8);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqu(xword[B - 0x70], xmm0);
        movdqu(xword[B - 0x50], xmm1);
        sub(B, -64);
        align(4);

        L(lf40);
        test(M, 0x4);
        jle(lfb0, T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        movd(xmm1, dword[A1 + LDA * 1 - 0x80]);
        movd(xmm2, dword[A1 + LDA * 2 - 0x80]);
        movd(xmm3, dword[A1 + LDA3 * 1 - 0x80]);
        sub(A1, -4);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        punpcklqdq(xmm0, xmm2);
        movdqu(xword[B - 0x80], xmm0);
        movd(xmm0, dword[A2 - 0x80]);
        movd(xmm1, dword[A2 + LDA * 1 - 0x80]);
        movd(xmm2, dword[A2 + LDA * 2 - 0x80]);
        movd(xmm3, dword[A2 + LDA3 * 1 - 0x80]);
        sub(A2, -4);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        punpcklqdq(xmm0, xmm2);
        movdqu(xword[B - 0x70], xmm0);
        sub(B, -32);
        align(4);

        L(lfb0);
        test(M, 0x2);
        jle(l1024, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        pinsrw(xmm0, eax, 0x0);
        mov(ax, word[A1 + LDA * 1 - 0x80]);
        pinsrw(xmm0, eax, 0x1);
        mov(ax, word[A1 + LDA * 2 - 0x80]);
        pinsrw(xmm0, eax, 0x2);
        mov(ax, word[A1 + LDA3 * 1 - 0x80]);
        sub(A1, -2);
        pinsrw(xmm0, eax, 0x3);
        mov(ax, word[A2 - 0x80]);
        pinsrw(xmm0, eax, 0x4);
        mov(ax, word[A2 + LDA * 1 - 0x80]);
        pinsrw(xmm0, eax, 0x5);
        mov(ax, word[A2 + LDA * 2 - 0x80]);
        pinsrw(xmm0, eax, 0x6);
        mov(ax, word[A2 + LDA3 * 1 - 0x80]);
        sub(A2, -2);
        pinsrw(xmm0, eax, 0x7);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l1024);
        test(M, 0x1);
        jle(l1090, T_NEAR);
        mov(al, byte[A1 - 0x80]);
        pinsrb(xmm0, eax, 0x0);
        mov(al, byte[A1 + LDA * 1 - 0x80]);
        pinsrb(xmm0, eax, 0x1);
        mov(al, byte[A1 + LDA * 2 - 0x80]);
        pinsrb(xmm0, eax, 0x2);
        mov(al, byte[A1 + LDA3 * 1 - 0x80]);
        pinsrb(xmm0, eax, 0x3);
        mov(al, byte[A2 - 0x80]);
        pinsrb(xmm0, eax, 0x4);
        mov(al, byte[A2 + LDA * 1 - 0x80]);
        pinsrb(xmm0, eax, 0x5);
        mov(al, byte[A2 + LDA * 2 - 0x80]);
        pinsrb(xmm0, eax, 0x6);
        mov(al, byte[A2 + LDA3 * 1 - 0x80]);
        pinsrb(xmm0, eax, 0x7);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l1090);
        sub(N, 0x8);
        cmp(N, 0x8);
        jge(ldb8, T_NEAR);
        align(4);

        L(l10a0);
        cmp(N, 0x4);
        jl(l1264, T_NEAR);
        align(4);

        L(l10ac);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 2]);
        lea(I, ptr[A1 + LDA * 4]);
        mov(A, I);
        mov(I, M);
        sar(I, 0x4);
        jle(l1138, T_NEAR);
        align(4);

        L(l10c4);
        movdqu(xmm0, xword[A1 - 0x80]);
        movdqu(xmm1, xword[A1 + LDA * 1 - 0x80]);
        sub(A1, -16);
        movdqu(xmm2, xword[A2 - 0x80]);
        movdqu(xmm3, xword[A2 + LDA * 1 - 0x80]);
        sub(A2, -16);
        movdqa(xmm4, xmm0);
        punpckldq(xmm0, xmm1);
        punpckhdq(xmm4, xmm1);
        movdqa(xmm5, xmm2);
        punpckldq(xmm2, xmm3);
        punpckhdq(xmm5, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqa(xmm3, xmm4);
        punpcklqdq(xmm4, xmm5);
        punpckhqdq(xmm3, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x70], xmm1);
        movdqu(xword[B - 0x60], xmm4);
        movdqu(xword[B - 0x50], xmm3);
        sub(B, -64);
        dec(I);
        jg(l10c4, T_NEAR);
        align(4);

        L(l1138);
        test(M, 0x8);
        jle(l1188, T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        movq(xmm1, qword[A1 + LDA * 1 - 0x80]);
        sub(A1, -8);
        movq(xmm2, qword[A2 - 0x80]);
        movq(xmm3, qword[A2 + LDA * 1 - 0x80]);
        sub(A2, -8);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x70], xmm1);
        sub(B, -32);
        align(4);

        L(l1188);
        test(M, 0x4);
        jle(l11cc, T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        movd(xmm1, dword[A1 + LDA * 1 - 0x80]);
        sub(A1, -4);
        movd(xmm2, dword[A2 - 0x80]);
        movd(xmm3, dword[A2 + LDA * 1 - 0x80]);
        sub(A2, -4);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        punpcklqdq(xmm0, xmm2);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l11cc);
        test(M, 0x2);
        jle(l1214, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        pinsrw(xmm0, eax, 0x0);
        mov(ax, word[A1 + LDA * 1 - 0x80]);
        sub(A1, -2);
        pinsrw(xmm0, eax, 0x1);
        mov(ax, word[A2 - 0x80]);
        pinsrw(xmm0, eax, 0x2);
        mov(ax, word[A2 + LDA * 1 - 0x80]);
        sub(A2, -2);
        pinsrw(xmm0, eax, 0x3);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l1214);
        test(M, 0x1);
        jle(l1254, T_NEAR);
        mov(al, byte[A1 - 0x80]);
        pinsrb(xmm0, eax, 0x0);
        mov(al, byte[A1 + LDA * 1 - 0x80]);
        pinsrb(xmm0, eax, 0x1);
        mov(al, byte[A2 - 0x80]);
        pinsrb(xmm0, eax, 0x2);
        mov(al, byte[A2 + LDA * 1 - 0x80]);
        pinsrb(xmm0, eax, 0x3);
        movd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(l1254);
        sub(N, 0x4);
        cmp(N, 0x4);
        jge(l10ac, T_NEAR);
        align(4);

        L(l1264);
        cmp(N, 0x2);
        jl(l1376, T_NEAR);
        align(4);

        L(l1270);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 1]);
        lea(I, ptr[A1 + LDA * 2]);
        mov(A, I);
        mov(I, M);
        sar(I, 0x4);
        jle(l12c0, T_NEAR);
        align(4);

        L(l1288);
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
        jg(l1288, T_NEAR);
        align(4);

        L(l12c0);
        test(M, 0x8);
        jle(l12ec, T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        sub(A1, -8);
        movq(xmm1, qword[A2 - 0x80]);
        sub(A2, -8);
        punpckldq(xmm0, xmm1);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l12ec);
        test(M, 0x4);
        jle(l1318, T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        sub(A1, -4);
        movd(xmm1, dword[A2 - 0x80]);
        sub(A2, -4);
        punpckldq(xmm0, xmm1);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l1318);
        test(M, 0x2);
        jle(l1348, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        sub(A1, -2);
        pinsrw(xmm0, eax, 0x0);
        mov(ax, word[A2 - 0x80]);
        sub(A2, -2);
        pinsrw(xmm0, eax, 0x1);
        movd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(l1348);
        test(M, 0x1);
        jle(l1368, T_NEAR);
        mov(al, byte[A1 - 0x80]);
        mov(byte[B - 0x80], al);
        mov(al, byte[A2 - 0x80]);
        mov(byte[B - 0x7f], al);
        sub(B, -2);
        align(4);

        L(l1368);
        sub(N, 0x2);
        cmp(N, 0x2);
        jge(l1270, T_NEAR);
        align(4);

        L(l1376);
        cmp(N, 0x1);
        jl(l1430, T_NEAR);
        align(4);

        L(l1380);
        mov(A1, A);
        add(A, LDA);
        mov(I, M);
        sar(I, 0x4);
        jle(l13ac, T_NEAR);
        align(4);

        L(l1390);
        movdqu(xmm0, xword[A1 - 0x80]);
        sub(A1, -16);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        dec(I);
        jg(l1390, T_NEAR);
        align(4);

        L(l13ac);
        test(M, 0x8);
        jle(l13cc, T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        sub(A1, -8);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l13cc);
        test(M, 0x4);
        jle(l13ec, T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        sub(A1, -4);
        movd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(l13ec);
        test(M, 0x2);
        jle(l1408, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        mov(word[B - 0x80], ax);
        sub(A1, -2);
        sub(B, -2);
        align(4);

        L(l1408);
        test(M, 0x1);
        jle(l1420, T_NEAR);
        mov(al, byte[A1 - 0x80]);
        mov(byte[B - 0x80], al);
        sub(B, -1);
        align(4);

        L(l1420);
        sub(N, 0x1);
        cmp(N, 0x1);
        jge(l1380, T_NEAR);
        align(4);

        L(l1430);
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
