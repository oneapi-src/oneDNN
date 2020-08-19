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

jit_avx2_vnni_u8_copy_an_kern::jit_avx2_vnni_u8_copy_an_kern()
    : jit_generator(nullptr, U8_COPY_KERNEL_CODE_SIZE) {}

void jit_avx2_vnni_u8_copy_an_kern::generate() {

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

        Xbyak::Label l138;
        Xbyak::Label l160;
        Xbyak::Label l170;
        Xbyak::Label l17c;
        Xbyak::Label l18c;
        Xbyak::Label l20;
        Xbyak::Label l204;
        Xbyak::Label l23c;
        Xbyak::Label l258;
        Xbyak::Label l268;
        Xbyak::Label l274;
        Xbyak::Label l288;
        Xbyak::Label l320;
        Xbyak::Label l34;
        Xbyak::Label l374;
        Xbyak::Label l3a0;
        Xbyak::Label l3bc;
        Xbyak::Label l3cc;
        Xbyak::Label l3d8;
        Xbyak::Label l3e8;
        Xbyak::Label l460;
        Xbyak::Label l4a4;
        Xbyak::Label l4d0;
        Xbyak::Label l4ec;
        Xbyak::Label l4fc;
        Xbyak::Label l508;
        Xbyak::Label l51c;
        Xbyak::Label l5b4;
        Xbyak::Label l608;
        Xbyak::Label l63c;
        Xbyak::Label l654;
        Xbyak::Label l662;
        Xbyak::Label l66c;
        Xbyak::Label l67c;
        Xbyak::Label l6f4;
        Xbyak::Label l73c;
        Xbyak::Label l760;
        Xbyak::Label l778;
        Xbyak::Label l788;
        Xbyak::Label le8;

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
        cmp(N, 0x18);
        jl(l170, T_NEAR);
        align(4);

        L(l20);
        mov(A1, A);
        add(A, 0x18);
        mov(I, M);
        sar(I, 0x2);
        jle(le8, T_NEAR);
        align(4);

        L(l34);
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
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x70], xmm6);
        movdqa(xmm6, xmm2);
        punpcklwd(xmm2, xmm3);
        punpckhwd(xmm6, xmm3);
        movdqu(xword[B - 0x60], xmm2);
        movdqu(xword[B - 0x50], xmm6);
        movdqa(xmm6, xmm4);
        punpcklwd(xmm4, xmm5);
        punpckhwd(xmm6, xmm5);
        movdqu(xword[B - 0x40], xmm4);
        movdqu(xword[B - 0x30], xmm6);
        sub(B, -96);
        dec(I);
        jg(l34, T_NEAR);
        align(4);

        L(le8);
        test(M, 0x2);
        jle(l138, T_NEAR);
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
        movdqu(xword[B - 0x80], xmm0);
        movdqu(xword[B - 0x70], xmm4);
        movdqu(xword[B - 0x60], xmm1);
        sub(B, -48);
        align(4);

        L(l138);
        test(M, 0x1);
        jle(l160, T_NEAR);
        movdqu(xmm0, xword[A1 - 0x80]);
        movq(xmm1, qword[A1 - 0x70]);
        add(A1, LDA);
        movdqu(xword[B - 0x80], xmm0);
        movq(qword[B - 0x70], xmm1);
        sub(B, -24);
        align(4);

        L(l160);
        sub(N, 0x18);
        cmp(N, 0x18);
        jge(l20, T_NEAR);
        align(4);

        L(l170);
        cmp(N, 0x10);
        jl(l268, T_NEAR);
        align(4);

        L(l17c);
        mov(A1, A);
        add(A, 0x10);
        mov(I, M);
        sar(I, 0x2);
        jle(l204, T_NEAR);
        align(4);

        L(l18c);
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
        jg(l18c, T_NEAR);
        align(4);

        L(l204);
        test(M, 0x2);
        jle(l23c, T_NEAR);
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

        L(l23c);
        test(M, 0x1);
        jle(l258, T_NEAR);
        movdqu(xmm0, xword[A1 - 0x80]);
        add(A1, LDA);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l258);
        sub(N, 0x10);
        cmp(N, 0x10);
        jge(l17c, T_NEAR);
        align(4);

        L(l268);
        cmp(N, 0x8);
        jl(l3cc, T_NEAR);
        align(4);

        L(l274);
        mov(A1, A);
        add(A, 0x8);
        mov(I, M);
        sar(I, 0x3);
        jle(l320, T_NEAR);
        align(4);

        L(l288);
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
        jg(l288, T_NEAR);
        align(4);

        L(l320);
        test(M, 0x4);
        jle(l374, T_NEAR);
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

        L(l374);
        test(M, 0x2);
        jle(l3a0, T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        add(A1, LDA);
        movq(xmm1, qword[A1 - 0x80]);
        add(A1, LDA);
        punpcklbw(xmm0, xmm1);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(l3a0);
        test(M, 0x1);
        jle(l3bc, T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        add(A1, LDA);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l3bc);
        sub(N, 0x8);
        cmp(N, 0x8);
        jge(l274, T_NEAR);
        align(4);

        L(l3cc);
        cmp(N, 0x4);
        jl(l4fc, T_NEAR);
        align(4);

        L(l3d8);
        mov(A1, A);
        add(A, 0x4);
        mov(I, M);
        sar(I, 0x3);
        jle(l460, T_NEAR);
        align(4);

        L(l3e8);
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
        jg(l3e8, T_NEAR);
        align(4);

        L(l460);
        test(M, 0x4);
        jle(l4a4, T_NEAR);
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

        L(l4a4);
        test(M, 0x2);
        jle(l4d0, T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        add(A1, LDA);
        movd(xmm1, dword[A1 - 0x80]);
        add(A1, LDA);
        punpcklbw(xmm0, xmm1);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(l4d0);
        test(M, 0x1);
        jle(l4ec, T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        movd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(l4ec);
        sub(N, 0x4);
        cmp(N, 0x4);
        jge(l3d8, T_NEAR);
        align(4);

        L(l4fc);
        cmp(N, 0x2);
        jl(l662, T_NEAR);
        align(4);

        L(l508);
        mov(A1, A);
        add(A, 0x2);
        mov(LDA3, M);
        sar(LDA3, 0x3);
        jle(l5b4, T_NEAR);
        align(4);

        L(l51c);
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
        jg(l51c, T_NEAR);
        align(4);

        L(l5b4);
        test(M, 0x4);
        jle(l608, T_NEAR);
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

        L(l608);
        test(M, 0x2);
        jle(l63c, T_NEAR);
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

        L(l63c);
        test(M, 0x1);
        jle(l654, T_NEAR);
        mov(ax, word[A1 - 0x80]);
        mov(word[B - 0x80], ax);
        sub(B, -2);
        align(4);

        L(l654);
        sub(N, 0x2);
        cmp(N, 0x2);
        jge(l508, T_NEAR);
        align(4);

        L(l662);
        cmp(N, 0x1);
        jl(l788, T_NEAR);
        align(4);

        L(l66c);
        mov(A1, A);
        add(A, 0x1);
        mov(LDA3, M);
        sar(LDA3, 0x3);
        jle(l6f4, T_NEAR);
        align(4);

        L(l67c);
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
        jg(l67c, T_NEAR);
        align(4);

        L(l6f4);
        test(M, 0x4);
        jle(l73c, T_NEAR);
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

        L(l73c);
        test(M, 0x2);
        jle(l760, T_NEAR);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        mov(byte[B - 0x80], al);
        mov(al, byte[A1 - 0x80]);
        add(A1, LDA);
        mov(byte[B - 0x7f], al);
        sub(B, -2);
        align(4);

        L(l760);
        test(M, 0x1);
        jle(l778, T_NEAR);
        mov(al, byte[A1 - 0x80]);
        mov(byte[B - 0x80], al);
        sub(B, -1);
        align(4);

        L(l778);
        sub(N, 0x1);
        cmp(N, 0x1);
        jge(l66c, T_NEAR);
        align(4);

        L(l788);
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
