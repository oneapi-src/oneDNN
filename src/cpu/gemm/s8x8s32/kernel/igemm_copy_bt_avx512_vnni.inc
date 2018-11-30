/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#ifndef _WIN32
#define M	rdi
#define N	rsi
#define A	rdx
#define LDA	rcx
#define ALPHA	r8
#define B	r9
#define I	rax
#define A1	r10
#define A2	r8
#define LDA3	r11
#define STACKSIZE	0
#else
#define M	rcx
#define N	rdx
#define A	r8
#define LDA	r9
#define ALPHA	rax
#define B	rdi
#define I	rax
#define A1	rsi
#define A2	r10
#define LDA3	r11
#define STACKSIZE	64
#define ARG_ALPHA	40+STACKSIZE+rsp
#define ARG_B	48+STACKSIZE+rsp
#endif

inLocalLabel();
{

Xbyak::Label l120;
Xbyak::Label l14c;
Xbyak::Label l168;
Xbyak::Label l178;
Xbyak::Label l184;
Xbyak::Label l194;
Xbyak::Label l20;
Xbyak::Label l20c;
Xbyak::Label l250;
Xbyak::Label l27c;
Xbyak::Label l298;
Xbyak::Label l2a8;
Xbyak::Label l2b4;
Xbyak::Label l2c8;
Xbyak::Label l34;
Xbyak::Label l364;
Xbyak::Label l3b8;
Xbyak::Label l3ec;
Xbyak::Label l404;
Xbyak::Label l412;
Xbyak::Label l41c;
Xbyak::Label l42c;
Xbyak::Label l4a4;
Xbyak::Label l4ec;
Xbyak::Label l510;
Xbyak::Label l528;
Xbyak::Label l538;
Xbyak::Label lcc;

#ifdef _WIN32
	sub(rsp, STACKSIZE);
	mov(ptr[rsp], rsi);
	mov(ptr[rsp+0x8], rdi);
	movups(ptr[rsp+0x10], xmm6);
	movups(ptr[rsp+0x20], xmm7);
	mov(ALPHA, ptr[ARG_ALPHA]);
	mov(B, ptr[ARG_B]);
#endif
	mov(M, ptr[M]);
	mov(N, ptr[N]);
	mov(LDA, ptr[LDA]);
	lea(LDA3, ptr[LDA+LDA*2]);
	sub(A, -128);
	sub(B, -128);
	cmp(N, 0x8);
	jl(l178, T_NEAR);
	align(4);

L(l20);
	mov(A1, A);
	add(A, 0x8);
	mov(I, M);
	sar(I, 0x3);
	jle(lcc, T_NEAR);
	align(4);

L(l34);
	movq(xmm0, ptr[A1-0x80]);
	add(A1, LDA);
	movq(xmm1, ptr[A1-0x80]);
	add(A1, LDA);
	movq(xmm2, ptr[A1-0x80]);
	add(A1, LDA);
	movq(xmm3, ptr[A1-0x80]);
	add(A1, LDA);
	punpcklbw(xmm0, xmm1);
	punpcklbw(xmm2, xmm3);
	movdqa(xmm1, xmm0);
	punpcklwd(xmm0, xmm2);
	punpckhwd(xmm1, xmm2);
	movdqu(ptr[B-0x80], xmm0);
	movdqu(ptr[B-0x70], xmm1);
	movq(xmm0, ptr[A1-0x80]);
	add(A1, LDA);
	movq(xmm1, ptr[A1-0x80]);
	add(A1, LDA);
	movq(xmm2, ptr[A1-0x80]);
	add(A1, LDA);
	movq(xmm3, ptr[A1-0x80]);
	add(A1, LDA);
	punpcklbw(xmm0, xmm1);
	punpcklbw(xmm2, xmm3);
	movdqa(xmm1, xmm0);
	punpcklwd(xmm0, xmm2);
	punpckhwd(xmm1, xmm2);
	movdqu(ptr[B-0x60], xmm0);
	movdqu(ptr[B-0x50], xmm1);
	sub(B, -64);
	dec(I);
	jg(l34, T_NEAR);
	align(4);

L(lcc);
	test(M, 0x4);
	jle(l120, T_NEAR);
	movq(xmm0, ptr[A1-0x80]);
	add(A1, LDA);
	movq(xmm1, ptr[A1-0x80]);
	add(A1, LDA);
	movq(xmm2, ptr[A1-0x80]);
	add(A1, LDA);
	movq(xmm3, ptr[A1-0x80]);
	add(A1, LDA);
	punpcklbw(xmm0, xmm1);
	punpcklbw(xmm2, xmm3);
	movdqa(xmm1, xmm0);
	punpcklwd(xmm0, xmm2);
	punpckhwd(xmm1, xmm2);
	movdqu(ptr[B-0x80], xmm0);
	movdqu(ptr[B-0x70], xmm1);
	sub(B, -32);
	align(4);

L(l120);
	test(M, 0x2);
	jle(l14c, T_NEAR);
	movq(xmm0, ptr[A1-0x80]);
	add(A1, LDA);
	movq(xmm1, ptr[A1-0x80]);
	add(A1, LDA);
	punpcklbw(xmm0, xmm1);
	movdqu(ptr[B-0x80], xmm0);
	sub(B, -16);
	align(4);

L(l14c);
	test(M, 0x1);
	jle(l168, T_NEAR);
	movq(xmm0, ptr[A1-0x80]);
	add(A1, LDA);
	movq(ptr[B-0x80], xmm0);
	sub(B, -8);
	align(4);

L(l168);
	sub(N, 0x8);
	cmp(N, 0x8);
	jge(l20, T_NEAR);
	align(4);

L(l178);
	cmp(N, 0x4);
	jl(l2a8, T_NEAR);
	align(4);

L(l184);
	mov(A1, A);
	add(A, 0x4);
	mov(I, M);
	sar(I, 0x3);
	jle(l20c, T_NEAR);
	align(4);

L(l194);
	movd(xmm0, ptr[A1-0x80]);
	add(A1, LDA);
	movd(xmm1, ptr[A1-0x80]);
	add(A1, LDA);
	movd(xmm2, ptr[A1-0x80]);
	add(A1, LDA);
	movd(xmm3, ptr[A1-0x80]);
	add(A1, LDA);
	punpcklbw(xmm0, xmm1);
	punpcklbw(xmm2, xmm3);
	punpcklwd(xmm0, xmm2);
	movdqu(ptr[B-0x80], xmm0);
	movd(xmm0, ptr[A1-0x80]);
	add(A1, LDA);
	movd(xmm1, ptr[A1-0x80]);
	add(A1, LDA);
	movd(xmm2, ptr[A1-0x80]);
	add(A1, LDA);
	movd(xmm3, ptr[A1-0x80]);
	add(A1, LDA);
	punpcklbw(xmm0, xmm1);
	punpcklbw(xmm2, xmm3);
	punpcklwd(xmm0, xmm2);
	movdqu(ptr[B-0x70], xmm0);
	sub(B, -32);
	dec(I);
	jg(l194, T_NEAR);
	align(4);

L(l20c);
	test(M, 0x4);
	jle(l250, T_NEAR);
	movd(xmm0, ptr[A1-0x80]);
	add(A1, LDA);
	movd(xmm1, ptr[A1-0x80]);
	add(A1, LDA);
	movd(xmm2, ptr[A1-0x80]);
	add(A1, LDA);
	movd(xmm3, ptr[A1-0x80]);
	add(A1, LDA);
	punpcklbw(xmm0, xmm1);
	punpcklbw(xmm2, xmm3);
	punpcklwd(xmm0, xmm2);
	movdqu(ptr[B-0x80], xmm0);
	sub(B, -16);
	align(4);

L(l250);
	test(M, 0x2);
	jle(l27c, T_NEAR);
	movd(xmm0, ptr[A1-0x80]);
	add(A1, LDA);
	movd(xmm1, ptr[A1-0x80]);
	add(A1, LDA);
	punpcklbw(xmm0, xmm1);
	movq(ptr[B-0x80], xmm0);
	sub(B, -8);
	align(4);

L(l27c);
	test(M, 0x1);
	jle(l298, T_NEAR);
	movd(xmm0, ptr[A1-0x80]);
	movd(ptr[B-0x80], xmm0);
	sub(B, -4);
	align(4);

L(l298);
	sub(N, 0x4);
	cmp(N, 0x4);
	jge(l184, T_NEAR);
	align(4);

L(l2a8);
	cmp(N, 0x2);
	jl(l412, T_NEAR);
	align(4);

L(l2b4);
	mov(A1, A);
	add(A, 0x2);
	mov(LDA3, M);
	sar(LDA3, 0x3);
	jle(l364, T_NEAR);
	align(4);

L(l2c8);
	mov(ax, ptr[A1-0x80]);
	add(A1, LDA);
	pinsrw(xmm0, eax, 0x0);
	mov(ax, ptr[A1-0x80]);
	add(A1, LDA);
	pinsrw(xmm1, eax, 0x0);
	mov(ax, ptr[A1-0x80]);
	add(A1, LDA);
	pinsrw(xmm2, eax, 0x0);
	mov(ax, ptr[A1-0x80]);
	add(A1, LDA);
	pinsrw(xmm3, eax, 0x0);
	punpcklbw(xmm0, xmm1);
	punpcklbw(xmm2, xmm3);
	punpcklwd(xmm0, xmm2);
	movq(ptr[B-0x80], xmm0);
	mov(ax, ptr[A1-0x80]);
	add(A1, LDA);
	pinsrw(xmm0, eax, 0x0);
	mov(ax, ptr[A1-0x80]);
	add(A1, LDA);
	pinsrw(xmm1, eax, 0x0);
	mov(ax, ptr[A1-0x80]);
	add(A1, LDA);
	pinsrw(xmm2, eax, 0x0);
	mov(ax, ptr[A1-0x80]);
	add(A1, LDA);
	pinsrw(xmm3, eax, 0x0);
	punpcklbw(xmm0, xmm1);
	punpcklbw(xmm2, xmm3);
	punpcklwd(xmm0, xmm2);
	movq(ptr[B-0x78], xmm0);
	sub(B, -16);
	dec(LDA3);
	jg(l2c8, T_NEAR);
	align(4);

L(l364);
	test(M, 0x4);
	jle(l3b8, T_NEAR);
	mov(ax, ptr[A1-0x80]);
	add(A1, LDA);
	pinsrw(xmm0, eax, 0x0);
	mov(ax, ptr[A1-0x80]);
	add(A1, LDA);
	pinsrw(xmm1, eax, 0x0);
	mov(ax, ptr[A1-0x80]);
	add(A1, LDA);
	pinsrw(xmm2, eax, 0x0);
	mov(ax, ptr[A1-0x80]);
	add(A1, LDA);
	pinsrw(xmm3, eax, 0x0);
	punpcklbw(xmm0, xmm1);
	punpcklbw(xmm2, xmm3);
	punpcklwd(xmm0, xmm2);
	movq(ptr[B-0x80], xmm0);
	sub(B, -8);
	align(4);

L(l3b8);
	test(M, 0x2);
	jle(l3ec, T_NEAR);
	mov(ax, ptr[A1-0x80]);
	add(A1, LDA);
	pinsrw(xmm0, eax, 0x0);
	mov(ax, ptr[A1-0x80]);
	add(A1, LDA);
	pinsrw(xmm1, eax, 0x0);
	punpcklbw(xmm0, xmm1);
	movd(ptr[B-0x80], xmm0);
	sub(B, -4);
	align(4);

L(l3ec);
	test(M, 0x1);
	jle(l404, T_NEAR);
	mov(ax, ptr[A1-0x80]);
	mov(ptr[B-0x80], ax);
	sub(B, -2);
	align(4);

L(l404);
	sub(N, 0x2);
	cmp(N, 0x2);
	jge(l2b4, T_NEAR);
	align(4);

L(l412);
	cmp(N, 0x1);
	jl(l538, T_NEAR);
	align(4);

L(l41c);
	mov(A1, A);
	add(A, 0x1);
	mov(LDA3, M);
	sar(LDA3, 0x3);
	jle(l4a4, T_NEAR);
	align(4);

L(l42c);
	mov(al, ptr[A1-0x80]);
	add(A1, LDA);
	pinsrb(xmm0, eax, 0x0);
	mov(al, ptr[A1-0x80]);
	add(A1, LDA);
	pinsrb(xmm0, eax, 0x1);
	mov(al, ptr[A1-0x80]);
	add(A1, LDA);
	pinsrb(xmm0, eax, 0x2);
	mov(al, ptr[A1-0x80]);
	add(A1, LDA);
	pinsrb(xmm0, eax, 0x3);
	mov(al, ptr[A1-0x80]);
	add(A1, LDA);
	pinsrb(xmm0, eax, 0x4);
	mov(al, ptr[A1-0x80]);
	add(A1, LDA);
	pinsrb(xmm0, eax, 0x5);
	mov(al, ptr[A1-0x80]);
	add(A1, LDA);
	pinsrb(xmm0, eax, 0x6);
	mov(al, ptr[A1-0x80]);
	add(A1, LDA);
	pinsrb(xmm0, eax, 0x7);
	movq(ptr[B-0x80], xmm0);
	sub(B, -8);
	dec(LDA3);
	jg(l42c, T_NEAR);
	align(4);

L(l4a4);
	test(M, 0x4);
	jle(l4ec, T_NEAR);
	mov(al, ptr[A1-0x80]);
	add(A1, LDA);
	pinsrb(xmm0, eax, 0x0);
	mov(al, ptr[A1-0x80]);
	add(A1, LDA);
	pinsrb(xmm0, eax, 0x1);
	mov(al, ptr[A1-0x80]);
	add(A1, LDA);
	pinsrb(xmm0, eax, 0x2);
	mov(al, ptr[A1-0x80]);
	add(A1, LDA);
	pinsrb(xmm0, eax, 0x3);
	movd(ptr[B-0x80], xmm0);
	sub(B, -4);
	align(4);

L(l4ec);
	test(M, 0x2);
	jle(l510, T_NEAR);
	mov(al, ptr[A1-0x80]);
	add(A1, LDA);
	mov(ptr[B-0x80], al);
	mov(al, ptr[A1-0x80]);
	add(A1, LDA);
	mov(ptr[B-0x7f], al);
	sub(B, -2);
	align(4);

L(l510);
	test(M, 0x1);
	jle(l528, T_NEAR);
	mov(al, ptr[A1-0x80]);
	mov(ptr[B-0x80], al);
	sub(B, -1);
	align(4);

L(l528);
	sub(N, 0x1);
	cmp(N, 0x1);
	jge(l41c, T_NEAR);
	align(4);

L(l538);
#ifdef _WIN32
	mov(rsi, ptr[rsp]);
	mov(rdi, ptr[rsp+0x8]);
	movups(xmm6, ptr[rsp+0x10]);
	movups(xmm7, ptr[rsp+0x20]);
	add(rsp, STACKSIZE);
#endif
	ret();
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
