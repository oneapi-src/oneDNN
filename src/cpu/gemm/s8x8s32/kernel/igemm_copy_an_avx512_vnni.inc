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

Xbyak::Label l170;
Xbyak::Label l1f0;
Xbyak::Label l20;
Xbyak::Label l224;
Xbyak::Label l234;
Xbyak::Label l240;
Xbyak::Label l254;
Xbyak::Label l32c;
Xbyak::Label l34;
Xbyak::Label l388;
Xbyak::Label l3b0;
Xbyak::Label l3c0;
Xbyak::Label l3cc;
Xbyak::Label l3dc;
Xbyak::Label l454;
Xbyak::Label l48c;
Xbyak::Label l4a8;
Xbyak::Label l4b8;
Xbyak::Label l4c4;
Xbyak::Label l4d8;
Xbyak::Label l570;
Xbyak::Label l5c4;
Xbyak::Label l5f0;
Xbyak::Label l60c;
Xbyak::Label l61c;
Xbyak::Label l628;
Xbyak::Label l638;
Xbyak::Label l6b0;
Xbyak::Label l6f4;
Xbyak::Label l720;
Xbyak::Label l73c;
Xbyak::Label l74c;
Xbyak::Label l758;
Xbyak::Label l76c;
Xbyak::Label l808;
Xbyak::Label l85c;
Xbyak::Label l890;
Xbyak::Label l8a8;
Xbyak::Label l8b6;
Xbyak::Label l8c0;
Xbyak::Label l8d0;
Xbyak::Label l948;
Xbyak::Label l990;
Xbyak::Label l9b4;
Xbyak::Label l9cc;
Xbyak::Label l9dc;

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
	cmp(N, 0x30);
	jl(l234, T_NEAR);
	align(4);

L(l20);
	mov(A1, A);
	add(A, 0x30);
	mov(I, M);
	sar(I, 0x2);
	jle(l170, T_NEAR);
	align(4);

L(l34);
	movdqu(xmm0, ptr[A1-0x80]);
	movdqu(xmm1, ptr[A1+LDA*1-0x80]);
	movdqu(xmm2, ptr[A1+LDA*2-0x80]);
	movdqu(xmm3, ptr[A1+LDA3*1-0x80]);
	movdqa(xmm4, xmm0);
	punpcklbw(xmm0, xmm1);
	punpckhbw(xmm4, xmm1);
	movdqa(xmm5, xmm2);
	punpcklbw(xmm2, xmm3);
	punpckhbw(xmm5, xmm3);
	movdqa(xmm1, xmm0);
	punpcklwd(xmm0, xmm2);
	punpckhwd(xmm1, xmm2);
	movdqa(xmm2, xmm4);
	punpcklwd(xmm4, xmm5);
	punpckhwd(xmm2, xmm5);
	movdqu(ptr[B-0x80], xmm0);
	movdqu(ptr[B-0x70], xmm1);
	movdqu(ptr[B-0x60], xmm4);
	movdqu(ptr[B-0x50], xmm2);
	movdqu(xmm0, ptr[A1-0x70]);
	movdqu(xmm1, ptr[A1+LDA*1-0x70]);
	movdqu(xmm2, ptr[A1+LDA*2-0x70]);
	movdqu(xmm3, ptr[A1+LDA3*1-0x70]);
	movdqa(xmm4, xmm0);
	punpcklbw(xmm0, xmm1);
	punpckhbw(xmm4, xmm1);
	movdqa(xmm5, xmm2);
	punpcklbw(xmm2, xmm3);
	punpckhbw(xmm5, xmm3);
	movdqa(xmm1, xmm0);
	punpcklwd(xmm0, xmm2);
	punpckhwd(xmm1, xmm2);
	movdqa(xmm2, xmm4);
	punpcklwd(xmm4, xmm5);
	punpckhwd(xmm2, xmm5);
	movdqu(ptr[B-0x40], xmm0);
	movdqu(ptr[B-0x30], xmm1);
	movdqu(ptr[B-0x20], xmm4);
	movdqu(ptr[B-0x10], xmm2);
	movdqu(xmm0, ptr[A1-0x60]);
	movdqu(xmm1, ptr[A1+LDA*1-0x60]);
	movdqu(xmm2, ptr[A1+LDA*2-0x60]);
	movdqu(xmm3, ptr[A1+LDA3*1-0x60]);
	lea(A1, ptr[A1+LDA*4]);
	movdqa(xmm4, xmm0);
	punpcklbw(xmm0, xmm1);
	punpckhbw(xmm4, xmm1);
	movdqa(xmm5, xmm2);
	punpcklbw(xmm2, xmm3);
	punpckhbw(xmm5, xmm3);
	movdqa(xmm1, xmm0);
	punpcklwd(xmm0, xmm2);
	punpckhwd(xmm1, xmm2);
	movdqa(xmm2, xmm4);
	punpcklwd(xmm4, xmm5);
	punpckhwd(xmm2, xmm5);
	movdqu(ptr[B], xmm0);
	movdqu(ptr[B+0x10], xmm1);
	movdqu(ptr[B+0x20], xmm4);
	movdqu(ptr[B+0x30], xmm2);
	sub(B, -192);
	dec(I);
	jg(l34, T_NEAR);
	align(4);

L(l170);
	test(M, 0x2);
	jle(l1f0, T_NEAR);
	movdqu(xmm0, ptr[A1-0x80]);
	movdqu(xmm1, ptr[A1-0x70]);
	movdqu(xmm2, ptr[A1-0x60]);
	add(A1, LDA);
	movdqu(xmm3, ptr[A1-0x80]);
	movdqu(xmm4, ptr[A1-0x70]);
	movdqu(xmm5, ptr[A1-0x60]);
	add(A1, LDA);
	movdqa(xmm6, xmm0);
	punpcklbw(xmm0, xmm3);
	punpckhbw(xmm6, xmm3);
	movdqu(ptr[B-0x80], xmm0);
	movdqu(ptr[B-0x70], xmm6);
	movdqa(xmm6, xmm1);
	punpcklbw(xmm1, xmm4);
	punpckhbw(xmm6, xmm4);
	movdqu(ptr[B-0x60], xmm1);
	movdqu(ptr[B-0x50], xmm6);
	movdqa(xmm6, xmm2);
	punpcklbw(xmm2, xmm5);
	punpckhbw(xmm6, xmm5);
	movdqu(ptr[B-0x40], xmm2);
	movdqu(ptr[B-0x30], xmm6);
	sub(B, -96);
	align(4);

L(l1f0);
	test(M, 0x1);
	jle(l224, T_NEAR);
	movdqu(xmm0, ptr[A1-0x80]);
	movdqu(xmm1, ptr[A1-0x70]);
	movdqu(xmm2, ptr[A1-0x60]);
	add(A1, LDA);
	movdqu(ptr[B-0x80], xmm0);
	movdqu(ptr[B-0x70], xmm1);
	movdqu(ptr[B-0x60], xmm2);
	sub(B, -48);
	align(4);

L(l224);
	sub(N, 0x30);
	cmp(N, 0x30);
	jge(l20, T_NEAR);
	align(4);

L(l234);
	cmp(N, 0x20);
	jl(l3c0, T_NEAR);
	align(4);

L(l240);
	mov(A1, A);
	add(A, 0x20);
	mov(I, M);
	sar(I, 0x2);
	jle(l32c, T_NEAR);
	align(4);

L(l254);
	movdqu(xmm0, ptr[A1-0x80]);
	movdqu(xmm1, ptr[A1+LDA*1-0x80]);
	movdqu(xmm2, ptr[A1+LDA*2-0x80]);
	movdqu(xmm3, ptr[A1+LDA3*1-0x80]);
	movdqa(xmm4, xmm0);
	punpcklbw(xmm0, xmm1);
	punpckhbw(xmm4, xmm1);
	movdqa(xmm5, xmm2);
	punpcklbw(xmm2, xmm3);
	punpckhbw(xmm5, xmm3);
	movdqa(xmm1, xmm0);
	punpcklwd(xmm0, xmm2);
	punpckhwd(xmm1, xmm2);
	movdqa(xmm2, xmm4);
	punpcklwd(xmm4, xmm5);
	punpckhwd(xmm2, xmm5);
	movdqu(ptr[B-0x80], xmm0);
	movdqu(ptr[B-0x70], xmm1);
	movdqu(ptr[B-0x60], xmm4);
	movdqu(ptr[B-0x50], xmm2);
	movdqu(xmm0, ptr[A1-0x70]);
	movdqu(xmm1, ptr[A1+LDA*1-0x70]);
	movdqu(xmm2, ptr[A1+LDA*2-0x70]);
	movdqu(xmm3, ptr[A1+LDA3*1-0x70]);
	lea(A1, ptr[A1+LDA*4]);
	movdqa(xmm4, xmm0);
	punpcklbw(xmm0, xmm1);
	punpckhbw(xmm4, xmm1);
	movdqa(xmm5, xmm2);
	punpcklbw(xmm2, xmm3);
	punpckhbw(xmm5, xmm3);
	movdqa(xmm1, xmm0);
	punpcklwd(xmm0, xmm2);
	punpckhwd(xmm1, xmm2);
	movdqa(xmm2, xmm4);
	punpcklwd(xmm4, xmm5);
	punpckhwd(xmm2, xmm5);
	movdqu(ptr[B-0x40], xmm0);
	movdqu(ptr[B-0x30], xmm1);
	movdqu(ptr[B-0x20], xmm4);
	movdqu(ptr[B-0x10], xmm2);
	sub(B, -128);
	dec(I);
	jg(l254, T_NEAR);
	align(4);

L(l32c);
	test(M, 0x2);
	jle(l388, T_NEAR);
	movdqu(xmm0, ptr[A1-0x80]);
	movdqu(xmm1, ptr[A1-0x70]);
	add(A1, LDA);
	movdqu(xmm2, ptr[A1-0x80]);
	movdqu(xmm3, ptr[A1-0x70]);
	add(A1, LDA);
	movdqa(xmm4, xmm0);
	punpcklbw(xmm0, xmm2);
	punpckhbw(xmm4, xmm2);
	movdqu(ptr[B-0x80], xmm0);
	movdqu(ptr[B-0x70], xmm4);
	movdqa(xmm4, xmm1);
	punpcklbw(xmm1, xmm3);
	punpckhbw(xmm4, xmm3);
	movdqu(ptr[B-0x60], xmm1);
	movdqu(ptr[B-0x50], xmm4);
	sub(B, -64);
	align(4);

L(l388);
	test(M, 0x1);
	jle(l3b0, T_NEAR);
	movdqu(xmm0, ptr[A1-0x80]);
	movdqu(xmm1, ptr[A1-0x70]);
	add(A1, LDA);
	movdqu(ptr[B-0x80], xmm0);
	movdqu(ptr[B-0x70], xmm1);
	sub(B, -32);
	align(4);

L(l3b0);
	sub(N, 0x20);
	cmp(N, 0x20);
	jge(l240, T_NEAR);
	align(4);

L(l3c0);
	cmp(N, 0x10);
	jl(l4b8, T_NEAR);
	align(4);

L(l3cc);
	mov(A1, A);
	add(A, 0x10);
	mov(I, M);
	sar(I, 0x2);
	jle(l454, T_NEAR);
	align(4);

L(l3dc);
	movdqu(xmm0, ptr[A1-0x80]);
	add(A1, LDA);
	movdqu(xmm1, ptr[A1-0x80]);
	add(A1, LDA);
	movdqu(xmm2, ptr[A1-0x80]);
	add(A1, LDA);
	movdqu(xmm3, ptr[A1-0x80]);
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
	movdqu(ptr[B-0x80], xmm0);
	movdqu(ptr[B-0x70], xmm3);
	movdqu(ptr[B-0x60], xmm4);
	movdqu(ptr[B-0x50], xmm2);
	sub(B, -64);
	dec(I);
	jg(l3dc, T_NEAR);
	align(4);

L(l454);
	test(M, 0x2);
	jle(l48c, T_NEAR);
	movdqu(xmm0, ptr[A1-0x80]);
	add(A1, LDA);
	movdqu(xmm1, ptr[A1-0x80]);
	add(A1, LDA);
	movdqa(xmm2, xmm0);
	punpcklbw(xmm0, xmm1);
	punpckhbw(xmm2, xmm1);
	movdqu(ptr[B-0x80], xmm0);
	movdqu(ptr[B-0x70], xmm2);
	sub(B, -32);
	align(4);

L(l48c);
	test(M, 0x1);
	jle(l4a8, T_NEAR);
	movdqu(xmm0, ptr[A1-0x80]);
	add(A1, LDA);
	movdqu(ptr[B-0x80], xmm0);
	sub(B, -16);
	align(4);

L(l4a8);
	sub(N, 0x10);
	cmp(N, 0x10);
	jge(l3cc, T_NEAR);
	align(4);

L(l4b8);
	cmp(N, 0x8);
	jl(l61c, T_NEAR);
	align(4);

L(l4c4);
	mov(A1, A);
	add(A, 0x8);
	mov(I, M);
	sar(I, 0x3);
	jle(l570, T_NEAR);
	align(4);

L(l4d8);
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
	jg(l4d8, T_NEAR);
	align(4);

L(l570);
	test(M, 0x4);
	jle(l5c4, T_NEAR);
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

L(l5c4);
	test(M, 0x2);
	jle(l5f0, T_NEAR);
	movq(xmm0, ptr[A1-0x80]);
	add(A1, LDA);
	movq(xmm1, ptr[A1-0x80]);
	add(A1, LDA);
	punpcklbw(xmm0, xmm1);
	movdqu(ptr[B-0x80], xmm0);
	sub(B, -16);
	align(4);

L(l5f0);
	test(M, 0x1);
	jle(l60c, T_NEAR);
	movq(xmm0, ptr[A1-0x80]);
	add(A1, LDA);
	movq(ptr[B-0x80], xmm0);
	sub(B, -8);
	align(4);

L(l60c);
	sub(N, 0x8);
	cmp(N, 0x8);
	jge(l4c4, T_NEAR);
	align(4);

L(l61c);
	cmp(N, 0x4);
	jl(l74c, T_NEAR);
	align(4);

L(l628);
	mov(A1, A);
	add(A, 0x4);
	mov(I, M);
	sar(I, 0x3);
	jle(l6b0, T_NEAR);
	align(4);

L(l638);
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
	jg(l638, T_NEAR);
	align(4);

L(l6b0);
	test(M, 0x4);
	jle(l6f4, T_NEAR);
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

L(l6f4);
	test(M, 0x2);
	jle(l720, T_NEAR);
	movd(xmm0, ptr[A1-0x80]);
	add(A1, LDA);
	movd(xmm1, ptr[A1-0x80]);
	add(A1, LDA);
	punpcklbw(xmm0, xmm1);
	movq(ptr[B-0x80], xmm0);
	sub(B, -8);
	align(4);

L(l720);
	test(M, 0x1);
	jle(l73c, T_NEAR);
	movd(xmm0, ptr[A1-0x80]);
	movd(ptr[B-0x80], xmm0);
	sub(B, -4);
	align(4);

L(l73c);
	sub(N, 0x4);
	cmp(N, 0x4);
	jge(l628, T_NEAR);
	align(4);

L(l74c);
	cmp(N, 0x2);
	jl(l8b6, T_NEAR);
	align(4);

L(l758);
	mov(A1, A);
	add(A, 0x2);
	mov(LDA3, M);
	sar(LDA3, 0x3);
	jle(l808, T_NEAR);
	align(4);

L(l76c);
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
	jg(l76c, T_NEAR);
	align(4);

L(l808);
	test(M, 0x4);
	jle(l85c, T_NEAR);
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

L(l85c);
	test(M, 0x2);
	jle(l890, T_NEAR);
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

L(l890);
	test(M, 0x1);
	jle(l8a8, T_NEAR);
	mov(ax, ptr[A1-0x80]);
	mov(ptr[B-0x80], ax);
	sub(B, -2);
	align(4);

L(l8a8);
	sub(N, 0x2);
	cmp(N, 0x2);
	jge(l758, T_NEAR);
	align(4);

L(l8b6);
	cmp(N, 0x1);
	jl(l9dc, T_NEAR);
	align(4);

L(l8c0);
	mov(A1, A);
	add(A, 0x1);
	mov(LDA3, M);
	sar(LDA3, 0x3);
	jle(l948, T_NEAR);
	align(4);

L(l8d0);
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
	jg(l8d0, T_NEAR);
	align(4);

L(l948);
	test(M, 0x4);
	jle(l990, T_NEAR);
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

L(l990);
	test(M, 0x2);
	jle(l9b4, T_NEAR);
	mov(al, ptr[A1-0x80]);
	add(A1, LDA);
	mov(ptr[B-0x80], al);
	mov(al, ptr[A1-0x80]);
	add(A1, LDA);
	mov(ptr[B-0x7f], al);
	sub(B, -2);
	align(4);

L(l9b4);
	test(M, 0x1);
	jle(l9cc, T_NEAR);
	mov(al, ptr[A1-0x80]);
	mov(ptr[B-0x80], al);
	sub(B, -1);
	align(4);

L(l9cc);
	sub(N, 0x1);
	cmp(N, 0x1);
	jge(l8c0, T_NEAR);
	align(4);

L(l9dc);
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
