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

jit_avx_kernel_gemm_s8u8s32_kern::jit_avx_kernel_gemm_s8u8s32_kern()
    : jit_generator(jit_name()) {}

void jit_avx_kernel_gemm_s8u8s32_kern::generate() {

#ifndef _WIN32

#define M rdi
#define N rsi
#define K rdx
#define A r8
#define B r9
#define C r10
#define LDC r11

#define AA rcx
#define I r12
#define J r13
#define H rax
#define AO r14
#define BO r15
#define CO1 rbx
#define CO2 rbp

#else

#define M rcx
#define N rdx
#define K r8
#define A rsi
#define B r9
#define C r10
#define LDC r11

#define AA rdi
#define I r12
#define J r13
#define H rax
#define AO r14
#define BO r15
#define CO1 rbx
#define CO2 rbp

#endif

#ifdef _WIN32
#define ARG_A (args_offset - 16) + rsp
#define ARG_B (args_offset - 8) + rsp
#endif
#define ARG_C ((args_offset + 0) + rsp)
#define ARG_LDC ((args_offset + 8) + rsp)

    inLocalLabel();
    {
        std::vector<Xbyak::Label> labels(91);

        auto stack_alloc_size = 32;
        auto args_offset = stack_alloc_size + get_size_of_abi_save_regs() + 8;
#ifdef _WIN32
        args_offset += 48;
#endif
        preamble();
        sub(rsp, stack_alloc_size);
#ifdef _WIN32
        mov(A, ptr[ARG_A]);
        mov(B, ptr[ARG_B]);
#endif

        mov(C, qword[ARG_C]);
        mov(LDC, qword[ARG_LDC]);
        sub(A, -128);
        sub(B, -128);
        mov(M, qword[M]);
        mov(N, qword[N]);
        mov(K, qword[K]);
        lea(LDC, ptr[LDC * 4 + 0x0]);
        vxorps(xmm8, xmm8, xmm8);
        vxorps(xmm9, xmm9, xmm9);
        vxorps(xmm10, xmm10, xmm10);
        vxorps(xmm11, xmm11, xmm11);
        vxorps(xmm12, xmm12, xmm12);
        vxorps(xmm13, xmm13, xmm13);
        vxorps(xmm14, xmm14, xmm14);
        vxorps(xmm15, xmm15, xmm15);
        mov(H, 0x10001);
        movq(xmm7, H);
        vpshufd(xmm7, xmm7, 0x0);
        mov(J, M);
        cmp(J, 0x10);
        jl(labels[73], T_NEAR);
        align(4);

        L(labels[70]);
        mov(CO1, C);
        add(C, 0x40);
        mov(BO, B);
        mov(AA, K);
        shl(AA, 0x20);
        lea(AA, ptr[A + AA * 1 + 0x200]);
        mov(I, N);
        cmp(I, 0x2);
        jl(labels[63], T_NEAR);
        align(4);

        L(labels[78]);
        mov(AO, A);
        vmovdqu(xmm0, xword[AO - 0x80]);
        vmovdqu(xmm1, xword[AO - 0x70]);
        vmovdqu(xmm2, xword[AO - 0x60]);
        vmovdqu(xmm3, xword[AO - 0x50]);
        vmovdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(labels[59], T_NEAR);
        sub(H, 0x8);
        jle(labels[57], T_NEAR);
        align(4);

        L(labels[88]);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm12, xmm12, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm14, xmm14, xmm6);
        prefetcht0(byte[BO]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm11, xmm11, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm13, xmm13, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm15, xmm15, xmm6);
        vmovdqu(xmm0, xword[AO - 0x40]);
        vmovdqu(xmm1, xword[AO - 0x30]);
        vmovdqu(xmm2, xword[AO - 0x20]);
        vmovdqu(xmm3, xword[AO - 0x10]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0xaa);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm12, xmm12, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm14, xmm14, xmm6);
        vpshufd(xmm4, xmm5, 0xff);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm11, xmm11, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm13, xmm13, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm15, xmm15, xmm6);
        vmovdqu(xmm5, xword[BO - 0x70]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO]);
        vmovdqu(xmm1, xword[AO + 0x10]);
        vmovdqu(xmm2, xword[AO + 0x20]);
        vmovdqu(xmm3, xword[AO + 0x30]);
        add(AA, 0x4);
        add(AO, 0x80);
        add(BO, 0x10);
        sub(H, 0x1);
        jg(labels[88], T_NEAR);
        align(4);

        L(labels[57]);
        prefetcht0(byte[CO1 + 0x3c]);
        prefetcht0(byte[CO1 + LDC * 1 + 0x3c]);
        add(H, 0x8);
        jle(labels[59], T_NEAR);
        align(4);

        L(labels[58]);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm12, xmm12, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm14, xmm14, xmm6);
        prefetcht0(byte[BO]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm11, xmm11, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm13, xmm13, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm15, xmm15, xmm6);
        vmovdqu(xmm0, xword[AO - 0x40]);
        vmovdqu(xmm1, xword[AO - 0x30]);
        vmovdqu(xmm2, xword[AO - 0x20]);
        vmovdqu(xmm3, xword[AO - 0x10]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0xaa);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm12, xmm12, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm14, xmm14, xmm6);
        vpshufd(xmm4, xmm5, 0xff);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm11, xmm11, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm13, xmm13, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm15, xmm15, xmm6);
        vmovdqu(xmm5, xword[BO - 0x70]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO]);
        vmovdqu(xmm1, xword[AO + 0x10]);
        vmovdqu(xmm2, xword[AO + 0x20]);
        vmovdqu(xmm3, xword[AO + 0x30]);
        add(AA, 0x4);
        add(AO, 0x80);
        add(BO, 0x10);
        sub(H, 0x1);
        jg(labels[58], T_NEAR);
        align(4);

        L(labels[59]);
        mov(H, K);
        test(H, 0x4);
        je(labels[60], T_NEAR);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm12, xmm12, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm14, xmm14, xmm6);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm11, xmm11, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm13, xmm13, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm15, xmm15, xmm6);
        add(AO, 0x40);
        add(BO, 0x8);
        align(4);

        L(labels[60]);
        mov(H, K);
        test(H, 0x2);
        je(labels[61], T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vmovdqu(xmm1, xword[AO - 0x80]);
        vpunpcklwd(xmm0, xmm1, xmm6);
        vpunpckhwd(xmm1, xmm1, xmm6);
        vmovdqu(xmm3, xword[AO - 0x70]);
        vpunpcklwd(xmm2, xmm3, xmm6);
        vpunpckhwd(xmm3, xmm3, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm12, xmm12, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm14, xmm14, xmm6);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm11, xmm11, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm13, xmm13, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm15, xmm15, xmm6);
        add(AO, 0x20);
        add(BO, 0x4);
        align(4);

        L(labels[61]);
        mov(H, K);
        test(H, 0x1);
        je(labels[62], T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vbroadcastss(xmm0, dword[AO - 0x80]);
        vpunpcklbw(xmm0, xmm0, xmm6);
        vpunpcklwd(xmm0, xmm0, xmm6);
        vbroadcastss(xmm1, dword[AO - 0x7c]);
        vpunpcklbw(xmm1, xmm1, xmm6);
        vpunpcklwd(xmm1, xmm1, xmm6);
        vbroadcastss(xmm2, dword[AO - 0x78]);
        vpunpcklbw(xmm2, xmm2, xmm6);
        vpunpcklwd(xmm2, xmm2, xmm6);
        vbroadcastss(xmm3, dword[AO - 0x74]);
        vpunpcklbw(xmm3, xmm3, xmm6);
        vpunpcklwd(xmm3, xmm3, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklbw(xmm5, xmm5, xmm5);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm12, xmm12, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm14, xmm14, xmm6);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm11, xmm11, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm13, xmm13, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm15, xmm15, xmm6);
        add(AO, 0x10);
        add(BO, 0x2);
        align(4);

        L(labels[62]);
        vmovdqu(xmm0, xword[CO1]);
        vpaddd(xmm8, xmm8, xmm0);
        vmovdqu(xword[CO1], xmm8);
        vxorps(xmm8, xmm8, xmm8);
        vmovdqu(xmm0, xword[CO1 + 0x10]);
        vpaddd(xmm10, xmm10, xmm0);
        vmovdqu(xword[CO1 + 0x10], xmm10);
        vxorps(xmm10, xmm10, xmm10);
        vmovdqu(xmm0, xword[CO1 + 0x20]);
        vpaddd(xmm12, xmm12, xmm0);
        vmovdqu(xword[CO1 + 0x20], xmm12);
        vxorps(xmm12, xmm12, xmm12);
        vmovdqu(xmm0, xword[CO1 + 0x30]);
        vpaddd(xmm14, xmm14, xmm0);
        vmovdqu(xword[CO1 + 0x30], xmm14);
        vxorps(xmm14, xmm14, xmm14);
        vmovdqu(xmm0, xword[CO1 + LDC * 1]);
        vpaddd(xmm9, xmm9, xmm0);
        vmovdqu(xword[CO1 + LDC * 1], xmm9);
        vxorps(xmm9, xmm9, xmm9);
        vmovdqu(xmm0, xword[CO1 + LDC * 1 + 0x10]);
        vpaddd(xmm11, xmm11, xmm0);
        vmovdqu(xword[CO1 + LDC * 1 + 0x10], xmm11);
        vxorps(xmm11, xmm11, xmm11);
        vmovdqu(xmm0, xword[CO1 + LDC * 1 + 0x20]);
        vpaddd(xmm13, xmm13, xmm0);
        vmovdqu(xword[CO1 + LDC * 1 + 0x20], xmm13);
        vxorps(xmm13, xmm13, xmm13);
        vmovdqu(xmm0, xword[CO1 + LDC * 1 + 0x30]);
        vpaddd(xmm15, xmm15, xmm0);
        vmovdqu(xword[CO1 + LDC * 1 + 0x30], xmm15);
        vxorps(xmm15, xmm15, xmm15);
        lea(CO1, ptr[CO1 + LDC * 2]);
        sub(I, 0x2);
        cmp(I, 0x2);
        jge(labels[78], T_NEAR);
        align(4);

        L(labels[63]);
        test(I, 0x1);
        jle(labels[72], T_NEAR);
        mov(AO, A);
        vmovdqu(xmm0, xword[AO - 0x80]);
        vmovdqu(xmm1, xword[AO - 0x70]);
        vmovdqu(xmm2, xword[AO - 0x60]);
        vmovdqu(xmm3, xword[AO - 0x50]);
        vmovdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(labels[67], T_NEAR);
        sub(H, 0x8);
        jle(labels[65], T_NEAR);
        align(4);

        L(labels[64]);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm12, xmm12, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm14, xmm14, xmm6);
        prefetcht0(byte[BO]);
        vmovdqu(xmm0, xword[AO - 0x40]);
        vmovdqu(xmm1, xword[AO - 0x30]);
        vmovdqu(xmm2, xword[AO - 0x20]);
        vmovdqu(xmm3, xword[AO - 0x10]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm12, xmm12, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm14, xmm14, xmm6);
        vmovdqu(xmm5, xword[BO - 0x78]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO]);
        vmovdqu(xmm1, xword[AO + 0x10]);
        vmovdqu(xmm2, xword[AO + 0x20]);
        vmovdqu(xmm3, xword[AO + 0x30]);
        add(AA, 0x4);
        add(AO, 0x80);
        add(BO, 0x8);
        sub(H, 0x1);
        jg(labels[64], T_NEAR);
        align(4);

        L(labels[65]);
        prefetcht0(byte[CO1 + 0x3c]);
        add(H, 0x8);
        jle(labels[67], T_NEAR);
        align(4);

        L(labels[66]);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm12, xmm12, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm14, xmm14, xmm6);
        prefetcht0(byte[BO]);
        vmovdqu(xmm0, xword[AO - 0x40]);
        vmovdqu(xmm1, xword[AO - 0x30]);
        vmovdqu(xmm2, xword[AO - 0x20]);
        vmovdqu(xmm3, xword[AO - 0x10]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm12, xmm12, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm14, xmm14, xmm6);
        vmovdqu(xmm5, xword[BO - 0x78]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO]);
        vmovdqu(xmm1, xword[AO + 0x10]);
        vmovdqu(xmm2, xword[AO + 0x20]);
        vmovdqu(xmm3, xword[AO + 0x30]);
        add(AA, 0x4);
        add(AO, 0x80);
        add(BO, 0x8);
        sub(H, 0x1);
        jg(labels[66], T_NEAR);
        align(4);

        L(labels[67]);
        mov(H, K);
        test(H, 0x4);
        je(labels[68], T_NEAR);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm12, xmm12, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm14, xmm14, xmm6);
        add(AO, 0x40);
        add(BO, 0x4);
        align(4);

        L(labels[68]);
        mov(H, K);
        test(H, 0x2);
        je(labels[69], T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vmovdqu(xmm1, xword[AO - 0x80]);
        vpunpcklwd(xmm0, xmm1, xmm6);
        vpunpckhwd(xmm1, xmm1, xmm6);
        vmovdqu(xmm3, xword[AO - 0x70]);
        vpunpcklwd(xmm2, xmm3, xmm6);
        vpunpckhwd(xmm3, xmm3, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm12, xmm12, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm14, xmm14, xmm6);
        add(AO, 0x20);
        add(BO, 0x2);
        align(4);

        L(labels[69]);
        mov(H, K);
        test(H, 0x1);
        je(labels[71], T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vbroadcastss(xmm0, dword[AO - 0x80]);
        vpunpcklbw(xmm0, xmm0, xmm6);
        vpunpcklwd(xmm0, xmm0, xmm6);
        vbroadcastss(xmm1, dword[AO - 0x7c]);
        vpunpcklbw(xmm1, xmm1, xmm6);
        vpunpcklwd(xmm1, xmm1, xmm6);
        vbroadcastss(xmm2, dword[AO - 0x78]);
        vpunpcklbw(xmm2, xmm2, xmm6);
        vpunpcklwd(xmm2, xmm2, xmm6);
        vbroadcastss(xmm3, dword[AO - 0x74]);
        vpunpcklbw(xmm3, xmm3, xmm6);
        vpunpcklwd(xmm3, xmm3, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklbw(xmm5, xmm5, xmm5);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm2);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm12, xmm12, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm3);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm14, xmm14, xmm6);
        add(AO, 0x10);
        add(BO, 0x1);
        align(4);

        L(labels[71]);
        vmovdqu(xmm0, xword[CO1]);
        vpaddd(xmm8, xmm8, xmm0);
        vmovdqu(xword[CO1], xmm8);
        vxorps(xmm8, xmm8, xmm8);
        vmovdqu(xmm0, xword[CO1 + 0x10]);
        vpaddd(xmm10, xmm10, xmm0);
        vmovdqu(xword[CO1 + 0x10], xmm10);
        vxorps(xmm10, xmm10, xmm10);
        vmovdqu(xmm0, xword[CO1 + 0x20]);
        vpaddd(xmm12, xmm12, xmm0);
        vmovdqu(xword[CO1 + 0x20], xmm12);
        vxorps(xmm12, xmm12, xmm12);
        vmovdqu(xmm0, xword[CO1 + 0x30]);
        vpaddd(xmm14, xmm14, xmm0);
        vmovdqu(xword[CO1 + 0x30], xmm14);
        vxorps(xmm14, xmm14, xmm14);
        lea(CO1, ptr[CO1 + LDC * 1]);
        align(4);

        L(labels[72]);
        mov(A, AO);
        sub(J, 0x10);
        cmp(J, 0x10);
        jge(labels[70], T_NEAR);
        align(4);

        L(labels[73]);
        test(J, 0x8);
        jle(labels[2], T_NEAR);
        mov(CO1, C);
        add(C, 0x20);
        mov(BO, B);
        mov(AA, K);
        shl(AA, 0x10);
        lea(AA, ptr[A + AA * 1 + 0x200]);
        mov(I, N);
        cmp(I, 0x2);
        jl(labels[83], T_NEAR);
        align(4);

        L(labels[74]);
        mov(AO, A);
        vmovdqu(xmm0, xword[AO - 0x80]);
        vmovdqu(xmm1, xword[AO - 0x70]);
        vmovdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(labels[79], T_NEAR);
        sub(H, 0x8);
        jle(labels[76], T_NEAR);
        align(4);

        L(labels[75]);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        prefetcht0(byte[BO]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm11, xmm11, xmm6);
        vmovdqu(xmm0, xword[AO - 0x60]);
        vmovdqu(xmm1, xword[AO - 0x50]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0xaa);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpshufd(xmm4, xmm5, 0xff);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm11, xmm11, xmm6);
        vmovdqu(xmm5, xword[BO - 0x70]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO - 0x40]);
        vmovdqu(xmm1, xword[AO - 0x30]);
        add(AA, 0x4);
        add(AO, 0x40);
        add(BO, 0x10);
        sub(H, 0x1);
        jg(labels[75], T_NEAR);
        align(4);

        L(labels[76]);
        prefetcht0(byte[CO1 + 0x3c]);
        prefetcht0(byte[CO1 + LDC * 1 + 0x3c]);
        add(H, 0x8);
        jle(labels[79], T_NEAR);
        align(4);

        L(labels[77]);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        prefetcht0(byte[BO]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm11, xmm11, xmm6);
        vmovdqu(xmm0, xword[AO - 0x60]);
        vmovdqu(xmm1, xword[AO - 0x50]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0xaa);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpshufd(xmm4, xmm5, 0xff);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm11, xmm11, xmm6);
        vmovdqu(xmm5, xword[BO - 0x70]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO - 0x40]);
        vmovdqu(xmm1, xword[AO - 0x30]);
        add(AA, 0x4);
        add(AO, 0x40);
        add(BO, 0x10);
        sub(H, 0x1);
        jg(labels[77], T_NEAR);
        align(4);

        L(labels[79]);
        mov(H, K);
        test(H, 0x4);
        je(labels[80], T_NEAR);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm11, xmm11, xmm6);
        add(AO, 0x20);
        add(BO, 0x8);
        align(4);

        L(labels[80]);
        mov(H, K);
        test(H, 0x2);
        je(labels[81], T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vmovdqu(xmm1, xword[AO - 0x80]);
        vpunpcklwd(xmm0, xmm1, xmm6);
        vpunpckhwd(xmm1, xmm1, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm11, xmm11, xmm6);
        add(AO, 0x10);
        add(BO, 0x4);
        align(4);

        L(labels[81]);
        mov(H, K);
        test(H, 0x1);
        je(labels[82], T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vbroadcastss(xmm0, dword[AO - 0x80]);
        vpunpcklbw(xmm0, xmm0, xmm6);
        vpunpcklwd(xmm0, xmm0, xmm6);
        vbroadcastss(xmm1, dword[AO - 0x7c]);
        vpunpcklbw(xmm1, xmm1, xmm6);
        vpunpcklwd(xmm1, xmm1, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklbw(xmm5, xmm5, xmm5);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm11, xmm11, xmm6);
        add(AO, 0x8);
        add(BO, 0x2);
        align(4);

        L(labels[82]);
        vmovdqu(xmm0, xword[CO1]);
        vpaddd(xmm8, xmm8, xmm0);
        vmovdqu(xword[CO1], xmm8);
        vxorps(xmm8, xmm8, xmm8);
        vmovdqu(xmm0, xword[CO1 + 0x10]);
        vpaddd(xmm10, xmm10, xmm0);
        vmovdqu(xword[CO1 + 0x10], xmm10);
        vxorps(xmm10, xmm10, xmm10);
        vmovdqu(xmm0, xword[CO1 + LDC * 1]);
        vpaddd(xmm9, xmm9, xmm0);
        vmovdqu(xword[CO1 + LDC * 1], xmm9);
        vxorps(xmm9, xmm9, xmm9);
        vmovdqu(xmm0, xword[CO1 + LDC * 1 + 0x10]);
        vpaddd(xmm11, xmm11, xmm0);
        vmovdqu(xword[CO1 + LDC * 1 + 0x10], xmm11);
        vxorps(xmm11, xmm11, xmm11);
        lea(CO1, ptr[CO1 + LDC * 2]);
        sub(I, 0x2);
        cmp(I, 0x2);
        jge(labels[74], T_NEAR);
        align(4);

        L(labels[83]);
        test(I, 0x1);
        jle(labels[1], T_NEAR);
        mov(AO, A);
        vmovdqu(xmm0, xword[AO - 0x80]);
        vmovdqu(xmm1, xword[AO - 0x70]);
        vmovdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(labels[87], T_NEAR);
        sub(H, 0x8);
        jle(labels[85], T_NEAR);
        align(4);

        L(labels[84]);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        prefetcht0(byte[BO]);
        vmovdqu(xmm0, xword[AO - 0x60]);
        vmovdqu(xmm1, xword[AO - 0x50]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vmovdqu(xmm5, xword[BO - 0x78]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO - 0x40]);
        vmovdqu(xmm1, xword[AO - 0x30]);
        add(AA, 0x4);
        add(AO, 0x40);
        add(BO, 0x8);
        sub(H, 0x1);
        jg(labels[84], T_NEAR);
        align(4);

        L(labels[85]);
        prefetcht0(byte[CO1 + 0x3c]);
        add(H, 0x8);
        jle(labels[87], T_NEAR);
        align(4);

        L(labels[86]);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        prefetcht0(byte[BO]);
        vmovdqu(xmm0, xword[AO - 0x60]);
        vmovdqu(xmm1, xword[AO - 0x50]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        vmovdqu(xmm5, xword[BO - 0x78]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO - 0x40]);
        vmovdqu(xmm1, xword[AO - 0x30]);
        add(AA, 0x4);
        add(AO, 0x40);
        add(BO, 0x8);
        sub(H, 0x1);
        jg(labels[86], T_NEAR);
        align(4);

        L(labels[87]);
        mov(H, K);
        test(H, 0x4);
        je(labels[89], T_NEAR);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        add(AO, 0x20);
        add(BO, 0x4);
        align(4);

        L(labels[89]);
        mov(H, K);
        test(H, 0x2);
        je(labels[90], T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vmovdqu(xmm1, xword[AO - 0x80]);
        vpunpcklwd(xmm0, xmm1, xmm6);
        vpunpckhwd(xmm1, xmm1, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        add(AO, 0x10);
        add(BO, 0x2);
        align(4);

        L(labels[90]);
        mov(H, K);
        test(H, 0x1);
        je(labels[0], T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vbroadcastss(xmm0, dword[AO - 0x80]);
        vpunpcklbw(xmm0, xmm0, xmm6);
        vpunpcklwd(xmm0, xmm0, xmm6);
        vbroadcastss(xmm1, dword[AO - 0x7c]);
        vpunpcklbw(xmm1, xmm1, xmm6);
        vpunpcklwd(xmm1, xmm1, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklbw(xmm5, xmm5, xmm5);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpmaddubsw(xmm6, xmm4, xmm1);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm10, xmm10, xmm6);
        add(AO, 0x8);
        add(BO, 0x1);
        align(4);

        L(labels[0]);
        vmovdqu(xmm0, xword[CO1]);
        vpaddd(xmm8, xmm8, xmm0);
        vmovdqu(xword[CO1], xmm8);
        vxorps(xmm8, xmm8, xmm8);
        vmovdqu(xmm0, xword[CO1 + 0x10]);
        vpaddd(xmm10, xmm10, xmm0);
        vmovdqu(xword[CO1 + 0x10], xmm10);
        vxorps(xmm10, xmm10, xmm10);
        lea(CO1, ptr[CO1 + LDC * 1]);
        align(4);

        L(labels[1]);
        mov(A, AO);
        align(4);

        L(labels[2]);
        test(J, 0x4);
        jle(labels[20], T_NEAR);
        mov(CO1, C);
        add(C, 0x10);
        mov(BO, B);
        mov(AA, K);
        shl(AA, 0x8);
        lea(AA, ptr[A + AA * 1 + 0x200]);
        mov(I, N);
        cmp(I, 0x2);
        jl(labels[11], T_NEAR);
        align(4);

        L(labels[3]);
        mov(AO, A);
        vmovdqu(xmm0, xword[AO - 0x80]);
        vmovdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(labels[7], T_NEAR);
        sub(H, 0x8);
        jle(labels[5], T_NEAR);
        align(4);

        L(labels[4]);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        prefetcht0(byte[BO]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vmovdqu(xmm0, xword[AO - 0x70]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0xaa);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpshufd(xmm4, xmm5, 0xff);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vmovdqu(xmm5, xword[BO - 0x70]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO - 0x60]);
        add(AA, 0x4);
        add(AO, 0x20);
        add(BO, 0x10);
        sub(H, 0x1);
        jg(labels[4], T_NEAR);
        align(4);

        L(labels[5]);
        prefetcht0(byte[CO1 + 0x3c]);
        prefetcht0(byte[CO1 + LDC * 1 + 0x3c]);
        add(H, 0x8);
        jle(labels[7], T_NEAR);
        align(4);

        L(labels[6]);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        prefetcht0(byte[BO]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vmovdqu(xmm0, xword[AO - 0x70]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0xaa);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpshufd(xmm4, xmm5, 0xff);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vmovdqu(xmm5, xword[BO - 0x70]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO - 0x60]);
        add(AA, 0x4);
        add(AO, 0x20);
        add(BO, 0x10);
        sub(H, 0x1);
        jg(labels[6], T_NEAR);
        align(4);

        L(labels[7]);
        mov(H, K);
        test(H, 0x4);
        je(labels[8], T_NEAR);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        add(AO, 0x10);
        add(BO, 0x8);
        align(4);

        L(labels[8]);
        mov(H, K);
        test(H, 0x2);
        je(labels[9], T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vmovdqu(xmm1, xword[AO - 0x80]);
        vpunpcklwd(xmm0, xmm1, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        add(AO, 0x8);
        add(BO, 0x4);
        align(4);

        L(labels[9]);
        mov(H, K);
        test(H, 0x1);
        je(labels[10], T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vbroadcastss(xmm0, dword[AO - 0x80]);
        vpunpcklbw(xmm0, xmm0, xmm6);
        vpunpcklwd(xmm0, xmm0, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklbw(xmm5, xmm5, xmm5);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        add(AO, 0x4);
        add(BO, 0x2);
        align(4);

        L(labels[10]);
        vmovdqu(xmm0, xword[CO1]);
        vpaddd(xmm8, xmm8, xmm0);
        vmovdqu(xword[CO1], xmm8);
        vxorps(xmm8, xmm8, xmm8);
        vmovdqu(xmm0, xword[CO1 + LDC * 1]);
        vpaddd(xmm9, xmm9, xmm0);
        vmovdqu(xword[CO1 + LDC * 1], xmm9);
        vxorps(xmm9, xmm9, xmm9);
        lea(CO1, ptr[CO1 + LDC * 2]);
        sub(I, 0x2);
        cmp(I, 0x2);
        jge(labels[3], T_NEAR);
        align(4);

        L(labels[11]);
        test(I, 0x1);
        jle(labels[19], T_NEAR);
        mov(AO, A);
        vmovdqu(xmm0, xword[AO - 0x80]);
        vmovdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(labels[15], T_NEAR);
        sub(H, 0x8);
        jle(labels[13], T_NEAR);
        align(4);

        L(labels[12]);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        prefetcht0(byte[BO]);
        vmovdqu(xmm0, xword[AO - 0x70]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vmovdqu(xmm5, xword[BO - 0x78]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO - 0x60]);
        add(AA, 0x4);
        add(AO, 0x20);
        add(BO, 0x8);
        sub(H, 0x1);
        jg(labels[12], T_NEAR);
        align(4);

        L(labels[13]);
        prefetcht0(byte[CO1 + 0x3c]);
        add(H, 0x8);
        jle(labels[15], T_NEAR);
        align(4);

        L(labels[14]);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        prefetcht0(byte[BO]);
        vmovdqu(xmm0, xword[AO - 0x70]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vmovdqu(xmm5, xword[BO - 0x78]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO - 0x60]);
        add(AA, 0x4);
        add(AO, 0x20);
        add(BO, 0x8);
        sub(H, 0x1);
        jg(labels[14], T_NEAR);
        align(4);

        L(labels[15]);
        mov(H, K);
        test(H, 0x4);
        je(labels[16], T_NEAR);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        add(AO, 0x10);
        add(BO, 0x4);
        align(4);

        L(labels[16]);
        mov(H, K);
        test(H, 0x2);
        je(labels[17], T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vmovdqu(xmm1, xword[AO - 0x80]);
        vpunpcklwd(xmm0, xmm1, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        add(AO, 0x8);
        add(BO, 0x2);
        align(4);

        L(labels[17]);
        mov(H, K);
        test(H, 0x1);
        je(labels[18], T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vbroadcastss(xmm0, dword[AO - 0x80]);
        vpunpcklbw(xmm0, xmm0, xmm6);
        vpunpcklwd(xmm0, xmm0, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklbw(xmm5, xmm5, xmm5);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        add(AO, 0x4);
        add(BO, 0x1);
        align(4);

        L(labels[18]);
        vmovdqu(xmm0, xword[CO1]);
        vpaddd(xmm8, xmm8, xmm0);
        vmovdqu(xword[CO1], xmm8);
        vxorps(xmm8, xmm8, xmm8);
        lea(CO1, ptr[CO1 + LDC * 1]);
        align(4);

        L(labels[19]);
        mov(A, AO);
        align(4);

        L(labels[20]);
        test(J, 0x2);
        jle(labels[38], T_NEAR);
        mov(CO1, C);
        add(C, 0x8);
        mov(BO, B);
        mov(AA, K);
        shl(AA, 0x4);
        lea(AA, ptr[A + AA * 1 + 0x200]);
        mov(I, N);
        cmp(I, 0x2);
        jl(labels[29], T_NEAR);
        align(4);

        L(labels[21]);
        mov(AO, A);
        vmovdqu(xmm0, xword[AO - 0x80]);
        vmovdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(labels[25], T_NEAR);
        sub(H, 0x8);
        jle(labels[23], T_NEAR);
        align(4);

        L(labels[22]);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        prefetcht0(byte[BO]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vmovdqu(xmm0, xword[AO - 0x78]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0xaa);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpshufd(xmm4, xmm5, 0xff);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vmovdqu(xmm5, xword[BO - 0x70]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO - 0x70]);
        add(AA, 0x4);
        add(AO, 0x10);
        add(BO, 0x10);
        sub(H, 0x1);
        jg(labels[22], T_NEAR);
        align(4);

        L(labels[23]);
        prefetcht0(byte[CO1 + 0x3c]);
        prefetcht0(byte[CO1 + LDC * 1 + 0x3c]);
        add(H, 0x8);
        jle(labels[25], T_NEAR);
        align(4);

        L(labels[24]);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        prefetcht0(byte[BO]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vmovdqu(xmm0, xword[AO - 0x78]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0xaa);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpshufd(xmm4, xmm5, 0xff);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vmovdqu(xmm5, xword[BO - 0x70]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO - 0x70]);
        add(AA, 0x4);
        add(AO, 0x10);
        add(BO, 0x10);
        sub(H, 0x1);
        jg(labels[24], T_NEAR);
        align(4);

        L(labels[25]);
        mov(H, K);
        test(H, 0x4);
        je(labels[26], T_NEAR);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        add(AO, 0x8);
        add(BO, 0x8);
        align(4);

        L(labels[26]);
        mov(H, K);
        test(H, 0x2);
        je(labels[27], T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vmovdqu(xmm1, xword[AO - 0x80]);
        vpunpcklwd(xmm0, xmm1, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        add(AO, 0x4);
        add(BO, 0x4);
        align(4);

        L(labels[27]);
        mov(H, K);
        test(H, 0x1);
        je(labels[28], T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vbroadcastss(xmm0, dword[AO - 0x80]);
        vpunpcklbw(xmm0, xmm0, xmm6);
        vpunpcklwd(xmm0, xmm0, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklbw(xmm5, xmm5, xmm5);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        add(AO, 0x2);
        add(BO, 0x2);
        align(4);

        L(labels[28]);
        vmovsd(xmm0, qword[CO1]);
        vpaddd(xmm8, xmm8, xmm0);
        vmovlps(qword[CO1], xmm8);
        vxorps(xmm8, xmm8, xmm8);
        vmovsd(xmm0, qword[CO1 + LDC * 1]);
        vpaddd(xmm9, xmm9, xmm0);
        vmovlps(qword[CO1 + LDC * 1], xmm9);
        vxorps(xmm9, xmm9, xmm9);
        lea(CO1, ptr[CO1 + LDC * 2]);
        sub(I, 0x2);
        cmp(I, 0x2);
        jge(labels[21], T_NEAR);
        align(4);

        L(labels[29]);
        test(I, 0x1);
        jle(labels[37], T_NEAR);
        mov(AO, A);
        vmovdqu(xmm0, xword[AO - 0x80]);
        vmovdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(labels[33], T_NEAR);
        sub(H, 0x8);
        jle(labels[31], T_NEAR);
        align(4);

        L(labels[30]);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        prefetcht0(byte[BO]);
        vmovdqu(xmm0, xword[AO - 0x78]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vmovdqu(xmm5, xword[BO - 0x78]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO - 0x70]);
        add(AA, 0x4);
        add(AO, 0x10);
        add(BO, 0x8);
        sub(H, 0x1);
        jg(labels[30], T_NEAR);
        align(4);

        L(labels[31]);
        prefetcht0(byte[CO1 + 0x3c]);
        add(H, 0x8);
        jle(labels[33], T_NEAR);
        align(4);

        L(labels[32]);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        prefetcht0(byte[BO]);
        vmovdqu(xmm0, xword[AO - 0x78]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vmovdqu(xmm5, xword[BO - 0x78]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO - 0x70]);
        add(AA, 0x4);
        add(AO, 0x10);
        add(BO, 0x8);
        sub(H, 0x1);
        jg(labels[32], T_NEAR);
        align(4);

        L(labels[33]);
        mov(H, K);
        test(H, 0x4);
        je(labels[34], T_NEAR);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        add(AO, 0x8);
        add(BO, 0x4);
        align(4);

        L(labels[34]);
        mov(H, K);
        test(H, 0x2);
        je(labels[35], T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vmovdqu(xmm1, xword[AO - 0x80]);
        vpunpcklwd(xmm0, xmm1, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        add(AO, 0x4);
        add(BO, 0x2);
        align(4);

        L(labels[35]);
        mov(H, K);
        test(H, 0x1);
        je(labels[36], T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vbroadcastss(xmm0, dword[AO - 0x80]);
        vpunpcklbw(xmm0, xmm0, xmm6);
        vpunpcklwd(xmm0, xmm0, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklbw(xmm5, xmm5, xmm5);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        add(AO, 0x2);
        add(BO, 0x1);
        align(4);

        L(labels[36]);
        vmovsd(xmm0, qword[CO1]);
        vpaddd(xmm8, xmm8, xmm0);
        vmovlps(qword[CO1], xmm8);
        vxorps(xmm8, xmm8, xmm8);
        lea(CO1, ptr[CO1 + LDC * 1]);
        align(4);

        L(labels[37]);
        mov(A, AO);
        align(4);

        L(labels[38]);
        test(J, 0x1);
        jle(labels[56], T_NEAR);
        mov(CO1, C);
        add(C, 0x4);
        mov(BO, B);
        mov(AA, K);
        shl(AA, 0x2);
        lea(AA, ptr[A + AA * 1 + 0x200]);
        mov(I, N);
        cmp(I, 0x2);
        jl(labels[47], T_NEAR);
        align(4);

        L(labels[39]);
        mov(AO, A);
        vmovdqu(xmm0, xword[AO - 0x80]);
        vmovdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(labels[43], T_NEAR);
        sub(H, 0x8);
        jle(labels[41], T_NEAR);
        align(4);

        L(labels[40]);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        prefetcht0(byte[BO]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vmovdqu(xmm0, xword[AO - 0x7c]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0xaa);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpshufd(xmm4, xmm5, 0xff);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vmovdqu(xmm5, xword[BO - 0x70]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO - 0x78]);
        add(AA, 0x4);
        add(AO, 0x8);
        add(BO, 0x10);
        sub(H, 0x1);
        jg(labels[40], T_NEAR);
        align(4);

        L(labels[41]);
        prefetcht0(byte[CO1 + 0x3c]);
        prefetcht0(byte[CO1 + LDC * 1 + 0x3c]);
        add(H, 0x8);
        jle(labels[43], T_NEAR);
        align(4);

        L(labels[42]);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        prefetcht0(byte[BO]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vmovdqu(xmm0, xword[AO - 0x7c]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0xaa);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpshufd(xmm4, xmm5, 0xff);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        vmovdqu(xmm5, xword[BO - 0x70]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO - 0x78]);
        add(AA, 0x4);
        add(AO, 0x8);
        add(BO, 0x10);
        sub(H, 0x1);
        jg(labels[42], T_NEAR);
        align(4);

        L(labels[43]);
        mov(H, K);
        test(H, 0x4);
        je(labels[44], T_NEAR);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        add(AO, 0x4);
        add(BO, 0x8);
        align(4);

        L(labels[44]);
        mov(H, K);
        test(H, 0x2);
        je(labels[45], T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vmovdqu(xmm1, xword[AO - 0x80]);
        vpunpcklwd(xmm0, xmm1, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        add(AO, 0x2);
        add(BO, 0x4);
        align(4);

        L(labels[45]);
        mov(H, K);
        test(H, 0x1);
        je(labels[46], T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vbroadcastss(xmm0, dword[AO - 0x80]);
        vpunpcklbw(xmm0, xmm0, xmm6);
        vpunpcklwd(xmm0, xmm0, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklbw(xmm5, xmm5, xmm5);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm9, xmm9, xmm6);
        add(AO, 0x1);
        add(BO, 0x2);
        align(4);

        L(labels[46]);
        vmovss(xmm0, dword[CO1]);
        vpaddd(xmm8, xmm8, xmm0);
        vmovss(dword[CO1], xmm8);
        vxorps(xmm8, xmm8, xmm8);
        vmovss(xmm0, dword[CO1 + LDC * 1]);
        vpaddd(xmm9, xmm9, xmm0);
        vmovss(dword[CO1 + LDC * 1], xmm9);
        vxorps(xmm9, xmm9, xmm9);
        lea(CO1, ptr[CO1 + LDC * 2]);
        sub(I, 0x2);
        cmp(I, 0x2);
        jge(labels[39], T_NEAR);
        align(4);

        L(labels[47]);
        test(I, 0x1);
        jle(labels[55], T_NEAR);
        mov(AO, A);
        vmovdqu(xmm0, xword[AO - 0x80]);
        vmovdqu(xmm5, xword[BO - 0x80]);
        mov(H, K);
        sar(H, 0x3);
        jle(labels[51], T_NEAR);
        sub(H, 0x8);
        jle(labels[49], T_NEAR);
        align(4);

        L(labels[48]);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        prefetcht0(byte[BO]);
        vmovdqu(xmm0, xword[AO - 0x7c]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vmovdqu(xmm5, xword[BO - 0x78]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO - 0x78]);
        add(AA, 0x4);
        add(AO, 0x8);
        add(BO, 0x8);
        sub(H, 0x1);
        jg(labels[48], T_NEAR);
        align(4);

        L(labels[49]);
        prefetcht0(byte[CO1 + 0x3c]);
        add(H, 0x8);
        jle(labels[51], T_NEAR);
        align(4);

        L(labels[50]);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        prefetcht0(byte[AO + 0x180]);
        prefetcht0(byte[BO]);
        vmovdqu(xmm0, xword[AO - 0x7c]);
        prefetcht0(byte[AO + 0x1c0]);
        vpshufd(xmm4, xmm5, 0x55);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        vmovdqu(xmm5, xword[BO - 0x78]);
        prefetcht1(byte[AA - 0x80]);
        vmovdqu(xmm0, xword[AO - 0x78]);
        add(AA, 0x4);
        add(AO, 0x8);
        add(BO, 0x8);
        sub(H, 0x1);
        jg(labels[50], T_NEAR);
        align(4);

        L(labels[51]);
        mov(H, K);
        test(H, 0x4);
        je(labels[52], T_NEAR);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        add(AO, 0x4);
        add(BO, 0x4);
        align(4);

        L(labels[52]);
        mov(H, K);
        test(H, 0x2);
        je(labels[53], T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vmovdqu(xmm1, xword[AO - 0x80]);
        vpunpcklwd(xmm0, xmm1, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        add(AO, 0x2);
        add(BO, 0x2);
        align(4);

        L(labels[53]);
        mov(H, K);
        test(H, 0x1);
        je(labels[54], T_NEAR);
        vxorps(xmm6, xmm6, xmm6);
        vbroadcastss(xmm0, dword[AO - 0x80]);
        vpunpcklbw(xmm0, xmm0, xmm6);
        vpunpcklwd(xmm0, xmm0, xmm6);
        vbroadcastss(xmm5, dword[BO - 0x80]);
        vpunpcklbw(xmm5, xmm5, xmm5);
        vpunpcklwd(xmm5, xmm5, xmm5);
        vpshufd(xmm4, xmm5, 0x0);
        vpmaddubsw(xmm6, xmm4, xmm0);
        vpmaddwd(xmm6, xmm7, xmm6);
        vpaddd(xmm8, xmm8, xmm6);
        add(AO, 0x1);
        add(BO, 0x1);
        align(4);

        L(labels[54]);
        vmovss(xmm0, dword[CO1]);
        vpaddd(xmm8, xmm8, xmm0);
        vmovss(dword[CO1], xmm8);
        vxorps(xmm8, xmm8, xmm8);
        lea(CO1, ptr[CO1 + LDC * 1]);
        align(4);

        L(labels[55]);
        mov(A, AO);
        align(4);

        L(labels[56]);
        add(rsp, stack_alloc_size);
        postamble();
    }
    outLocalLabel();

#undef M
#undef N
#undef K
#undef A
#undef B
#undef C
#undef LDC
#undef AA
#undef I
#undef J
#undef H
#undef AO
#undef BO
#undef CO1
#undef CO2
#ifdef _WIN32
#undef ARG_A
#undef ARG_B
#endif
#undef ARG_C
#undef ARG_LDC
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
