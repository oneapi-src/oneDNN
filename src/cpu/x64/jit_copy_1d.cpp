/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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

#include <typeinfo>

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"

#include "jit_copy_1d.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace spmd {

template <typename T, int batch>
struct jit_copy_1d {
    bool is_generic() const { return true; }
    template <typename... Args>
    void operator()(Args...) {}
};

template <typename T, int batch>
struct jit_zero_1d {
    bool is_generic() const { return true; }
    template <typename... Args>
    void operator()(Args...) {}
};

// Skip windows, for now
#ifndef _WIN32
using namespace Xbyak;

template <typename derived>
struct jit_t : public jit_generator {
    void generate() override {
        if (mayiuse(avx512_common)) {
            reinterpret_cast<derived *>(this)->generate_avx512();
        } else if (mayiuse(avx2)) {
            reinterpret_cast<derived *>(this)->generate_avx2();
        } else {
            use_generic_ = true;
        }
    }

    bool is_generic() const { return use_generic_; }
    jit_t() {
        create_kernel();
        jit_ker_ += ker_off_;
    }

    const char *name() const override { return typeid(derived).name(); }
    const char *source_file() const override { return __FILE__; }

protected:
    size_t ker_off_ {0};
    bool use_generic_ {false};
};

template <>
struct jit_copy_1d<float, 1> : public jit_t<jit_copy_1d<float, 1>> {
    using CodeGenerator::L;
    void generate_avx2() {
        align(2 << 5); // -- begin function copy_1d_simd
        Label LCPI6_0 = L();
        dd(0); // 0x0
        dd(1); // 0x1
        dd(2); // 0x2
        dd(3); // 0x3
        dd(4); // 0x4
        dd(5); // 0x5
        dd(6); // 0x6
        dd(7); // 0x7
        align(2 << 2);
        Label LCPI6_1 = L();
        dd(8); // 0x8
        align(2 << 4);
        ker_off_ = this->getSize(); // @copy_1d_simd
        // %bb.0:                   // %allocas
        // kill: def $edx killed $edx def $rdx
        vmovd(xmm0, r8d);
        vpbroadcastd(ymm0, xmm0);
        vpcmpgtd(ymm1, ymm0, yword[rip + LCPI6_0]);
        vmovmskps(eax, ymm1);
        test(al, al);
        Label LBB6_5;
        je(LBB6_5, T_NEAR);
        // %bb.1:                     // %for_loop.lr.ph
        vmovdqa(ymm1, yword[rip + LCPI6_0]); // ymm1 = [0,1,2,3,4,5,6,7]
        vpcmpgtd(ymm2, ymm0, ymm1);
        vmovd(xmm3, edx);
        vpbroadcastd(ymm3, xmm3);
        vmovd(xmm4, ecx);
        vpbroadcastd(ymm4, xmm4);
        shl(edx, 2);
        xor_(eax, eax);
        vpcmpeqd(ymm5, ymm5, ymm5);
        vxorps(xmm6, xmm6, xmm6);
        vpbroadcastd(ymm7, dword[rip + LCPI6_1]); // ymm7 = [8,8,8,8,8,8,8,8]
                // implicit-def: $ymm8
        Label LBB6_2;
        jmp(LBB6_2);
        align(2 << 4);
        Label LBB6_4;
        L(LBB6_4); // $select_done
        //   in Loop: Header=BB6_2 Depth=1
        vblendvps(ymm9, ymm6, ymm8, ymm9);
        cdqe();
        vmaskmovps(yword[rdi + rax], ymm2, ymm9);
        vpaddd(ymm1, ymm1, ymm7);
        vpcmpgtd(ymm9, ymm0, ymm1);
        vpand(ymm2, ymm9, ymm2);
        vmovmskps(ecx, ymm2);
        add(eax, 32);
        test(cl, cl);
        je(LBB6_5, T_NEAR);
        L(LBB6_2); // %for_loop
                // =>This Inner Loop Header: Depth=1
        vpaddd(ymm9, ymm1, ymm3);
        vpcmpgtd(ymm10, ymm9, ymm5);
        vpcmpgtd(ymm9, ymm4, ymm9);
        vpand(ymm9, ymm9, ymm10);
        vpand(ymm10, ymm9, ymm2);
        vmovmskps(ecx, ymm10);
        test(cl, cl);
        je(LBB6_4);
        // %bb.3:                       // %select_eval_expr
        //    in Loop: Header=BB6_2 Depth=1
        lea(ecx, ptr[rdx + rax]);
        movsxd(rcx, ecx);
        vmaskmovps(ymm8, ymm10, yword[rsi + rcx]);
        jmp(LBB6_4);
        L(LBB6_5);
        vzeroupper();
        ret();
    }
    void generate_avx512() {
        align(2 << 5);
        Label LCPI6_0 = L();
        dd(0); // 0x0
        dd(1); // 0x1
        dd(2); // 0x2
        dd(3); // 0x3
        dd(4); // 0x4
        dd(5); // 0x5
        dd(6); // 0x6
        dd(7); // 0x7
        align(2 << 2);
        Label LCPI6_1 = L();
        dd(8); // 0x8
        align(2 << 4);
        ker_off_ = getSize(); // @copy_1d_simd
        // %bb.0:                 // %allocas
        // kill: def $edx killed $edx def $rdx
        vpbroadcastd(ymm0, r8d);
        vpcmpgtd(k1, ymm0, yword[rip + LCPI6_0]);
        kortestb(k1, k1);
        Label LBB6_5;
        je(LBB6_5);
        // %bb.1:                 // %for_loop.lr.ph
        vpbroadcastd(ymm1, edx);
        vpbroadcastd(ymm2, ecx);
        shl(edx, 2);
        vmovdqa(ymm3, yword[rip + LCPI6_0]); // ymm3 = [0,1,2,3,4,5,6,7]
        xor_(eax, eax);
        vpcmpeqd(ymm4, ymm4, ymm4);
        vpbroadcastd(ymm5, dword[rip + LCPI6_1]); // ymm5 = [8,8,8,8,8,8,8,8]
                // implicit-def: $ymm6
        Label LBB6_2;
        jmp(LBB6_2);
        align(2 << 4);
        Label LBB6_4 = L(); // %select_done
                //   in Loop: Header=BB6_2 Depth=1
        vmovaps(ymm7 | k2 | T_z, ymm6);
        cdqe();
        vmovups(yword[rdi + rax] | k1, ymm7);
        vpaddd(ymm3, ymm3, ymm5);
        vpcmpgtd(k1 | k1, ymm0, ymm3);
        add(eax, 32);
        kortestb(k1, k1);
        je(LBB6_5);
        L(LBB6_2); // %for_loop
                // =>This Inner Loop Header: Depth=1
        vpaddd(ymm7, ymm3, ymm1);
        vpcmpgtd(k2, ymm7, ymm4);
        vpcmpgtd(k2 | k2, ymm2, ymm7);
        kandb(k3, k2, k1);
        kortestb(k3, k3);
        je(LBB6_4);
        // %BB.3:                             // %select_eval_expr
        //   in Loop: Header=BB6_2 Depth=1
        lea(ecx, ptr[rdx + rax]);
        movsxd(rcx, ecx);
        vmovups(ymm6 | k3 | T_z, yword[rsi + rcx]);
        jmp(LBB6_4);
        L(LBB6_5); // %for_exit
        vzeroupper();
        ret();
    }
};

template <>
struct jit_copy_1d<float, 4> : public jit_t<jit_copy_1d<float, 4>> {
    using CodeGenerator::L;
    void generate_avx2() {
        align(2 << 5);
        Label LCPI7_0 = L();
        dd(0); // 0x0
        dd(1); // 0x1
        dd(2); // 0x2
        dd(3); // 0x3
        dd(4); // 0x4
        dd(5); // 0x5
        dd(6); // 0x6
        dd(7); // 0x7
        align(2 << 2);
        Label LCPI7_1 = L();
        dd(8); // 0x8
        align(2 << 4);
        ker_off_ = getSize(); // @copy_1d_simd_c_unroll_4
        // %bb.0:                               // %allocas
        push(rbp);
        push(r15);
        push(r14);
        push(rbx);
        // kill: def $r8d killed $r8d def $r8
        vpbroadcastd(ymm0, dword[rsp + 40]);
        vpcmpgtd(ymm1, ymm0, yword[rip + LCPI7_0]);
        vmovmskps(eax, ymm1);
        test(al, al);
        Label LBB7_5;
        je(LBB7_5, T_NEAR);
        // %bb.1:                               // %for_loop.lr.ph
        vmovdqa(ymm1, yword[rip + LCPI7_0]); // ymm1 = [0,1,2,3,4,5,6,7]
        vpcmpgtd(ymm2, ymm0, ymm1);
        vmovd(xmm3, r8d);
        vpbroadcastd(ymm3, xmm3);
        vmovd(xmm4, r9d);
        vpbroadcastd(ymm4, xmm4);
        shl(r8d, 2);
        xor_(eax, eax);
        vpcmpeqd(ymm5, ymm5, ymm5);
        vxorps(xmm6, xmm6, xmm6);
        movsxd(r15, esi);
        lea(r9, ptr[rdi + 4 * r15]);
        lea(esi, ptr[r15 + r15]);
        movsxd(rsi, esi);
        lea(rsi, ptr[rdi + 4 * rsi]);
        lea(ebx, ptr[r15 + 2 * r15]);
        movsxd(rbx, ebx);
        lea(r10, ptr[rdi + 4 * rbx]);
        vpbroadcastd(ymm7, dword[rip + LCPI7_1]); // ymm7 = [8,8,8,8,8,8,8,8]
        movsxd(rcx, ecx);
        lea(r11, ptr[rdx + 4 * rcx]);
        lea(ebx, ptr[rcx + rcx]);
        movsxd(rbx, ebx);
        lea(r14, ptr[rdx + 4 * rbx]);
        lea(ebx, ptr[r15 + 2 * r15]);
        movsxd(rbx, ebx);
        lea(r15, ptr[rdi + 4 * rbx]);
        lea(ecx, ptr[rcx + 2 * rcx]);
        movsxd(rcx, ecx);
        lea(rcx, ptr[rdx + 4 * rcx]);
        // implicit-def: $ymm8
        Label LBB7_2;
        jmp(LBB7_2);
        align(2 << 4);
        Label LBB7_3 = L(); // %select_done.3
                //    in Loop: Header=BB7_2 Depth=1
        lea(ebp, ptr[r8 + rbx]);
        movsxd(rbp, ebp);
        vmaskmovps(ymm8, ymm10, yword[rdx + rbp]);
        vblendvps(ymm8, ymm6, ymm8, ymm9);
        vmaskmovps(yword[rdi + rbx], ymm2, ymm8);
        vmaskmovps(ymm8, ymm10, yword[rbp + r11]);
        vblendvps(ymm8, ymm6, ymm8, ymm9);
        vmaskmovps(yword[rbx + r9], ymm2, ymm8);
        vmaskmovps(ymm8, ymm10, yword[rbp + r14]);
        vblendvps(ymm8, ymm6, ymm8, ymm9);
        vmaskmovps(yword[rbx + rsi], ymm2, ymm8);
        vmaskmovps(ymm8, ymm10, yword[rbp + rcx]);
        vblendvps(ymm9, ymm6, ymm8, ymm9);
        vmaskmovps(yword[rbx + r15], ymm2, ymm9);
        Label LBB7_4 = L(); // %for_exit22
                //   in Loop: Header=BB7_2 Depth=1
        vpaddd(ymm1, ymm1, ymm7);
        vpcmpgtd(ymm9, ymm0, ymm1);
        vpand(ymm2, ymm9, ymm2);
        vmovmskps(ebx, ymm2);
        add(eax, 32);
        test(bl, bl);
        je(LBB7_5);
        L(LBB7_2); // %for_loop
                // =>This Inner Loop Header: Depth=1
        vpaddd(ymm9, ymm1, ymm3);
        vpcmpgtd(ymm10, ymm9, ymm5);
        vpcmpgtd(ymm9, ymm4, ymm9);
        vpand(ymm9, ymm9, ymm10);
        vpand(ymm10, ymm9, ymm2);
        vmovmskps(ebp, ymm10);
        movsxd(rbx, eax);
        test(bpl, bpl);
        jne(LBB7_3);
        // %bb.6:                       // %select_done.us.3
        //                              //   in Loop: Header=BB7_2 Depth=1
        vblendvps(ymm9, ymm6, ymm8, ymm9);
        vmaskmovps(yword[rdi + rbx], ymm2, ymm9);
        vmaskmovps(yword[rbx + r9], ymm2, ymm9);
        vmaskmovps(yword[rbx + rsi], ymm2, ymm9);
        vmaskmovps(yword[rbx + r10], ymm2, ymm9);
        jmp(LBB7_4);
        L(LBB7_5); // %for_exit
        pop(rbx);
        pop(r14);
        pop(r15);
        pop(rbp);
        vzeroupper();
        ret();
    }
    void generate_avx512() {
        align(2 << 5);
        Label LCPI7_0 = L();
        dd(0); // 0x0
        dd(1); // 0x1
        dd(2); // 0x2
        dd(3); // 0x3
        dd(4); // 0x4
        dd(5); // 0x5
        dd(6); // 0x6
        dd(7); // 0x7
        align(2 << 2);
        Label LCPI7_1 = L();
        dd(8); // 0x8
        align(2 << 4);
        ker_off_ = getSize(); // @copy_1d_simd_c_unroll_4
        // %bb.0:                   // %allocas
        push(rbp);
        push(r15);
        push(r14);
        push(rbx);
        // kill: def $r8d killed $r8d def $r8
        vpbroadcastd(ymm0, dword[rsp + 40]);
        vpcmpgtd(k1, ymm0, yword[rip + LCPI7_0]);
        kortestb(k1, k1);
        Label LBB7_5;
        je(LBB7_5, T_NEAR);
        // %bb.1:                   // %for_loop.lr.ph
        vpbroadcastd(ymm1, r8d);
        vpbroadcastd(ymm2, r9d);
        vmovdqa(ymm3, yword[rip + LCPI7_0]); // ymm3 = [0,1,2,3,4,5,6,7]
        shl(r8d, 2);
        xor_(eax, eax);
        vpcmpeqd(ymm4, ymm4, ymm4);
        movsxd(r15, esi);
        lea(r9, ptr[rdi + 4 * r15]);
        lea(esi, ptr[r15 + r15]);
        movsxd(rsi, esi);
        lea(rsi, ptr[rdi + 4 * rsi]);
        lea(ebx, ptr[r15 + 2 * r15]);
        movsxd(rbx, ebx);
        vpbroadcastd(ymm5, dword[rip + LCPI7_1]); // ymm5 = [8,8,8,8,8,8,8,8]
        lea(r10, ptr[rdi + 4 * rbx]);
        movsxd(rcx, ecx);
        lea(r11, ptr[rdx + 4 * rcx]);
        lea(ebx, ptr[rcx + rcx]);
        movsxd(rbx, ebx);
        lea(r14, ptr[rdx + 4 * rbx]);
        lea(ebx, ptr[r15 + 2 * r15]);
        movsxd(rbx, ebx);
        lea(r15, ptr[rdi + 4 * rbx]);
        lea(ecx, ptr[rcx + 2 * rcx]);
        movsxd(rcx, ecx);
        lea(rcx, ptr[rdx + 4 * rcx]);
        // implicit-def: $ymm6
        Label LBB7_2;
        jmp(LBB7_2);
        align(2 << 4);
        Label LBB7_3 = L(); // %select_done.3
                //   in Loop: Header=BB7_2 Depth=1
        lea(ebp, ptr[r8 + rbx]);
        movsxd(rbp, ebp);
        vmovups(ymm6 | k3 | T_z, yword[rdx + rbp]);
        vmovaps(ymm6 | k2 | T_z, ymm6);
        vmovups(yword[rdi + rbx] | k1, ymm6);
        vmovups(ymm6 | k3 | T_z, yword[rbp + r11]);
        vmovaps(ymm6 | k2 | T_z, ymm6);
        vmovups(yword[rbx + r9] | k1, ymm6);
        vmovups(ymm6 | k3 | T_z, yword[rbp + r14]);
        vmovaps(ymm6 | k2 | T_z, ymm6);
        vmovups(yword[rbx + rsi] | k1, ymm6);
        vmovups(ymm6 | k3 | T_z, yword[rbp + rcx]);
        vmovaps(ymm7 | k2 | T_z, ymm6);
        vmovups(yword[rbx + r15] | k1, ymm7);
        Label LBB7_4 = L(); // %for_exit20
                //   in Loop: Header=BB7_2 Depth=1
        vpaddd(ymm3, ymm3, ymm5);
        vpcmpgtd(k1 | k1, ymm0, ymm3);
        add(eax, 32);
        kortestb(k1, k1);
        je(LBB7_5);
        L(LBB7_2); // %for_loop
                // =>This Inner Loop Header: Depth=1
        vpaddd(ymm7, ymm3, ymm1);
        vpcmpgtd(k2, ymm7, ymm4);
        vpcmpgtd(k2 | k2, ymm2, ymm7);
        kandb(k3, k2, k1);
        kortestb(k3, k3);
        movsxd(rbx, eax);
        jne(LBB7_3);
        // %bb.6:                     // %select_done.us.3
        //   in Loop: Header=BB7_2 Depth=1
        vmovaps(ymm7 | k2 | T_z, ymm6);
        vmovups(yword[rdi + rbx] | k1, ymm7);
        vmovups(yword[rbx + r9] | k1, ymm7);
        vmovups(yword[rbx + rsi] | k1, ymm7);
        vmovups(yword[rbx + r10] | k1, ymm7);
        jmp(LBB7_4);
        L(LBB7_5); // %for_exit
        pop(rbx);
        pop(r14);
        pop(r15);
        pop(rbp);
        vzeroupper();
        ret();
    }
};

template <>
struct jit_copy_1d<float, 8> : public jit_t<jit_copy_1d<float, 8>> {
    using CodeGenerator::L;
    void generate_avx2() {
        align(2 << 5);
        Label LCPI8_0 = L();
        dd(0); // 0x0
        dd(1); // 0x1
        dd(2); // 0x2
        dd(3); // 0x3
        dd(4); // 0x4
        dd(5); // 0x5
        dd(6); // 0x6
        dd(7); // 0x7
        align(2 << 2);
        Label LCPI8_1 = L();
        dd(8); // 0x8
        align(2 << 4);
        ker_off_ = getSize(); // @copy_1d_simd_c_untoll_8
        // %bb.0:                 // %allocas
        push(rbp);
        push(r15);
        push(r14);
        push(r13);
        push(r12);
        push(rbx);
        // kill: def $r8d killed $r8d def $r8
        // kill: def $ecx killed $ecx def $rcx
        // kill: def $esi killed $esi def $rsi
        vpbroadcastd(ymm0, dword[rsp + 56]);
        vpcmpgtd(ymm1, ymm0, yword[rip + LCPI8_0]);
        vmovmskps(eax, ymm1);
        test(al, al);
        Label LBB8_5;
        je(LBB8_5, T_NEAR);
        // %bb.1:                 // %for_loop.lr.ph
        vmovdqa(ymm1, yword[rip + LCPI8_0]); // ymm1 = [0,1,2,3,4,5,6,7]
        vpcmpgtd(ymm2, ymm0, ymm1);
        vmovd(xmm3, r8d);
        vpbroadcastd(ymm3, xmm3);
        vmovd(xmm4, r9d);
        vpbroadcastd(ymm4, xmm4);
        shl(r8d, 2);
        mov(qword[rsp - 8], r8); // 8-byte Spill
        xor_(eax, eax);
        vpcmpeqd(ymm5, ymm5, ymm5);
        vxorps(xmm6, xmm6, xmm6);
        movsxd(r9, esi);
        lea(r10, ptr[rdi + 4 * r9]);
        lea(ebp, ptr[rsi + rsi]);
        movsxd(r11, ebp);
        lea(r15, ptr[rdi + 4 * r11]);
        lea(ebp, ptr[rsi + 2 * rsi]);
        movsxd(rbp, ebp);
        lea(rbp, ptr[rdi + 4 * rbp]);
        mov(qword[rsp - 32], rbp); // 8-byte Spill
        lea(ebp, ptr[4 * rsi]);
        movsxd(rbp, ebp);
        lea(r13, ptr[rdi + 4 * rbp]);
        lea(ebp, ptr[rsi + 4 * rsi]);
        movsxd(rbp, ebp);
        lea(rbp, ptr[rdi + 4 * rbp]);
        mov(qword[rsp - 40], rbp); // 8-byte Spill
        vpbroadcastd(ymm7, dword[rip + LCPI8_1]); // ymm7 = [8,8,8,8,8,8,8,8]
        movsxd(r12, ecx);
        lea(rbp, ptr[rdx + 4 * r12]);
        mov(qword[rsp - 16], rbp); // 8-byte Spill
        lea(ebp, ptr[r12 + r12]);
        movsxd(r14, ebp);
        lea(rbp, ptr[rdx + 4 * r14]);
        mov(qword[rsp - 24], rbp); // 8-byte Spill
                // Implicit-def: $ymm10
        Label LBB8_2;
        jmp(LBB8_2, T_NEAR);
        align(2 << 4);
        Label LBB8_3 = L(); // %select_done.7
                //   in Loop: Header=BB8_2 Depth=1
        mov(rbx, qword[rsp - 8]); // 8-byte Reload
        add(ebx, ebp);
        movsxd(r8, ebx);
        vmaskmovps(ymm10, ymm9, yword[rdx + r8]);
        vblendvps(ymm10, ymm6, ymm10, ymm8);
        vmaskmovps(yword[rdi + rbp], ymm2, ymm10);
        mov(rbx, qword[rsp - 16]); // 8-byte Reload
        vmaskmovps(ymm10, ymm9, yword[r8 + rbx]);
        vblendvps(ymm10, ymm6, ymm10, ymm8);
        vmaskmovps(yword[rbp + r10], ymm2, ymm10);
        mov(rbx, qword[rsp - 24]); // 8-byte Reload
        vmaskmovps(ymm10, ymm9, yword[r8 + rbx]);
        vblendvps(ymm10, ymm6, ymm10, ymm8);
        vmaskmovps(yword[rbp + r15], ymm2, ymm10);
        lea(ebx, ptr[r12 + 2 * r12]);
        movsxd(rbx, ebx);
        lea(rbx, ptr[rdx + 4 * rbx]);
        vmaskmovps(ymm10, ymm9, yword[r8 + rbx]);
        lea(ebx, ptr[r9 + 2 * r9]);
        movsxd(rbx, ebx);
        lea(rbx, ptr[rdi + 4 * rbx]);
        vblendvps(ymm10, ymm6, ymm10, ymm8);
        vmaskmovps(yword[rbp + rbx], ymm2, ymm10);
        lea(ebx, ptr[4 * rcx]);
        movsxd(rbx, ebx);
        lea(rbx, ptr[rdx + 4 * rbx]);
        vmaskmovps(ymm10, ymm9, yword[r8 + rbx]);
        vblendvps(ymm10, ymm6, ymm10, ymm8);
        vmaskmovps(yword[rbp + r13], ymm2, ymm10);
        lea(ebx, ptr[r12 + 4 * r12]);
        movsxd(rbx, ebx);
        lea(rbx, ptr[rdx + 4 * rbx]);
        vmaskmovps(ymm10, ymm9, yword[r8 + rbx]);
        lea(ebx, ptr[r9 + 4 * r9]);
        movsxd(rbx, ebx);
        lea(rbx, ptr[rdi + 4 * rbx]);
        vblendvps(ymm10, ymm6, ymm10, ymm8);
        vmaskmovps(yword[rbp + rbx], ymm2, ymm10);
        lea(ebx, ptr[r14 + 2 * r14]);
        movsxd(rbx, ebx);
        lea(rbx, ptr[rdx + 4 * rbx]);
        vmaskmovps(ymm10, ymm9, yword[r8 + rbx]);
        lea(ebx, ptr[r11 + 2 * r11]);
        movsxd(rbx, ebx);
        lea(rbx, ptr[rdi + 4 * rbx]);
        vblendvps(ymm10, ymm6, ymm10, ymm8);
        vmaskmovps(yword[rbp + rbx], ymm2, ymm10);
        lea(ebx, ptr[8 * rcx]);
        sub(ebx, ecx);
        movsxd(rbx, ebx);
        lea(rbx, ptr[rdx + 4 * rbx]);
        vmaskmovps(ymm10, ymm9, yword[r8 + rbx]);
        lea(ebx, ptr[8 * rsi]);
        sub(ebx, esi);
        movsxd(rbx, ebx);
        lea(rbx, ptr[rdi + 4 * rbx]);
        vblendvps(ymm8, ymm6, ymm10, ymm8);
        Label LBB8_4 = L(); // %for_exit22
                //   in Loop: Header=BB8_2 Depth=1
        vmaskmovps(yword[rbp + rbx], ymm2, ymm8);
        vpaddd(ymm1, ymm1, ymm7);
        vpcmpgtd(ymm8, ymm0, ymm1);
        vpand(ymm2, ymm8, ymm2);
        vmovmskps(ebp, ymm2);
        add(eax, 32);
        test(bpl, bpl);
        je(LBB8_5, T_NEAR);
        L(LBB8_2); // %for_loop
                // =>This Inner Loop Header: Depth=1
        vpaddd(ymm8, ymm1, ymm3);
        vpcmpgtd(ymm9, ymm8, ymm5);
        vpcmpgtd(ymm8, ymm4, ymm8);
        vpand(ymm8, ymm8, ymm9);
        vpand(ymm9, ymm8, ymm2);
        vmovmskps(ebx, ymm9);
        movsxd(rbp, eax);
        test(bl, bl);
        jne(LBB8_3);
        // %bb.6:                     // %select_done.us.7
        //   in Loop: Header=BB8_2 Depth=1
        vblendvps(ymm8, ymm6, ymm10, ymm8);
        vmaskmovps(yword[rdi + rbp], ymm2, ymm8);
        vmaskmovps(yword[rbp + r10], ymm2, ymm8);
        vmaskmovps(yword[rbp + r15], ymm2, ymm8);
        mov(rbx, qword[rsp - 32]); // 8-byte Reload
        vmaskmovps(yword[rbp + rbx], ymm2, ymm8);
        vmaskmovps(yword[rbp + r13], ymm2, ymm8);
        mov(rbx, qword[rsp - 40]); // 8-byte Reload
        vmaskmovps(yword[rbp + rbx], ymm2, ymm8);
        lea(ebx, ptr[r11 + 2 * r11]);
        movsxd(rbx, ebx);
        lea(rbx, ptr[rdi + 4 * rbx]);
        vmaskmovps(yword[rbp + rbx], ymm2, ymm8);
        lea(ebx, ptr[8 * rsi]);
        sub(ebx, esi);
        movsxd(rbx, ebx);
        lea(rbx, ptr[rdi + 4 * rbx]);
        jmp(LBB8_4);
        L(LBB8_5); // %for_exit
        pop(rbx);
        pop(r12);
        pop(r13);
        pop(r14);
        pop(r15);
        pop(rbp);
        vzeroupper();
        ret();
    }
    void generate_avx512() {
        align(2 << 5);
        Label LCPI8_0 = L();
        dd(0); // 0x0
        dd(1); // 0x1
        dd(2); // 0x2
        dd(3); // 0x3
        dd(4); // 0x4
        dd(5); // 0x5
        dd(6); // 0x6
        dd(7); // 0x7
        align(2 << 2);
        Label LCPI8_1 = L();
        dd(8); // 0x8
        align(2 << 4);
        ker_off_ = getSize(); // @copy_1d_simd_c_unroll_4
        // %bb.0:                   // %allocas
        push(rbp);
        push(r15);
        push(r14);
        push(r13);
        push(r12);
        push(rbx);
        // kill: def $r8d killed $r8d def $r8
        // kill: def $ecx killed $ecx def $rcx
        // kill: def $esi killed $esi def $rsi
        vpbroadcastd(ymm0, dword[rsp + 56]);
        vpcmpgtd(k1, ymm0, yword[rip + LCPI8_0]);
        kortestb(k1, k1);
        Label LBB8_5;
        je(LBB8_5, T_NEAR);
        // %bb.1:                   // %for_loop.lr.ph
        vpbroadcastd(ymm1, r8d);
        vpbroadcastd(ymm2, r9d);
        shl(r8d, 2);
        mov(qword[rsp - 8], r8); // 8-byte Spill
        vmovdqa(ymm3, yword[rip + LCPI8_0]); // ymm3 = [0,1,2,3,4,5,6,7]
        xor_(eax, eax);
        vpcmpeqd(ymm4, ymm4, ymm4);
        movsxd(r9, esi);
        lea(r10, ptr[rdi + 4 * r9]);
        lea(ebp, ptr[rsi + rsi]);
        movsxd(rbx, ebp);
        lea(r15, ptr[rdi + 4 * rbx]);
        lea(ebp, ptr[rsi + 2 * rsi]);
        movsxd(rbp, ebp);
        lea(rbp, ptr[rdi + 4 * rbp]);
        mov(qword[rsp - 24], rbp); // 8-byte Spill
        lea(ebp, ptr[4 * rsi]);
        movsxd(rbp, ebp);
        lea(r8, ptr[rdi + 4 * rbp]);
        lea(ebp, ptr[rsi + 4 * rsi]);
        movsxd(rbp, ebp);
        lea(rbp, ptr[rdi + 4 * rbp]);
        mov(qword[rsp - 32], rbp); // 8-byte Spill
        mov(qword[rsp - 16], rbx); // 8-byte Spill
        lea(ebp, ptr[rbx + 2 * rbx]);
        movsxd(rbp, ebp);
        lea(rbp, ptr[rdi + 4 * rbp]);
        mov(qword[rsp - 40], rbp); // 8-byte Spill
        vpbroadcastd(ymm5, dword[rip + LCPI8_1]); // ymm5 = [8,8,8,8,8,8,8,8]
        movsxd(r12, ecx);
        lea(r14, ptr[rdx + 4 * r12]);
        mov(ebp, r12d);
        add(ebp, ebp);
        // implicit-def: $ymm6
        Label LBB8_2;
        jmp(LBB8_2, T_NEAR);
        align(2 << 4);
        Label LBB8_3 = L(); // %select_done.7
                //   in Loop: Header=BB8_2 Depth=1
        mov(rbx, qword[rsp - 8]); // 8-byte Reload
        add(ebx, r13d);
        movsxd(r11, ebx);
        movsxd(rbx, ebp);
        lea(rbx, ptr[rdx + 4 * rbx]);
        vmovups(ymm7 | k3 | T_z, yword[r11 + rbx]);
        lea(ebx, ptr[r12 + 2 * r12]);
        movsxd(rbx, ebx);
        lea(rbx, ptr[rdx + 4 * rbx]);
        vmovups(ymm8 | k3 | T_z, yword[r11 + rbx]);
        lea(ebx, ptr[4 * rcx]);
        movsxd(rbx, ebx);
        lea(rbx, ptr[rdx + 4 * rbx]);
        vmovups(ymm9 | k3 | T_z, yword[r11 + rbx]);
        lea(ebx, ptr[r12 + 4 * r12]);
        movsxd(rbx, ebx);
        lea(rbx, ptr[rdx + 4 * rbx]);
        vmovups(ymm10 | k3 | T_z, yword[r11 + rbx]);
        lea(ebx, ptr[rbp + 2 * rbp]);
        movsxd(rbx, ebx);
        lea(rbx, ptr[rdx + 4 * rbx]);
        vmovups(ymm11 | k3 | T_z, yword[r11 + rbx]);
        lea(ebx, ptr[8 * rcx]);
        sub(ebx, ecx);
        movsxd(rbx, ebx);
        lea(rbx, ptr[rdx + 4 * rbx]);
        vmovups(ymm6 | k3 | T_z, yword[r11 + rbx]);
        vmovups(ymm12 | k3 | T_z, yword[rdx + r11]);
        vmovaps(ymm12 | k2 | T_z, ymm12);
        vmovups(yword[rdi + r13] | k1, ymm12);
        vmovups(ymm12 | k3 | T_z, yword[r11 + r14]);
        vmovaps(ymm12 | k2 | T_z, ymm12);
        vmovups(yword[r13 + r10] | k1, ymm12);
        vmovaps(ymm7 | k2 | T_z, ymm7);
        vmovups(yword[r13 + r15] | k1, ymm7);
        lea(ebx, ptr[r9 + 2 * r9]);
        movsxd(rbx, ebx);
        lea(rbx, ptr[rdi + 4 * rbx]);
        vmovaps(ymm7 | k2 | T_z, ymm8);
        vmovups(yword[r13 + rbx] | k1, ymm7);
        vmovaps(ymm7 | k2 | T_z, ymm9);
        vmovups(yword[r13 + r8] | k1, ymm7);
        lea(ebx, ptr[r9 + 4 * r9]);
        movsxd(rbx, ebx);
        lea(rbx, ptr[rdi + 4 * rbx]);
        vmovaps(ymm7 | k2 | T_z, ymm10);
        vmovups(yword[r13 + rbx] | k1, ymm7);
        mov(rbx, qword[rsp - 16]); // 8-byte Reload
        lea(ebx, ptr[rbx + 2 * rbx]);
        movsxd(rbx, ebx);
        lea(rbx, ptr[rdi + 4 * rbx]);
        vmovaps(ymm7 | k2 | T_z, ymm11);
        vmovups(yword[r13 + rbx] | k1, ymm7);
        lea(ebx, ptr[8 * rsi]);
        sub(ebx, esi);
        movsxd(rbx, ebx);
        lea(rbx, ptr[rdi + 4 * rbx]);
        vmovaps(ymm7 | k2 | T_z, ymm6);
        Label LBB8_4 = L(); // %for_exit20
                //   in Loop: Header=BB8_2 Depth=1
        vmovups(yword[r13 + rbx] | k1, ymm7);
        vpaddd(ymm3, ymm3, ymm5);
        vpcmpgtd(k1 | k1, ymm0, ymm3);
        add(eax, 32);
        kortestb(k1, k1);
        je(LBB8_5, T_NEAR);
        L(LBB8_2); // %for_loop
                // => This Inner Loop Header: Depth=1
        vpaddd(ymm7, ymm3, ymm1);
        vpcmpgtd(k2, ymm7, ymm4);
        vpcmpgtd(k2 | k2, ymm2, ymm7);
        kandb(k3, k2, k1);
        kortestb(k3, k3);
        movsxd(r13, eax);
        jne(LBB8_3);
        // %bb.6:               // %select_done.us.7
        //   in Loop: Header=BB8_2 Depth=1
        vmovaps(ymm7 | k2 | T_z, ymm6);
        vmovups(yword[rdi + r13] | k1, ymm7);
        vmovups(yword[r13 + r10] | k1, ymm7);
        vmovups(yword[r13 + r15] | k1, ymm7);
        mov(rbx, qword[rsp - 24]); // 8-byte Reload
        vmovups(yword[r13 + rbx] | k1, ymm7);
        vmovups(yword[r13 + r8] | k1, ymm7);
        mov(rbx, qword[rsp - 32]); // 8-byte Reload
        vmovups(yword[r13 + rbx] | k1, ymm7);
        mov(rbx, qword[rsp - 40]); // 8-byte Reload
        vmovups(yword[r13 + rbx] | k1, ymm7);
        lea(ebx, ptr[8 * rsi]);
        sub(ebx, esi);
        movsxd(rbx, ebx);
        lea(rbx, ptr[rdi + 4 * rbx]);
        jmp(LBB8_4);
        L(LBB8_5); // %for_exit
        pop(rbx);
        pop(r12);
        pop(r13);
        pop(r14);
        pop(r15);
        pop(rbp);
        vzeroupper();
        ret();
    }
};

template <>
struct jit_zero_1d<float, 1> : public jit_t<jit_zero_1d<float, 1>> {
    using CodeGenerator::L;
    void generate_avx2() {
        align(2 << 5);
        Label LCPI9_0 = L();
        dd(0); // 0x0
        dd(1); // 0x1
        dd(2); // 0x2
        dd(3); // 0x3
        dd(4); // 0x4
        dd(5); // 0x5
        dd(6); // 0x6
        dd(7); // 0x7
        align(2 << 2);
        Label LCPI9_1 = L();
        dd(8); // 0x8
        align(2 << 4);
        ker_off_ = getSize(); // @zero_1d_simd
        // %bb.0:                 // %allocas
        vmovd(xmm0, esi);
        vpbroadcastd(ymm0, xmm0);
        vpcmpgtd(ymm1, ymm0, yword[rip + LCPI9_0]);
        vmovmskps(eax, ymm1);
        test(al, al);
        Label LBB9_3;
        je(LBB9_3);
        // %bb.1:                 // %for_loop.lr.ph
        vmovdqa(ymm1, yword[rip + LCPI9_0]); // ymm1 = [0,1,2,3,4,5,6,7]
        vpcmpgtd(ymm2, ymm0, ymm1);
        xor_(eax, eax);
        vxorps(xmm3, xmm3, xmm3);
        vpbroadcastd(ymm4, dword[rip + LCPI9_1]); // ymm4 = [8,8,8,8,8,8,8,8]
        align(2 << 4);
        Label LBB9_2 = L(); // %for_loop
                // =>This Inner Loop Header: Depth=1
        cdqe();
        vmaskmovps(yword[rdi + rax], ymm2, ymm3);
        vpaddd(ymm1, ymm1, ymm4);
        vpcmpgtd(ymm5, ymm0, ymm1);
        vpand(ymm2, ymm5, ymm2);
        vmovmskps(ecx, ymm2);
        add(eax, 32);
        test(cl, cl);
        jne(LBB9_2);
        L(LBB9_3); // %for_exit
        vzeroupper();
        ret();
    }
    void generate_avx512() {
        align(2 << 5);
        Label LCPI9_0 = L();
        dd(0); // 0x0
        dd(1); // 0x1
        dd(2); // 0x2
        dd(3); // 0x3
        dd(4); // 0x4
        dd(5); // 0x5
        dd(6); // 0x6
        dd(7); // 0x7
        align(2 << 2);
        Label LCPI9_1 = L();
        dd(8); // 0x8
        align(2 << 4);
        ker_off_ = getSize(); // @zero_1d_simd
        // %bb.0:               // %allocas
        vpbroadcastd(ymm0, esi);
        vpcmpgtd(k1, ymm0, yword[rip + LCPI9_0]);
        kortestb(k1, k1);
        Label LBB9_3;
        je(LBB9_3);
        // %bb.1:               // %for_loop.lr.ph
        vmovdqa(ymm1, yword[rip + LCPI9_0]); // ymm1 = [0,1,2,3,4,5,6,7]
        xor_(eax, eax);
        vxorps(xmm2, xmm2, xmm2);
        vpbroadcastd(ymm3, dword[rip + LCPI9_1]); // ymm3 = [8,8,8,8,8,8,8,8]
        align(2 << 4);
        Label LBB9_2 = L(); // %for_loop
                // =>This Inner Loop Header: Depth=1
        vpaddd(ymm1, ymm1, ymm3);
        cdqe();
        vmovups(yword[rdi + rax] | k1, ymm2);
        vpcmpgtd(k1 | k1, ymm0, ymm1);
        add(eax, 32);
        kortestb(k1, k1);
        jne(LBB9_2);
        L(LBB9_3); // %for_exit
        vzeroupper();
        ret();
    }
};

template <>
struct jit_zero_1d<float, 4> : public jit_t<jit_zero_1d<float, 4>> {
    using CodeGenerator::L;
    void generate_avx2() {
        align(2 << 5);
        Label LCPI10_0 = L();
        dd(0); // 0x0
        dd(1); // 0x1
        dd(2); // 0x2
        dd(3); // 0x3
        dd(4); // 0x4
        dd(5); // 0x5
        dd(6); // 0x6
        dd(7); // 0x7
        align(2 << 2);
        Label LCPI10_1 = L();
        dd(8); // 0x8
        align(2 << 4);
        ker_off_ = getSize(); // @zero_1d_simd_c_unroll_4
        // %bb.0:               // %allocas
        vmovd(xmm0, edx);
        vpbroadcastd(ymm0, xmm0);
        vpcmpgtd(ymm1, ymm0, yword[rip + LCPI10_0]);
        vmovmskps(eax, ymm1);
        test(al, al);
        Label LBB10_3;
        je(LBB10_3);
        // %bb.1:               // %for_test14.preheader.lr.ph
        vmovdqa(ymm1, yword[rip + LCPI10_0]); // ymm1 = [0,1,2,3,4,5,6,7]
        vpcmpgtd(ymm2, ymm0, ymm1);
        xor_(eax, eax);
        vxorps(xmm3, xmm3, xmm3);
        movsxd(rsi, esi);
        lea(r8, ptr[rdi + 4 * rsi]);
        lea(edx, ptr[rsi + rsi]);
        movsxd(rdx, edx);
        lea(rdx, ptr[rdi + 4 * rdx]);
        lea(esi, ptr[rsi + 2 * rsi]);
        movsxd(rsi, esi);
        lea(rsi, ptr[rdi + 4 * rsi]);
        vpbroadcastd(ymm4, dword[rip + LCPI10_1]); // ymm4 = [8,8,8,8,8,8,8,8]
        align(2 << 4);
        Label LBB10_2 = L(); // %for_test14.preheader
                // =>This Inner Loop Header: Depth=1
        cdqe();
        vmaskmovps(yword[rdi + rax], ymm2, ymm3);
        vmaskmovps(yword[rax + r8], ymm2, ymm3);
        vmaskmovps(yword[rax + rdx], ymm2, ymm3);
        vmaskmovps(yword[rax + rsi], ymm2, ymm3);
        vpaddd(ymm1, ymm1, ymm4);
        vpcmpgtd(ymm5, ymm0, ymm1);
        vpand(ymm2, ymm5, ymm2);
        vmovmskps(ecx, ymm2);
        add(eax, 32);
        test(cl, cl);
        jne(LBB10_2);
        L(LBB10_3); // %for_exit
        vzeroupper();
        ret();
    }
    void generate_avx512() {
        align(2 << 5);
        Label LCPI10_0 = L();
        dd(0); // 0x0
        dd(1); // 0x1
        dd(2); // 0x2
        dd(3); // 0x3
        dd(4); // 0x4
        dd(5); // 0x5
        dd(6); // 0x6
        dd(7); // 0x7
        align(2 << 2);
        Label LCPI10_1 = L();
        dd(8); // 0x8
        align(2 << 4);
        ker_off_ = getSize(); // @zero_1d_simd_c_unroll_4
        // %bb.0:               // %allocas
        vpbroadcastd(ymm0, edx);
        vpcmpgtd(k1, ymm0, yword[rip + LCPI10_0]);
        kortestb(k1, k1);
        Label LBB10_3;
        je(LBB10_3);
        // %bb.1:               // %for_test12.preheader.lr.ph
        vmovdqa(ymm1, yword[rip + LCPI10_0]); // ymm1 = [0,1,2,3,4,5,6,7]
        xor_(eax, eax);
        vxorps(xmm2, xmm2, xmm2);
        movsxd(rsi, esi);
        lea(rcx, ptr[rdi + 4 * rsi]);
        lea(edx, ptr[rsi + rsi]);
        movsxd(rdx, edx);
        lea(rdx, ptr[rdi + 4 * rdx]);
        lea(esi, ptr[rsi + 2 * rsi]);
        movsxd(rsi, esi);
        lea(rsi, ptr[rdi + 4 * rsi]);
        vpbroadcastd(ymm3, dword[rip + LCPI10_1]); // ymm3 = [8,8,8,8,8,8,8,8]
        align(2 << 4);
        Label LBB10_2 = L(); // %for_test12.preheader
                // =>This Inner Loop Header: Depth=1
        cdqe();
        vmovups(yword[rdi + rax] | k1, ymm2);
        vmovups(yword[rax + rcx] | k1, ymm2);
        vmovups(yword[rax + rdx] | k1, ymm2);
        vpaddd(ymm1, ymm1, ymm3);
        vmovups(yword[rax + rsi] | k1, ymm2);
        vpcmpgtd(k1 | k1, ymm0, ymm1);
        add(eax, 32);
        kortestb(k1, k1);
        jne(LBB10_2);
        L(LBB10_3); // %for_exit
        vzeroupper();
        ret();
    }
};

template <>
struct jit_zero_1d<float, 8> : public jit_t<jit_zero_1d<float, 8>> {
    using CodeGenerator::L;
    void generate_avx2() {
        align(2 << 5);
        Label LCPI11_0 = L();
        dd(0); // 0x0
        dd(1); // 0x1
        dd(2); // 0x2
        dd(3); // 0x3
        dd(4); // 0x4
        dd(5); // 0x5
        dd(6); // 0x6
        dd(7); // 0x7
        align(2 << 2);
        Label LCPI11_1 = L();
        dd(8); // 0x8
        align(2 << 4);
        ker_off_ = getSize(); // @zero_1d_simd_c_unroll_8
        // %bb.0:               // %allocas
        push(rbx);
        vmovd(xmm0, edx);
        vpbroadcastd(ymm0, xmm0);
        vpcmpgtd(ymm1, ymm0, yword[rip + LCPI11_0]);
        vmovmskps(eax, ymm1);
        test(al, al);
        Label LBB11_3;
        je(LBB11_3, T_NEAR);
        // %bb.1:               // %for_test14.preheader.lr.ph
        vmovdqa(ymm1, yword[rip + LCPI11_0]); // ymm1 = [0,1,2,3,4,5,6,7]
        vpcmpgtd(ymm2, ymm0, ymm1);
        xor_(eax, eax);
        vxorps(xmm3, xmm3, xmm3);
        movsxd(rcx, esi);
        lea(r8, ptr[rdi + 4 * rcx]);
        lea(edx, ptr[rcx + rcx]);
        movsxd(rsi, edx);
        lea(r9, ptr[rdi + 4 * rsi]);
        lea(edx, ptr[rcx + 2 * rcx]);
        movsxd(rdx, edx);
        lea(r10, ptr[rdi + 4 * rdx]);
        lea(edx, ptr[4 * rcx]);
        movsxd(rdx, edx);
        lea(r11, ptr[rdi + 4 * rdx]);
        lea(edx, ptr[rcx + 4 * rcx]);
        movsxd(rdx, edx);
        lea(rdx, ptr[rdi + 4 * rdx]);
        lea(esi, ptr[rsi + 2 * rsi]);
        movsxd(rsi, esi);
        lea(rsi, ptr[rdi + 4 * rsi]);
        lea(ebx, ptr[8 * rcx]);
        sub(ebx, ecx);
        movsxd(rcx, ebx);
        lea(rcx, ptr[rdi + 4 * rcx]);
        vpbroadcastd(ymm4, dword[rip + LCPI11_1]); // ymm4 = [8,8,8,8,8,8,8,8]
        align(2 << 4);
        Label LBB11_2 = L(); // %for_test14.preheader
                // =>This Inner Loop Header: Depth=1
        cdqe();
        vmaskmovps(yword[rdi + rax], ymm2, ymm3);
        vmaskmovps(yword[rax + r8], ymm2, ymm3);
        vmaskmovps(yword[rax + r9], ymm2, ymm3);
        vmaskmovps(yword[rax + r10], ymm2, ymm3);
        vmaskmovps(yword[rax + r11], ymm2, ymm3);
        vmaskmovps(yword[rax + rdx], ymm2, ymm3);
        vmaskmovps(yword[rax + rsi], ymm2, ymm3);
        vmaskmovps(yword[rax + rcx], ymm2, ymm3);
        vpaddd(ymm1, ymm1, ymm4);
        vpcmpgtd(ymm5, ymm0, ymm1);
        vpand(ymm2, ymm5, ymm2);
        vmovmskps(ebx, ymm2);
        add(eax, 32);
        test(bl, bl);
        jne(LBB11_2);
        L(LBB11_3); // %for_exit
        pop(rbx);
        vzeroupper();
        ret();
    }
    void generate_avx512() {
        align(2 << 5);
        Label LCPI11_0 = L();
        dd(0); // 0x0
        dd(1); // 0x1
        dd(2); // 0x2
        dd(3); // 0x3
        dd(4); // 0x4
        dd(5); // 0x5
        dd(6); // 0x6
        dd(7); // 0x7
        align(2 << 2);
        Label LCPI11_1 = L();
        dd(8); // 0x8
        align(2 << 4);
        ker_off_ = getSize(); // @zero_1d_simd_c_unroll_8
        // %bb.0:               // %allocas
        push(rbx);
        vpbroadcastd(ymm0, edx);
        vpcmpgtd(k1, ymm0, yword[rip + LCPI11_0]);
        kortestb(k1, k1);
        Label LBB11_3;
        je(LBB11_3, T_NEAR);
        // %bb.1:               // %for_test12.preheader.lr.ph
        vmovdqa(ymm1, yword[rip + LCPI11_0]); // ymm1 = [0,1,2,3,4,5,6,7]
        xor_(eax, eax);
        vxorps(xmm2, xmm2, xmm2);
        movsxd(rcx, esi);
        lea(r8, ptr[rdi + 4 * rcx]);
        lea(edx, ptr[rcx + rcx]);
        movsxd(rsi, edx);
        lea(r9, ptr[rdi + 4 * rsi]);
        lea(edx, ptr[rcx + 2 * rcx]);
        movsxd(rdx, edx);
        lea(r10, ptr[rdi + 4 * rdx]);
        lea(edx, ptr[4 * rcx]);
        movsxd(rdx, edx);
        lea(r11, ptr[rdi + 4 * rdx]);
        lea(edx, ptr[rcx + 4 * rcx]);
        movsxd(rdx, edx);
        lea(rdx, ptr[rdi + 4 * rdx]);
        lea(esi, ptr[rsi + 2 * rsi]);
        movsxd(rsi, esi);
        lea(rsi, ptr[rdi + 4 * rsi]);
        lea(ebx, ptr[8 * rcx]);
        sub(ebx, ecx);
        movsxd(rcx, ebx);
        lea(rcx, ptr[rdi + 4 * rcx]);
        vpbroadcastd(ymm3, dword[rip + LCPI11_1]); // ymm3 = [8,8,8,8,8,8,8,8]
        align(2 << 4);
        Label LBB11_2 = L(); // %for_test12.preheader
                // =>This Inner Loop Header: Depth=1
        cdqe();
        vmovups(yword[rdi + rax] | k1, ymm2);
        vmovups(yword[rax + r8] | k1, ymm2);
        vmovups(yword[rax + r9] | k1, ymm2);
        vmovups(yword[rax + r10] | k1, ymm2);
        vmovups(yword[rax + r11] | k1, ymm2);
        vmovups(yword[rax + rdx] | k1, ymm2);
        vmovups(yword[rax + rsi] | k1, ymm2);
        vpaddd(ymm1, ymm1, ymm3);
        vmovups(yword[rax + rcx] | k1, ymm2);
        vpcmpgtd(k1 | k1, ymm0, ymm1);
        add(eax, 32);
        kortestb(k1, k1);
        jne(LBB11_2);
        L(LBB11_3); // %for_exit
        pop(rbx);
        vzeroupper();
        ret();
    }
};

#endif

template <typename T>
void copy_1d(T *dst, const T *src, int start_off, int src_width, int nelem) {
    static jit_copy_1d<T, 1> j;
    if (!j.is_generic())
        return j(dst, src, start_off, src_width, nelem);
    else {
        PRAGMA_OMP_SIMD()
        for (int i = 0; i < nelem; ++i) {
            auto _off = i + start_off;
            dst[i] = (_off >= 0 && _off < src_width) ? src[_off] : (T)0;
        }
    }
}

template <typename T>
void copy_1d_batch_4(T *dst, int d_stride, const T *src, int s_stride,
        int start_off, int src_width, int nelem) {
    static jit_copy_1d<T, 4> j;
    if (!j.is_generic())
        return j(dst, d_stride, src, s_stride, start_off, src_width, nelem);
    else {
        PRAGMA_OMP_SIMD()
        for (int i = 0; i < nelem; ++i) {
            auto p_off = i + start_off;
            for (int batch = 0; batch < 4; ++batch) {
                (dst + batch * d_stride)[i] = (p_off >= 0 && p_off < src_width)
                        ? (src + batch * s_stride)[p_off]
                        : (T)0;
            }
        }
    }
}

template <typename T>
void copy_1d_batch_8(T *dst, int d_stride, const T *src, int s_stride,
        int start_off, int src_width, int nelem) {
    static jit_copy_1d<T, 8> j;
    if (!j.is_generic())
        return j(dst, d_stride, src, s_stride, start_off, src_width, nelem);
    else {
        PRAGMA_OMP_SIMD()
        for (int i = 0; i < nelem; ++i) {
            auto p_off = i + start_off;
            for (int batch = 0; batch < 8; ++batch) {
                (dst + batch * d_stride)[i] = (p_off >= 0 && p_off < src_width)
                        ? (src + batch * s_stride)[p_off]
                        : (T)0;
            }
        }
    }
}

template <typename T>
void zero_1d(T *dst, int nelem) {
    static jit_zero_1d<T, 1> j;
    if (!j.is_generic())
        return j(dst, nelem);
    else {
        PRAGMA_OMP_SIMD()
        for (int pid = 0; pid < nelem; ++pid) {
            dst[pid] = (T)0;
        }
    }
}

template <typename T>
void zero_1d_batch_4(T *dst, int stride, int nelem) {
    static jit_zero_1d<T, 4> j;
    if (!j.is_generic())
        return j(dst, stride, nelem);
    else {
        for (int batch = 0; batch < 4; ++batch) {
            PRAGMA_OMP_SIMD()
            for (int pid = 0; pid < nelem; ++pid) {
                (dst + batch * stride)[pid] = (T)0;
            }
        }
    }
}

template <typename T>
void zero_1d_batch_8(T *dst, int stride, int nelem) {
    static jit_zero_1d<T, 8> j;
    if (!j.is_generic())
        return j(dst, stride, nelem);
    else {
        for (int batch = 0; batch < 8; ++batch) {
            PRAGMA_OMP_SIMD()
            for (int pid = 0; pid < nelem; ++pid) {
                (dst + batch * stride)[pid] = (T)0;
            }
        }
    }
}

// Instantiate what we need
template void copy_1d<float>(
        float *dst, const float *src, int start_off, int src_width, int nelem);
template void copy_1d_batch_4<float>(float *dst, int d_stride, const float *src,
        int s_stride, int start_off, int src_width, int nelem);
template void copy_1d_batch_8<float>(float *dst, int d_stride, const float *src,
        int s_stride, int start_off, int src_width, int nelem);

template void zero_1d<float>(float *dst, int nelem);
template void zero_1d_batch_4<float>(float *dst, int stride, int nelem);
template void zero_1d_batch_8<float>(float *dst, int stride, int nelem);

// bfloat16
template void copy_1d<uint16_t>(uint16_t *dst, const uint16_t *src,
        int start_off, int src_width, int nelem);
template void copy_1d_batch_4<uint16_t>(uint16_t *dst, int d_stride,
        const uint16_t *src, int s_stride, int start_off, int src_width,
        int nelem);
template void copy_1d_batch_8<uint16_t>(uint16_t *dst, int d_stride,
        const uint16_t *src, int s_stride, int start_off, int src_width,
        int nelem);

template void zero_1d<uint16_t>(uint16_t *dst, int nelem);
template void zero_1d_batch_4<uint16_t>(uint16_t *dst, int stride, int nelem);
template void zero_1d_batch_8<uint16_t>(uint16_t *dst, int stride, int nelem);

} // namespace spmd
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
