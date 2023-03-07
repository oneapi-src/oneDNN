/*******************************************************************************
 * Copyright 2021-2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_X86_64_REGISTERS_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_X86_64_REGISTERS_HPP

#include <compiler/jit/xbyak/configured_xbyak.hpp>
#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {
namespace x86_64 {

/// Shortened names for some of the x86-64 registers, defined here
/// for notational convenience.
/// Suitable for comparison / assignment to instance members such as
/// Xbyak::CodeGenerator::r8.
namespace regs {
static const Xbyak::Reg64 rax(Xbyak::Operand::Code::RAX);
static const Xbyak::Reg64 rcx(Xbyak::Operand::Code::RCX);
static const Xbyak::Reg64 rdx(Xbyak::Operand::Code::RDX);
static const Xbyak::Reg64 rbx(Xbyak::Operand::Code::RBX);
static const Xbyak::Reg64 rsp(Xbyak::Operand::Code::RSP);
static const Xbyak::Reg64 rbp(Xbyak::Operand::Code::RBP);
static const Xbyak::Reg64 rsi(Xbyak::Operand::Code::RSI);
static const Xbyak::Reg64 rdi(Xbyak::Operand::Code::RDI);
static const Xbyak::Reg64 r8(Xbyak::Operand::Code::R8);
static const Xbyak::Reg64 r9(Xbyak::Operand::Code::R9);
static const Xbyak::Reg64 r10(Xbyak::Operand::Code::R10);
static const Xbyak::Reg64 r11(Xbyak::Operand::Code::R11);
static const Xbyak::Reg64 r12(Xbyak::Operand::Code::R12);
static const Xbyak::Reg64 r13(Xbyak::Operand::Code::R13);
static const Xbyak::Reg64 r14(Xbyak::Operand::Code::R14);
static const Xbyak::Reg64 r15(Xbyak::Operand::Code::R15);

/// NOTE: According to this Wikipedia page:
/// https://en.wikipedia.org/wiki/Advanced_Vector_Extensions
/// the there were only 16 (not 32) Xmm registers in AVX.
/// Some IA extension after AVX (perhaps AVX512?) introduced
/// xmm16...xmm31.
/// We'll assume all 32 Xmm registers are accessible.
static const Xbyak::Xmm xmm0(0);
static const Xbyak::Xmm xmm1(1);
static const Xbyak::Xmm xmm2(2);
static const Xbyak::Xmm xmm3(3);
static const Xbyak::Xmm xmm4(4);
static const Xbyak::Xmm xmm5(5);
static const Xbyak::Xmm xmm6(6);
static const Xbyak::Xmm xmm7(7);
static const Xbyak::Xmm xmm8(8);
static const Xbyak::Xmm xmm9(9);
static const Xbyak::Xmm xmm10(10);
static const Xbyak::Xmm xmm11(11);
static const Xbyak::Xmm xmm12(12);
static const Xbyak::Xmm xmm13(13);
static const Xbyak::Xmm xmm14(14);
static const Xbyak::Xmm xmm15(15);
static const Xbyak::Xmm xmm16(16);
static const Xbyak::Xmm xmm17(17);
static const Xbyak::Xmm xmm18(18);
static const Xbyak::Xmm xmm19(19);
static const Xbyak::Xmm xmm20(20);
static const Xbyak::Xmm xmm21(21);
static const Xbyak::Xmm xmm22(22);
static const Xbyak::Xmm xmm23(23);
static const Xbyak::Xmm xmm24(24);
static const Xbyak::Xmm xmm25(25);
static const Xbyak::Xmm xmm26(26);
static const Xbyak::Xmm xmm27(27);
static const Xbyak::Xmm xmm28(28);
static const Xbyak::Xmm xmm29(29);
static const Xbyak::Xmm xmm30(30);
static const Xbyak::Xmm xmm31(31);

// OP Mask Reg for AVX512
static const Xbyak::Opmask k0(0);
static const Xbyak::Opmask k1(1);
static const Xbyak::Opmask k2(2);
static const Xbyak::Opmask k3(3);
static const Xbyak::Opmask k4(4);
static const Xbyak::Opmask k5(5);
static const Xbyak::Opmask k6(6);
static const Xbyak::Opmask k7(7);

// Tile Regs for AMX
static const Xbyak::Tmm tmm0(0);
static const Xbyak::Tmm tmm1(1);
static const Xbyak::Tmm tmm2(2);
static const Xbyak::Tmm tmm3(3);
static const Xbyak::Tmm tmm4(4);
static const Xbyak::Tmm tmm5(5);
static const Xbyak::Tmm tmm6(6);
static const Xbyak::Tmm tmm7(7);

} // namespace regs

/// Convert gp reg to specific type
inline Xbyak::Reg8 to_reg8(const Xbyak::Reg &r) {
    COMPILE_ASSERT(r.isREG(), "Not a GP reg: " << r.toString());
    return r.cvt8();
}
inline Xbyak::Reg16 to_reg16(const Xbyak::Reg &r) {
    COMPILE_ASSERT(r.isREG(), "Not a GP reg: " << r.toString());
    return r.cvt16();
}
inline Xbyak::Reg32 to_reg32(const Xbyak::Reg &r) {
    COMPILE_ASSERT(r.isREG(), "Not a GP reg: " << r.toString());
    return r.cvt32();
}
inline Xbyak::Reg64 to_reg64(const Xbyak::Reg &r) {
    COMPILE_ASSERT(r.isREG(), "Not a GP reg: " << r.toString());
    return r.cvt64();
}

/// Convert fp reg to specific type
inline Xbyak::Xmm to_xmm(const Xbyak::Reg &r) {
    COMPILE_ASSERT(r.isXMM() || r.isYMM() || r.isZMM(),
            "Not an [XYZ]MM reg: " << r.toString());
    return Xbyak::Xmm(r.getIdx());
}
inline Xbyak::Ymm to_ymm(const Xbyak::Reg &r) {
    COMPILE_ASSERT(r.isXMM() || r.isYMM() || r.isZMM(),
            "Not an [XYZ]MM reg: " << r.toString());
    return Xbyak::Ymm(r.getIdx());
}
inline Xbyak::Zmm to_zmm(const Xbyak::Reg &r) {
    COMPILE_ASSERT(r.isXMM() || r.isYMM() || r.isZMM(),
            "Not an [XYZ]MM reg: " << r.toString());
    return Xbyak::Zmm(r.getIdx());
}

/// Convert mask reg to Opmask
inline Xbyak::Opmask to_mask(const Xbyak::Reg &r) {
    COMPILE_ASSERT(r.isOPMASK(), "Not an OPMASK reg: " << r.toString());
    return Xbyak::Opmask(r.getIdx());
}

/// Convert tile reg to AMX tile
inline Xbyak::Tmm to_tmm(const Xbyak::Reg &r) {
    COMPILE_ASSERT(r.isTMM(), "Not an AMX tile reg: " << r.toString());
    return Xbyak::Tmm(r.getIdx());
}

} // namespace x86_64
} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
