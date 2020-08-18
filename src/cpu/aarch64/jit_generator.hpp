/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
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

#ifndef CPU_AARCH64_JIT_GENERATOR_HPP
#define CPU_AARCH64_JIT_GENERATOR_HPP

#define XBYAK_CODE_PTR uint32

#include <limits.h>

#include "common/bit_cast.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/cpu_isa_traits.hpp"

#define CGA64 CodeGeneratorAArch64

#if defined(_WIN32) && !defined(__GNUC__)
#define STRUCT_ALIGN(al, ...) __declspec(align(al)) __VA_ARGS__
#else
#define STRUCT_ALIGN(al, ...) __VA_ARGS__ __attribute__((__aligned__(al)))
#endif

#if defined(_WIN32)
#define OFFSET_SHADOWSPACE 0x28
#endif

#define DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_name) \
    const char *name() const override { return STRINGIFY(jit_name); } \
    const char *source_file() const override { return __FILE__; }

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

#define CGA64 CodeGeneratorAArch64
namespace xa = Xbyak::Xbyak_aarch64;
// TODO: move this to jit_generator class?
namespace {

typedef enum {
    MAX_CODE_SIZE = 256 * 1024,
} max_code_size_t;

// TODO: move this somewhere else? Although this is only used by jit kernels
// (Roma)
static inline int float2int(float x) {
    return utils::bit_cast<int>(x);
}


// TODO: A GPR class that hides ABI details from the JIT kernels and allows
// numbering registers from 0 to 14 (x86_64) / 6 (x32) (gpr0, gpr1, ...) and
// stack register (sr).
//
// This will allow using syntax like this:
//
// param = gpr0;
// reg_input = gpr0;
// reg_output =  gpr1;
// ...
//
// #ifndef XBYAK64
// mov(param, ptr[sr])
// #endif
//
// (Roma)

// Callee-saved registers
  constexpr Xbyak::Xbyak_aarch64::Operand::Code abi_save_gpr_regs_aarch64[] = { Xbyak::Xbyak_aarch64::Operand::X19,
    Xbyak::Xbyak_aarch64::Operand::X20, Xbyak::Xbyak_aarch64::Operand::X21, Xbyak::Xbyak_aarch64::Operand::X22,
    Xbyak::Xbyak_aarch64::Operand::X23, Xbyak::Xbyak_aarch64::Operand::X24, Xbyak::Xbyak_aarch64::Operand::X25,
    Xbyak::Xbyak_aarch64::Operand::X26, Xbyak::Xbyak_aarch64::Operand::X27, Xbyak::Xbyak_aarch64::Operand::X28 };

// See "Procedure Call Standsard for the ARM 64-bit Architecture (AArch64)"
static const Xbyak::Xbyak_aarch64::XReg abi_param1_aarch64(Xbyak::Xbyak_aarch64::Operand::X0),
        abi_param2_aarch64(Xbyak::Xbyak_aarch64::Operand::X1),
        abi_param3_aarch64(Xbyak::Xbyak_aarch64::Operand::X2),
        abi_param4_aarch64(Xbyak::Xbyak_aarch64::Operand::X3),
        abi_param5_aarch64(Xbyak::Xbyak_aarch64::Operand::X4),
        abi_param6_aarch64(Xbyak::Xbyak_aarch64::Operand::X5),
        abi_param7_aarch64(Xbyak::Xbyak_aarch64::Operand::X6),
        abi_param8_aarch64(Xbyak::Xbyak_aarch64::Operand::X7),
        abi_not_param1_aarch64(Xbyak::Xbyak_aarch64::Operand::X15); // Fujitsu uses X15 on A64FX as
                                             // abi_not_param1 on x64.
#ifdef XBYAK64
constexpr Xbyak::Operand::Code abi_save_gpr_regs[] = {
    Xbyak::Operand::RBX, Xbyak::Operand::RBP, Xbyak::Operand::R12,
    Xbyak::Operand::R13, Xbyak::Operand::R14, Xbyak::Operand::R15,
#ifdef _WIN32
    Xbyak::Operand::RDI, Xbyak::Operand::RSI,
#endif
};

#ifdef _WIN32
static const Xbyak::Reg64 abi_param1(Xbyak::Operand::RCX),
             abi_param2(Xbyak::Operand::RDX),
             abi_param3(Xbyak::Operand::R8),
             abi_param4(Xbyak::Operand::R9),
             abi_not_param1(Xbyak::Operand::RDI);
#else
static const Xbyak::Reg64 abi_param1(Xbyak::Operand::RDI),
             abi_param2(Xbyak::Operand::RSI),
             abi_param3(Xbyak::Operand::RDX),
             abi_param4(Xbyak::Operand::RCX),
             abi_param5(Xbyak::Operand::R8),
             abi_param6(Xbyak::Operand::R9),
             abi_not_param1(Xbyak::Operand::RCX);
#endif
#endif //#ifdef XBYAK64


} // namespace

class jit_generator : public Xbyak::CodeGenerator {
private:
    const size_t xmm_len = 16;
#ifdef _WIN32
    const size_t xmm_to_preserve_start = 6;
    const size_t xmm_to_preserve = 10;
#else
    const size_t xmm_to_preserve_start = 0;
    const size_t xmm_to_preserve = 0;
#endif

    const size_t xreg_len = 8;
    const size_t vreg_len_preserve = 8; // Only bottom 8byte must be preserved.
    const size_t vreg_to_preserve = 8; // VREG8 - VREG15 

    const size_t num_abi_save_gpr_regs_aarch64
            = sizeof(abi_save_gpr_regs_aarch64) / sizeof(abi_save_gpr_regs_aarch64[0]);

    const size_t size_of_abi_save_regs_aarch64
            = (num_abi_save_gpr_regs_aarch64 + 2)* x0.getBit() / 8
            + xmm_to_preserve * xmm_len;
    
    const size_t preserved_stack_size
            = xreg_len * (2 + num_abi_save_gpr_regs_aarch64)
            + vreg_len_preserve * vreg_to_preserve;

    const size_t num_abi_save_gpr_regs
        = sizeof(abi_save_gpr_regs) / sizeof(abi_save_gpr_regs[0]);

    const size_t size_of_abi_save_regs
        = num_abi_save_gpr_regs * rax.getBit() / 8
        + xmm_to_preserve * xmm_len;

public:
    enum {
        _cmp_eq_oq = 0u,
        _cmp_lt_os = 1u,
        _cmp_le_os = 2u,
        _cmp_neq_uq = 4u,
        _cmp_nlt_us = 5u,
        _cmp_nle_us = 6u,

        _op_floor = 1u,
        _op_mxcsr = 4u,
    };

    xa::XReg param1_aarch64 = abi_param1_aarch64;
    class XRegValue : public xa::XReg
    {
        public:
        int64_t value_;
        explicit XRegValue(uint32_t idx, int64_t value)
            : Xbyak::Xbyak_aarch64::XReg(idx), value_(value) {}
        explicit XRegValue(uint32_t idx)
            : Xbyak::Xbyak_aarch64::XReg(idx), value_(0xFFFFFFFFFFFFFFFF) {}
    };

    inline size_t get_size_of_abi_save_regs_aarch64() { 
                    return size_of_abi_save_regs_aarch64; 
    }
    Xbyak::Reg64 param1 = abi_param1;
    const int EVEX_max_8b_offt = 0x200;
    const Xbyak::Reg64 reg_EVEX_max_8b_offt = rbp;

    inline size_t get_size_of_abi_save_regs() {
        return size_of_abi_save_regs;
    }


    void preamble() {
        CGA64::stp(x29, x30, pre_ptr(CGA64::sp, -16));
        /* x29 is a frame pointer. */
        CGA64::mov(x29, CGA64::sp);
        CGA64::sub(CGA64::sp, CGA64::sp, static_cast<int64_t>(preserved_stack_size) - 16);

        /* x9 can be used as a temporal register. */
        CGA64::mov(x9, CGA64::sp);

        if (vreg_to_preserve) {
            CGA64::st4((v8.d - v11.d)[0], post_ptr(x9, vreg_len_preserve*4));
            CGA64::st4((v12.d - v15.d)[0], post_ptr(x9, vreg_len_preserve*4));
        }
        for (size_t i = 0; i < num_abi_save_gpr_regs_aarch64; i += 2) {
            CGA64::stp(xa::XReg(abi_save_gpr_regs_aarch64[i]),
                xa::XReg(abi_save_gpr_regs_aarch64[i + 1]),
                post_ptr(x9, xreg_len*2));
        }
        CGA64::ptrue(P_ALL_ONE.b);
        CGA64::ptrue(P_MSB_384.b, xa::VL16);
        CGA64::ptrue(P_MSB_256.b, xa::VL32);
        CGA64::not_(P_MSB_384.b, P_ALL_ONE/xa::T_z, P_MSB_384.b);
        CGA64::not_(P_MSB_256.b, P_ALL_ONE/xa::T_z, P_MSB_256.b);
        
        /* arg values are passed different registers between x86_64 and aarch64. */
        CGA64::mov(x7, x0); /* First arg. */
        CGA64::mov(x6, x1); /* Sedond arg. */
        CGA64::mov(x2, x2);
        CGA64::mov(x1, x3);
        CGA64::mov(x8, x4);
        CGA64::mov(x9, x5); /* 6-th arg. */
        /* Note:If # of args is more than 6, 7-th, 8-th, ..., args are passed by stack. */
        CGA64::mov(x4, CGA64::sp); /* Intel64's stack register is 4-th register. */
        CGA64::sub_imm(X_TRANSLATOR_STACK, x4, xt_stack_offset, X_TMP_0);
        CGA64::mov_imm(X_TMP_0, getTranslatorVersion()); /*get translator version info */

        if (mayiuse(avx512_common)) {
            mov(reg_EVEX_max_8b_offt, 2 * EVEX_max_8b_offt);
        }
 
    }

//TODO:
#if 0
    // This function returns the address on the stack of the fist argument
    // that is not passed by register
    // By default it assumes to be called after the prologue
    // Note: that we cannot use RBP inside as we override it in preamble
    // for address computation in EVEX instructions
    inline const Xbyak::RegExp get_stack_params_address(
            bool after_prolog = true) {
        int saved_regs_size = after_prolog ? get_size_of_abi_save_regs() : 0;
#ifdef _WIN32
        // Using stack layout described in MS ABI
        // (https://docs.microsoft.com/en-us/cpp/build/stack-usage?view=vs-2019)
        // here, the return address and the first 4 parameters are allocated
        // on the stack
        int first_params_and_return_addr_size = 40;
#else
        // In System V ABI, only the return address is stacked
        // before the arguments
        int first_params_and_return_addr_size = 8;
#endif
        return x0 + saved_regs_size + first_params_and_return_addr_size;
    }
#endif

    void mic_prefetcht0(Xbyak::Address a) {
        if (mayiuse(avx512_mic))
            prefetcht0(a);
    }

    void mic_prefetcht1(Xbyak::Address a) {
        if (mayiuse(avx512_mic))
            prefetcht1(a);
    }

    void mic_prefetcht2(Xbyak::Address a) {
        if (mayiuse(avx512_mic))
            prefetcht2(a);
    }

    void uni_vzeroupper() {
        if (mayiuse(avx) && !mayiuse(avx512_mic))
            vzeroupper();
    }

    void postamble() {
        CGA64::mov(x9, CGA64::sp);
        CGA64::eor(P_ALL_ONE.b, P_ALL_ONE/xa::T_z, P_ALL_ONE.b, P_ALL_ONE.b);
        CGA64::eor(P_MSB_384.b, P_MSB_384/xa::T_z, P_MSB_384.b, P_MSB_384.b);
        CGA64::eor(P_MSB_256.b, P_MSB_256/xa::T_z, P_MSB_256.b, P_MSB_256.b);

        if (vreg_to_preserve) {
            CGA64::ld4((v8.d - v11.d)[0], post_ptr(x9, vreg_len_preserve*4));
            CGA64::ld4((v12.d - v15.d)[0], post_ptr(x9, vreg_len_preserve*4));
        }

        for (size_t i = 0; i < num_abi_save_gpr_regs_aarch64; i += 2) {
            CGA64::ldp(xa::XReg(abi_save_gpr_regs_aarch64[i]),
                xa::XReg(abi_save_gpr_regs_aarch64[i + 1]),
                post_ptr(x9, xreg_len*2));
        }

        CGA64::add(CGA64::sp, CGA64::sp, static_cast<int64_t>(preserved_stack_size) - 16);
        CGA64::ldp(x29, x30, post_ptr(CGA64::sp, 16));
        CGA64::ret();
    }

    template <typename T>
    Xbyak::Address EVEX_compress_addr(Xbyak::Reg64 base,
            T raw_offt, bool bcast = false)
    {
        using Xbyak::Zmm;
        using Xbyak::Reg64;
        using Xbyak::Address;
        using Xbyak::RegExp;

        assert(raw_offt <= INT_MAX);
        auto offt = static_cast<int>(raw_offt);

        int scale = 0;

        if (EVEX_max_8b_offt <= offt && offt < 3 * EVEX_max_8b_offt) {
            offt = offt - 2 * EVEX_max_8b_offt;
            scale = 1;
        } else if (3 * EVEX_max_8b_offt <= offt && offt < 5 * EVEX_max_8b_offt) {
            offt = offt - 4 * EVEX_max_8b_offt;
            scale = 2;
        }

        auto re = RegExp() + base + offt;
        if (scale)
            re = re + reg_EVEX_max_8b_offt * scale;

        if (bcast)
            return zword_b [re];
        else
            return zword [re];
    }

    Xbyak::Address make_safe_addr(const Xbyak::Reg64 &reg_out, size_t offt,
        const Xbyak::Reg64 &tmp_reg, bool bcast = false) {
        if (offt > INT_MAX) {
            mov(tmp_reg, offt);
            return bcast ? ptr_b[reg_out + tmp_reg] : ptr[reg_out + tmp_reg];
        } else {
            return bcast ? ptr_b[reg_out + offt] : ptr[reg_out + offt];
        }
    }

    Xbyak::Address EVEX_compress_addr_safe(const Xbyak::Reg64 &base,
        size_t raw_offt, const Xbyak::Reg64 &reg_offt, bool bcast = false) {
        if (raw_offt > INT_MAX) {
            return make_safe_addr(base, raw_offt, reg_offt, bcast);
        } else {
            return EVEX_compress_addr(base, raw_offt, bcast);
        }
    }

    void safe_add(const Xbyak::Reg64 &base, size_t raw_offt,
        const Xbyak::Reg64 &reg_offt) {
        if (raw_offt > INT_MAX) {
            mov(reg_offt, raw_offt);
            add(base, reg_offt);
        } else {
            add(base, raw_offt);
        }
    }

    void safe_sub(const Xbyak::Reg64 &base, size_t raw_offt,
        const Xbyak::Reg64 &reg_offt) {
        if (raw_offt > INT_MAX) {
            mov(reg_offt, raw_offt);
            sub(base, reg_offt);
        } else {
            sub(base, raw_offt);
        }
    }

    // Disallow char-based labels completely
    void L(const char *label) = delete;
    void L(Xbyak::Label& label) { Xbyak::CodeGenerator::L(label); }

    void L_aligned(Xbyak::Label &label, int alignment = 16) {
        align(alignment);
        L(label);
    }

    void uni_vpxor(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
                   const Xbyak::Operand &op) {
        assert(x1.getIdx() == x2.getIdx());
        pxor(x2, op);
    }
    void uni_vpxor(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
                   const Xbyak::Operand &op) {
        if (mayiuse(avx2)) {
            vpxor(x1, x2, op);
        } else {
            vxorps(x1, x2, op);
        }
    }
    void uni_vpxor(const Xbyak::Zmm &x1, const Xbyak::Zmm &x2,
                   const Xbyak::Operand &op) {
        vpxord(x1, x2, op);
    }

    void uni_vmovss(const Xbyak::Address& addr, const Xbyak::Xmm &x) {
        movss(addr, x);
    }
    void uni_vmovss(const Xbyak::Address& addr, const Xbyak::Ymm &x) {
        vmovss(addr, Xbyak::Xmm(x.getIdx()));
    }
    void uni_vmovss(const Xbyak::Xmm &x, const Xbyak::Address& addr) {
        movss(x, addr);
    }
    void uni_vmovss(const Xbyak::Ymm &x, const Xbyak::Address& addr) {
        vmovss(Xbyak::Xmm(x.getIdx()), addr);
    }

    void uni_vmovsd(const Xbyak::Address& addr, const Xbyak::Xmm &x) {
        movsd(addr, x);
    }
    void uni_vmovsd(const Xbyak::Address& addr, const Xbyak::Ymm &x) {
        vmovsd(addr, x);
    }
    void uni_vmovsd(const Xbyak::Xmm &x, const Xbyak::Address& addr) {
        movsd(x, addr);
    }
    void uni_vmovsd(const Xbyak::Ymm &x, const Xbyak::Address& addr) {
        vmovsd(x, addr);
    }

    void uni_vmovdqu(const Xbyak::Address &addr, const Xbyak::Xmm &x) {
        movdqu(addr, x);
    }
    void uni_vmovdqu(const Xbyak::Address &addr, const Xbyak::Ymm &x) {
        vmovdqu(addr, x);
    }
    void uni_vmovdqu(const Xbyak::Address &addr, const Xbyak::Zmm &x) {
        vmovdqu32(addr, x);
    }

    void uni_vmovdqu(const Xbyak::Xmm &x, const Xbyak::Address &addr) {
        movdqu(x, addr);
    }
    void uni_vmovdqu(const Xbyak::Ymm &x, const Xbyak::Address &addr) {
        vmovdqu(x, addr);
    }
    void uni_vmovdqu(const Xbyak::Zmm &x, const Xbyak::Address &addr) {
        vmovdqu32(x, addr);
    }

    void uni_vmovups(const Xbyak::Address &addr, const Xbyak::Xmm &x) {
        movups(addr, x);
    }
    void uni_vmovups(const Xbyak::Address &addr, const Xbyak::Ymm &x) {
        vmovups(addr, x);
    }

    void uni_vmovups(const Xbyak::Xmm &x, const Xbyak::Operand &op) {
        movups(x, op);
    }
    void uni_vmovups(const Xbyak::Ymm &x, const Xbyak::Operand &op) {
        vmovups(x, op);
    }

    void uni_vmovups_tail(const Xbyak::Address &addr, const Xbyak::Ymm &mask,
            const Xbyak::Ymm &x) {
        vmaskmovps(addr, mask, x);
    }
    void uni_vmovups_tail(const Xbyak::Ymm &x, const Xbyak::Ymm &mask,
            const Xbyak::Address &addr) {
        vmaskmovps(x, mask, addr);
    }

    void uni_vmovups_tail(const Xbyak::Address &addr, const Xbyak::Opmask &mask,
            const Xbyak::Zmm &x) {
        vmovups(addr | mask, x);
    }
    void uni_vmovups_tail(const Xbyak::Zmm &x, const Xbyak::Opmask &mask,
            const Xbyak::Address &addr) {
        vmovups(x | mask | T_z, addr);
    }

    void uni_vmovntps(const Xbyak::Address &addr, const Xbyak::Xmm &x) {
        movntps(addr, x);
    }
    void uni_vmovntps(const Xbyak::Address &addr, const Xbyak::Ymm &x) {
        vmovntps(addr, x);
    }

    void uni_vbroadcastss(const Xbyak::Xmm &x, const Xbyak::Operand &op) {
        movss(x, op);
        shufps(x, x, 0x0);
    }
    void uni_vbroadcastss(const Xbyak::Ymm &x, const Xbyak::Operand &op) {
        if (op.isMEM() || mayiuse(avx2)) {
            vbroadcastss(x, op);
        } else {
            Xbyak::Xmm t(x.getIdx());
            if (t.getIdx() != op.getIdx()) movss(t, op);
            vinsertf128(x, x, t, 1);
            vshufps(x, x, x, 0);
        }
    }

    void uni_vpbroadcastd(const Xbyak::Xmm &x, const Xbyak::Operand &op) {
        movsd(x, op);
        pshufd(x, x, 0x0);
    }
    void uni_vpbroadcastd(const Xbyak::Ymm &x, const Xbyak::Operand &op) {
        if (mayiuse(avx2)) {
            vpbroadcastd(x, op);
        } else {
            Xbyak::Xmm t(x.getIdx());
            if (t.getIdx() != op.getIdx()) movsd(t, op);
            vinsertf128(x, x, t, 1);
            vshufps(x, x, x, 0);
        }
    }

    void uni_vrcpss(const Xbyak::Xmm &x, const Xbyak::Operand &op) {
        rcpss(x, op);
    }
    void uni_vrcpss(const Xbyak::Ymm &x1, const Xbyak::Xmm &x2) {
        Xbyak::Xmm x1_(x1.getIdx());
        Xbyak::Xmm x2_(x2.getIdx());
        vrcpss(x1_, x1_, x2_);
    }
    void uni_vrcpss(const Xbyak::Ymm &x, const Xbyak::Address &op) {
        Xbyak::Xmm x_(x.getIdx());
        vrcpss(x_, x_, op);
    }

    void uni_vrcpps(const Xbyak::Xmm &x, const Xbyak::Operand &op) {
        rcpps(x, op);
    }
    void uni_vrcpps(const Xbyak::Ymm &x, const Xbyak::Operand &op) {
        vrcpps(x, op);
    }
    void uni_vrcpps(const Xbyak::Zmm &x, const Xbyak::Operand &op) {
        vrcp14ps(x, op);
    }

    void uni_vdivps(const Xbyak::Xmm &x, const Xbyak::Operand &op1,
                    const Xbyak::Operand &op2 = Xbyak::Operand()) {
        assert(x.getIdx() == op1.getIdx());
        divps(x, op2);
    }
    void uni_vdivps(const Xbyak::Ymm &x, const Xbyak::Operand &op1,
                    const Xbyak::Operand &op2 = Xbyak::Operand()) {
        vdivps(x, op1, op2);
    }

    void uni_vdivps(const Xbyak::Xmm &x, const Xbyak::Operand &op1,
                    const Xbyak::Operand &op2, const Xbyak::Xmm &buf) {
        movups(buf, op1);
        divps(buf, op2);
        if (x.getIdx() != buf.getIdx()) {
            movups(x, buf);
        }
    }

    void uni_vdivps(const Xbyak::Ymm &x, const Xbyak::Operand &op1,
                    const Xbyak::Operand &op2, const Xbyak::Ymm &buf) {
        vdivps(x, op1, op2);
    }

    void uni_vaddps(const Xbyak::Xmm &x, const Xbyak::Operand &op1,
                    const Xbyak::Operand &op2 = Xbyak::Operand()) {
        assert(x.getIdx() == op1.getIdx());
        addps(x, op2);
    }
    void uni_vaddps(const Xbyak::Ymm &x, const Xbyak::Operand &op1,
                    const Xbyak::Operand &op2 = Xbyak::Operand()) {
        vaddps(x, op1, op2);
    }
    void uni_vaddss(const Xbyak::Xmm &x, const Xbyak::Operand &op1,
                    const Xbyak::Operand &op2 = Xbyak::Operand()) {
        assert(x.getIdx() == op1.getIdx());
        addss(x, op2);
    }
    void uni_vaddss(const Xbyak::Ymm &x, const Xbyak::Operand &op1,
                    const Xbyak::Operand &op2 = Xbyak::Operand()) {
        vaddss(x, op1, op2);
    }

    void uni_vpsignd(const Xbyak::Xmm& x1, const Xbyak::Xmm& x2,
                     const Xbyak::Operand& op) {
        assert(x1.getIdx() == x2.getIdx());
        psignd(x1, op);
    }
    void uni_vpsignd(const Xbyak::Ymm& x1, const Xbyak::Ymm& x2,
                     const Xbyak::Operand& op) {
        vpsignd(x1, x2, op);
    }

    void uni_vsubss(const Xbyak::Xmm &x, const Xbyak::Operand &op1,
                    const Xbyak::Operand &op2 = Xbyak::Operand()) {
        assert(x.getIdx() == op1.getIdx());
        subps(x, op2);
    }
    void uni_vsubss(const Xbyak::Ymm &x, const Xbyak::Operand &op1,
                    const Xbyak::Operand &op2 = Xbyak::Operand()) {
        vsubss(x, Xbyak::Xmm(op1.getIdx()), Xbyak::Xmm(op2.getIdx()));
    }

    void uni_vsubps(const Xbyak::Xmm &x, const Xbyak::Operand &op1,
                    const Xbyak::Operand &op2 = Xbyak::Operand()) {
        assert(x.getIdx() == op1.getIdx());
        subps(x, op2);
    }
    void uni_vsubps(const Xbyak::Ymm &x, const Xbyak::Operand &op1,
                    const Xbyak::Operand &op2 = Xbyak::Operand()) {
        vsubps(x, op1, op2);
    }

    void uni_vsubps(const Xbyak::Xmm &x, const Xbyak::Operand &op1,
                    const Xbyak::Operand &op2, const Xbyak::Xmm &buf) {
        movups(buf, op1);
        subps(buf, op2);
        if (x.getIdx() != buf.getIdx()) {
            movups(x, buf);
        }
    }

    void uni_vsubps(const Xbyak::Ymm &x, const Xbyak::Operand &op1,
                    const Xbyak::Operand &op2, const Xbyak::Ymm &buf) {
        vsubps(x, op1, op2);
    }

    void uni_vmulps(const Xbyak::Xmm &x, const Xbyak::Operand &op1,
                    const Xbyak::Operand &op2 = Xbyak::Operand()) {
        assert(x.getIdx() == op1.getIdx());
        mulps(x, op2);
    }
    void uni_vmulps(const Xbyak::Ymm &x, const Xbyak::Operand &op1,
                    const Xbyak::Operand &op2 = Xbyak::Operand()) {
        vmulps(x, op1, op2);
    }

    void uni_vmulss(const Xbyak::Xmm &x, const Xbyak::Operand &op1,
                    const Xbyak::Operand &op2 = Xbyak::Operand()) {
        assert(x.getIdx() == op1.getIdx());
        mulss(x, op2);
    }
    void uni_vmulss(const Xbyak::Ymm &x, const Xbyak::Operand &op1,
                    const Xbyak::Address &op2) {
        vmulss(x, Xbyak::Xmm(op1.getIdx()), op2);
    }
    void uni_vmulss(const Xbyak::Ymm &x, const Xbyak::Operand &op1,
                    const Xbyak::Ymm &op2) {
        vmulss(x, Xbyak::Xmm(op1.getIdx()), Xbyak::Xmm(op2.getIdx()));
    }

    void uni_vfmadd213ps(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
                         const Xbyak::Operand &op) {
        mulps(x1, x2);
        addps(x1, op);
    }
    void uni_vfmadd213ps(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
                         const Xbyak::Operand &op) {
        vfmadd213ps(x1, x2, op);
    }

    void uni_vfmadd213ss(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
                         const Xbyak::Operand &op) {
        mulss(x1, x2);
        addss(x1, op);
    }
    void uni_vfmadd213ss(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
                         const Xbyak::Operand &op) {
        vfmadd213ss(x1, x2, op);
    }

    void uni_vfmadd231ps(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
                         const Xbyak::Operand &op) {
        mulps(x2, op);
        addps(x1, x2);
    }
    void uni_vfmadd231ps(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
                         const Xbyak::Operand &op) {
        vfmadd231ps(x1, x2, op);
    }
    void uni_vfmadd231ss(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
                         const Xbyak::Operand &op) {
        mulss(x2, op);
        addss(x1, x2);
    }
    void uni_vfmadd231ss(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
                         const Xbyak::Operand &op) {
        vfmadd231ss(Xbyak::Xmm(x1.getIdx()), Xbyak::Xmm(x2.getIdx()), op);
    }

    void uni_vfnmadd231ps(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
                           const Xbyak::Operand &op) {
        mulps(x2, op);
        subps(x1, x2);
    }

    void uni_vfnmadd231ps(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
                           const Xbyak::Operand &op) {
        vfnmadd231ps(x1, x2, op);
    }

    void uni_vsqrtps(const Xbyak::Xmm &x, const Xbyak::Operand &op) {
        sqrtps(x, op);
    }
    void uni_vsqrtps(const Xbyak::Ymm &x, const Xbyak::Operand &op) {
        vsqrtps(x, op);
    }

    void uni_vpaddd(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
                    const Xbyak::Operand &op) {
        assert(x1.getIdx() == x2.getIdx());
        paddd(x2, op);
    }
    void uni_vpaddd(const Xbyak::Ymm &x1, const Xbyak::Xmm &x2,
                    const Xbyak::Operand &op) {
        vpaddd(x1, x2, op);
    }

    void uni_vandps(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
                    const Xbyak::Operand &op = Xbyak::Operand()) {
        assert(x1.getIdx() == x2.getIdx());
        andps(x1, op);
    }
    void uni_vandps(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
                    const Xbyak::Operand &op = Xbyak::Operand()) {
        if (!mayiuse(avx512_common) || x1.getBit() < 512)
            vandps(x1, x2, op);
        else
            vpandd(x1, x2, op);
    }

    void uni_vorps(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
                    const Xbyak::Operand &op = Xbyak::Operand()) {
        assert(x1.getIdx() == x2.getIdx());
        orps(x1, op);
    }
    void uni_vorps(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
                    const Xbyak::Operand &op = Xbyak::Operand()) {
        if (!mayiuse(avx512_common) || x1.getBit() < 512)
            vorps(x1, x2, op);
        else
            vpord(x1, x2, op);
    }

    void uni_vpslld(const Xbyak::Xmm &x, const Xbyak::Operand &op,
                    const int imm) {
        assert(x.getIdx() == op.getIdx());
        pslld(x, imm);
    }
    void uni_vpslld(const Xbyak::Ymm &x, const Xbyak::Operand &op,
                    const int imm) {
        vpslld(x, op, imm);
    }

    void uni_vpsrld(const Xbyak::Xmm &x, const Xbyak::Operand &op,
                    const int imm) {
        assert(x.getIdx() == op.getIdx());
        psrld(x, imm);
    }
    void uni_vpsrld(const Xbyak::Ymm &x, const Xbyak::Operand &op,
                    const int imm) {
        vpsrld(x, op, imm);
    }

    void uni_vmaxps(const Xbyak::Xmm &x, const Xbyak::Operand &op1,
                    const Xbyak::Operand &op2 = Xbyak::Operand()) {
        assert(x.getIdx() == op1.getIdx());
        maxps(x, op2);
    }
    void uni_vmaxps(const Xbyak::Ymm &x, const Xbyak::Operand &op1,
                    const Xbyak::Operand &op2 = Xbyak::Operand()) {
        vmaxps(x, op1, op2);
    }

    void uni_vminps(const Xbyak::Xmm &x, const Xbyak::Operand &op1,
                    const Xbyak::Operand &op2 = Xbyak::Operand()) {
        assert(x.getIdx() == op1.getIdx());
        minps(x, op2);
    }
    void uni_vminps(const Xbyak::Ymm &x, const Xbyak::Operand &op1,
                    const Xbyak::Operand &op2 = Xbyak::Operand()) {
        vminps(x, op1, op2);
    }

    void uni_vcmpgtps(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
                      const Xbyak::Operand &op) {
        assert(x1.getIdx() == x2.getIdx());
        cmpps(x1, op, _cmp_nle_us);
    }

    void uni_vcmpgtps(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
                      const Xbyak::Operand &op) {
        vcmpgtps(x1, x2, op);
    }

    void uni_vcmpgeps(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
                      const Xbyak::Operand &op) {
        assert(x1.getIdx() == x2.getIdx());
        cmpps(x1, op, _cmp_nlt_us);
    }

    void uni_vcmpgeps(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
                      const Xbyak::Operand &op) {
        vcmpps(x1, x2, op, _cmp_nlt_us);
    }

    void uni_vtestps(const Xbyak::Xmm &x1, const Xbyak::Operand &op) {
        ptest(x1, op);
    }

    void uni_vtestps(const Xbyak::Ymm &x1, const Xbyak::Operand &op) {
        assert(!(x1.isZMM() || op.isZMM()));
        vtestps(x1, op);
    }

    void uni_vblendvps(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
                       const Xbyak::Operand &op, const Xbyak::Xmm &msk) {
        assert(x1.getIdx() == x2.getIdx());
        assert(msk.getIdx() == 0);
        blendvps(x1, op);
    }
    void uni_vblendvps(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
                       const Xbyak::Operand &op, const Xbyak::Ymm &msk) {
        vblendvps(x1, x2, op, msk);
    }

    void uni_vroundps(const Xbyak::Xmm &x, const Xbyak::Operand &op,
                      const int imm) {
        roundps(x, op, imm);
    }
    void uni_vroundps(const Xbyak::Ymm &x, const Xbyak::Operand &op,
                      const int imm) {
        vroundps(x, op, imm);
    }
    void uni_vroundps(const Xbyak::Zmm &x, const Xbyak::Operand &op,
                      const int imm) {
        vrndscaleps(x, op, imm & 0x3);
    }

    void uni_vcvtps2dq(const Xbyak::Xmm &x, const Xbyak::Operand &op) {
        cvtps2dq(x, op);
    }
    void uni_vcvtps2dq(const Xbyak::Ymm &x, const Xbyak::Operand &op) {
        vcvtps2dq(x, op);
    }

    void uni_vcvtdq2ps(const Xbyak::Xmm &x, const Xbyak::Operand &op) {
        cvtdq2ps(x, op);
    }
    void uni_vcvtdq2ps(const Xbyak::Ymm &x, const Xbyak::Operand &op) {
        vcvtdq2ps(x, op);
    }

    void uni_vmovmskps(const Xbyak::Reg &x1, const Xbyak::Xmm &x2) {
        movmskps(x1.cvt64(), x2);
    }
    void uni_vmovmskps(const Xbyak::Reg &x1, const Xbyak::Ymm &x2) {
        vmovmskps(x1, x2);
    }

    void uni_vmovq(const Xbyak::Xmm &x, const Xbyak::Reg64 &r) {
        if (mayiuse(avx))
            vmovq(x, r);
        else
            movq(x, r);
    }
    void uni_vmovq(const Xbyak::Address &addr, const Xbyak::Xmm &x) {
        if (mayiuse(avx))
            vmovq(addr, x);
        else
            movq(addr, x);
    }

    void uni_vpackssdw(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2, const Xbyak::Operand &op){
        assert(x1.getIdx() == x1.getIdx());
        packssdw(x1, op);
    }
    void uni_vpackssdw(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2, const Xbyak::Operand &op){
        vpackssdw(x1, x2, op);
    }

    void uni_vpackuswb(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2, const Xbyak::Operand &op){
        assert(x1.getIdx() == x1.getIdx());
        packuswb(x1, op);
    }
    void uni_vpackuswb(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2, const Xbyak::Operand &op){
        vpackuswb(x1, x2, op);
    }


    void mul_by_const(const Xbyak::Reg &out,
            const Xbyak::Reg64 &tmp, int value) {
        // Generates a shift + add sequence for multiplicating contents of the
        // out register by a known JIT-time value. Clobbers the tmp register.
        //
        // Pros compared to mul/imul:
        // - does not require using known registers
        // - not microcoded on Intel(R) Xeon Phi(TM) processors
        // Still, there are probably a lot of cases when mul/imul is faster on
        // Intel(R) Core(TM) processors. Not intended for critical path.

        // TODO: detect when overflow is emminent (Roma)
        // TODO: detect when using mul/imul is a better option (Roma)

        int p = 0; // the current power of 2
        int old_p = 0; // the last seen power of 2 such that value[old_p] != 0

        xor_(tmp, tmp);
        while (value) {
            if (value & 1) {
                int shift = p - old_p;
                if (shift) {
                    shl(out, shift);
                    old_p = p;
                }
                add(tmp, out);
            }
            value >>= 1;
            p++;
        }
        mov(out, tmp);
    }


    void dump_code32(const Xbyak::XBYAK_CODE_PTR *code) const {
        if (code) {
            static int counter = 0;
#define MAX_FNAME_LEN 256
            char fname[MAX_FNAME_LEN + 1];
            snprintf(fname, MAX_FNAME_LEN, "dnnl_dump_%s.%d.bin", name(),
                    counter);
            counter++;

            FILE *fp = fopen(fname, "w+");
            // Failure to dump code is not fatal
            if (fp) {
#ifdef DNNL_INDIRECT_JIT_AARCH64
                size_t unused = fwrite(code, getSize() * 4, 1, fp);
#else
                size_t unused = fwrite(code, getSize(), 1, fp);
#endif
                UNUSED(unused);
                fclose(fp);
            }
        }
#undef MAX_FNAME_LEN
    }

    void dump_code(const Xbyak::uint8 *code) const {
        if (code) {
            static int counter = 0;
#define MAX_FNAME_LEN 256
            char fname[MAX_FNAME_LEN + 1];
            snprintf(fname, MAX_FNAME_LEN, "dnnl_dump_%s.%d.bin", name(),
                    counter);
            counter++;

            FILE *fp = fopen(fname, "w+");

	    // Failure to dump code is not fatal
            if (fp) {
                size_t unused = fwrite(code, getSize()*4, 1, fp);
                UNUSED(unused);
                fclose(fp);
            }
        }
#undef MAX_FNAME_LEN
    }


    static unsigned int get_A64FX_cache_size(int level, bool per_core = true, int nthreads = 1) {
        unsigned int l = level - 1;
        // Currently, if XByak is not able to fetch the cache topology
        // we default to 64KiB of L1 per core, 8MiB of L2 per 1CMG.
        if (cpu.getDataCacheLevels() == 0) {
            const int L1_cache_per_core = 65536;
            const int L2_cache_per_CMG = 8388608;
            int num_cores = per_core ? 1 : nthreads;
            switch (l) {
            case (0): return L1_cache_per_core;
            case (1): return L2_cache_per_CMG * utils::div_up(num_cores, 12);
            default: return 0;
            }
        }
        if (l < cpu.getDataCacheLevels()) {
           return cpu.getDataCacheSize(l)
                    / (per_core ? cpu.getCoresSharingDataCache(l) : 1);
        } else
        return 0;
    }

    DNNL_DISALLOW_COPY_AND_ASSIGN(jit_generator);

public:
    jit_generator(void *code_ptr = nullptr, size_t code_size = MAX_CODE_SIZE,
            bool use_autogrow = true)
        : Xbyak::CodeGenerator(code_size,
          code_ptr) {}
#if 0
                (code_ptr == nullptr && use_autogrow) ? Xbyak::Xbyak_aarch64::AutoGrow
                                                      : code_ptr) {}
#endif
    virtual ~jit_generator() {}

    virtual const char *name() const = 0;
    virtual const char *source_file() const = 0;

    const uint32_t *getCode32() {
        this->ready();
        const uint32_t *code = CodeGeneratorAArch64::getCode32();
      
        if( get_jit_dump() ) dump_code32(code);

        return code;
    }

    // XXX: use normal_case name and update all callees (?)
    const Xbyak::uint8 *getCode() {
        const Xbyak::uint8 *code = CodeGenerator::getCode();

        if (get_jit_dump())
            dump_code(code);

        return code;
    }


    template <typename F>
    const F getCode() {
        return (const F)getCode32();
    }
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
