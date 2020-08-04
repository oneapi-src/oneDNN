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

#include <limits.h>

#include "common/bit_cast.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/cpu_isa_traits.hpp"

#include "cpu/aarch64/jit_utils/jit_utils.hpp"

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

static inline void tc_configure_tile(
        tileconfig_t *tc, int t, int rows, int cols) {
    tc->rows[t] = rows;
    tc->cols[t] = cols;
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

} // namespace

class jit_generator : public Xbyak::Xbyak_aarch64::CodeGeneratorAArch64 {
private:
    const size_t xmm_len = 16;
#ifdef _WIN32
    const size_t xmm_to_preserve_start = 6;
    const size_t xmm_to_preserve = 10;
#else
    const size_t xmm_to_preserve_start = 0;
    const size_t xmm_to_preserve = 0;
#endif

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

    Xbyak::Xbyak_aarch64::XReg param1 = abi_param1;
    const int EVEX_max_8b_offt = 0x200;

    inline size_t get_size_of_abi_save_regs() { return size_of_abi_save_regs; }

    void preamble() {
        CodeGeneratorAArch64::stp(x29, x30,
                                  pre_ptr(CodeGeneratorAArch64::sp,
                                  -16));
        /* x29 is a frame pointer. */
        CodeGeneratorAArch64::mov(x29, CodeGeneratorAArch64::sp);
        CodeGeneratorAArch64::sub(sp, sp, static_cast<int64_t>(preserved_stack_size) - 16);

        /* x9 can be used as a temporal register. */
        CodeGeneratorAArch64::mov(x9, CodeGeneratorAArch64::sp);

        if (vreg_to_preserve) {
            CodeGeneratorAArch64::st4((v8.d - v11.d)[0], post_ptr(x9, vreg_len_preserve*4));
            CodeGeneratorAArch64::st4((v12.d - v15.d)[0], post_ptr(x9, vreg_len_preserve*4));
        }
        for (size_t i = 0; i < num_abi_save_gpr_regs; i += 2) {
            CodeGeneratorAArch64::stp(Xbyak::Xbyak_aarch64::XReg(abi_save_gpr_regs_aarch64[i]),
                Xbyak::Xbyak_aarch64::XReg(abi_save_gpr_regs_aarch64[i + 1]),
                post_ptr(x9, xreg_len*2));
        }
    }

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
        return rsp + saved_regs_size + first_params_and_return_addr_size;
    }

    void uni_vzeroupper() {
        // TODO
        if (mayiuse(avx) && !mayiuse(avx512_mic) && !mayiuse(sve)) assert(NULL);
    }

    void postamble() {
        CodeGeneratorAArch64::mov(x9, CodeGeneratorAArch64::sp);

        if (vreg_to_preserve) {
            CodeGeneratorAArch64::ld4((v8.d - v11.d)[0], post_ptr(x9, vreg_len_preserve*4));
            CodeGeneratorAArch64::ld4((v12.d - v15.d)[0], post_ptr(x9, vreg_len_preserve*4));
        }

        for (size_t i = 0; i < num_abi_save_gpr_regs; i += 2) {
            CodeGeneratorAArch64::ldp(Xbyak::Xbyak_aarch64::XReg(abi_save_gpr_regs_aarch64[i]),
                Xbyak::Xbyak_aarch64::XReg(abi_save_gpr_regs_aarch64[i + 1]),
                post_ptr(x9, xreg_len*2));
        }

        CodeGeneratorAArch64::add(sp, sp, static_cast<int64_t>(preserved_stack_size) - 16);
        CodeGeneratorAArch64::ldp(x29, x30, post_ptr(CodeGeneratorAArch64::sp, 16));
        CodeGeneratorAArch64::ret();
    }

    // Disallow char-based labels completely
    void L(const char *label) = delete;
    void L(Xbyak::Xbyak_aarch64::LabelAArch64 &label) { 
          Xbyak::Xbyak_aarch64::CodeGeneratorAArch64::L_aarch64(label);
    }

    void L_aligned(Xbyak::Xbyak_aarch64::LabelAArch64 &label, int alignment = 16) {
        align(alignment);
        L(label);
    }

    DNNL_DISALLOW_COPY_AND_ASSIGN(jit_generator);

public:
    jit_generator(void *code_ptr = nullptr, size_t code_size = MAX_CODE_SIZE,
            bool use_autogrow = true)
        : Xbyak::Xbyak_aarch64::CodeGeneratorAArch64(code_size,
                (code_ptr == nullptr && use_autogrow) ? Xbyak::AutoGrow
                                                      : code_ptr) {}
    virtual ~jit_generator() {}

    virtual const char *name() const = 0;
    virtual const char *source_file() const = 0;

    void register_jit_code(const Xbyak::uint8 *code, size_t code_size) const {
        jit_utils::register_jit_code(code, code_size, name(), source_file());
    }

    const uint32_t *getCode32() {
        this->ready();
        const uint32_t *code = CodeGeneratorAArch64::getCode32();
        register_jit_code(code, getSize());
        return code;
    }

    template <typename F>
    const F getCode() {
        return (const F)getCode32();
    }
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
