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

    Xbyak::Xbyak_aarch64::XReg param1 = abi_param1_aarch64;
    class XRegValue : public Xbyak::Xbyak_aarch64::XReg
    {
        public:
        int64_t value_;
        explicit XRegValue(uint32_t idx, int64_t value)
            : Xbyak::Xbyak_aarch64::XReg(idx), value_(value) {}
        explicit XRegValue(uint32_t idx)
            : Xbyak::Xbyak_aarch64::XReg(idx), value_(0xFFFFFFFFFFFFFFFF) {}
    };


    const int EVEX_max_8b_offt = 0x200;

    inline size_t get_size_of_abi_save_regs_aarch64() { return size_of_abi_save_regs_aarch64; }

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
        for (size_t i = 0; i < num_abi_save_gpr_regs_aarch64; i += 2) {
            CodeGeneratorAArch64::stp(Xbyak::Xbyak_aarch64::XReg(abi_save_gpr_regs_aarch64[i]),
                Xbyak::Xbyak_aarch64::XReg(abi_save_gpr_regs_aarch64[i + 1]),
                post_ptr(x9, xreg_len*2));
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

    void uni_vzeroupper() {
        // TODO
        assert(NULL);
    }

    void postamble() {
        CodeGeneratorAArch64::mov(x9, CodeGeneratorAArch64::sp);

        if (vreg_to_preserve) {
            CodeGeneratorAArch64::ld4((v8.d - v11.d)[0], post_ptr(x9, vreg_len_preserve*4));
            CodeGeneratorAArch64::ld4((v12.d - v15.d)[0], post_ptr(x9, vreg_len_preserve*4));
        }

        for (size_t i = 0; i < num_abi_save_gpr_regs_aarch64; i += 2) {
            CodeGeneratorAArch64::ldp(Xbyak::Xbyak_aarch64::XReg(abi_save_gpr_regs_aarch64[i]),
                Xbyak::Xbyak_aarch64::XReg(abi_save_gpr_regs_aarch64[i + 1]),
                post_ptr(x9, xreg_len*2));
        }

        CodeGeneratorAArch64::add(sp, sp, static_cast<int64_t>(preserved_stack_size) - 16);
        CodeGeneratorAArch64::ldp(x29, x30, post_ptr(CodeGeneratorAArch64::sp, 16));
        CodeGeneratorAArch64::ret();
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

    const uint32_t *getCode32() {
        this->ready();
        const uint32_t *code = CodeGeneratorAArch64::getCode32();
      
        if( get_jit_dump() ) dump_code32(code);

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
