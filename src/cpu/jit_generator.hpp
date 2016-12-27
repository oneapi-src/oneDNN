/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#ifndef CPU_JIT_AVX2_GENERATOR_HPP
#define CPU_JIT_AVX2_GENERATOR_HPP

#define XBYAK64
#define XBYAK_NO_OP_NAMES
/* in order to make selinux happy memory that would be marked with X-bit should
 * be obtained with mmap */
#define XBYAK_USE_MMAP_ALLOCATOR
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

namespace mkldnn {
namespace impl {
namespace cpu {


// TODO: move this to jit_generator class?
namespace {

// TODO: move this somewhere else? Although this is only used by jit kernels
// (Roma)
static inline int float2int(float x) {
    union {
        float vfloat;
        int vint;
    } cvt;
    cvt.vfloat = x;
    return cvt.vint;
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

#ifdef XBYAK64
constexpr Xbyak::Operand::Code abi_save_regs[] = {
    Xbyak::Operand::RBX, Xbyak::Operand::RSP, Xbyak::Operand::RBP,
    Xbyak::Operand::R12, Xbyak::Operand::R13, Xbyak::Operand::R14,
    Xbyak::Operand::R15,
#ifdef _WIN
    Xbyak::Operand::RDI, Xbyak::Operand::RSI,
#endif
};
constexpr size_t num_abi_save_regs
    = sizeof(abi_save_regs) / sizeof(abi_save_regs[0]);

#ifdef _WIN
static const Xbyak::Reg64 abi_param1(Xbyak::Operand::RCX),
             abi_param2(Xbyak::Operand::RDX),
             abi_param3(Xbyak::Operand::R8),
             abi_param4(Xbyak::Operand::R9);
#else
static const Xbyak::Reg64 abi_param1(Xbyak::Operand::RDI),
             abi_param2(Xbyak::Operand::RSI),
             abi_param3(Xbyak::Operand::RDX),
             abi_param4(Xbyak::Operand::RCX);
#endif
#endif

typedef enum {
    avx2,
    avx512_mic,
} cpu_isa_t;

static inline bool mayiuse(const cpu_isa_t cpu_isa) {
    using namespace Xbyak::util;
    static Cpu cpu;

    switch (cpu_isa) {
    case avx2:
        return cpu.has(Cpu::tAVX2);
    case avx512_mic:
        return true
            && cpu.has(Cpu::tAVX512F)
            && cpu.has(Cpu::tAVX512CD)
            && cpu.has(Cpu::tAVX512ER)
            && cpu.has(Cpu::tAVX512PF);
    }
    return false;
}

}

class jit_generator : public Xbyak::CodeGenerator
{
protected:
    Xbyak::Reg64 param1 = abi_param1;

    void preamble() {
        for (size_t i = 0; i < num_abi_save_regs; ++i)
            push(Xbyak::Reg64(abi_save_regs[i]));
    }

    void postamble() {
        for (size_t i = 0; i < num_abi_save_regs; ++i)
            pop(Xbyak::Reg64(abi_save_regs[num_abi_save_regs - 1 - i]));
        ret();
    }

public:
    jit_generator(
        void *code_ptr = nullptr,
        size_t code_size = 32 * 1024 // size of a typical IC$
        ) : Xbyak::CodeGenerator(code_size, code_ptr)
    {
    }
};

}
}
}

#endif
