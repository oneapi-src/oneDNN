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
#include "xbyak/xbyak.h"

#define XBYAK_VERSION 0x5000

#if XBYAK_VERSION >= 0x5000
    #define ZWORD   zword
    #define ZWORD_b zword_b
    #define YWORD   yword
    #define YWORD_b yzword_b
#else
    #define ZWORD zmmword
    #define YWORD ymmword
#endif

namespace mkldnn {
namespace impl {
namespace cpu {

namespace {
using namespace Xbyak;
#ifdef XBYAK64
static const Operand::Code reg_to_preserve[] = {
    Operand::RBX, Operand::RSP, Operand::RBP,
    Operand::R12, Operand::R13, Operand::R14, Operand::R15,
#ifdef _WIN
    Operand::RDI, Operand::RSI,
#endif
};
#ifdef _WIN
static const Reg64 cdecl_param1(Operand::RCX), cdecl_param2(Operand::RDX),
             cdecl_param3(Operand::R8), cdecl_param4(Operand::R9);
#else
static const Reg64 cdecl_param1(Operand::RDI), cdecl_param2(Operand::RSI),
             cdecl_param3(Operand::RDX), cdecl_param4(Operand::RCX);
#endif
#endif
}

class jit_generator : public Xbyak::CodeGenerator
{
protected:
    Xbyak::Reg64 param1 = cdecl_param1;
    void preamble() {
        const size_t nregs = sizeof(reg_to_preserve)/sizeof(reg_to_preserve[0]);
        for (size_t i = 0; i < nregs; ++i)
            push(Xbyak::Reg64(reg_to_preserve[i]));
    }
    void postamble() {
        const size_t nregs = sizeof(reg_to_preserve)/sizeof(reg_to_preserve[0]);
        for (size_t i = 0; i < nregs; ++i)
            pop(Xbyak::Reg64(reg_to_preserve[nregs - 1 - i]));
        ret();
    }

public:
    jit_generator(
        void* code_ptr = nullptr,
        size_t code_size = 8 * Xbyak::DEFAULT_MAX_CODE_SIZE
        ) : Xbyak::CodeGenerator(code_size, code_ptr)
    {
    }
};

}
}
}

#endif
