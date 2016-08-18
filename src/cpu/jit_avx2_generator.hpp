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

#include "xbyak_proxy.hpp"

namespace mkldnn { namespace impl { namespace cpu {

class jit_avx2_generator : public Xbyak::CodeGenerator
{
protected:
    Xbyak::Reg64 param1 = Xbyak::util::cdecl_param1;
    void preamble() {
        using Xbyak::util::reg_to_preserve;
        const size_t nregs = sizeof(reg_to_preserve)/sizeof(reg_to_preserve[0]);
        for (size_t i = 0; i < nregs; ++i)
            push(Xbyak::Reg64(reg_to_preserve[i]));
    }
    void postamble() {
        using Xbyak::util::reg_to_preserve;
        const size_t nregs = sizeof(reg_to_preserve)/sizeof(reg_to_preserve[0]);
        for (size_t i = 0; i < nregs; ++i)
            pop(Xbyak::Reg64(reg_to_preserve[nregs - 1 - i]));
        ret();
    }

public:
    jit_avx2_generator(
        void* code_ptr = nullptr,
        size_t code_size = 8 * Xbyak::DEFAULT_MAX_CODE_SIZE
        ) : Xbyak::CodeGenerator(code_size, code_ptr)
    {
    }
};

}}}

#endif
