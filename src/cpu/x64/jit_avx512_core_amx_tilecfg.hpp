/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef CPU_X64_JIT_AVX512_CORE_AMX_TILECFG_HPP
#define CPU_X64_JIT_AVX512_CORE_AMX_TILECFG_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"
#include "cpu/x64/jit_uni_eltwise.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_avx512_core_amx_tilecfg_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_amx_tilecfg_t)

    jit_avx512_core_amx_tilecfg_t(jit_conv_conf_t ajcp) { generate(); }

private:
    void generate() {
        preamble();

        tilerelease();
        ldtilecfg(ptr[abi_param1]);

        postamble();
    }
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
