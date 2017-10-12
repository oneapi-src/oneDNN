/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
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

#ifndef CPU_ISA_HPP
#define CPU_ISA_HPP

namespace mkldnn {
namespace impl {
namespace cpu {

typedef enum {
    isa_any,
    sse42,
    avx2,
    avx512_common,
    avx512_core,
    avx512_mic,
    avx512_mic_4ops,
} cpu_isa_t;

#if defined(TARGET_VANILLA)
// should not include jit_generator.hpp (or any other jit stuff)
static inline constexpr bool mayiuse(const cpu_isa_t /*cpu_isa*/) {
    return true;
}
#endif

}
}
}
#endif // CPU_ISA_HPP
