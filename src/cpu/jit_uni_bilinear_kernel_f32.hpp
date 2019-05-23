/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#ifndef CPU_JIT_UNI_BILINEAR_KERNEL_F32_HPP
#define CPU_JIT_UNI_BILINEAR_KERNEL_F32_HPP

#include <cfloat>

#include "c_types_map.hpp"
#include "resize_bilinear_pd.hpp"
#include "type_helpers.hpp"

#include "jit_generator.hpp"
#include "jit_primitive_conf.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace Xbyak;

template <cpu_isa_t isa>
struct jit_uni_bilinear_kernel_f32: public jit_generator {
    jit_uni_bilinear_kernel_f32(jit_bilinear_conf_t ajpp): jpp(ajpp) {}

    jit_bilinear_conf_t jpp;

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_bilinear_kernel_f32)

    void operator()(jit_bilinear_call_s *arg) { jit_ker(arg); }
    static status_t init_conf(jit_bilinear_conf_t &jbp,
            const resize_bilinear_pd_t *ppd);
private:
    void (*jit_ker)(jit_bilinear_call_s *);
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
