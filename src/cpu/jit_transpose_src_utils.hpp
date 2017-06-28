/*******************************************************************************
* Copyright 2017 Intel Corporation
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

#ifndef CPU_JIT_TRANSPOSE_SRC_HPP
#define CPU_JIT_TRANSPOSE_SRC_HPP

#include "jit_generator.hpp"
#include "jit_primitive_conf.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct jit_src_transpose_s {
    const void *src;
    const void *tr_src;
    const void *src_prf;
    const void *tr_src_prf;
};

struct jit_transpose_src: public jit_generator
{
    jit_transpose_src(jit_conv_conf_t *aparams)
        : params(aparams)
    {
        this->generate();
        jit_ker = (decltype(jit_ker))this->getCode();
    }

    jit_conv_conf_t *params;
    void (*jit_ker)(jit_src_transpose_s *);

    void operator()(jit_src_transpose_s *arg) { jit_ker(arg); }

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using opmask_t = const Xbyak::Opmask;

    enum { typesize = sizeof(float), transpose_size = 16, small_spatial = 14 };
    int src_stride, tr_src_stride;
    int tail;
    bool enable_prefetch;

    opmask_t k3333 = k1;
    opmask_t k5555 = k2;
    opmask_t kAAAA = k3;
    opmask_t kCCCC = k4;
    opmask_t k0F0F = k5;
    opmask_t kF0F0 = k6;
    opmask_t kTail = k7;

    reg64_t reg_src = r8;
    reg64_t reg_tr_src = r9;
    reg64_t reg_src_prf = r10;
    reg64_t reg_tr_src_prf = r11;
    reg64_t reg_loop = r12;
    reg64_t reg_tr_src_tmp = r13;
    reg32_t regw_tmp = r14d;

    void transpose(int nrows, int l_pad, int r_pad, bool nontemporal_stores);
    void generate();

};

}
}
}

#endif
