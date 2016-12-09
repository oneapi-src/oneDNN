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

#ifndef JIT_AVX2_RELU_KERNEL_F32_HPP
#define JIT_AVX2_RELU_KERNEL_F32_HPP

#include "c_types_map.hpp"
#include "jit_generator.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct jit_relu_conf_t {
    size_t vector_length;
    size_t unroll_factor;
    size_t jit_n_runs;

    size_t n_elems;
    size_t block_size;
    size_t main_loop_iters;
    size_t unrolled_size;
    size_t remainder_size;
    size_t remainder_main_loop_iters;

    float negative_slope;
};

struct __attribute__((__packed__)) jit_relu_call_s {
    const float *src;
    const float *dst; /* hack, non-const for forward */
    size_t main_loop_iters;
    size_t process_remainder;
};

struct jit_avx2_relu_kernel_f32: public jit_generator {
    jit_avx2_relu_kernel_f32(jit_relu_conf_t ajrp, void *code_ptr = nullptr,
            size_t code_size = 8 * Xbyak::DEFAULT_MAX_CODE_SIZE)
        : jrp(ajrp)
    {
        this->generate();
        jit_ker = (void (*)(jit_relu_call_s *))this->getCode();
    }

    void operator()(jit_relu_call_s *arg) { jit_ker(arg); }
    static status_t init_conf(jit_relu_conf_t &jrp,
            const relu_desc_t &rd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &dst_d,
            float negative_slope = 0.f);

    jit_relu_conf_t jrp;
    void (*jit_ker)(jit_relu_call_s *);

private:
    using reg64_t = const Xbyak::Reg64;
    reg64_t reg_src = rax;
    reg64_t reg_dst = r8;
    reg64_t reg_main_loop_iterator = r9;
    reg64_t reg_remainder = rdx;
    reg64_t imm_addr64 = rbx;

    inline Xbyak::Address get_address(Xbyak::Reg64 base, int offset);
    inline void step(bool vectorize, size_t shift);

    void generate();
};

}
}
}

#endif
