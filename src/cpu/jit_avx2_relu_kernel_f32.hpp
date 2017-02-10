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

#ifndef JIT_AVX2_RELU_KERNEL_F32_HPP
#define JIT_AVX2_RELU_KERNEL_F32_HPP

#include "c_types_map.hpp"
#include "jit_generator.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct jit_relu_conf_t {
    bool isBackward;

    int vector_length;
    int unroll_factor;
    int jit_n_runs;

    int n_elems;
    int block_size;
    int main_loop_iters;
    int unrolled_size;
    int remainder_size;
    int remainder_main_loop_iters;

    float negative_slope;
};

struct __attribute__((__packed__)) jit_relu_call_s {
    const float *for_comparison;
    const float *from;
    const float *to;
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
            const relu_desc_t &rd, const memory_desc_wrapper &data_d,
            bool isBackward);

    jit_relu_conf_t jrp;
    void (*jit_ker)(jit_relu_call_s *);

private:
    using reg64_t = Xbyak::Reg64;
    using reg_ymm = Xbyak::Ymm;
    using reg_xmm = Xbyak::Xmm;

    reg64_t reg_for_comparison;
    reg64_t reg_from;
    reg64_t reg_to;
    reg64_t reg_main_loop_iterator;
    reg64_t reg_remainder;
    reg64_t imm_addr64;

    reg_ymm ymm_ns;
    reg_ymm ymm_zero;
    reg_ymm ymm_mask;

    reg_ymm ymm_for_comparison;
    reg_xmm xmm_for_comparison;
    reg_ymm ymm_from;
    reg_xmm xmm_from;
    reg_ymm ymm_to;
    reg_xmm xmm_to;


    inline void setupRegisters();
    inline void step(bool vectorize, int shift);

    void generate();
};

}
}
}

#endif
