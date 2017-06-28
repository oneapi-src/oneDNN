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

#ifndef JIT_AVX512_COMMON_1x1_CONV_KERNEL_HPP
#define JIT_AVX512_COMMON_1x1_CONV_KERNEL_HPP

#include "c_types_map.hpp"
#include "jit_generator.hpp"
#include "jit_primitive_conf.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct jit_avx512_common_1x1_conv_kernel : public jit_generator {
    enum { REDUCE_FLAG_FIRST = 1, REDUCE_FLAG_LAST = 2 };

    jit_avx512_common_1x1_conv_kernel(jit_1x1_conv_conf_t ajcp) : jcp(ajcp)
    {
        this->generate();
        jit_ker = (void (*)(jit_1x1_conv_call_s *)) this->getCode();
    }

    static status_t init_conf(jit_1x1_conv_conf_t &jcp,
                                const convolution_desc_t &cd,
                                const memory_desc_wrapper &src_d,
                                const memory_desc_wrapper &weights_d,
                                const memory_desc_wrapper &dst_d,
                                bool with_relu, double relu_negative_slope,
                                int nthreads, bool reduce_src);

    static status_t init_conf(jit_1x1_conv_conf_t &jcp,
                              const convolution_desc_t &cd,
                              const memory_desc_wrapper &src_d,
                              const memory_desc_wrapper &weights_d,
                              const memory_desc_wrapper &dst_d,
                              int nthreads, bool reduce_src)
    {
        return init_conf(jcp, cd, src_d, weights_d, dst_d, false, 0.0,
            nthreads, reduce_src);
    }

    jit_1x1_conv_conf_t jcp;
    void (*jit_ker)(jit_1x1_conv_call_s *);

  private:
    using reg64_t = const Xbyak::Reg64;
    using zmm_t = const Xbyak::Zmm;
    using mask_t = const Xbyak::Opmask;

    reg64_t reg_bcast_data = r8;
    reg64_t reg_load_data = r10;
    reg64_t reg_output_data = r9;
    reg64_t aux_reg_bcast_data = r14;
    reg64_t aux1_reg_bcast_data = rbx;
    reg64_t aux_reg_load_data = r15;
    reg64_t aux_reg_output_data = rcx;
    reg64_t reg_load_loop_work = rsi;
    reg64_t reg_reduce_loop_work = r11;
    reg64_t bcast_loop_iter = rdx;
    reg64_t reduce_loop_iter = rdi;
    reg64_t reg_reduce_pos_flag = rax;
    reg64_t reg_output_stride = r12;
    reg64_t reg_bias_data = r12;
    reg64_t reg_relu_ns = r13;
    reg64_t reg_bcast_loop_work = aux1_reg_bcast_data;
    reg64_t reg_diff_bias_data = bcast_loop_iter;
    mask_t vmask = k7;

    Xbyak::Zmm zmm_relu_ns = Xbyak::Zmm(30);
    Xbyak::Zmm zmm_zero = Xbyak::Zmm(31);

    int reg_diff_bias_data_stack_offt = 0;
    int bcast_loop_work_offt = 16;
    int stack_space_needed = 32;

    void bcast_loop(int load_loop_blk);
    void reduce_loop(int load_loop_blk, int ur, int substep, bool wraparound);

    void generate();
};
}
}
}

#endif
