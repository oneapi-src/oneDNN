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

#ifndef JIT_AVX2_1x1_CONV_KERNEL_F32_HPP
#define JIT_AVX2_1x1_CONV_KERNEL_F32_HPP

#include "c_types_map.hpp"
#include "jit_generator.hpp"
#include "jit_primitive_conf.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct jit_avx2_1x1_conv_kernel_f32: public jit_generator {
    jit_avx2_1x1_conv_kernel_f32(jit_1x1_conv_conf_t ajcp): jcp(ajcp)
    {
        this->generate();
        jit_ker = (void (*)(jit_1x1_conv_call_s *))this->getCode();
    }

    static status_t init_conf(jit_1x1_conv_conf_t &jcp,
            const convolution_desc_t &cd,
            const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d,
            bool with_relu, double relu_negative_slope);

    static status_t init_conf(jit_1x1_conv_conf_t &jcp,
            const convolution_desc_t &cd,
            const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d)
    {
        return init_conf(jcp, cd, src_d, weights_d, dst_d, false, 0.0);
    }

    jit_1x1_conv_conf_t jcp;
    void (*jit_ker)(jit_1x1_conv_call_s *);

private:
    using reg64_t = const Xbyak::Reg64;
    using ymm_t = const Xbyak::Ymm;

    reg64_t reg_bcast_data = rax;
    reg64_t reg_load_data = rsi;
    reg64_t reg_output_data = rbx;
    reg64_t aux_reg_bcast_data = rdx;
    reg64_t aux1_reg_bcast_data = rcx;
    reg64_t aux_reg_load_data = rdi;
    reg64_t aux_reg_output_data = rbp;
    reg64_t reg_load_loop_work = r9;
    reg64_t reg_bcast_loop_work = r10;
    reg64_t reg_reduce_loop_work = r11;
    reg64_t load_loop_iter = r13;
    reg64_t bcast_loop_iter = r14;
    reg64_t reduce_loop_iter = r15;
    reg64_t imm_addr64 = reduce_loop_iter;
    reg64_t reg_reduce_pos_flag = r8;
    reg64_t reg_output_stride = r12;
    reg64_t reg_bias_data = r12;
    reg64_t reg_diff_bias_data = bcast_loop_iter;

    int reg_diff_bias_data_stack_offt = 0;
    int stack_space_needed = 8;

    ymm_t vreg_bcast = ymm_t(15);
    Xbyak::Xmm xmm_relu_ns = Xbyak::Xmm(13);
    Xbyak::Ymm ymm_relu_ns = Xbyak::Ymm(13);
    Xbyak::Ymm ymm_res_ns = Xbyak::Ymm(12);
    Xbyak::Ymm vzero = Xbyak::Ymm(15);
    Xbyak::Ymm vmask = Xbyak::Ymm(14);

    void bcast_loop(int load_loop_blk, char load_loop_tag);
    void reduce_loop(int load_loop_blk, int ur, char load_loop_tag,
            char bcast_loop_tag);
    void diff_bias_loop(int load_loop_blk, char load_loop_tag);

    void generate();
};

}
}
}

#endif
