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

#ifndef CPU_JIT_AVX2_POOL_KERNEL_F32_HPP
#define CPU_JIT_AVX2_POOL_KERNEL_F32_HPP

#include <cfloat>

#include "c_types_map.hpp"
#include "jit_generator.hpp"
#include "type_helpers.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct jit_pool_conf_t {
    int mb, c;
    int ih, iw, oh, ow;
    int stride_h, stride_w;
    int kh, kw;
    int t_pad, l_pad;
    bool is_max;
    bool is_training;

    int nb_c, c_block;
    int ur_h, ur_w;
    int ur_w_tail;
};

struct __attribute__ ((__packed__)) jit_pool_call_s {
    const float *src;
    const float *dst;
    const int *indices;
    const float *src_prf;
    const float *dst_prf;
    const int *indices_prf;
    size_t kh_padding;
    size_t kh_padding_prf;
    size_t kw_padding;
    const float* init_value;
    int* init_array;
};

struct jit_avx2_pool_kernel_f32: public jit_generator {
    jit_avx2_pool_kernel_f32(jit_pool_conf_t ajpp): jpp(ajpp)
    {
        this->generate();
        jit_ker = (decltype(jit_ker))this->getCode();
    }

    jit_pool_conf_t jpp;
    void operator()(jit_pool_call_s *arg) { jit_ker(arg); }
    static status_t init_conf(jit_pool_conf_t &jbp,
            const pooling_desc_t &pd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &dst_d, bool is_training);

private:
    using reg64_t = const Xbyak::Reg64;
    reg64_t reg_input      = r8;
    reg64_t aux_reg_input  = r9;
    reg64_t reg_index      = r10;
    reg64_t aux_reg_index  = r11;
    reg64_t reg_output     = r12;
    reg64_t reg_arr_init   = r13;

    reg64_t kj      = r14;
    reg64_t oi_iter = r15;
    reg64_t reg_kh  = rax;
    reg64_t tmp_gpr = rcx;
    reg64_t tmp_gpr2 = rdx;

    void (*jit_ker)(jit_pool_call_s *);

    void avg_oh_step(int ur_w, int pad_l, int pad_r, const char *kh_label);
    void max_oh_step(int ur_w, int pad_l, int pad_r, const char *kh_label);

    void oh_step(int ur_w, int pad_l, int pad_r, const char *kh_label) {
        if (jpp.is_max)
            max_oh_step(ur_w, pad_l, pad_r, kh_label);
        else
            avg_oh_step(ur_w, pad_l, pad_r, kh_label);
    }

    void generate();
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
