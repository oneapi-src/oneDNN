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
#ifndef CPU_JIT_AVX2_POOLING_GENERATOR_F32_HPP
#define CPU_JIT_AVX2_POOLING_GENERATOR_F32_HPP

#include <cfloat>

#include "c_types_map.hpp"
#include "jit_generator.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

typedef struct {
    int c;
    uint32_t mb;
    uint32_t ih, iw, oh, ow;
    int l_pad, t_pad;
    uint32_t kh, kw;
    uint32_t stride_h, stride_w;
    uint32_t nb_c, c_block;
    uint32_t ur_h, ur_w;
    uint32_t ur_w_tail;
} jit_pooling_param_t;

typedef struct __attribute__ ((__packed__))
jit_pooling_kernel_s {
    float *src;
    float *dst;
    uint32_t *indices;
    float *src_prf;
    float *dst_prf;
    uint32_t *indices_prf;
    size_t kh_padding;
    size_t kh_padding_prf;
    size_t kw_padding;
    float* init_value;
    uint32_t* init_array;
}  jit_pooling_kernel_t;

class jit_avx2_pooling_generator_f32 : public jit_generator {
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

    const bool _is_training;

    inline void oh_step(jit_pooling_param_t *params, int ur_w,
                         int pad_l, int pad_r, const char* kh_lable);
public:
    jit_avx2_pooling_generator_f32(
        jit_pooling_param_t *params,
        bool is_training,
        void* code_ptr = nullptr,
        size_t code_size = 8 * Xbyak::DEFAULT_MAX_CODE_SIZE);
};
}
}
}

#endif
