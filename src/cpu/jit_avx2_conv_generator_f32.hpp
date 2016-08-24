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

#ifndef CPU_JIT_AVX2_CONV_GENERATOR_F32_HPP
#define CPU_JIT_AVX2_CONV_GENERATOR_F32_HPP

#include "c_types_map.hpp"
#include "jit_generator.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct jit_convolution_param_t {
    int ic, oc;
    uint32_t mb;
    uint32_t ih, iw, oh, ow;
    uint32_t ihp, iwp, ohp, owp;
    int l_pad, t_pad;
    uint32_t kh, kw;
    uint32_t stride_h, stride_w;
    uint32_t nb_ic, ic_block;
    uint32_t nb_oc, oc_block;
    uint32_t nb_ic_blocking, nb_oc_blocking; // blocking of nb_ic and nb_ic
    uint32_t ur_h, ur_w;
    uint32_t ur_w_tail;
    uint32_t ngroups;
    memory_format_t src_fmt;
    bool with_bias;
    bool with_relu;
    double relu_negative_slope;
};

typedef struct __attribute__((__packed__)) jit_convolution_kernel_s {
    const float *src; /* hack, non-const for backward_data */
    const float *dst; /* hack, non-const for forward */
    const float *filt; /* hack, non-const for backward_weights */
    const float *bias; /* hack, non-const for backward_bias */
    const float *src_prf;
    const float *dst_prf;
    const float *filt_prf;
    size_t kh_padding;
    size_t kh_padding_prf;
    size_t kw_padding;
    int32_t ic_flag;
} jit_convolution_kernel_t;

class jit_avx2_conv_generator_f32 : public jit_generator {
private:
    using reg64_t = const Xbyak::Reg64;
    reg64_t reg_input = rax;
    reg64_t aux_reg_input = r8;
    reg64_t reg_kernel = rdx;
    reg64_t aux_reg_kernel = r9;
    reg64_t reg_output = rsi;
    reg64_t reg_bias = rbx;

    reg64_t kj = r10;
    reg64_t oi_iter = r11;
    reg64_t ki_iter = r12;
    reg64_t reg_kh = rcx;
    Xbyak::Reg32 reg_ci_flag = r13d;

    const bool _src_in_nchw;

    inline void oh_step_unroll(jit_convolution_param_t *params, int ur_w,
            int pad_l, int pad_r);

    inline void oh_step_nopad(jit_convolution_param_t *params, int ur_w,
            int pad_l, int pad_r, char pad_label);

    inline void width_blk_step(jit_convolution_param_t *params, int ur_w,
            int pad_l, int pad_r, char pad_label);

    inline void init_jit_params(const convolution_desc_t &cd,
            const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d);
    inline void generate();
public:
    enum { IC_FLAG_FIRST = 1, IC_FLAG_LAST = 2 };
    jit_avx2_conv_generator_f32(const convolution_primitive_desc_t &cpd,
            void *code_ptr = nullptr,
            size_t code_size = 8 * Xbyak::DEFAULT_MAX_CODE_SIZE);
    jit_avx2_conv_generator_f32(
            const convolution_relu_primitive_desc_t &crpd,
            void *code_ptr = nullptr,
            size_t code_size = 8 * Xbyak::DEFAULT_MAX_CODE_SIZE);

    jit_convolution_param_t jcp;
    void (*jit_ker)(void *);

    static bool is_applicable(const convolution_desc_t &conv_d);
    static bool is_applicable(const convolution_relu_desc_t &conv_relu_d);
};
}
}
}

#endif
