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

#include "mkldnn_types.h"

#define XBYAK_NO_OP_NAMES
#include "utils_xbyak.hpp"
#include "jit_avx2_generator.hpp"

namespace mkldnn { namespace impl { namespace cpu {

typedef struct {
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
    int SIMD_W;
    mkldnn_memory_format_t src_fmt;
} jit_convolution_param_t;

typedef struct __attribute__ ((__packed__))
jit_convolution_kernel_s {
    float *src;
    float *dst;
    float *filt;
    float *src_prf;
    float *dst_prf;
    float *filt_prf;
    size_t kh_padding;
    size_t kh_padding_prf;
    size_t kw_padding;
}  jit_convolution_kernel_t;

class jit_avx2_conv_generator_f32 : public jit_avx2_generator
{
private:
    Xbyak::Reg64 reg_input      = rax;
    Xbyak::Reg64 aux_reg_input  = r8;
    Xbyak::Reg64 reg_kernel     = rdx;
    Xbyak::Reg64 aux_reg_kernel = r9;
    Xbyak::Reg64 reg_output     = rsi;

    Xbyak::Reg64 kj      = r10;
    Xbyak::Reg64 oi_iter = r11;
    Xbyak::Reg64 ki_iter = r12;
    Xbyak::Reg64 reg_kh  = rcx;

    inline void hsw_iter(jit_convolution_param_t *params, uint32_t ur_w,
                         int pad_l, int pad_r, const char* kh_lable, const char* kw_lable);

public:

    jit_avx2_conv_generator_f32(
        jit_convolution_param_t *params,
        void* code_ptr = nullptr,
        size_t code_size = 8 * Xbyak::DEFAULT_MAX_CODE_SIZE
        );
};

}}}

#endif
