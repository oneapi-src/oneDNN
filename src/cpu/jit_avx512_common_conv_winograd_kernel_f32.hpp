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

#ifndef JIT_AVX512_COMMON_CONV_WINOGRAD_KERNEL_F32_HPP
#define JIT_AVX512_COMMON_CONV_WINOGRAD_KERNEL_F32_HPP

#include "c_types_map.hpp"
#include "jit_generator.hpp"
#include "jit_primitive_conf.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct _jit_avx512_common_conv_winograd_data_kernel_f32 : public jit_generator
{
    _jit_avx512_common_conv_winograd_data_kernel_f32(
            jit_conv_winograd_conf_t ajcp) : jcp(ajcp) {
        this->gemm_loop_generate();
        gemm_loop_ker =
            (void (*)(float *, const float *, const float *)) this->getCode();
    }

    static status_t init_conf_common(jit_conv_winograd_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d);

    jit_conv_winograd_conf_t jcp;
    void (*gemm_loop_ker)(float *, const float *, const float *);
    void (*input_transform_ker)(const float *, const float *);

  private:
    using reg64_t = const Xbyak::Reg64;
    enum { typesize = sizeof(float) };

    void gemm_compute_kernel(bool first_iter, bool last_iter);
    void gemm_loop_generate();

    /* registers used for GEMM */
    reg64_t reg_dstC_const = abi_param1;
    reg64_t reg_srcA = abi_param2;
    reg64_t reg_srcB = abi_param3;

    reg64_t reg_dstC = r9;
    reg64_t reg_nb_Xc = r10;
    reg64_t reg_loop_cpt = r11;
};

struct jit_avx512_common_conv_winograd_fwd_kernel_f32
    : _jit_avx512_common_conv_winograd_data_kernel_f32 {
    using _jit_avx512_common_conv_winograd_data_kernel_f32::
        _jit_avx512_common_conv_winograd_data_kernel_f32;

    static status_t init_conf(jit_conv_winograd_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d, bool with_relu = false,
            double relu_negative_slope = 0.);
};

struct jit_avx512_common_conv_winograd_bwd_data_kernel_f32
    : public _jit_avx512_common_conv_winograd_data_kernel_f32 {
    using _jit_avx512_common_conv_winograd_data_kernel_f32::
        _jit_avx512_common_conv_winograd_data_kernel_f32;

    static status_t init_conf(jit_conv_winograd_conf_t &jcp,
            const convolution_desc_t &cd,
            const memory_desc_wrapper &diff_src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &diff_dst_d);
};

struct jit_avx512_common_conv_winograd_bwd_weights_kernel_f32
    : public jit_generator {
    jit_avx512_common_conv_winograd_bwd_weights_kernel_f32(
        jit_conv_winograd_conf_t ajcp) : jcp(ajcp) {
        this->gemm_loop_generate(true);
        gemm_loop_ker_first_img =
            (void (*)(float *, const float *, const float *)) this->getCode();
        this->align();
        gemm_loop_ker =
            (void (*)(float *, const float *, const float *)) this->getCurr();
        this->gemm_loop_generate(false);
    }

    static status_t init_conf(jit_conv_winograd_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &diff_dst_d,
            const memory_desc_wrapper &diff_weights_d);

    jit_conv_winograd_conf_t jcp;
    void (*gemm_loop_ker)(float *, const float *, const float *);
    void (*gemm_loop_ker_first_img)(float *, const float *, const float *);

  private:
    using reg64_t = const Xbyak::Reg64;
    enum { typesize = sizeof(float) };

    void gemm_loop_generate(bool first_img);
    void gemm_compute_kernel();

    reg64_t reg_dstC = abi_param1;
    reg64_t reg_srcA_const = abi_param2;
    reg64_t reg_srcB = abi_param3;

    reg64_t reg_sp = rsp;
    reg64_t reg_srcA = r9;
    reg64_t reg_nb_ic = r10;
    reg64_t reg_loop_cpt = r11;
};

}
}
}

#endif
