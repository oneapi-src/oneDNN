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

#ifndef CPU_JIT_AVX2_BATCH_NORM_GENERATOR_F32_HPP
#define CPU_JIT_AVX2_BATCH_NORM_GENERATOR_F32_HPP

#include "c_types_map.hpp"
#include "jit_generator.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct jit_batch_normalization_param_t {
    int c;
    int mb;
    int h, w;
    int c_block;
    int nb_c;
    int wh_block;
    int wh_block_tail;
    bool _is_training;
    float eps;
};

typedef struct __attribute__((__packed__)) jit_batch_normalization_kernel_s {
    const float *src;
    const float *dst;
    const float *scaleshift;
    const float *workspace;
} jit_batch_normalization_kernel_t;

class jit_avx2_batch_norm_generator_f32 : public jit_generator {
private:
    using reg64_t = const Xbyak::Reg64;
    using reg_ymm = const Xbyak::Ymm;
    using reg_xmm = const Xbyak::Xmm;

    reg64_t reg_src= rax;
    reg64_t reg_scaleshift = rdx;
    reg64_t reg_dst = rsi;
    reg64_t reg_workspace = rbx;

    reg64_t aux_ptr = r8;
    reg64_t save_ptr = r9;
    reg64_t aux_dst_ptr = r10;
    reg64_t save_dst_ptr = r11;
    reg64_t n_iter = r12;
    reg64_t sp_iter = r13;
    reg64_t tmp_gpr = r14;

    reg_ymm ymm_mean = Ymm(15);
    reg_ymm ymm_variance = Ymm(14);
    reg_ymm ymm_mean_mul_variance = Ymm(13);
    reg_ymm ymm_epsilon = Ymm(12);
    reg_xmm xmm_epsilon = Xmm(12);
    reg_ymm ymm_scale = Ymm(11);
    reg_ymm ymm_shift = Ymm(10);
    reg_ymm ymm_spatial_n = Ymm(9);
    reg_xmm xmm_spatial_n = Xmm(9);
    reg_ymm ymm_one = Ymm(8);
    reg_xmm xmm_one = Xmm(8);

    inline void init_jit_params(const batch_normalization_desc_t &bnd,
            const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &scaleshift_d,
            const memory_desc_wrapper &dst_d,
            const bool _is_training);
    inline void mean_compute(int block_size,
        jit_batch_normalization_param_t *params);
    inline void variance_compute(int block_size,
        jit_batch_normalization_param_t *params);
    inline void dst_compute(int block_size,
        jit_batch_normalization_param_t *params);

    inline void generate();
public:
    jit_avx2_batch_norm_generator_f32(
        const batch_normalization_primitive_desc_t &bnpd,
        const bool _is_training,
        void *code_ptr = nullptr,
        size_t code_size = 8 * Xbyak::DEFAULT_MAX_CODE_SIZE);

    jit_batch_normalization_param_t jbnp;
    void (*jit_ker)(void *);

    static bool is_applicable(const batch_normalization_desc_t &bnorm_d);
};

}
}
}

#endif
