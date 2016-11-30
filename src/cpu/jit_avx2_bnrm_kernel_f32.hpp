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

#ifndef CPU_JIT_AVX2_BNRM_KERNEL_F32_HPP
#define CPU_JIT_AVX2_BNRM_KERNEL_F32_HPP

#include "c_types_map.hpp"
#include "jit_generator.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct jit_bnrm_conf_t {
    int mb, c, h, w;
    float eps;
    bool is_training;
    bool stats_is_src;
    bool use_scaleshift;

    int c_block;
    int nb_c;
    int wh_block;
    int wh_block_tail;
};

struct __attribute__((__packed__)) jit_bnrm_call_s {
    const float *src, *dst;
    const float *mean;
    const float *variance;
    const float *scaleshift;
};

struct jit_avx2_bnrm_kernel_f32: public jit_generator {
    jit_avx2_bnrm_kernel_f32(const jit_bnrm_conf_t &ajbp): jbp(ajbp)
    {
        this->generate();
        jit_ker = (decltype(jit_ker))this->getCode();
    }

    jit_bnrm_conf_t jbp;
    void operator()(jit_bnrm_call_s *arg) { jit_ker(arg); }
    static status_t init_conf(jit_bnrm_conf_t &jbp,
            const batch_normalization_desc_t &bnd,
            const memory_desc_wrapper &data_d,
            const memory_desc_wrapper &scaleshift_d,
            bool is_training, bool stats_is_src, bool use_scaleshift);

private:
    // TODO: move to jit_kernel.h and use Xbyak names (Roma)
    using reg64_t = const Xbyak::Reg64;
    using reg_ymm = const Xbyak::Ymm;
    using reg_xmm = const Xbyak::Xmm;

    reg64_t reg_src = rax;
    reg64_t reg_scaleshift = rdx;
    reg64_t reg_dst = rsi;
    reg64_t reg_mean = rbx;
    reg64_t reg_variance = rcx;

    reg64_t aux_ptr = r8;
    reg64_t save_ptr = r9;
    reg64_t aux_dst_ptr = r10;
    reg64_t save_dst_ptr = r11;
    reg64_t n_iter = r12;
    reg64_t sp_iter = r13;
    reg64_t tmp_gpr = r14;

    reg_ymm ymm_mean = reg_ymm(15);
    reg_ymm ymm_variance = reg_ymm(14);
    reg_ymm ymm_mean_mul_variance = reg_ymm(13);
    reg_ymm ymm_epsilon = reg_ymm(12);
    reg_xmm xmm_epsilon = reg_xmm(12);
    reg_ymm ymm_scale = reg_ymm(11);
    reg_ymm ymm_shift = reg_ymm(10);
    reg_ymm ymm_spatial_n = reg_ymm(9);
    reg_xmm xmm_spatial_n = reg_xmm(9);
    reg_ymm ymm_one = reg_ymm(8);
    reg_xmm xmm_one = reg_xmm(8);

    void (*jit_ker)(jit_bnrm_call_s *);

    inline void mean_compute(int block_size);
    inline void variance_compute(int block_size);
    inline void dst_compute(int block_size);

    void generate();
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
