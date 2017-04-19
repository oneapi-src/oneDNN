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

#ifndef CPU_JIT_UNI_POOL_KERNEL_F32_HPP
#define CPU_JIT_UNI_POOL_KERNEL_F32_HPP

#include <cfloat>

#include "c_types_map.hpp"
#include "jit_generator.hpp"
#include "type_helpers.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace Xbyak;

struct jit_pool_conf_t {
    int mb, c;
    int ih, iw, oh, ow;
    int stride_h, stride_w;
    int kh, kw;
    int t_pad, l_pad;
    alg_kind_t alg;
    bool is_training;
    bool pad_w_is_null;
    bool is_backward;

    int nb_c, c_block;
    int ur_w;
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
    size_t kh_padding_shift;
    size_t kw_padding;
    const float* init_value;
    float ker_area_h;
};

template <cpu_isa_t isa>
struct jit_uni_pool_kernel_f32: public jit_generator {
    jit_uni_pool_kernel_f32(jit_pool_conf_t ajpp): jpp(ajpp)
    {
        this->generate();
        jit_ker = (decltype(jit_ker))this->getCode();
    }

    jit_pool_conf_t jpp;

    void operator()(jit_pool_call_s *arg) { jit_ker(arg); }
    static status_t init_conf(jit_pool_conf_t &jbp,
            const pooling_desc_t &pd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &dst_d);

private:

    using Vmm = typename utils::conditional<isa == avx2, Ymm, Zmm>::type;
    const AddressFrame &vmmword = (isa == avx2) ? yword : zword;
    void uni_vpxor(const Xmm& x1, const Xmm& x2, const Operand& op)
        { if (isa == avx2) vpxor(x1, x2, op); else vpxord(x1, x2, op); }
    void uni_vmovdqu(const Address& addr, const Xmm& x)
        { if (isa == avx2) vmovdqu(addr, x); else vmovdqu32(addr, x); }
    void uni_vmovdqu(const Xmm& x, const Address& addr)
        { if (isa == avx2) vmovdqu(x, addr); else vmovdqu32(x, addr); }
    Vmm vreg(int idx) { return Vmm((isa == avx2 ? 15 : 31) - idx); }

    Xmm xmm_ker_area_h = Xmm(1);
    Xmm xmm_one = Xmm(1);
    Xmm xmm_tmp = Xmm(2);

    Vmm vmm_ker_area_h = Vmm(1);
    Vmm vmm_one = Vmm(1);
    Vmm vmm_tmp = Vmm(2);

    Vmm vmm_k_offset = Vmm(0);

    Opmask k_store_mask = Opmask(7);

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
    reg64_t reg_k_shift  = rbx;
    reg64_t tmp_gpr = rcx;
    reg64_t reg_ker_area_h = rdx;

    int prev_kw;
    void (*jit_ker)(jit_pool_call_s *);

    void maybe_recalculate_divisor(int jj, int ur_w, int pad_l, int pad_r);
    void avg_step(int ur_w, int pad_l, int pad_r, const char *kh_label);
    void max_step_fwd(int ur_w, int pad_l, int pad_r, const char *kh_label);
    void max_step_bwd(int ur_w, int pad_l, int pad_r, const char *kh_label);

    void step(int ur_w, int pad_l, int pad_r, const char *kh_label) {
        if (jpp.alg == alg_kind::pooling_max) {
            if(jpp.is_backward)
                max_step_bwd(ur_w, pad_l, pad_r, kh_label);
            else
                max_step_fwd(ur_w, pad_l, pad_r, kh_label);
        }
        else
            avg_step(ur_w, pad_l, pad_r, kh_label);
    }

    void generate();
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
