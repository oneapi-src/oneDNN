/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef CPU_X64_JIT_BRGEMM_CONV_COMP_PAD_KERNEL_HPP
#define CPU_X64_JIT_BRGEMM_CONV_COMP_PAD_KERNEL_HPP

#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace jit_avx512_core_brgemm_conv_comp_pad_kernel {
struct jit_brgemm_conv_comp_pad_call_s {
    const void *ptr_in;
    void *ptr_zp_out;
    void *ptr_cp_out;
    size_t kd_l;
    size_t kh_l;
    size_t ow_l;
};

struct jit_avx512_core_brgemm_conv_comp_pad_kernel_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_brgemm_conv_comp_pad_kernel_t)

    using reg64_t = const Xbyak::Reg64;

    jit_avx512_core_brgemm_conv_comp_pad_kernel_t(
            const jit_brgemm_conv_conf_t &ajcp);

    ~jit_avx512_core_brgemm_conv_comp_pad_kernel_t() = default;

protected:
    jit_brgemm_conv_conf_t jcp_;
    const int inp_dsz_;
    const int out_dsz_;
    const size_t nb_ic_;
    const size_t inp_ic_sz_;
    const size_t inp_kh_sz_;
    const size_t inp_kd_sz_;
    const size_t inp_ocb_sz_;
    const size_t out_ow_sz_;

    // Register decomposition
    const reg64_t param1 = abi_param1;
    const reg64_t reg_in = r15;
    const reg64_t reg_comp_out = r14;
    const reg64_t reg_zp_comp_out = r13;

    const reg64_t reg_mb_count = r12;
    const reg64_t reg_kd_l = r11;
    const reg64_t reg_kh_l = r10;
    const reg64_t reg_ow_l = r9;
    const reg64_t reg_icb = r8;

    const reg64_t reg_aux_in = rsi;
    const reg64_t reg_aux_kd_in = rbx;
    const reg64_t reg_tmp = rax;

    const reg64_t reg_aux_comp_out = rsi;
    const reg64_t reg_aux_zp_comp_out = rdx;

    Xbyak::Zmm zmm_one_bytes = Xbyak::Zmm(30);
    Xbyak::Zmm zmm_zp_shift = Xbyak::Zmm(28);
    Xbyak::Zmm zmm_cp_shift = Xbyak::Zmm(29);

    const int last_ic_block_ = 4;
    const int n_block2_ = 4;
    const int m_block2_ = 16;
    const int max_regs_ = 28;
    const int n_max_regs_ = 4;

    const Xbyak::Zmm &zmm_tmp_1() const noexcept { return this->zmm31; }

    Xbyak::Zmm accum(const int n_block, const int m, const int n) const;
    size_t out_oc_offset(const int m, const int n) const;
    size_t inp_ic_offset(const int icb, const int m, const int n) const;

    void store_accumulators(const int m_block, const int n_block);
    void zero_accumulators(const int m_block, const int n_block);
    void store(const int m_block, const int n_block);
    void compute(const int m_block, const int n_block);
    void kdh_loop(const int m_block, const int n_block);
    void load_params();
    void generate() override;
};

} // namespace jit_avx512_core_brgemm_conv_comp_pad_kernel

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
