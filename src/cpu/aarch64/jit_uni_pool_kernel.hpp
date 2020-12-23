/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
* Copyright 2018 YANDEX LLC
* Copyright 2020 FUJITSU LIMITED
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

#ifndef CPU_AARCH64_JIT_UNI_POOL_KERNEL_HPP
#define CPU_AARCH64_JIT_UNI_POOL_KERNEL_HPP

#include <cfloat>
#include <functional>
#include <memory>

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/aarch64/jit_generator.hpp"

#include "cpu/aarch64/jit_primitive_conf.hpp"

using namespace Xbyak_aarch64;

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

template <cpu_isa_t isa>
struct jit_uni_pool_kernel : public jit_generator {

    jit_uni_pool_kernel(
            const jit_pool_conf_t &ajpp, const memory_desc_t *dst_md);
    jit_pool_conf_t jpp;
    ~jit_uni_pool_kernel();

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_pool_kernel)

    static status_t init_conf(jit_pool_conf_t &jbp,
            memory_tracking::registrar_t &scratchpad, const pooling_pd_t *ppd,
            int nthreads);

private:
    using TReg = typename cpu_isa_traits<isa>::TReg;
    using TRegS = typename cpu_isa_traits<isa>::TRegS;

    int vmm_idx_upper_bound() const noexcept { return 31; }

    int reg_idx(int idx) const noexcept { return vmm_idx_upper_bound() - idx; }

    VReg xreg(int idx) const noexcept { return VReg(reg_idx(idx)); }
    ZReg yreg(int idx) const noexcept { return ZReg(reg_idx(idx)); }
    ZReg zreg(int idx) const noexcept { return ZReg(reg_idx(idx)); }
    TReg vreg(int idx) const noexcept { return TReg(reg_idx(idx)); }

    VReg vmm_mask = VReg(0);
    ZReg ymm_tmp_1 = ZReg(0);
    TRegS vmm_tmp_1 = TRegS(0);

    // Used only for avx and if c tail is present
    TReg vmm_c_tail_mask = TReg(2);

    VReg xmm_ker_area_h = VReg(2);
    VReg xmm_one = VReg(2);
    VReg xmm_tmp = VReg(3);

    TRegS vmm_ker_area_h = TRegS(2);
    TRegS vmm_one = TRegS(2);
    TReg vmm_tmp = TReg(3);
    ZReg ymm_tmp = ZReg(3);

    TRegS vmm_k_offset = TRegS(1);

    inline uint32_t reg_idx() {
        if (!jpp.is_backward) {
            return (jpp.is_training) ? 4 : 1;
        } else
            return 4;
    }

    const std::vector<uint32_t> tmp_vec_idx = {4, 5, 6, 7};
    ZReg z_tmp0 = z4;
    ZReg z_tmp1 = z5;
    ZReg z_tmp2 = z6;
    ZReg z_tmp3 = z7;

    PReg k_c_tail_mask = p4;
    PReg k_mask_cvt = p5;
    PReg k_store_mask = p6;

    /* Caution: Chose predicate registers not used by x64's implementation. */
    PReg p_256 = p1;
    PReg p_512 = p2;
    PReg p_tmp0 = p3;
    PReg p_128 = p7;
    PReg p_lsb = p2;
    PReg p_tmp1 = p11;
    PReg p_tmp2 = p12;
    PReg P_MSB_256 = p13;
    PReg P_MSB_384 = p14;
    PReg P_ALL_ONE = p15;

    // Here be some (tame) dragons. This kernel does not follow the regular
    // OS-agnostic ABI pattern because when isa is sse41 it uses maskmovdqu
    // instruction which has its destination hardcoded in rdi. Therefore:
    // - all registers are hardcoded
    // - on Windows rdi and rcx are swapped to mimic the Unix x86_64 ABI
    //
    // While this is only required by the backward pass, the quirk above
    // is applied to the forward pass as well to keep things simpler.
    using xreg_t = const XReg;
    xreg_t reg_param = x0; // Always mimic the Unix ABI
    xreg_t reg_input = x4;
    xreg_t aux_reg_input = x5;
    xreg_t reg_index = x10;
    xreg_t reg_output = x12;
    xreg_t reg_kd_pad_shift = x13;
    xreg_t dst_ptr = x0; // Must be rdi due to maskmovdqu

    xreg_t kj = x14;
    xreg_t oi_iter = x15;
    xreg_t reg_kh = x7;
    xreg_t reg_k_shift = x3;
    xreg_t tmp_gpr = x6;
    ; // Must be rcx because rdi is used above
    xreg_t reg_ker_area_h = x2;
    xreg_t reg_nbc = x1;

    xreg_t reg_zero_ptr = x5;
    xreg_t reg_zero_id = x13;
    xreg_t reg_zero_ih = x14;
    xreg_t aux_reg_zero_ih = x15;
    xreg_t ki = x12;
    xreg_t aux_reg_input_d = x4;

    using wreg_t = const WReg;
    wreg_t w_tmp_0 = w23;
    wreg_t W_TMP_0 = w23;

    xreg_t aux_xreg_input = x5;
    xreg_t xreg_output = x12;
    xreg_t xreg_index = x10;
    xreg_t xreg_zero_ptr = x5;
    xreg_t x_tmp_addr = x28;
    xreg_t x_tmp_0 = x23;
    xreg_t X_TMP_0 = x23;
    xreg_t X_TRANSLATOR_STACK = x22;
    xreg_t reg_adrimm = x24;

    WReg reg_shuf_mask = w1;

    bool sse_high_half = false;
    bool disable_postops_when_sse_high_half_processed_ = false;

    int prev_kw;

    void prepare_tail_mask();
    void put_one_in_vmm();
    void uni_broadcast_reg_val(const int reg_idx, const int vmm_idx);
    void push_vmm_val(const int idx);
    void pop_vmm_val(const int idx);
    void load(const int idx, const xreg_t &reg_ptr, const int offset,
            const bool is_c_tail_proccessing);
    void store(const int idx, const xreg_t &reg_ptr, const int offset,
            const bool is_c_tail_proccessing);

    void maybe_recalculate_divisor(int jj, int ur_w, int pad_l, int pad_r,
            bool with_c_tail_proccessing);
    void avg_step(int ur_w, int ur_bc, int pad_l, int pad_r,
            bool with_c_tail_proccessing);
    void max_step_fwd(int ur_w, int ur_bc, int pad_l, int pad_r,
            bool with_c_tail_proccessing);
    void max_step_bwd(int ur_w, int ur_bc, int pad_l, int pad_r,
            bool with_c_tail_proccessing);

    void zero_diff_src(int ur_bc, bool with_c_tail_proccessing);

    void step(int ur_w, int ur_bc, int pad_l, int pad_r,
            bool with_c_tail_proccessing) {
        if (jpp.alg == alg_kind::pooling_max) {
            if (jpp.is_backward)
                max_step_bwd(
                        ur_w, ur_bc, pad_l, pad_r, with_c_tail_proccessing);
            else
                max_step_fwd(
                        ur_w, ur_bc, pad_l, pad_r, with_c_tail_proccessing);
        } else
            avg_step(ur_w, ur_bc, pad_l, pad_r, with_c_tail_proccessing);
    }

    void generate() override;

    static bool post_ops_ok(jit_pool_conf_t &jpp, const primitive_attr_t &attr,
            const memory_desc_wrapper &dst_d);
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
