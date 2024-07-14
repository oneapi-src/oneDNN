/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#ifndef CPU_X64_JIT_BRGEMM_POST_OPS_HPP
#define CPU_X64_JIT_BRGEMM_POST_OPS_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/cpu_engine.hpp"

#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_avx512_core_fp8cvt.hpp"
#include "cpu/x64/jit_brgemm_primitive_conf.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct brgemm_kernel_diff_bias_t {
    brgemm_kernel_diff_bias_t()
        : ptr_diff_dst(nullptr)
        , ptr_diff_bias_acc(nullptr)
        , ptr_diff_bias(nullptr)
        , flags(0) {};

    void *ptr_diff_dst;
    void *ptr_diff_bias_acc;
    void *ptr_diff_bias;
    int flags;
};

template <typename Vmm>
struct jit_brgemm_kernel_diff_bias_t : public jit_generator {
    jit_brgemm_kernel_diff_bias_t(const jit_brgemm_primitive_conf_t &ajbgp,
            const brgemm_desc_t &abrg);

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_kernel_diff_bias_t)

private:
    brgemm_desc_t brg_;
    data_type_t ddst_dt_;
    data_type_t bia_dt_;
    data_type_t acc_dt_;

    int ddst_typesize_;
    int bia_typesize_;
    int acc_typesize_;
    int mult_;

    using Vmm_lower_t = typename vreg_traits<Vmm>::Vmm_lower_t;
    using reg64_t = const Xbyak::Reg64;
    // Register decomposition
    const reg64_t param1 = abi_param1;
    const reg64_t reg_ddst = r15;
    const reg64_t reg_bias = r14;
    const reg64_t reg_bias_acc = r13;
    const reg64_t aux_reg_ddst = r12;
    const reg64_t reg_k_iter = r11;
    const reg64_t reg_flag = r10;
    const reg64_t reg_mask = rax;

    Xbyak::Label f16_perm_table_;
    Xbyak::Label mask_label_;
    Xbyak::Opmask k_full_mask = Xbyak::Opmask(2);
    Xbyak::Opmask k_tail_mask = Xbyak::Opmask(3);
    Xbyak::Opmask k_f16_perm_mask = Xbyak::Opmask(4);
    Vmm vreg_unit = Vmm(31);
    Vmm vreg_perm = Vmm(30);
    Vmm vmm_tail_mask = Vmm(15); // use for avx tail loads

    const int n_max_regs_ = 4;

    Vmm vmm_mask(const Vmm vmm_in, bool mask_flag, bool store,
            Xbyak::Opmask ktail_mask);
    Vmm get_bias_reg(int n) const { return Vmm(n); }
    Vmm_lower_t get_bias_reg_lower(int n) const { return Vmm_lower_t(n); }
    Vmm get_ddst_reg(int n) const { return Vmm(n + n_max_regs_); }

    void accumulate_bias(int idx, bool mask_flag);
    void store(int idx, bool mask_flag);
    void loop_by_N(int n_loop, int nb_tail);
    void init_masks(int tail_length);
    void generate() override;
};

struct brgemm_kernel_post_ops_args_t {
    void *ptr_in;
    void *ptr_out;
    void *ptr_bias;
    void *ptr_scales;
    const void *ptr_binary_post_ops_rhs;
    size_t apply_comp = 0;
    int32_t a_comp_val = 1;
    int32_t *a_zp_compensation;
    int32_t *c_zp_values;
    int32_t *s8s8_compensation;
    const void *dst_orig;
    void *ptr_dst_scales;
};

// This is a shim user interface that allows to create a template-free object
// of post-ops class.
struct jit_brgemm_kernel_post_ops_base_t {
    // `isa` argument specifies the `Vmm` type the kernel to be generated for.
    // Rest arguments are propagated as is to the underlying class.
    static jit_brgemm_kernel_post_ops_base_t *create(cpu_isa_t isa,
            const brgemm_desc_t &abrg, const primitive_attr_t &aattr);

    virtual ~jit_brgemm_kernel_post_ops_base_t() = default;

    virtual status_t generate_kernel() = 0;

    virtual void operator()(brgemm_kernel_post_ops_args_t *args) const = 0;

    virtual int get_bcast_dim() const = 0;
};

// An implementation class for post-ops based on `Vmm` template argument.
// `Vmm` is propagated further to uni_postops injector class.
// Shouldn't be called directly on implementation side.
template <typename Vmm>
struct jit_brgemm_kernel_post_ops_t : public jit_brgemm_kernel_post_ops_base_t,
                                      public jit_generator {

    // TODO: the proper design should replace `brgemm_desc_t` argument and
    // introduce a dedicated struct with members properly initialized. This will
    // let avoiding a `brgemm_desc_t` object copy which is unsafe due to `attr`
    // member.
    jit_brgemm_kernel_post_ops_t(
            const brgemm_desc_t &abrg, const primitive_attr_t &aattr);

    // These two methods are required for a base class to work since it's not
    // derived from the jit_generator.
    status_t generate_kernel() override {
        return jit_generator::create_kernel();
    }
    void operator()(brgemm_kernel_post_ops_args_t *args) const override {
        return jit_generator::operator()(args);
    }

    ~jit_brgemm_kernel_post_ops_t() = default;

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_kernel_post_ops_t)

    // Used for assertion on implementation side in debug mode.
    int get_bcast_dim() const override { return brg_.bcast_dim; }

private:
    // This can't be a reference, otherwise, `get_bcast_dim()` would return
    // rubbish due to brgemm_desc argument is a copy on stack (see comment
    // above).
    // This means a copy at construction time.
    // This class is ridiculously broken.
    brgemm_desc_t brg_;
    const primitive_attr_t &attr_;

    data_type_t inp_dt_;
    data_type_t out_dt_;
    data_type_t bia_dt_;

    using Vmm_lower_t = typename vreg_traits<Vmm>::Vmm_lower_t;
    using Vmm_lower2_t = typename vreg_traits<Vmm_lower_t>::Vmm_lower_t;
    using po_injector_t = injector::jit_uni_postops_injector_base_t<Vmm>;
    std::unique_ptr<po_injector_t> postops_injector_;
    std::unique_ptr<bf16_emulation_t> bf16_emu_;
    std::unique_ptr<fp8_emulation_e5m2_t> f8_e5m2_emulator_;
    std::unique_ptr<fp8_emulation_e4m3_t> f8_e4m3_emulator_;

    int max_vregs_;
    const bool with_binary_non_scalar_bcast_;

    int inp_typesize_;
    int out_typesize_;
    int bia_typesize_;

    int is_oc_scale_;

    using reg64_t = const Xbyak::Reg64;

    // Register decomposition
    const reg64_t reg_reserved_eltwise = rax;
    const reg64_t param1 = abi_param1;
    const reg64_t reg_in = r15;
    const reg64_t reg_out = r14;
    const reg64_t aux_reg_in = r13;
    const reg64_t aux_reg_out = r12;

    const reg64_t reg_bias = r11;
    const reg64_t aux_reg_bias = r10;

    const reg64_t reg_scales = r9;
    const reg64_t aux_reg_scales = r8;

    const reg64_t reg_ptr_sum_scale = rdx;
    const reg64_t reg_ptr_sum_zp = rsi;

    const reg64_t reg_zp_c_values = rbx;
    const reg64_t aux_reg_zp_c_values = rbx;
    const reg64_t reg_zp_a_comp = rbx;
    const reg64_t aux_reg_zp_a_comp = rbx;
    const reg64_t reg_s8s8_comp = rbx;
    const reg64_t aux_reg_s8s8_comp = rbx;
    const reg64_t reg_zp_a_val = rbx;
    const reg64_t reg_apply_comp = rbx;
    const reg64_t reg_dst_scales = rbx;
    const reg64_t aux_reg_dst_scales = rbx;
    const reg64_t reg_tmp = abi_not_param1;

    constexpr static int reg_zp_c_values_offs_ = 0;
    constexpr static int aux_reg_zp_c_values_offs_ = 8;
    constexpr static int reg_zp_a_comp_offs_ = 16;
    constexpr static int aux_reg_zp_a_comp_offs_ = 24;
    constexpr static int reg_s8s8_comp_offs_ = 32;
    constexpr static int aux_reg_s8s8_comp_offs_ = 40;
    constexpr static int reg_zp_a_val_offs_ = 48;
    constexpr static int reg_apply_comp_offs_ = 56;
    constexpr static int reg_dst_scales_offs_ = 64;
    constexpr static int stack_space_needed_ = 72;

    /* bf16 emulation */
    Xbyak::Zmm emu_reserv_1 = Xbyak::Zmm(27);
    Xbyak::Zmm emu_reserv_2 = Xbyak::Zmm(26);
    Xbyak::Zmm emu_reserv_3 = Xbyak::Zmm(25);
    Xbyak::Zmm emu_reserv_4 = Xbyak::Zmm(24);
    Xbyak::Zmm emu_reserv_5 = Xbyak::Zmm(23);
    reg64_t emu_scratch = reg_tmp;
    Xbyak::Opmask emu_mask = Xbyak::Opmask(4);

    Xbyak::Opmask k_full_mask = Xbyak::Opmask(2);
    Xbyak::Opmask k_tail_mask = Xbyak::Opmask(3);

    const int n_block2_ = 4;

    Vmm vmm_tmp(int i) const { return Vmm(max_vregs_ - 1 - i); }

    int zp_c_values_offset(int n, bool is_tail = false) const noexcept;
    int zp_comp_a_vpad_offset(
            int n, int m, bool is_tail = false) const noexcept;
    int mb_zp_comp_a_offset(int m_block) const noexcept;
    int compensation_vpad_offset(
            int n, int m, bool is_tail = false) const noexcept;
    int mb_compensation_offset(int m_block) const noexcept {
        return sizeof(int32_t) * m_block * brg_.LDB;
    }

    template <typename T>
    const T maybe_mask(const T vmm_in, bool mask_flag, bool store,
            Xbyak::Opmask ktail_mask) {
        assert(IMPLICATION(mask_flag, isa_has_masks(brg_.isa_impl)));
        return mask_flag
                ? (store ? vmm_in | ktail_mask : vmm_in | ktail_mask | T_z)
                : vmm_in;
    }

    void cvt2ps(data_type_t type_in, const Vmm vmm_in, const Xbyak::Operand &op,
            int tail_size, bool store, Xbyak::Opmask ktail_mask,
            bool skip_cvt2ps = false);

    Vmm vector(int m, int n, int n_block) { return Vmm(m * n_block + n); };

    void inject_attr_postops(int m_block, int n_block, int tail = 0);
    void apply_comp(int m_block, int n_block, int tail = 0);
    void maybe_apply_comp(int m_block, int n_block, int tail = 0);
    void apply_post_ops(int m_block, int n_block, int tail = 0);
    void loop_by_N(int m_block, int nb2, int nb2_tail, int nb_tail);
    void generate() override;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
