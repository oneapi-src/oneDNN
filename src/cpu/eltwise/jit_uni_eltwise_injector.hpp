/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#ifndef CPU_JIT_UNI_ELTWISE_INJECTOR_HPP
#define CPU_JIT_UNI_ELTWISE_INJECTOR_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "primitive_attr.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <cpu_isa_t isa>
struct jit_uni_eltwise_injector_f32 {
    using Vmm = typename cpu_isa_traits<isa>::Vmm;

    jit_uni_eltwise_injector_f32(jit_generator *host, alg_kind_t alg,
            float alpha, float beta, float scale, bool save_state = true,
            Xbyak::Reg64 p_table = Xbyak::util::rax,
            Xbyak::Opmask k_mask = Xbyak::Opmask(1))
        : alg_(alg)
        , alpha_(alpha)
        , beta_(beta)
        , scale_(scale)
        , h(host)
        , save_state_(save_state)
        , p_table(p_table)
        , k_mask(k_mask) {
        using namespace alg_kind;
        assert(utils::one_of(isa, sse41, avx2, avx512_common, avx512_core));
        assert(utils::one_of(alg_, eltwise_relu, eltwise_tanh, eltwise_elu,
                eltwise_square, eltwise_abs, eltwise_sqrt, eltwise_linear,
                eltwise_bounded_relu, eltwise_soft_relu, eltwise_logistic,
                eltwise_exp, eltwise_gelu_tanh, eltwise_swish, eltwise_log,
                eltwise_clip, eltwise_pow, eltwise_gelu_erf,
                eltwise_relu_use_dst_for_bwd, eltwise_tanh_use_dst_for_bwd,
                eltwise_elu_use_dst_for_bwd, eltwise_sqrt_use_dst_for_bwd,
                eltwise_logistic_use_dst_for_bwd, eltwise_exp_use_dst_for_bwd));
        register_table_entries();
    }

    jit_uni_eltwise_injector_f32(jit_generator *host,
            const post_ops_t::entry_t::eltwise_t &eltwise,
            bool save_state = true, Xbyak::Reg64 p_table = Xbyak::util::rax,
            Xbyak::Opmask k_mask = Xbyak::Opmask(1))
        : jit_uni_eltwise_injector_f32(host, eltwise.alg, eltwise.alpha,
                eltwise.beta, eltwise.scale, save_state, p_table, k_mask) {}

    void compute_vector_range(size_t start_idx, size_t end_idx);
    void compute_vector(size_t idx) { compute_vector_range(idx, idx + 1); }
    void prepare_table(bool gen_table = true);
    void load_table_addr() { h->mov(p_table, l_table); }

private:
    const alg_kind_t alg_;
    const float alpha_;
    const float beta_;
    const float scale_;

    jit_generator *const h;

    const bool save_state_;
    const Xbyak::Reg64 p_table;
    const Xbyak::Opmask k_mask;
    Xbyak::Label l_table;

    // if only the injector was inherited from jit_generator...
    enum {
        _cmp_eq_oq = jit_generator::_cmp_eq_oq,
        _cmp_lt_os = jit_generator::_cmp_lt_os,
        _cmp_le_os = jit_generator::_cmp_le_os,
        _cmp_ge_os = jit_generator::_cmp_nlt_us,
        _cmp_gt_os = jit_generator::_cmp_nle_us,
        _op_floor = jit_generator::_op_floor
    };

    static constexpr bool has_avx512() {
        return utils::one_of(isa, avx512_common, avx512_core);
    }

    static constexpr size_t vlen = cpu_isa_traits<isa>::vlen;
    static constexpr size_t preserved_vecs_max = 5;
    static constexpr size_t vecs_count = has_avx512() ? 32 : 16;
    static constexpr int n_mantissa_bits = 23;
    static constexpr int k_mask_size = 8;

    size_t vecs_to_preserve = 0;
    size_t preserved_vecs_count = 0;
    size_t preserved_vec_idxs[preserved_vecs_max] = {0};
    size_t start_idx_tail = 0;

    Vmm vmm_mask, vmm_aux0, vmm_aux1, vmm_aux2, vmm_aux3, vmm_aux4;

    size_t aux_vecs_count();

    void compute_body(size_t start_idx, size_t end_idx);
    void injector_preamble(size_t start_idx, size_t end_idx);
    void injector_preamble_tail(size_t start_idx);
    void injector_postamble();
    void assign_regs();
    void compute_cmp_mask(const Vmm &vmm_src,
            const Xbyak::Operand &compare_operand, int cmp_predicate);
    void blend_with_mask(const Vmm &vmm_dst, const Xbyak::Operand &src);
    void test_mask();

    void exp_compute_vector_fwd(const Vmm &vmm_src);
    void relu_compute_vector_fwd(const Vmm &vmm_src);
    void relu_zero_ns_compute_vector_fwd(const Vmm &vmm_src);
    void elu_compute_vector_fwd(const Vmm &vmm_src);
    void tanh_compute_vector_fwd(const Vmm &vmm_src);
    void square_compute_vector_fwd(const Vmm &vmm_src);
    void abs_compute_vector_fwd(const Vmm &vmm_src);
    void sqrt_compute_vector_fwd(const Vmm &vmm_src);
    void linear_compute_vector_fwd(const Vmm &vmm_src);
    void bounded_relu_compute_vector_fwd(const Vmm &vmm_src);
    void soft_relu_compute_vector_fwd(const Vmm &vmm_src);
    void logistic_compute_vector_fwd(const Vmm &vmm_src);
    void gelu_tanh_compute_vector_fwd(const Vmm &vmm_src);
    void swish_compute_vector_fwd(const Vmm &vmm_src);
    void log_compute_vector_fwd(const Vmm &vmm_src);
    void clip_compute_vector_fwd(const Vmm &vmm_src);
    void pow_compute_vector_fwd(const Vmm &vmm_src);
    void gelu_erf_compute_vector_fwd(const Vmm &vmm_src);

    enum key_t {
        scale = 0, // scale argument
        alpha, // alpha argument
        beta, // beta argument
        zero, // 0.f
        half, // 0.5f
        one, // 1.f  or  mask for exponent bits
        minus_two, // -2.f
        ln2f, // 0.69314718f
        positive_mask, // changes sign to positive
        sign_mask, // gets sign value
        exponent_bias, // (127 = 2^7 - 1), gets exponent bits
        exp_log2ef, // 1.44269502f - formula-based for approx
        exp_ln_flt_max_f, // logf(FLT_MAX) - max normal value
        exp_ln_flt_min_f, // logf(FLT_MIN) - min normal value
        exp_pol, // see correspondent table for float values
        tanh_bound_x, // arg below which tanh(x) = x
        tanh_bound_pol, // arg below which polynomial approx is valid
        tanh_bound_one, // arg after which tanh(x) = 1.f
        tanh_pol, // see correspondent table for float values
        soft_relu_one_twenty_six, // 126.f
        soft_relu_change_sign_mask, // changes sign to opposite
        soft_relu_mantissa_sign_mask, // mask for mantissa bits and sign
        soft_relu_pol, // see correspondent table for float values
        gelu_tanh_fitting_const, // 0.044715f
        gelu_tanh_sqrt_two_over_pi, // sqrtf(2.f/pi) = 0.797884f
        gelu_erf_approx_const, // 0.3275911f - implementation based for approx
        gelu_erf_one_over_sqrt_two, // 1.f / sqrtf(2.f)
        gelu_erf_pol, // see correspondent table for float values
        log_minus_inf, // -inf
        log_qnan, // qnan
        log_mantissa_mask, // gets mantissa bits
        log_full_k_reg_mask, // sets k_register with all bits of 1
        log_full_vector_reg_mask, // sets vector register will all bits of 1
        log_five_bit_offset, // 5 bits off (31 = 2^5 - 1)
        log_pol, // see correspondent table for float values
        log_predefined_vals, // see correspondent table for float values
    };

    Xbyak::Address table_val(key_t key, size_t key_off_val_shift = 0) {
        const auto it = entry_map_.find(key); // search an entry for a key
        assert(it != entry_map_.end());
        const auto &t_e = (*it).second;
        const auto t_e_offset = t_e.first;
        const auto index = t_e_offset + key_off_val_shift;
        return h->ptr[p_table + index * vlen];
    }

    using table_entry_t = std::pair<size_t, size_t>; // {offset, hex_value}
    using table_t = std::multimap<key_t, table_entry_t>; // {key, table_entry}
    void register_table_entries();
    table_t entry_map_;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
