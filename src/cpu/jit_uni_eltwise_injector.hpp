/*******************************************************************************
* Copyright 2019 Intel Corporation
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
                eltwise_exp, eltwise_gelu, eltwise_swish, eltwise_log,
                eltwise_clip, eltwise_pow));
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

    size_t vecs_to_preserve = 0;
    size_t preserved_vecs_count = 0;
    size_t preserved_vec_idxs[preserved_vecs_max] = {0};
    size_t start_idx_tail = 0;

    Vmm vmm_mask, vmm_aux0, vmm_aux1, vmm_aux2, vmm_aux3, vmm_aux4;

    Xbyak::Address table_val(int index) {
        size_t scale_offt_ = vlen;
        return h->ptr[p_table + scale_offt_ + index * vlen];
    }

    size_t aux_vecs_count(alg_kind_t alg);

    void compute_body(size_t start_idx, size_t end_idx);
    void injector_preamble(size_t start_idx, size_t end_idx);
    void injector_preamble_tail(size_t start_idx);
    void injector_postamble();
    void assign_regs();
    void compute_cmp_mask(const Vmm &vmm_src,
            const Xbyak::Operand &compare_operand, int cmp_predicate);
    void blend_with_mask(const Vmm &vmm_dst, const Xbyak::Operand &src);
    void test_mask();

    void exp_compute_vector(const Vmm &vmm_src);
    void relu_compute_vector(const Vmm &vmm_src);
    void relu_zero_ns_compute_vector(const Vmm &vmm_src);
    void elu_compute_vector(const Vmm &vmm_src);
    void tanh_compute_vector(const Vmm &vmm_src);
    void square_compute_vector(const Vmm &vmm_src);
    void abs_compute_vector(const Vmm &vmm_src);
    void sqrt_compute_vector(const Vmm &vmm_src);
    void linear_compute_vector(const Vmm &vmm_src);
    void bounded_relu_compute_vector(const Vmm &vmm_src);
    void soft_relu_compute_vector(const Vmm &vmm_src);
    void logistic_compute_vector(const Vmm &vmm_src);
    void gelu_compute_vector(const Vmm &vmm_src);
    void swish_compute_vector(const Vmm &vmm_src);
    void log_compute_vector(const Vmm &vmm_src);
    void clip_compute_vector(const Vmm &vmm_src);
    void pow_compute_vector(const Vmm &vmm_src);

    void relu_prepare_table();
    void elu_prepare_table();
    void soft_relu_prepare_table();
    void abs_prepare_table();
    void linear_prepare_table();
    void log_prepare_table();
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
