/*******************************************************************************
* Copyright 2020 Intel Corporation
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
#include "cpu/x64/prelu/jit_prelu_reduction_kernel.hpp"
#include "common/nstl.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

static constexpr dim_t alignment
        = platform::get_cache_line_size() / sizeof(float);
static dim_t get_C(const cpu_prelu_bwd_pd_t *pd) {
    const memory_desc_wrapper src_diff_d {pd->diff_src_md(0)};
    return src_diff_d.ndims() >= 2 ? src_diff_d.dims()[1] : 1;
}

jit_prelu_reduction_kernel_t::jit_prelu_reduction_kernel_t(
        const cpu_prelu_bwd_pd_t *pd, int simd_w)
    : simd_w_(simd_w)
    , scratchpad_c_block_offset_(
              utils::rnd_up(get_C(pd), alignment) * sizeof(float))
    , data_type_(pd->diff_weights_md(0)->data_type)
    , tail_size_(get_C(pd) % simd_w_) {}

#define PARAM_OFF(x) offsetof(call_params_t, x)

void jit_prelu_reduction_kernel_t::load_kernel_call_params() {
    mov(reg_reduction_blocks_, ptr[abi_param1 + PARAM_OFF(reduction_blocks)]);
    mov(reg_weights_diff_scratch_,
            ptr[abi_param1 + PARAM_OFF(weights_diff_scratch)]);
    mov(reg_weights_diff_, ptr[abi_param1 + PARAM_OFF(weights_diff)]);
    mov(reg_tail_, byte[abi_param1 + PARAM_OFF(tail)]);
}

#undef PARAM_OFF

void jit_prelu_reduction_kernel_t::generate() {
    Xbyak::Label tail, end;

    preamble();
    load_kernel_call_params();

    if (tail_size_) {
        cmp(reg_tail_, 1);
        je(tail, T_NEAR);

        generate(false /* tail*/);
        jmp(end, T_NEAR);

        L(tail);
        generate(true /* tail*/);

        L(end);
    } else
        generate(false /* tail*/);

    postamble();
}

void jit_prelu_reduction_kernel_t::generate(bool tail) {

    Xbyak::Label unroll_loop, unroll_loop_tail, end;
    const auto unrolling_factor = get_unrolling_factor(tail);

    prepare_kernel_const_vars(tail);
    xor_(reg_offset_, reg_offset_);
    L(unroll_loop);
    {
        const size_t offt = unrolling_factor * scratchpad_c_block_offset_;
        cmp(reg_reduction_blocks_, unrolling_factor);
        jl(unroll_loop_tail, T_NEAR);
        compute_dst(unrolling_factor, tail);
        sub(reg_reduction_blocks_, unrolling_factor);
        add(reg_offset_, offt);
        jmp(unroll_loop);
    }

    L(unroll_loop_tail);
    {
        cmp(reg_reduction_blocks_, 0);
        jle(end, T_NEAR);
        compute_dst(1, tail);
        sub(reg_reduction_blocks_, 1);
        add(reg_offset_, scratchpad_c_block_offset_);
        jmp(unroll_loop_tail);
    }

    L(end);

    finalize(tail);
}

Xbyak::Address jit_prelu_reduction_kernel_t::diff_scratch_ptr(
        int unrolling_group) const {
    return ptr[reg_weights_diff_scratch_ + reg_offset_
            + unrolling_group * scratchpad_c_block_offset_];
}

template <typename Vmm>
jit_uni_prelu_reduction_kernel_t<Vmm>::jit_uni_prelu_reduction_kernel_t(
        const cpu_prelu_bwd_pd_t *pd, const cpu_isa_t &isa)
    : jit_prelu_reduction_kernel_t(pd, prelu::get_vlen(isa) / sizeof(float))
    , isa_(isa)
    , io_(this, isa, data_type_, tail_size_, tail_opmask_, tail_vmm_mask_,
              reg_tmp_) {}

template <typename Vmm>
size_t jit_uni_prelu_reduction_kernel_t<Vmm>::get_unrolling_factor(
        bool tail) const {
    const size_t max_num_threads = dnnl_get_max_threads();
    const size_t n_vregs = prelu::get_n_vregs(isa_);
    int numer_reserved_regs = 1; // accumulator
    if (tail && utils::one_of(isa_, avx, avx2))
        ++numer_reserved_regs;
    else if (data_type_ == data_type::bf16 && isa_ == avx512_core)
        numer_reserved_regs += 4;
    const size_t number_of_available_regs = n_vregs - numer_reserved_regs;

    return nstl::min(number_of_available_regs, max_num_threads);
}

template <typename Vmm>
void jit_uni_prelu_reduction_kernel_t<Vmm>::finalize(bool tail) {
    io_.store(accumulator_, ptr[reg_weights_diff_], tail);
}

template <typename Vmm>
void jit_uni_prelu_reduction_kernel_t<Vmm>::prepare_kernel_const_vars(
        bool tail) {
    uni_vxorps(accumulator_, accumulator_, accumulator_);

    if (tail) io_.prepare_tail_mask();
}

template <typename Vmm>
void jit_uni_prelu_reduction_kernel_t<Vmm>::compute_dst(
        int unrolling_factor, bool tail) {

    const int vmm_begin = tail && utils::one_of(isa_, avx, avx2) ? 2 : 1;

    for (int unrolling_group = 0; unrolling_group < unrolling_factor;
            ++unrolling_group) {
        const Vmm load_vmm {vmm_begin + unrolling_group};
        uni_vmovups(load_vmm, diff_scratch_ptr(unrolling_group));
        uni_vaddps(accumulator_, accumulator_, load_vmm);
    }
}

jit_prelu_reduction_kernel_t *jit_prelu_reduction_kernel_t::create(
        const cpu_prelu_bwd_pd_t *pd) {

    const auto isa = prelu::get_supported_isa();

    if (is_superset(isa, avx512_common))
        return new jit_uni_prelu_reduction_kernel_t<Xbyak::Zmm>(pd, isa);
    else if (is_superset(isa, avx))
        return new jit_uni_prelu_reduction_kernel_t<Xbyak::Ymm>(pd, isa);
    else if (isa == sse41)
        return new jit_uni_prelu_reduction_kernel_t<Xbyak::Xmm>(pd, isa);

    return nullptr;
}

template class jit_uni_prelu_reduction_kernel_t<Xbyak::Zmm>;
template class jit_uni_prelu_reduction_kernel_t<Xbyak::Ymm>;
template class jit_uni_prelu_reduction_kernel_t<Xbyak::Xmm>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
