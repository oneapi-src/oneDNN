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
#include <cstddef>
#include "cpu/x64/prelu/jit_uni_prelu_forward_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

jit_prelu_forward_kernel_t::jit_prelu_forward_kernel_t(
        const cpu_prelu_fwd_pd_t *pd, int vlen)
    : pd_(pd)
    , simd_w_(vlen / sizeof(float))
    , bcast_(prelu::get_bcast_type(memory_desc_wrapper(pd_->src_md()),
              memory_desc_wrapper(pd_->weights_md())))
    , tail_size_(calc_tail_size())
    , data_type_(pd_->src_md()->data_type) {}

size_t jit_prelu_forward_kernel_t::simd_w() const noexcept {
    return simd_w_;
}

prelu::bcast jit_prelu_forward_kernel_t::get_bcast() const noexcept {
    return bcast_;
}

size_t jit_prelu_forward_kernel_t::calc_tail_size() const noexcept {

    const auto src_d = memory_desc_wrapper(pd_->src_md());
    dim_t nelems = 0;
    const auto &ndims = src_d.ndims();
    if (bcast_ == prelu::bcast::full)
        nelems = src_d.nelems();
    else if (bcast_ == prelu::bcast::per_oc_n_spatial_c)
        return src_d.dims()[1];
    else if (bcast_ == prelu::bcast::per_oc_n_c_spatial && ndims >= 3)
        nelems = utils::array_product(src_d.dims() + 2, ndims - 2);
    return nelems % simd_w_;
}

void jit_prelu_forward_kernel_t::generate() {
    Xbyak::Label unroll_loop, unroll_loop_tail, nelems_tail, end;
    const auto dt_size = types::data_type_size(data_type_);
    const auto dt_vec_size = simd_w_ * dt_size;
    const auto unrolling_factor = get_unrolling_factor();
    preamble();
    load_kernel_call_params();
    prepare_kernel_const_vars();

    xor_(reg_offset_, reg_offset_);
    L(unroll_loop);
    {
        const size_t offt = unrolling_factor * dt_vec_size;
        cmp(reg_data_size_, offt);
        jl(unroll_loop_tail, T_NEAR);

        compute_dst(unrolling_factor, false /*tail*/);
        sub(reg_data_size_, offt);
        add(reg_offset_, offt);
        jmp(unroll_loop);
    }

    static constexpr size_t single_unrolling = 1u;
    L(unroll_loop_tail);
    {
        cmp(reg_data_size_, dt_vec_size);
        jl(nelems_tail, T_NEAR);

        compute_dst(single_unrolling, false /*tail*/);
        sub(reg_data_size_, dt_vec_size);
        add(reg_offset_, dt_vec_size);
        jmp(unroll_loop_tail);
    }

    L(nelems_tail);
    {
        cmp(reg_data_size_, 1);
        jl(end, T_NEAR);

        compute_dst(single_unrolling, true /*tail*/);
    }

    L(end);

    postamble();
}

#define PARAM_OFF(x) offsetof(call_params_t, x)

void jit_prelu_forward_kernel_t::load_kernel_call_params() {
    mov(reg_src_, ptr[abi_param1 + PARAM_OFF(src)]);
    mov(reg_weights_, ptr[abi_param1 + PARAM_OFF(weights)]);
    mov(reg_dst_, ptr[abi_param1 + PARAM_OFF(dst)]);
    mov(reg_data_size_, ptr[abi_param1 + PARAM_OFF(compute_data_size)]);
}

#undef PARAM_OFF

Xbyak::Address jit_prelu_forward_kernel_t::data_ptr(int arg_num, size_t offt) {
    switch (arg_num) {
        case DNNL_ARG_SRC: return ptr[reg_src_ + reg_offset_ + offt];
        case DNNL_ARG_WEIGHTS: return ptr[reg_weights_ + reg_offset_ + offt];
        case DNNL_ARG_DST: return ptr[reg_dst_ + reg_offset_ + offt];
        default: assert(!"unsupported arg_num"); break;
    }
    return Xbyak::Address(0);
}

template <typename Vmm>
jit_uni_prelu_forward_kernel_t<Vmm>::jit_uni_prelu_forward_kernel_t(
        const cpu_prelu_fwd_pd_t *pd, const cpu_isa_t &isa)
    : jit_prelu_forward_kernel_t(pd, prelu::get_vlen(isa))
    , isa_(isa)
    , number_vmms_reserved_const_vars_(0)
    , vmm_zeros_(reserve_vmm())
    , tail_vmm_mask_(tail_size_ && utils::one_of(isa, avx, avx2) ? reserve_vmm()
                                                                 : Vmm(0))
    , weights_const_vmm_(utils::one_of(bcast_, prelu::bcast::per_oc_n_c_spatial,
                                 prelu::bcast::per_oc_blocked)
                      ? reserve_vmm()
                      : Vmm(0))
    , number_vmm_single_compute_(
              utils::one_of(isa, sse41, avx) || data_type_ == data_type::bf16
                      ? 4u
                      : 3u)
    , unrolling_factor_(calc_unrolling_factor())
    , io_(this, isa, data_type_, tail_size_, tail_opmask_, tail_vmm_mask_,
              reg_tmp_) {}

template <typename Vmm>
jit_uni_prelu_forward_kernel_t<Vmm>::~jit_uni_prelu_forward_kernel_t()
        = default;

template <typename Vmm>
void jit_uni_prelu_forward_kernel_t<Vmm>::prepare_kernel_const_vars() {
    uni_vxorps(vmm_zeros_, vmm_zeros_, vmm_zeros_);
    if (tail_size_) io_.prepare_tail_mask();
    if (bcast_ == prelu::bcast::per_oc_n_c_spatial)
        io_.broadcast(ptr[reg_weights_], weights_const_vmm_);
    else if (bcast_ == prelu::bcast::per_oc_blocked)
        io_.load(ptr[reg_weights_], weights_const_vmm_, false /*tail*/);
}

template <typename Vmm>
Vmm jit_uni_prelu_forward_kernel_t<Vmm>::reserve_vmm() {
    return Vmm(number_vmms_reserved_const_vars_++);
}

template <typename Vmm>
size_t jit_uni_prelu_forward_kernel_t<Vmm>::get_number_reserved_vmms() const
        noexcept {
    return number_vmms_reserved_const_vars_;
}

template <>
size_t
jit_uni_prelu_forward_kernel_t<Xbyak::Zmm>::get_number_reserved_vmms() const
        noexcept {
    static constexpr size_t number_vmm_reserved_bf16_process = 4u;
    const bool process_bf16_with_emu
            = data_type_ == data_type::bf16 && isa_ == avx512_core;

    return number_vmms_reserved_const_vars_
            + (process_bf16_with_emu ? number_vmm_reserved_bf16_process : 0);
}

template <typename Vmm>
size_t jit_uni_prelu_forward_kernel_t<Vmm>::calc_unrolling_factor() const
        noexcept {
    static const auto n_vregs = prelu::get_n_vregs(isa_);
    const size_t number_of_available_regs
            = n_vregs - get_number_reserved_vmms();
    const size_t max_unrolling_factor
            = number_of_available_regs / number_vmm_single_compute_;

    const auto src_d = memory_desc_wrapper(pd_->src_md());
    size_t single_thread_estimated_elems = 0;
    const auto &dims = src_d.dims();
    const auto &ndims = src_d.ndims();
    const dim_t D = ndims >= 5 ? dims[ndims - 3] : 1;
    const dim_t H = ndims >= 4 ? dims[ndims - 2] : 1;
    const dim_t W = ndims >= 3 ? dims[ndims - 1] : 1;
    const dim_t SP = D * H * W;

    if (bcast_ == prelu::bcast::full) {
        const size_t nelems = src_d.nelems();
        single_thread_estimated_elems = nelems / dnnl_get_max_threads();
    } else if (bcast_ == prelu::bcast::per_oc_n_spatial_c) {
        single_thread_estimated_elems = src_d.dims()[1];
    } else if (bcast_ == prelu::bcast::per_oc_blocked) {
        single_thread_estimated_elems = SP * simd_w_;
    } else if (bcast_ == prelu::bcast::per_oc_n_c_spatial) {
        single_thread_estimated_elems = SP;
    }

    const size_t estimated_vectors_used = nstl::max(
            static_cast<size_t>(
                    std::floor(single_thread_estimated_elems / simd_w_)),
            static_cast<size_t>(1));

    return nstl::min(max_unrolling_factor, estimated_vectors_used);
}

template <typename Vmm>
size_t jit_uni_prelu_forward_kernel_t<Vmm>::get_unrolling_factor() const {
    return unrolling_factor_;
}

template <typename Vmm>
const Xbyak::Operand &jit_uni_prelu_forward_kernel_t<Vmm>::get_or_load_weights(
        const Xbyak::Address &src_addr, const Vmm &weights_vmm, bool tail) {
    if (utils::one_of(bcast_, prelu::bcast::per_oc_n_c_spatial,
                prelu::bcast::per_oc_blocked))
        return weights_const_vmm_;
    else if (data_type_ == data_type::bf16) {
        io_.load(src_addr, weights_vmm, tail);
        return weights_vmm;
    }
    return src_addr;
}

template <>
const Xbyak::Operand &
jit_uni_prelu_forward_kernel_t<Xbyak::Ymm>::get_or_load_weights(
        const Xbyak::Address &src_addr, const Xbyak::Ymm &weights_vmm,
        bool tail) {
    if (utils::one_of(bcast_, prelu::bcast::per_oc_n_c_spatial,
                prelu::bcast::per_oc_blocked))
        return weights_const_vmm_;
    else if (tail || isa_ == avx) {
        io_.load(src_addr, weights_vmm, tail);
        return weights_vmm;
    }

    return src_addr;
}

template <>
const Xbyak::Operand &
jit_uni_prelu_forward_kernel_t<Xbyak::Xmm>::get_or_load_weights(
        const Xbyak::Address &src_addr, const Xbyak::Xmm &weights_vmm,
        bool tail) {

    if (utils::one_of(bcast_, prelu::bcast::per_oc_n_c_spatial,
                prelu::bcast::per_oc_blocked))
        return weights_const_vmm_;

    io_.load(src_addr, weights_vmm, tail);
    return weights_vmm;
}

template <typename Vmm>
void jit_uni_prelu_forward_kernel_t<Vmm>::uni_vfmadd132ps(
        const Vmm &x1, const Vmm &x2, const Xbyak::Operand &op, bool tail) {
    if (tail && op.isMEM())
        uni_vfmadd132ps(x1 | tail_opmask_, x2, op);
    else
        uni_vfmadd132ps(x1, x2, op);
}

template <typename Vmm>
void jit_uni_prelu_forward_kernel_t<Vmm>::compute_dst(
        int unrolling_factor, int tail) {
    static constexpr size_t max_idx = 0;
    static constexpr size_t min_idx = 1;
    static constexpr size_t src_idx = 2;
    static constexpr size_t weights_idx = 3;
    const auto dt_size = types::data_type_size(data_type_);

    for (size_t unroll_group = 0; unroll_group < unrolling_factor;
            ++unroll_group) {
        const Vmm max_vmm = get_compute_vmm(max_idx, unroll_group);
        const Vmm min_vmm = get_compute_vmm(min_idx, unroll_group);
        const Vmm src_vmm = get_compute_vmm(src_idx, unroll_group);
        const Vmm weights_vmm = get_compute_vmm(weights_idx, unroll_group);

        const auto offset = unroll_group * simd_w_ * dt_size;
        io_.load(data_ptr(DNNL_ARG_SRC, offset), src_vmm, tail);
        const auto &weights_operand = get_or_load_weights(
                data_ptr(DNNL_ARG_WEIGHTS, offset), weights_vmm, tail);
        uni_vmaxps(max_vmm, vmm_zeros_, src_vmm);
        uni_vminps(min_vmm, vmm_zeros_, src_vmm);
        const auto &dst_vmm = min_vmm;
        uni_vfmadd132ps(dst_vmm, max_vmm, weights_operand, tail);
        io_.store(dst_vmm, data_ptr(DNNL_ARG_DST, offset), tail);
    }
}

template <typename Vmm>
Vmm jit_uni_prelu_forward_kernel_t<Vmm>::get_compute_vmm(
        size_t base_idx, size_t unroll_group) {
    return Vmm(number_vmms_reserved_const_vars_ + base_idx
            + unroll_group * number_vmm_single_compute_);
}

jit_prelu_forward_kernel_t *jit_prelu_forward_kernel_t::create(
        const cpu_prelu_fwd_pd_t *pd) {

    const auto isa = prelu::get_supported_isa();

    if (utils::one_of(isa, avx512_core_bf16, avx512_core, avx512_common))
        return new jit_uni_prelu_forward_kernel_t<Xbyak::Zmm>(pd, isa);
    else if (utils::one_of(isa, avx, avx2))
        return new jit_uni_prelu_forward_kernel_t<Xbyak::Ymm>(pd, isa);
    else if (isa == sse41)
        return new jit_uni_prelu_forward_kernel_t<Xbyak::Xmm>(pd, isa);

    return nullptr;
}

template class jit_uni_prelu_forward_kernel_t<Xbyak::Zmm>;
template class jit_uni_prelu_forward_kernel_t<Xbyak::Ymm>;
template class jit_uni_prelu_forward_kernel_t<Xbyak::Xmm>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
