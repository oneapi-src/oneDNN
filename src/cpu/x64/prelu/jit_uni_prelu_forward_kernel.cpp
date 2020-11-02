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
        const cpu_prelu_fwd_pd_t *pd, const cpu_isa_t &isa,
        size_t number_vmm_single_compute)
    : jit_prelu_base_kernel_t(isa,
            prelu::get_bcast_type(memory_desc_wrapper(pd->src_md(0)),
                    memory_desc_wrapper(pd->weights_md(0))),
            memory_desc_wrapper(pd->src_md(0)), number_vmm_single_compute)
    , pd_(pd) {}

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
    : jit_prelu_forward_kernel_t(pd, isa,
            utils::one_of(isa, sse41, avx) || data_type_ == data_type::bf16
                    ? 4u
                    : 3u)
    , vmm_zeros_(reserve_vmm())
    , tail_vmm_mask_(
              tail_size_ && utils::one_of(isa, avx, avx2) ? reserve_vmm() : 0)
    , weights_const_vmm_(utils::one_of(bcast_, prelu::bcast::per_oc_n_c_spatial,
                                 prelu::bcast::per_oc_blocked)
                      ? reserve_vmm()
                      : 0)
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
    uni_vfmadd132ps(x1, x2, op);
}

template <>
void jit_uni_prelu_forward_kernel_t<Xbyak::Zmm>::uni_vfmadd132ps(
        const Xbyak::Zmm &x1, const Xbyak::Zmm &x2, const Xbyak::Operand &op,
        bool tail) {
    if (tail && op.isMEM())
        vfmadd132ps(x1 | tail_opmask_, x2, op);
    else
        vfmadd132ps(x1, x2, op);
}

template <typename Vmm>
void jit_uni_prelu_forward_kernel_t<Vmm>::compute_dst(
        size_t unrolling_factor, bool tail) {
    static constexpr size_t max_idx = 0;
    static constexpr size_t min_idx = 1;
    static constexpr size_t src_idx = 2;
    static constexpr size_t weights_idx = 3;
    const auto dt_size = types::data_type_size(data_type_);

    for (size_t unroll_group = 0; unroll_group < unrolling_factor;
            ++unroll_group) {
        const Vmm max_vmm {get_compute_vmm(max_idx, unroll_group)};
        const Vmm min_vmm {get_compute_vmm(min_idx, unroll_group)};
        const Vmm src_vmm {get_compute_vmm(src_idx, unroll_group)};
        const Vmm weights_vmm {get_compute_vmm(weights_idx, unroll_group)};

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

jit_prelu_forward_kernel_t *jit_prelu_forward_kernel_t::create(
        const cpu_prelu_fwd_pd_t *pd) {

    const auto isa = prelu::get_supported_isa();
    if (is_superset(isa, avx512_common))
        return new jit_uni_prelu_forward_kernel_t<Xbyak::Zmm>(pd, isa);
    else if (is_superset(isa, avx))
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
