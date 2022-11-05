/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
* Copyright 2022 FUJITSU LIMITED
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
#include <numeric>
#include "common/broadcast_strategy.hpp"
#include "cpu/aarch64/injectors/injector_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace injector_utils {

template <cpu_isa_t isa>
register_preserve_guard_t<isa>::register_preserve_guard_t(jit_generator *host,
        std::initializer_list<Xbyak_aarch64::XReg> reg64_to_preserve,
        std::initializer_list<Xbyak_aarch64::VReg> vmm_to_preserve)
    : host_(host)
    , reg64_stack_(reg64_to_preserve)
    , vmm_stack_(vmm_to_preserve)
    , vmm_to_preserve_size_bytes_(
              calc_vmm_to_preserve_size_bytes(vmm_to_preserve)) {

    for (const auto &reg : reg64_to_preserve)
        host_->str(reg, pre_ptr(host_->X_SP, -8));

    if (!vmm_stack_.empty()) {
        host_->sub(host_->X_SP, host_->X_SP, vmm_to_preserve_size_bytes_);

        uint32_t stack_offset = vmm_to_preserve_size_bytes_;
        for (const auto &vmm : vmm_to_preserve) {
            const uint32_t bytes = cpu_isa_traits<isa>::vlen;
            stack_offset -= bytes;
            const auto idx = vmm.getIdx();
            if (is_superset(isa, sve_256)) {
                if (stack_offset % cpu_sveLen_ == 0) {
                    host_->st1w(Xbyak_aarch64::ZRegS(idx), host_->P_ALL_ONE,
                            ptr(host_->X_SP, stack_offset / bytes,
                                    Xbyak_aarch64::MUL_VL));
                } else {
                    host_->add_imm(host_->X_DEFAULT_ADDR, host_->X_SP,
                            stack_offset, host_->X_TMP_0);
                    host_->st1w(Xbyak_aarch64::ZRegS(idx), host_->P_ALL_ONE,
                            ptr(host_->X_DEFAULT_ADDR));
                }
            } else {
                host_->str(Xbyak_aarch64::QReg(idx),
                        ptr(host_->X_SP, stack_offset));
            }
        }
    }
}

template <cpu_isa_t isa>
register_preserve_guard_t<isa>::~register_preserve_guard_t() {

    uint32_t tmp_stack_offset = 0;

    while (!vmm_stack_.empty()) {
        const Xbyak_aarch64::VReg &vmm = vmm_stack_.top();
        const uint32_t bytes = cpu_isa_traits<isa>::vlen;
        const auto idx = vmm.getIdx();
        if (is_superset(isa, sve_256)) {
            if (tmp_stack_offset % cpu_sveLen_ == 0) {
                host_->ld1w(Xbyak_aarch64::ZRegS(idx), host_->P_ALL_ONE,
                        ptr(host_->X_SP, tmp_stack_offset / bytes,
                                Xbyak_aarch64::MUL_VL));
            } else {
                host_->add_imm(host_->X_DEFAULT_ADDR, host_->X_SP,
                        tmp_stack_offset, host_->X_TMP_0);
                host_->ld1w(Xbyak_aarch64::ZRegS(idx), host_->P_ALL_ONE,
                        ptr(host_->X_SP, host_->X_DEFAULT_ADDR));
            }
        } else {
            host_->ldr(Xbyak_aarch64::QReg(idx),
                    ptr(host_->X_SP, tmp_stack_offset));
        }

        tmp_stack_offset += bytes;
        vmm_stack_.pop();
    }

    if (vmm_to_preserve_size_bytes_)
        host_->add_imm(host_->X_SP, host_->X_SP, vmm_to_preserve_size_bytes_,
                host_->X_TMP_0);

    while (!reg64_stack_.empty()) {
        host_->ldr(reg64_stack_.top(), post_ptr(host_->X_SP, 8));
        reg64_stack_.pop();
    }
}

template <cpu_isa_t isa>
size_t register_preserve_guard_t<isa>::calc_vmm_to_preserve_size_bytes(
        const std::initializer_list<Xbyak_aarch64::VReg> &vmm_to_preserve)
        const {

    return std::accumulate(vmm_to_preserve.begin(), vmm_to_preserve.end(),
            std::size_t(0u),
            [](std::size_t accum, const Xbyak_aarch64::VReg &vmm) {
                return accum + cpu_isa_traits<isa>::vlen;
            });
}

template <cpu_isa_t isa>
size_t register_preserve_guard_t<isa>::stack_space_occupied() const {
    constexpr static size_t reg64_size = 8;
    const size_t stack_space_occupied
            = vmm_to_preserve_size_bytes_ + reg64_stack_.size() * reg64_size;

    return stack_space_occupied;
};

template <cpu_isa_t isa>
conditional_register_preserve_guard_t<
        isa>::conditional_register_preserve_guard_t(bool condition_to_be_met,
        jit_generator *host,
        std::initializer_list<Xbyak_aarch64::XReg> reg64_to_preserve,
        std::initializer_list<Xbyak_aarch64::VReg> vmm_to_preserve)
    : register_preserve_guard_t<isa> {condition_to_be_met
                    ? register_preserve_guard_t<isa> {host, reg64_to_preserve,
                            vmm_to_preserve}
                    : register_preserve_guard_t<isa> {nullptr, {}, {}}} {};

template class register_preserve_guard_t<sve_512>;
template class register_preserve_guard_t<sve_256>;
template class register_preserve_guard_t<sve_128>;
template class conditional_register_preserve_guard_t<sve_512>;
template class conditional_register_preserve_guard_t<sve_256>;
template class conditional_register_preserve_guard_t<sve_128>;

} // namespace injector_utils
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
