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

#ifndef CPU_X64_PRELU_JIT_PRELU_BACKWARD_KERNEL_HPP
#define CPU_X64_PRELU_JIT_PRELU_BACKWARD_KERNEL_HPP

#include <memory>
#include "cpu/cpu_prelu_pd.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/prelu/jit_prelu_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

class jit_prelu_backward_kernel_t : public jit_generator {
public:
    static jit_prelu_backward_kernel_t *create(const cpu_prelu_bwd_pd_t *pd);

    struct call_params_t {
        const void *src = nullptr, *weights = nullptr, *dst_diff = nullptr;
        void *src_diff = nullptr, *weights_diff = nullptr;
        size_t compute_data_size = 0u;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_prelu_backward_kernel_t)

    void operator()(jit_prelu_backward_kernel_t::call_params_t *params) {
        jit_generator::operator()(params);
    }
    size_t simd_w() const noexcept;
    prelu::bcast get_bcast() const noexcept;

protected:
    jit_prelu_backward_kernel_t(const cpu_prelu_bwd_pd_t *pd, int vlen);
    Xbyak::Address data_ptr(int arg_num, size_t offt = 0);

    const cpu_prelu_bwd_pd_t *pd_;
    const size_t simd_w_ = 0;
    const prelu::bcast bcast_;
    const size_t tail_size_ = 0u;
    const data_type_t data_type_;

private:
    void generate() override;
    void load_kernel_call_params();
    virtual void prepare_kernel_const_vars() = 0;
    virtual void compute_dst(int unrolling_factor, int tail) = 0;
    virtual size_t get_unrolling_factor() const = 0;

    size_t calc_tail_size() const noexcept;

    const Xbyak::Reg64 &reg_data_size_ = r8;
    const Xbyak::Reg64 &reg_offset_ = r9;
    const Xbyak::Reg64 &reg_src_ = r10;
    const Xbyak::Reg64 &reg_weights_ = r11;
    const Xbyak::Reg64 &reg_src_diff_ = r12;
    const Xbyak::Reg64 &reg_weights_diff_ = r13;
    const Xbyak::Reg64 &reg_dst_diff_ = r14;
};

template <typename Vmm>
class jit_uni_prelu_backward_kernel_t : public jit_prelu_backward_kernel_t {
public:
    jit_uni_prelu_backward_kernel_t(
            const cpu_prelu_bwd_pd_t *pd, const cpu_isa_t &isa);
    ~jit_uni_prelu_backward_kernel_t() override;

private:
    void prepare_kernel_const_vars() override;
    void compute_dst(int unrolling_factor, int tail) override;
    size_t get_unrolling_factor() const override;
    Vmm reserve_vmm();
    size_t get_number_reserved_vmms() const noexcept;
    size_t calc_unrolling_factor() const noexcept;
    Vmm get_compute_vmm(size_t base_idx, size_t unroll_group);

    const cpu_isa_t isa_;
    size_t number_vmms_reserved_const_vars_ = 0;
    const Vmm vmm_zeros_;
    const Vmm tail_vmm_mask_;
    const Vmm vmm_ones_;
    const Xbyak::Opmask &tail_opmask_ = k0;
    const Xbyak::Reg64 &reg_tmp_ = r15;
    static constexpr size_t number_vmm_single_compute_
            = std::is_same<Vmm, Xbyak::Zmm>::value ? 4 : 6u;
    const size_t unrolling_factor_ = 0;

    prelu::jit_prelu_io_helper<Vmm> io_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
