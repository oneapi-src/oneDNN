
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

#ifndef CPU_X64_PRELU_JIT_PRELU_UTILS_HPP
#define CPU_X64_PRELU_JIT_PRELU_UTILS_HPP

#include <memory>
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct bf16_emulation_t;

namespace prelu {

enum class bcast {
    full,
    per_oc_blocked,
    per_oc_n_spatial_c,
    per_oc_n_c_spatial,
    unsupported
};

bcast get_bcast_type(
        const memory_desc_wrapper &lhs, const memory_desc_wrapper &rhs);
cpu_isa_t get_supported_isa();
int get_vlen(const cpu_isa_t &isa) noexcept;
int get_n_vregs(const cpu_isa_t &isa) noexcept;

template <typename Vmm>
class jit_prelu_io_helper {
public:
    jit_prelu_io_helper(jit_generator *host, const cpu_isa_t &isa,
            const data_type_t &data_type, std::size_t tail_size,
            const Xbyak::Opmask &tail_opmask, const Vmm &tail_vmm_mask,
            const Xbyak::Reg64 &reg_tmp);
    ~jit_prelu_io_helper();
    void prepare_tail_mask();
    void broadcast(const Xbyak::Address &src_addr, const Vmm &dst_vmm);
    void load(const Xbyak::Address &src_addr, const Vmm &dst_vmm, bool tail);
    void store(const Vmm &src_vmm, const Xbyak::Address &dst_addr, bool tail);

private:
    void load_tail(const Xbyak::Address &src_addr, const Vmm &dst_vmm);
    void store_tail(const Vmm &src_vmm, const Xbyak::Address &dst_addr);

    jit_generator *host_;
    const cpu_isa_t isa_;
    const data_type_t data_type_;
    const std::size_t tail_size_;
    const Xbyak::Opmask tail_opmask_;
    const Vmm tail_vmm_mask_;
    const Xbyak::Reg64 reg_tmp_;
    const bool bf16_supported_;
    const std::unique_ptr<bf16_emulation_t> bf16_emu_;
};

} // namespace prelu
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
