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

#ifndef CPU_X64_UNI_RESAMPLING_KERNEL_HPP
#define CPU_X64_UNI_RESAMPLING_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_resampling_pd.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"
#include "cpu/x64/utils/jit_io_helper.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

static bcast_set_t get_supported_bcast_strategies() {
    return {broadcasting_strategy_t::scalar, broadcasting_strategy_t::per_oc,
            broadcasting_strategy_t::per_oc_spatial};
}

template <cpu_isa_t isa>
struct jit_uni_resampling_kernel : public jit_generator {

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_resampling)

    jit_uni_resampling_kernel(
            const jit_resampling_conf_t conf, const memory_desc_t *dst_md);

    virtual ~jit_uni_resampling_kernel() = default;

protected:
    using Xmm = Xbyak::Xmm;
    using Ymm = Xbyak::Ymm;
    using Zmm = Xbyak::Zmm;
    using Opmask = Xbyak::Opmask;
    using Reg64 = Xbyak::Reg64;

    using Vmm = typename cpu_isa_traits<isa>::Vmm;

    constexpr int vmm_idx(int idx) const {
        return (cpu_isa_traits<avx>::n_vregs - 1) - idx;
    }

    void apply_sum(const int data_idx, const bool is_tail);

    void apply_postops(const int data_idx, const bool is_tail,
            const Reg64 *reg_c = nullptr);

    void nearest_ncsp_format();
    void nearest_c_oriented_format();
    void linear_ncsp_format();
    void linear_c_oriented_format();

    void generate() override;

    // Used only for avx and if c tail is present.
    const Vmm vmm_tail_mask_ = Vmm(0);
    // Used only for avx2 and if ncsp format is present.
    // Vgatherdps always gets data using a conditional mask.
    // This register contains all bits set to 1, allowing
    // to get the maximum number of values available to the register
    const Vmm vmm_full_mask_ = Vmm(1);
    const Vmm vmm_src_ = Vmm(2);
    const Vmm vmm_weights_ = Vmm(3);
    const Vmm vmm_indices_ = Vmm(4);
    const Vmm vmm_sum_scale_ = Vmm(7);
    const Vmm vmm_tmp_ = Vmm(8);
    const Vmm vmm_post_op_helper_ = Vmm(9);

    const Opmask &k_tail_mask_ = k3;
    const Opmask &k_full_mask_ = k4;

    const Reg64 &reg_tmp_ = rax;
    const Reg64 &reg_dst_ = rbx;
    const Reg64 &reg_work_ = rdx;
    const Reg64 &reg_indices_ = rsi;
    const Reg64 &reg_c_offset = rbp;
    const Reg64 &reg_param = abi_param1;
    const Reg64 &reg_weights = abi_not_param1;
    const Reg64 &reg_src_ = r8;
    const Reg64 &reg_aux_src_0_ = r9;
    const Reg64 &reg_aux_src_1_ = r10;
    const Reg64 &reg_aux_src_2_ = r11;
    const Reg64 &reg_tmp1_ = r15;

    // Registers which are used only for linear algorithm
    // and for channel oriented formats.
    // Meaning of shortcuts:
    // f - front, b - back
    // t - top, b - bottom
    // l - left, r - right
    // Example:
    // src_ftl_ - source tensor data for the front top left corner
    // reg_src_ftl_ - register which contains address of source
    //                tensor data for the front top left corner
    const Vmm weight_left_ = Vmm(1);
    const Vmm weight_right_ = Vmm(2);
    const Vmm weight_top_ = Vmm(3);
    const Vmm weight_bottom_ = Vmm(4);
    const Vmm weight_front_ = Vmm(5);
    const Vmm weight_back_ = Vmm(6);
    const Vmm src_ftl_ = Vmm(vmm_idx(0));
    const Vmm src_ftr_ = Vmm(vmm_idx(1));
    const Vmm src_fbl_ = Vmm(vmm_idx(2));
    const Vmm src_fbr_ = Vmm(vmm_idx(3));
    const Vmm src_btl_ = Vmm(vmm_idx(4));
    const Vmm src_btr_ = Vmm(vmm_idx(5));
    const Vmm src_bbl_ = Vmm(vmm_idx(6));
    const Vmm src_bbr_ = Vmm(vmm_idx(7));

    const Reg64 &reg_src_ftl_ = reg_src_;
    const Reg64 &reg_src_ftr_ = reg_aux_src_0_;
    const Reg64 &reg_src_fbl_ = reg_aux_src_1_;
    const Reg64 &reg_src_fbr_ = reg_aux_src_2_;
    const Reg64 &reg_src_btl_ = r12;
    const Reg64 &reg_src_btr_ = r13;
    const Reg64 &reg_src_bbl_ = r14;
    const Reg64 &reg_src_bbr_ = r15;

    jit_resampling_conf_t conf_;
    io::jit_io_helper_t<Vmm> io_;
    std::unique_ptr<injector::jit_uni_postops_injector_t<isa>>
            postops_injector_;
};
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
