/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef GPU_INTEL_JIT_JIT_REDUCTION_INJECTOR_HPP
#define GPU_INTEL_JIT_JIT_REDUCTION_INJECTOR_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/utils.hpp"

#include "gpu/intel/jit/codegen/register_allocator.hpp"
#include "gpu/intel/jit/emulation.hpp"
#include "gpu/intel/jit/jit_generator.hpp"
#include "gpu/intel/jit/ngen/ngen_core.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

inline bool jit_reduction_injector_f32_is_supported(alg_kind_t alg) {
    using namespace alg_kind;
    return utils::one_of(alg, reduction_sum, reduction_mean, reduction_max,
            reduction_min, reduction_mul);
}

template <gpu_gen_t hw>
struct jit_reduction_injector_f32 {
    jit_reduction_injector_f32(jit_generator<hw> &host, alg_kind_t alg,
            reg_allocator_t &ra_, int stepping_id)
        : emu_strategy(hw, stepping_id), alg_(alg), h(host), ra(ra_) {
        assert(jit_reduction_injector_f32_is_supported(alg_));
    }

    // src_ptr: GRF whose 1st qword subregister holds the first address to be loaded from
    // acc: Potentially uninitialized GRFRange to store values in
    // stride: Number of elements to increment the pointer by between iterations
    // iters: Number of reduction iterations
    void compute(const ngen::GRF &src_ptr, const ngen::GRFRange &acc,
            dim_t stride, dim_t iters);

private:
    void initialize(int simd, const ngen::GRF &reg);
    // Load data from a contiguous range in global memory into a contiguous
    // range of registers (block load)
    void eload(const ngen::GRFRange &dst, const ngen::GRF &base_src_addr);

    // Emulation functions
    void emov(jit_generator<hw> &host, const ngen::InstructionModifier &mod,
            const ngen::RegData &dst, const ngen::Immediate &src0);
    void emov(jit_generator<hw> &host, const ngen::InstructionModifier &mod,
            const ngen::RegData &dst, const ngen::RegData &src0);
    void eadd(jit_generator<hw> &host, const ngen::InstructionModifier &mod,
            const ngen::RegData &dst, const ngen::RegData &src0,
            const ngen::Immediate &src1);
    void eadd(jit_generator<hw> &host, const ngen::InstructionModifier &mod,
            const ngen::RegData &dst, const ngen::RegData &src0,
            const ngen::RegData &src1);
    void emul(jit_generator<hw> &host, const ngen::InstructionModifier &mod,
            const ngen::RegData &dst, const ngen::RegData &src0,
            const ngen::Immediate &src1);
    void emul(jit_generator<hw> &host, const ngen::InstructionModifier &mod,
            const ngen::RegData &dst, const ngen::RegData &src0,
            const ngen::RegData &src1);
    EmulationStrategy emu_strategy;

    const alg_kind_t alg_;
    jit_generator<hw> &h;
    reg_allocator_t &ra;

    void sum_fwd(int simd, const ngen::GRF &acc, const ngen::GRF &val);
    void max_fwd(int simd, const ngen::GRF &acc, const ngen::GRF &val);
    void min_fwd(int simd, const ngen::GRF &acc, const ngen::GRF &val);
    void mul_fwd(int simd, const ngen::GRF &acc, const ngen::GRF &val);
};

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_INTEL_JIT_JIT_REDUCTION_INJECTOR_HPP
