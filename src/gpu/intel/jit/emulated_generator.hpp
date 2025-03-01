/*******************************************************************************
 * Copyright 2024-2025 Intel Corporation
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

#ifndef GPU_INTEL_JIT_EMULATED_GENERATOR_HPP
#define GPU_INTEL_JIT_EMULATED_GENERATOR_HPP

// Must be included before emulation.hpp
#include "ngen/ngen.hpp"

#include "gpu/intel/compute/device_info.hpp"
#include "gpu/intel/jit/codegen/register_allocator.hpp"
#include "gpu/intel/jit/emulation.hpp"
#include "gpu/intel/jit/generator.hpp"
#include "ngen/ngen_core.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

template <gpu_gen_t hw>
class emulated_generator_t : public generator_t<hw> {
    friend struct EmulationImplementation;

protected:
    NGEN_FORWARD_OPENCL(hw);

public:
    emulated_generator_t(const compute::device_info_t &device_info,
            const std::string &name, const debug_config_t &debug_config)
        : generator_t<hw>(debug_config)
        , ra_(hw, name)
        , emu_strategy(hw, device_info.stepping_id()) {}

protected:
    reg_allocator_t ra_;

private:
    EmulationStrategy emu_strategy;

protected:
    reg_allocator_t &ra() { return ra_; }

    void emov(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0);
    void emov(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::Immediate &src0);
    void emul(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, const ngen::RegData &src1);
    void emul(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, const ngen::Immediate &src1);
    void eadd(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, const ngen::RegData &src1);
    void eadd(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, const ngen::Immediate &src1);
    void emad(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, const ngen::RegData &src1,
            const ngen::RegData &src2);
    void emad(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, const ngen::RegData &src1,
            const ngen::Immediate &src2);
    void eadd3(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, const ngen::RegData &src1,
            const ngen::Immediate &src2);
    void eadd3(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, const ngen::RegData &src1,
            const ngen::RegData &src2);

private:
    enum class OperandType { dst, src0, src1, src2 };

    // ngen operand support
    bool supports_operand(const ngen::InstructionModifier &mod,
            const ngen::RegData &rd, OperandType op_type);
    ngen::DataType exec_type(const ngen::DataType &src);
    ngen::DataType exec_type(
            const ngen::DataType &src, const ngen::DataType &src1);
    ngen::DataType exec_type(const ngen::DataType &src,
            const ngen::DataType &src1, const ngen::DataType &src2);

    bool supports_exectype(
            const ngen::RegData &dst, const ngen::DataType &execType);

    // Query for arch-specific (but instruction-agnostic) support of operands and their combination
    bool supports_signature(const ngen::InstructionModifier &mod,
            const ngen::RegData &dst, const ngen::RegData &src);
    bool supports_signature(const ngen::InstructionModifier &mod,
            const ngen::RegData &dst, const ngen::Immediate &src);
    bool supports_signature(const ngen::InstructionModifier &mod,
            const ngen::RegData &dst, const ngen::RegData &src0,
            const ngen::RegData &src1);
    bool supports_signature(const ngen::InstructionModifier &mod,
            const ngen::RegData &dst, const ngen::RegData &src0,
            const ngen::Immediate &src1);
    bool supports_signature(const ngen::InstructionModifier &mod,
            const ngen::RegData &dst, const ngen::RegData &src0,
            const ngen::RegData &src1, const ngen::RegData &src2);
    bool supports_signature(const ngen::InstructionModifier &mod,
            const ngen::RegData &dst, const ngen::RegData &src0,
            const ngen::RegData &src1, const ngen::Immediate &src2);
};

// clang-format off
#define FORWARD_EMULATION(hw) \
reg_allocator_t &ra() { return emulated_generator_t<hw>::ra(); } \
void emov(const ngen::InstructionModifier &mod, const ngen::RegData &dst, const ngen::RegData &src0) { emulated_generator_t<hw>::emov(mod, dst, src0); } \
void emov(const ngen::InstructionModifier &mod, const ngen::RegData &dst, const ngen::Immediate &src0) { emulated_generator_t<hw>::emov(mod, dst, src0); } \
void emul(const ngen::InstructionModifier &mod, const ngen::RegData &dst, const ngen::RegData &src0, const ngen::RegData &src1) { emulated_generator_t<hw>::emul(mod, dst, src0, src1); } \
void emul(const ngen::InstructionModifier &mod, const ngen::RegData &dst, const ngen::RegData &src0, const ngen::Immediate &src1) { emulated_generator_t<hw>::emul(mod, dst, src0, src1); } \
void eadd(const ngen::InstructionModifier &mod, const ngen::RegData &dst, const ngen::RegData &src0, const ngen::RegData &src1) { emulated_generator_t<hw>::eadd(mod, dst, src0, src1); } \
void eadd(const ngen::InstructionModifier &mod, const ngen::RegData &dst, const ngen::RegData &src0, const ngen::Immediate &src1) { emulated_generator_t<hw>::eadd(mod, dst, src0, src1); } \
void emad(const ngen::InstructionModifier &mod, const ngen::RegData &dst, const ngen::RegData &src0, const ngen::RegData &src1, const ngen::RegData &src2) { emulated_generator_t<hw>::emad(mod, dst, src0, src1, src2); } \
void emad(const ngen::InstructionModifier &mod, const ngen::RegData &dst, const ngen::RegData &src0, const ngen::RegData &src1, const ngen::Immediate &src2) { emulated_generator_t<hw>::emad(mod, dst, src0, src1, src2); } \
void eadd3(const ngen::InstructionModifier &mod, const ngen::RegData &dst, const ngen::RegData &src0, const ngen::RegData &src1, const ngen::RegData &src2) { emulated_generator_t<hw>::eadd3(mod, dst, src0, src1, src2); } \
void eadd3(const ngen::InstructionModifier &mod, const ngen::RegData &dst, const ngen::RegData &src0, const ngen::RegData &src1, const ngen::Immediate &src2) { emulated_generator_t<hw>::eadd3(mod, dst, src0, src1, src2); }
// clang-format on

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_INTEL_JIT_EMULATED_GENERATOR_HPP
