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

#include "gpu/intel/jit/emulated_generator.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

using namespace ngen;

bool is_floating_point(const DataType &dt) {
    return utils::one_of(dt, DataType::f, DataType::hf, DataType::bf,
            DataType::df, DataType::vf);
}

template <typename ngen::HW hw>
void emulated_generator_t<hw>::emov(const InstructionModifier &mod,
        const RegData &dst, const RegData &src0) {
    EmulationImplementation::emov(*this, mod, dst, src0, emu_strategy);
}

template <typename ngen::HW hw>
void emulated_generator_t<hw>::emov(const InstructionModifier &mod,
        const RegData &dst, const Immediate &src0) {
    EmulationImplementation::emov(*this, mod, dst, src0, emu_strategy);
}
// TODO: Change EmulationState register allocation so it can be handled
// by the EmulationImplementation directly, instead of maintaining allocation
// for the entire lifetime of the EmulationState. This would eliminate overeager
// register allocation when using several injectors which each have emulation.
template <typename ngen::HW hw>
void emulated_generator_t<hw>::emul(const InstructionModifier &mod,
        const RegData &dst, const RegData &src0, const RegData &src1) {
    EmulationState state;
    state.temp[0] = ra_.alloc();
    state.temp[1] = ra_.alloc();
    EmulationImplementation::emul(
            *this, mod, dst, src0, src1, emu_strategy, state);
    ra_.release(state.temp[0]);
    ra_.release(state.temp[1]);
}

template <typename ngen::HW hw>
void emulated_generator_t<hw>::emul(const InstructionModifier &mod,
        const RegData &dst, const RegData &src0, const Immediate &src1) {
    EmulationState state;
    state.temp[0] = ra_.alloc();
    state.temp[1] = ra_.alloc();
    EmulationImplementation::emul(
            *this, mod, dst, src0, src1, emu_strategy, state);
    ra_.release(state.temp[0]);
    ra_.release(state.temp[1]);
}

template <typename ngen::HW hw>
void emulated_generator_t<hw>::eadd(const InstructionModifier &mod,
        const RegData &dst, const RegData &src0, const RegData &src1) {
    if (!supports_operand(mod, src1, OperandType::src1)) {
        // src1 does not support bytes
        gpu_assert(src1.getType() == ngen::DataType::b)
                << "Expected src1 to be b";
        ngen::Subregister src1_w = ra_.alloc_sub(ngen::DataType::w);
        emov(mod, src1_w, src1);
        gpu_assert(supports_operand(mod, src1_w, OperandType::src1))
                << "Unable to emulate eadd";
        eadd(mod, dst, src0, src1_w);
        ra_.release(src1_w);
        return;
    }

    // Use regular eadd implementation
    EmulationState state;
    state.temp[0] = ra_.alloc();
    state.temp[1] = ra_.alloc();
    EmulationImplementation::eadd(
            *this, mod, dst, src0, src1, emu_strategy, state);
    ra_.release(state.temp[0]);
    ra_.release(state.temp[1]);
}

template <typename ngen::HW hw>
void emulated_generator_t<hw>::eadd(const InstructionModifier &mod,
        const RegData &dst, const RegData &src0, const Immediate &src1) {
    EmulationState state;
    state.temp[0] = ra_.alloc();
    state.temp[1] = ra_.alloc();
    EmulationImplementation::eadd(
            *this, mod, dst, src0, src1, emu_strategy, state);
    ra_.release(state.temp[0]);
    ra_.release(state.temp[1]);
}

template <typename ngen::HW hw>
void emulated_generator_t<hw>::emad(const InstructionModifier &mod,
        const RegData &dst, const RegData &src0, const RegData &src1,
        const RegData &src2) {
    // mad is only supported for dw/w types
    auto supported = [](const RegData &data) -> bool {
        return EmulationImplementation::isDW(data)
                || EmulationImplementation::isW(data);
    };
    bool src2_supported = EmulationImplementation::isW(src2);
    if (supported(dst) && supported(src0) && supported(src1)
            && src2_supported) {
        gpu_assert(supports_signature(mod, dst, src0, src1, src2))
                << "Invalid instruction";
        mad(mod, dst, src0, src1, src2);
    } else {
        // emulate with separate mul/add
        if (src0 == dst) {
            Subregister tmp = ra_.alloc_sub(dst.getType());
            emul(mod, tmp, src1, src2);
            eadd(mod, dst, tmp, src0);
            ra_.release(tmp);
        } else {
            emul(mod, dst, src1, src2);
            eadd(mod, dst, dst, src0);
        }
    }
}

template <typename ngen::HW hw>
void emulated_generator_t<hw>::emad(const InstructionModifier &mod,
        const RegData &dst, const RegData &src0, const RegData &src1,
        const Immediate &src2) {
    auto supported = [](const RegData &data) -> bool {
        return EmulationImplementation::isDW(data)
                || EmulationImplementation::isW(data);
    };
    bool src2_supported = EmulationImplementation::isW(src2);
    bool imm_supported = getBytes(src2.getType()) <= 2 && src2_supported;
    bool mad_supported = supported(dst) && supported(src0) && supported(src1)
            && imm_supported;
    if (mad_supported) {
        gpu_assert(supports_signature(mod, dst, src0, src1, src2))
                << "Invalid instruction";
        mad(mod, dst, src0, src1, src2);
    } else {
        // emulate with separate mul/add
        if (src0 == dst) {
            Subregister tmp = ra_.alloc_sub(dst.getType());
            emul(mod, tmp, src1, src2);
            eadd(mod, dst, tmp, src0);
            ra_.release(tmp);
        } else {
            emul(mod, dst, src1, src2);
            eadd(mod, dst, dst, src0);
        }
    }
}

template <typename ngen::HW hw>
void emulated_generator_t<hw>::eadd3(const InstructionModifier &mod,
        const RegData &dst, const RegData &src0, const RegData &src1,
        const Immediate &src2) {
    // add3 only supports dw/w types - emulate other options with 2 adds
    auto supported = [](const RegData &data) -> bool {
        return EmulationImplementation::isDW(data)
                || EmulationImplementation::isW(data);
    };
    bool src2_supported = utils::one_of(src2.getType(), DataType::uw,
            DataType::w, DataType::ud, DataType::d);
    if (supported(dst) && supported(src0) && supported(src1)
            && src2_supported) {
        gpu_assert(supports_signature(mod, dst, src0, src1, src2))
                << "Invalid instruction";
        add3(mod, dst, src0, src1, src2);
    } else {
        eadd(mod, dst, src0, src1);
        eadd(mod, dst, dst, src2);
    }
}

template <typename ngen::HW hw>
void emulated_generator_t<hw>::eadd3(const InstructionModifier &mod,
        const RegData &dst, const RegData &src0, const RegData &src1,
        const RegData &src2) {
    // add3 only supports dw/w types - emulate other options with 2 adds
    auto supported = [](const RegData &data) -> bool {
        return EmulationImplementation::isDW(data)
                || EmulationImplementation::isW(data);
    };
    if (supported(dst) && supported(src0) && supported(src1)
            && supported(src2)) {
        gpu_assert(supports_signature(mod, dst, src0, src1, src2))
                << "Unsupported signature";
        add3(mod, dst, src0, src1, src2);
    } else {
        if (src2 == dst) {
            Subregister tmp = ra_.alloc_sub(dst.getType());
            eadd(mod, tmp, src0, src1);
            eadd(mod, dst, tmp, src2);
            ra_.release(tmp);
        } else {
            eadd(mod, dst, src0, src1);
            eadd(mod, dst, dst, src2);
        }
    }
}

template <typename ngen::HW hw>
DataType emulated_generator_t<hw>::exec_type(const DataType &src) {
    if (is_floating_point(src)) return src;
    if (isSigned(src)) return src;

    // convert unsigned to signed
    switch (src) {
        case DataType::ub: return DataType::b;
        case DataType::uw: return DataType::w;
        case DataType::ud: return DataType::d;
        case DataType::uq: return DataType::q;
        default: break;
    }
    return DataType::invalid;
}

template <typename ngen::HW hw>
DataType emulated_generator_t<hw>::exec_type(
        const DataType &src0, const DataType &src1) {
    if (is_floating_point(src0) || is_floating_point(src1)) {
        if (src0 != src1) return DataType::invalid;
        return src0;
    }
    return exec_type(getBytes(src0) > getBytes(src1) ? src0 : src1);
}

template <typename ngen::HW hw>
DataType emulated_generator_t<hw>::exec_type(
        const DataType &src0, const DataType &src1, const DataType &src2) {
    if (is_floating_point(src0) || is_floating_point(src1)
            || is_floating_point(src2)) {
        if (!utils::everyone_is(src0, src1, src2)) return DataType::invalid;
        return src0;
    }
    DataType dt = src0;
    if (getBytes(src1) > getBytes(dt)) dt = src1;
    if (getBytes(src2) > getBytes(dt)) dt = src2;
    return exec_type(dt);
}

template <typename ngen::HW hw>
bool emulated_generator_t<hw>::supports_exectype(
        const RegData &dst, const DataType &dt) {
    // dst must be aligned to the execution data type's size
    int size_ratio = getBytes(dt) / getBytes(dst.getType());
    return size_ratio == 0 || dst.getHS() % size_ratio == 0;
}

#define REQUIRE(stmt) \
    if (!(stmt)) return false

template <typename ngen::HW hw>
bool emulated_generator_t<hw>::supports_operand(
        const InstructionModifier &mod, const RegData &rd, OperandType opType) {
    int execSize = mod.getExecSize();

    // execution size has an upper bound
    int maxExecSize = hw < HW::XeHPC ? 64 : 128;
    int dtSize = getBytes(rd.getType());
    REQUIRE(execSize * dtSize <= maxExecSize);

    // regioning parameter restrictions
    int HS = rd.getHS();
    int VS = rd.getVS();
    int W = rd.getWidth();
    if (opType == OperandType::dst) {
        REQUIRE(HS == 0);
    } else {
        // Generic source operand requirements
        REQUIRE(execSize >= W);
        if (execSize == W && HS != 0) REQUIRE(VS == W * HS);
        if (W == 1) {
            REQUIRE(HS == 0);
            if (execSize == 1) REQUIRE(VS == 0 && HS == 0);
        }
        if (VS == 0 && HS == 0) REQUIRE(W == 1);

        // Elements within a row cannot cross GRF boundaries
        int byteOff = rd.getByteOffset();
        REQUIRE(byteOff + W * dtSize <= GRF::bytes(hw));

        // Specific source operand requirements
        if (opType == OperandType::src1) {
            // src1 doesn't support byte types (b/ub)
            REQUIRE(rd.getBytes() > 1);
        }
        if (opType == OperandType::src2) {
            // src2 doesn't support byte types (b/ub)
            REQUIRE(rd.getBytes() > 1);
            REQUIRE(rd.isScalar() || rd.getByteOffset() % 64 == 0);
        }
    }

    return true;
}

template <typename ngen::HW hw>
bool emulated_generator_t<hw>::supports_signature(
        const InstructionModifier &mod, const RegData &dst,
        const RegData &src) {
    // Check operand-specific support
    REQUIRE(supports_operand(mod, dst, OperandType::dst));
    REQUIRE(supports_operand(mod, src, OperandType::src0));

    // Check dst/exec dt for support
    DataType exec_dt = exec_type(src.getType());
    gpu_assert(exec_dt != DataType::invalid) << "Invalid execution data type";
    REQUIRE(supports_exectype(dst, exec_dt));

    return true;
}

template <typename ngen::HW hw>
bool emulated_generator_t<hw>::supports_signature(
        const InstructionModifier &mod, const RegData &dst,
        const Immediate &src) {
    // Check operand-specific support
    REQUIRE(supports_operand(mod, dst, OperandType::dst));

    // Check dst/exec dt for support
    DataType exec_dt = exec_type(src.getType());
    gpu_assert(exec_dt != DataType::invalid) << "Invalid execution data type";
    REQUIRE(supports_exectype(dst, exec_dt));

    return true;
}

template <typename ngen::HW hw>
bool emulated_generator_t<hw>::supports_signature(
        const InstructionModifier &mod, const RegData &dst, const RegData &src0,
        const RegData &src1) {
    // Check operand-specific support
    REQUIRE(supports_operand(mod, dst, OperandType::dst));
    REQUIRE(supports_operand(mod, src0, OperandType::src0));
    REQUIRE(supports_operand(mod, src1, OperandType::src1));

    // All source operands have to be floating point or not
    bool fp_case = utils::everyone_is(true, is_floating_point(src0.getType()),
            is_floating_point(src1.getType()));
    bool int_case = utils::everyone_is(false, is_floating_point(src0.getType()),
            is_floating_point(src1.getType()));
    REQUIRE(fp_case || int_case);

    // Check dst/exec dt for support
    DataType exec_dt = exec_type(src0.getType(), src1.getType());
    gpu_assert(exec_dt != DataType::invalid) << "Invalid execution data type";
    REQUIRE(supports_exectype(dst, exec_dt));

    return true;
}

template <typename ngen::HW hw>
bool emulated_generator_t<hw>::supports_signature(
        const InstructionModifier &mod, const RegData &dst, const RegData &src0,
        const Immediate &src1) {
    // Check operand-specific support
    REQUIRE(supports_operand(mod, dst, OperandType::dst));
    REQUIRE(supports_operand(mod, src0, OperandType::src0));

    // All source operands have to be floating point or not
    bool fp_case = utils::everyone_is(true, is_floating_point(src0.getType()),
            is_floating_point(src1.getType()));
    bool int_case = utils::everyone_is(false, is_floating_point(src0.getType()),
            is_floating_point(src1.getType()));
    REQUIRE(fp_case || int_case);

    // Check dst/exec dt for support
    DataType exec_dt = exec_type(src0.getType(), src1.getType());
    gpu_assert(exec_dt != DataType::invalid) << "Invalid execution data type";
    REQUIRE(supports_exectype(dst, exec_dt));

    return true;
}

template <typename ngen::HW hw>
bool emulated_generator_t<hw>::supports_signature(
        const InstructionModifier &mod, const RegData &dst, const RegData &src0,
        const RegData &src1, const RegData &src2) {
    // Check operand-specific support
    REQUIRE(supports_operand(mod, dst, OperandType::dst));
    REQUIRE(supports_operand(mod, src0, OperandType::src0));
    REQUIRE(supports_operand(mod, src1, OperandType::src1));
    REQUIRE(supports_operand(mod, src2, OperandType::src2));

    // All source operands have to be floating point or not
    bool fp_case = utils::everyone_is(true, is_floating_point(src0.getType()),
            is_floating_point(src1.getType()),
            is_floating_point(src2.getType()));
    bool int_case = utils::everyone_is(false, is_floating_point(src0.getType()),
            is_floating_point(src1.getType()),
            is_floating_point(src2.getType()));
    REQUIRE(fp_case || int_case);

    // Check dst/exec dt for support
    DataType exec_dt
            = exec_type(src0.getType(), src1.getType(), src2.getType());
    gpu_assert(exec_dt != DataType::invalid) << "Invalid execution data type";
    REQUIRE(supports_exectype(dst, exec_dt));

    return true;
}

template <typename ngen::HW hw>
bool emulated_generator_t<hw>::supports_signature(
        const InstructionModifier &mod, const RegData &dst, const RegData &src0,
        const RegData &src1, const Immediate &src2) {
    // Check operand-specific support
    REQUIRE(supports_operand(mod, dst, OperandType::dst));
    REQUIRE(supports_operand(mod, src0, OperandType::src0));
    REQUIRE(supports_operand(mod, src1, OperandType::src1));

    // All source operands have to be floating point or not
    bool fp_case = utils::everyone_is(true, is_floating_point(src0.getType()),
            is_floating_point(src1.getType()));
    bool int_case = utils::everyone_is(false, is_floating_point(src0.getType()),
            is_floating_point(src1.getType()));
    REQUIRE(fp_case || int_case);

    // Check dst/exec dt for support
    DataType exec_dt = exec_type(src0.getType(), src1.getType());
    gpu_assert(exec_dt != DataType::invalid) << "Invalid execution data type";
    REQUIRE(supports_exectype(dst, exec_dt));

    return true;
}

#undef REQUIRE

REG_GEN9_ISA(template class emulated_generator_t<gpu_gen9>);
REG_GEN11_ISA(template class emulated_generator_t<gpu_gen11>);
REG_XELP_ISA(template class emulated_generator_t<gpu_xe_lp>);
REG_XEHP_ISA(template class emulated_generator_t<gpu_xe_hp>);
REG_XEHPG_ISA(template class emulated_generator_t<gpu_xe_hpg>);
REG_XEHPC_ISA(template class emulated_generator_t<gpu_xe_hpc>);
REG_XE2_ISA(template class emulated_generator_t<gpu_xe2>);

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
