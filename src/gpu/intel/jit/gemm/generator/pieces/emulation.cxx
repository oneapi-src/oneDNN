/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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


#include "generator.hpp"
#include "ngen_object_helpers.hpp"

using namespace ngen;

#include "internal/namespace_start.hxx"


// Remove (sat) from an InstructionModifier.
static inline InstructionModifier unsaturated(InstructionModifier mod)
{
    if (mod.isSaturate())
        return mod ^ InstructionModifier::createSaturate();
    else
        return mod;
}

// Modify the signedness of an integer type.
static inline DataType withSignedness(DataType dt, bool signedType)
{
    switch (dt) {
        case DataType::b:
        case DataType::ub: return signedType ? DataType::b : DataType::ub;
        case DataType::w:
        case DataType::uw: return signedType ? DataType::w : DataType::uw;
        case DataType::d:
        case DataType::ud: return signedType ? DataType::d : DataType::ud;
        case DataType::q:
        case DataType::uq: return signedType ? DataType::q : DataType::uq;
        default: return dt;
    }
}

// Three-argument add.
template <HW hw>
template <typename DT, typename S0, typename S2>
void BLASKernelGenerator<hw>::eadd3(const InstructionModifier &mod, const RegData &dst, const S0 &src0, const RegData &src1, const S2 &src2)
{
    if ((hw >= HW::XeHP) && !(dst.getOffset() & 1))
        add3<DT>(mod, dst, src0, src1, src2);
    else {
        add<DT>(mod, dst, src1, src0);
        add<DT>(mod, dst, dst, src2);
    }
}

template <HW hw>
template <typename S0>
void BLASKernelGenerator<hw>::ecsel(const InstructionModifier &mod, const InstructionModifier &cmod, const FlagRegister &flag,
                                    const RegData &dst,  const S0 &src0,
                                    const RegData &src1, const RegData &src2)
{
    if (hw == HW::Gen9 || dst.getByteOffset() & 7) {
        cmp(mod | cmod | flag, src2, 0);
        sel(mod | ~flag, dst, src1, src0);
    } else
        csel(mod | cmod | flag, dst, src0, src1, src2);
};

template <HW hw>
template <typename DT>
void BLASKernelGenerator<hw>::emov(const ngen::InstructionModifier &mod, ngen::RegData dst, ngen::RegData src0, const CommonStrategy &strategy, CommonState &state)
{
    EmulationImplementation::applyDefaultType<DT>(dst);
    EmulationImplementation::applyDefaultType<DT>(src0);

    if (dst.getType() == DataType::tf32 && src0.getType() == DataType::tf32) {
        dst.setType(DataType::f);
        src0.setType(DataType::f);
    }

    if (hw >= HW::XeHP && one_of(src0.getType(), DataType::hf, DataType::f, DataType::bf)
            && src0.getType() == dst.getType()
            && ((src0.getHS() != dst.getHS()) || (src0.getOffset() != dst.getOffset()) || (src0.getHS() != 1 && getBytes(src0.getType()) == 2))) {
        moveToIntPipe(mod.getExecSize(), dst);
        moveToIntPipe(mod.getExecSize(), src0);
    }

    if (dst.getType() == DataType::f && src0.getType() == DataType::bf) {
        dst.setType(DataType::ud);
        src0.setType(DataType::uw);
        shl(mod, dst, src0, 16);
    } else if (!strategy.systolicAvailable && dst.getType() == DataType::bf && src0.getType() == DataType::f) {
        // Emulated f32->bf16 RTNE conversion.
        auto flag = state.emulate.flag;
        if (!flag.isValid()) stub();
        dst.setType(DataType::uw);
        src0.setType(DataType::ud);
        add(mod, src0, src0, -0x8000);
        and_(mod | nz | flag, null.ud(), src0, 0x1FFFF);
        mov(mod, dst, EmulationImplementation::highWord(src0));
        // add(mod, src0, src0, 0x8000);       // Preserve src0 -- if nondestructive mov -- not needed
        add(mod | flag, dst, dst, 1);
    } else
        EmulationImplementation::emov(*this, mod, dst, src0, strategy.emulate);
}

template <HW hw>
template <typename DT>
void BLASKernelGenerator<hw>::eadd(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const CommonStrategy &strategy, CommonState &state)
{
    if (dst.getType() == DataType::f && src0.getType() == DataType::f && src1.getType() == DataType::bf && src1.getHS() != 1) {
        GRF alloced, temp = state.emulate.temp[0];
        if (temp.isInvalid())
            temp = alloced = state.ra.alloc();

        auto src1UW = src1;
        src1UW.setType(DataType::uw);
        mov(mod, temp.uw(0)(1), src1UW);
        add(mod, dst, src0, temp.bf(0)(1));

        state.ra.safeRelease(alloced);
    } else
        EmulationImplementation::eadd<DT>(*this, mod, dst, src0, src1, strategy.emulate, state.emulate);
}

template <HW hw>
template <typename S0>
void BLASKernelGenerator<hw>::emad(const InstructionModifier &mod, const RegData &dst, const S0 &src0, RegData src1, RegData src2, const CommonStrategy &strategy, CommonState &state)
{
    bool sub = false;
    if (src1.getNeg()) {
        src1 = -src1;
        sub = !sub;
    };
    if (src2.getNeg()) {
        src2 = -src2;
        sub = !sub;
    }
    emad(mod, dst, src0, src1, src2, strategy, state, sub);
}

template <HW hw>
template <typename S0, typename S2>
void BLASKernelGenerator<hw>::emad(const InstructionModifier &mod, const RegData &dst, const S0 &src0, const RegData &src1, const S2 &src2, const CommonStrategy &strategy, CommonState &state, bool sub)
{
    auto dstType = dst.getType();
    if ((hw >= HW::Gen10 && !sub && !(dst.getByteOffset() & 7) && !one_of(dstType, DataType::q, DataType::uq) && !one_of(src2.getType(), DataType::d, DataType::ud))
            || one_of(dstType, DataType::hf, DataType::f, DataType::df)) {
        mad(mod, dst, src0, src1, src2);
    } else {
        auto ttype = withSignedness(dst.getType(), isSigned(src1.getType()) || isSigned(src2.getType()));
        RegData temp;
        Subregister tempSub;
        GRFRange tempRange;
        if (mod.getExecSize() == 1)
            temp = tempSub = state.ra.alloc_sub(ttype);
        else {
            tempRange = state.ra.alloc_range(div_up(mod.getExecSize(), elementsPerGRF(hw, ttype)));
            temp = tempRange[0].retype(ttype);
        }

        emul(unsaturated(mod), temp, src1, src2, strategy, state);
        eadd(mod, dst, sub ? -temp : temp, src0, strategy, state);

        state.ra.safeRelease(tempSub);
        state.ra.safeRelease(tempRange);
    }
}

template <HW hw>
template <typename S0>
void BLASKernelGenerator<hw>::emad(const InstructionModifier &mod, const RegData &dst, const S0 &src0, const RegData &src1, const Immediate &src2, const CommonStrategy &strategy, CommonState &state)
{
    emad(mod, dst, src0, src1, src2, strategy, state, false);
}

template <HW hw>
template <typename S0>
void BLASKernelGenerator<hw>::emad(const InstructionModifier &mod, const RegData &dst, const S0 &src0, const RegData &src1, int32_t src2, const CommonStrategy &strategy, CommonState &state)
{
    auto dstType = dst.getType();
    if (src2 == 0)
        emov(mod, dst, src0, strategy, state);
    else if (src2 == 1)
        eadd(mod, dst, src1, src0, strategy, state);
    else if (hw >= HW::Gen10 && !(dst.getByteOffset() & 7) && (src2 >= -0x8000 && src2 < 0x10000) && !one_of(dstType, DataType::q, DataType::uq)) {
        mad(mod, dst, src0, src1, src2);
    } else {
        auto ttype = isSigned(src1.getType()) ? DataType::d : DataType::ud;
        Subregister tempScalar;
        GRFRange tempGRFs;
        RegData temp;
        if (mod.getExecSize() == 1)
            temp = tempScalar = state.ra.alloc_sub(ttype);
        else {
            tempGRFs = state.ra.alloc_range(2);
            temp = tempGRFs[0].retype(ttype);
        }
        emulConstant(unsaturated(mod), temp, src1, src2, strategy, state);
        eadd(mod, dst, temp, src0, strategy, state);
        state.ra.safeRelease(tempScalar);
        state.ra.safeRelease(tempGRFs);
    }
}

template <HW hw>
template <typename S0>
void BLASKernelGenerator<hw>::eaddScaled(const InstructionModifier &mod, const RegData &dst, const S0 &src0, const RegData &src1, Type src2, const CommonStrategy &strategy, CommonState &state)
{
    if (src2.isInt4()) {
        auto tmpRange = state.ra.alloc_range(2);
        auto tmp = tmpRange[0].retype(src1.getType());
        eshr(mod, tmp, src1, 1, strategy, state);
        eadd(mod, dst, tmp, src0, strategy, state);
        state.ra.safeRelease(tmpRange);
    } else
        emad(mod, dst, src0, src1, src2.size(), strategy, state);
}

template <HW hw>
template <typename DT>
void BLASKernelGenerator<hw>::emulConstant(const ngen::InstructionModifier &mod, const ngen::RegData &dst, const ngen::RegData &src0, Type src1, const CommonStrategy &strategy, const CommonState &state)
{
    if (src1.isInt4())
        eshr<DT>(mod, dst, src0, 1, strategy, state);
    else
        emulConstant<DT>(mod, dst, src0, src1.size(), strategy, state);
}

template <HW hw>
template <typename DT>
void BLASKernelGenerator<hw>::emath(const InstructionModifier &mod, MathFunction fc, const RegData &dst, const RegData &src0, const GEMMStrategy &strategy, CommonState &state)
{
    if (hw == HW::XeHP && strategy.systolic && mod.getExecSize() <= 8) {
        // Workaround for DPAS + SIMD8 EM hang: use SIMD16 arithmetic.
        auto mod16 = mod;
        mod16.setExecSize(16);

        auto temp = state.ra.alloc_range(2);
        auto tt = temp[0].retype(src0.getType());

        mov(mod.getExecSize(), tt, src0);
        math(mod16, fc, tt, tt);
        mov(mod.getExecSize(), dst, tt);

        state.ra.safeRelease(temp);
    } else
        math(mod, fc, dst, src0);
}

template <HW hw>
void BLASKernelGenerator<hw>::ejmpi(InstructionModifier mod, Label &dst)
{
    if (hw >= HW::XeHPC && mod.getPredCtrl() == PredCtrl::anyv && !mod.isPredInv()) {
        mod.setPredCtrl(PredCtrl::Normal);
        jmpi(mod, dst);
        auto flag = mod.getFlagReg();
        flag.setBase(flag.getBase() ^ 1);
        mod.setFlagReg(flag);
        jmpi(mod, dst);
    } else
        jmpi(mod, dst);
}


#include "internal/namespace_end.hxx"
