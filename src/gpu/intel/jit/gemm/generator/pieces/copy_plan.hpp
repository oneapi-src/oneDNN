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

#ifndef GEMMSTONE_GUARD_COPY_PLAN_HPP
#define GEMMSTONE_GUARD_COPY_PLAN_HPP

#include <array>
#include <cstdint>
#include <functional>
#include <vector>

#include "internal/ngen_includes.hpp"
#include "internal/utils.hpp"

#include "internal/namespace_start.hxx"


class CopyPlan;

struct CopyOperand
{
    int16_t grf = 0;
    uint8_t offset = 0;
    int8_t stride = 1;
    uint8_t inVS = 0, inW = 0;                          // Incoming region parameters.
    ngen::DataType type = ngen::DataType::invalid;
    ngen::DataType range = ngen::DataType::invalid;
    enum : uint8_t {GRF, Immediate, Flag, Null} kind = Null;
    bool temp = false;                                  // Operand is a temporary?
    bool overwrite = false;                             // Operand can be trashed?
    bool overwriteStride = false;                       // Padding area between strides can be trashed?
    bool neg = false;                                   // Negate (-) rd operator
    bool abs = false;                                   // Absolute Value abs() rd operator
    bool inv = false;                                   // Invert (~) rd operator
    uint64_t value = 0;                                 // Immediate value, or temporary index

    bool isNull() const { return kind == Null; }
    operator bool() const { return !isNull(); }
    bool operator!() const { return isNull(); }

    int byteOffset() const { return offset * getBytes(type); }
    int absByteOffset(ngen::HW hw) const { return byteOffset() + ngen::GRF::bytes(hw) * grf; }

    int byteStride() const { return stride * getBytes(type); }

    ngen::RegData ngen() const;
    ngen::Immediate ngenImmediate() const;
    ngen::FlagRegister ngenFlag() const;

    CopyOperand() {}
    CopyOperand(ngen::RegData rd);
    CopyOperand(ngen::Immediate imm) : type(imm.getType()), kind(Immediate), value(imm) {}
    CopyOperand(int imm) : CopyOperand(ngen::Immediate(imm)) {}

    CopyOperand operator-() const;
    CopyOperand operator~() const;

};

struct CopyInstruction
{
    ngen::Opcode op;
    uint8_t ctrl;
    int simd = 0;
    int16_t cnumMin, cnumMax;
    uint16_t phase = 0, spread = 0;
    CopyOperand dst, src0, src1, src2, flag;
    ngen::ConditionModifier cmod = ngen::ConditionModifier::none;
    bool atomic = false, sat = false;
    int16_t cnumSub = 0;

    void invalidate()       { simd = 0; }
    bool isInvalid()  const { return (simd == 0); }

    bool hasCMod()    const { return cmod != ngen::ConditionModifier::none; }

    void moveToIntegerPipe();

    ngen::InstructionModifier ngenModifiers() const;

    template <typename Generator>
    inline void execute(Generator &g);

};

struct CopyTemporary
{
    friend class CopyPlan;

    int bytes = 0, align = 0, offset = 0;
    bool flag = false;
    int16_t cnumMin = 0x7FFF;
    int16_t cnumMax = -1;
    int assignment = -1;

    explicit CopyTemporary(int bytes_, int align_, int offset_ = 0)
            : bytes(bytes_), align(align_), offset(offset_) {}

    static CopyTemporary createFlag(int bits = 16);

protected:
    void usedBy(const CopyInstruction &i) {
        cnumMin = std::min(cnumMin, i.cnumMin);
        cnumMax = std::max(cnumMax, i.cnumMax);
    }

private:
    CopyTemporary() {}
};

class CopyPlan
{
public:
    using GRFAllocator = std::function<void(int count, ngen::GRFRange &range)>;
    using FlagAllocator = std::function<void(int bytes, ngen::FlagRegister &flag)>;

    CopyPlan(ngen::HW hw_, bool systolicAvailable_) : hw(hw_), systolicAvailable(systolicAvailable_) {}

    CopyInstruction &append(CopyInstruction &&i);
    CopyInstruction &append(ngen::Opcode op, int simd, const CopyOperand &dst, const CopyOperand &src0, const CopyOperand &src1 = CopyOperand(), const CopyOperand &src2 = CopyOperand());
    CopyInstruction &append(ngen::Opcode op, int simd, ngen::InstructionModifier mod, const CopyOperand &dst, const CopyOperand &src0, const CopyOperand &src1 = CopyOperand(), const CopyOperand &src2 = CopyOperand());
    CopyInstruction &append(int phase, ngen::Opcode op, int simd, const CopyOperand &dst, const CopyOperand &src0, const CopyOperand &src1 = CopyOperand(), const CopyOperand &src2 = CopyOperand());
    CopyInstruction &append(int phase, ngen::Opcode op, int simd, ngen::InstructionModifier mod, const CopyOperand &dst, const CopyOperand &src0, const CopyOperand &src1 = CopyOperand(), const CopyOperand &src2 = CopyOperand());
    CopyInstruction &appendDestructiveMov(int simd, const CopyOperand &dst, const CopyOperand &src0, bool overwriteStride = false);
    CopyInstruction &appendDestructiveMov(int simd, ngen::InstructionModifier mod, const CopyOperand &dst, const CopyOperand &src0, bool overwriteStride = false);

    void transform();
    void materializeTemps(const GRFAllocator &grfAllocator, const FlagAllocator &flagAllocator);

    template <typename Generator>
    inline void execute(Generator &g);

    int tempFlagBytes() const;

protected:
    ngen::HW hw;
    bool systolicAvailable;
    std::vector<CopyInstruction> insns, newInsns;
    std::vector<CopyTemporary> temps;

    enum class SortType {
        PhaseOnly, Register, SourceOrder
    };

    CopyOperand newTemp(ngen::DataType type, int elems, int stride, int align = 0, int offset = 0);
    CopyOperand newFlag(int bits = 16);

    CopyInstruction &split(CopyInstruction &i, bool sequenced = true);
    template <int n>
    std::array<CopyInstruction*, n> splitMultiple(CopyInstruction &i);
    CopyInstruction &join(CopyInstruction &i1, CopyInstruction &i2);
    void mergeChanges();

    void copyThrough(CopyInstruction &i, ngen::DataType type, int stride = 0, bool strideOff0 = false);
    void restrideSrc0(CopyInstruction &i, int stride, bool strideOff0 = false);
    void restrideDst(CopyInstruction &i, int stride, bool strideOff0 = false);

    void repositionSrc(CopyInstruction &i, int n, int stride, int offset);
    void repositionDst(CopyInstruction &i, int stride, int offset);

    void checkNoSubbytes();
    void collapseCNums();

    void distributePhases();
    void split2DRegions();
    void planTypeConversions();
    void planEarlyInt4Upconversions();
    void planEmulatedHalveFloat(CopyInstruction &i);
    void planSmallUWToHF(CopyInstruction &i);
    void planBToHF(CopyInstruction &i);
    void planS4ToHF(CopyInstruction &i);
    void planEmulatedE3M0ToHF(CopyInstruction &i);
    void planEmulatedF4E2M1ToHF(CopyInstruction &i);
    void planEmulatedHFToF4E2M1(CopyInstruction &i);
    void planInt4Upconversion(CopyInstruction &i);
    void planEmulatedHF8ToHF(CopyInstruction &i);
    void planEmulatedHFToHF8(CopyInstruction &i);
    void planFP8SIMD1Mov(CopyInstruction &i);
    void planEmulatedFP8E8M0ToHF(CopyInstruction &i);
    void legalizeSIMD(bool initial = false);
    void legalizeRegions();
    void legalizeNegation();
    void legalizeImmediateTypes();
    void sort(SortType type);
    void optimizeZip();
    void optimizeZipAdjacent();
    void optimizeWidenIntegers();
    void optimizeConcatenate(bool initial = false);
    void optimizeWriteCombine();
    void optimizeWriteSpread();
    void optimizeIntegerDownconvert();
    void optimizeSaturate();
};


template <typename Generator>
void CopyPlan::execute(Generator &g)
{
    for (auto &i: insns) i.execute(g);
}

template <typename Generator>
void CopyInstruction::execute(Generator &g)
{
#define UNARY_OP_CASE(o)                                                            \
    case ngen::Opcode::o:                                                           \
        if (src0.kind == CopyOperand::Immediate)                                    \
            g.o(ngenModifiers(), dst.ngen(), src0.ngenImmediate());                 \
        else                                                                        \
            g.o(ngenModifiers(), dst.ngen(), src0.ngen());                          \
        break;
#define BINARY_OP_CASE(o)                                                           \
    case ngen::Opcode::o:                                                           \
        if (src1.kind == CopyOperand::Immediate)                                    \
            g.o(ngenModifiers(), dst.ngen(), src0.ngen(), src1.ngenImmediate());    \
        else                                                                        \
            g.o(ngenModifiers(), dst.ngen(), src0.ngen(), src1.ngen());             \
        break;
#define TERNARY_OP_CASE(o)                                                          \
    case ngen::Opcode::o:                                                           \
        if (src0.kind == CopyOperand::Immediate) {                                  \
            if (src2.kind == CopyOperand::Immediate)                                \
                g.o(ngenModifiers(), dst.ngen(), src0.ngenImmediate(), src1.ngen(), src2.ngenImmediate()); \
            else                                                                    \
                g.o(ngenModifiers(), dst.ngen(), src0.ngenImmediate(), src1.ngen(), src2.ngen()); \
        } else {                                                                    \
            if (src2.kind == CopyOperand::Immediate)                                \
                g.o(ngenModifiers(), dst.ngen(), src0.ngen(), src1.ngen(), src2.ngenImmediate()); \
            else                                                                    \
                g.o(ngenModifiers(), dst.ngen(), src0.ngen(), src1.ngen(), src2.ngen()); \
        }                                                                           \
        break;
#define BFN_OP_CASE(o)                                                              \
    case ngen::Opcode::o:                                                           \
        if (src0.kind == CopyOperand::Immediate) {                                  \
            if (src2.kind == CopyOperand::Immediate)                                \
                g.o(ngenModifiers(), ctrl, dst.ngen(), src0.ngenImmediate(), src1.ngen(), src2.ngenImmediate()); \
            else                                                                    \
                g.o(ngenModifiers(), ctrl, dst.ngen(), src0.ngenImmediate(), src1.ngen(), src2.ngen()); \
        } else {                                                                    \
            if (src2.kind == CopyOperand::Immediate)                                \
                g.o(ngenModifiers(), ctrl, dst.ngen(), src0.ngen(), src1.ngen(), src2.ngenImmediate()); \
            else                                                                    \
                g.o(ngenModifiers(), ctrl, dst.ngen(), src0.ngen(), src1.ngen(), src2.ngen()); \
        }                                                                           \
        break;

    switch (op) {
        UNARY_OP_CASE(mov)
        BINARY_OP_CASE(add)
        BINARY_OP_CASE(mul)
        BINARY_OP_CASE(cmp)
        BINARY_OP_CASE(and_)
        BINARY_OP_CASE(or_)
        BINARY_OP_CASE(xor_)
        BINARY_OP_CASE(shl)
        BINARY_OP_CASE(shr)
        BINARY_OP_CASE(asr)
        BINARY_OP_CASE(sel)
        TERNARY_OP_CASE(mad)
        TERNARY_OP_CASE(csel)
        BFN_OP_CASE(bfn)
        default: stub("Unsupported opcode");
    }

#undef UNARY_OP_CASE
#undef BINARY_OP_CASE
#undef TERNARY_OP_CASE
#undef BFN_OP_CASE
}

#include "internal/namespace_end.hxx"

#endif /* header guard */
