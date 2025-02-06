/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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


#include "copy_plan.hpp"
#include "internal/utils.hpp"
#include "ngen_object_helpers.hpp"

#include <algorithm>

#include "internal/namespace_start.hxx"


using namespace ngen;
using namespace ngen::utils;


/********************/
/* Utility routines */
/********************/

static bool isBitwise(Opcode op) {
    return one_of(op, Opcode::mov, Opcode::and_, Opcode::or_, Opcode::xor_);
}
static bool isBroadcast(const CopyOperand &op) {
    return (op.kind != op.GRF) || (op.stride == 0);
}

// Check if a CopyOperand spans multiple registers.
static bool multiGRF(HW hw, const CopyInstruction &i, const CopyOperand &op)
{
    if (op.kind != op.GRF) return false;
    return elementsToBytes(op.offset + op.stride * (i.simd - 1), op.type) >= GRF::bytes(hw);
}

// Check if two CopyOperands may overlap.
static bool mayOverlap(HW hw, const CopyInstruction &i, const CopyOperand &op1, const CopyOperand &op2)
{
    if (!op1 || !op2) return false;
    if (op1.kind != op2.kind) return false;
    if (op1.temp != op2.temp) return false;
    if (op1.temp && op1.value != op2.value) return false;

    if (op1.kind == CopyOperand::Flag) return op1.grf == op2.grf;
    if (op1.kind != CopyOperand::GRF) return false;

    int bs1 = op1.byteStride();
    int bs2 = op2.byteStride();
    int boStart1 = op1.absByteOffset(hw);
    int boStart2 = op2.absByteOffset(hw);
    int boEnd1 = boStart1 + bs1 * i.simd;
    int boEnd2 = boStart2 + bs2 * i.simd;

    if (boEnd2 <= boStart1 || boEnd1 <= boStart2) return false;
    if (bs1 != bs2) return true;

    int slotA = boStart1 % bs1;
    int slotB = boStart2 % bs1;
    auto dbytesA = getBytes(op1.type);
    auto dbytesB = getBytes(op2.type);

    if (slotA > slotB) {
        std::swap(slotA, slotB);
        std::swap(dbytesA, dbytesB);
    }

    if (slotB + dbytesB > bs1) return true;
    if (slotA + dbytesA > slotB) return true;
    return false;
}

// Return a DataType representing the potential numerical range of
//  a conversion from one data type to another.
static DataType conversionRange(DataType from, DataType to)
{
    return (getBytes(from) < getBytes(to)) ? from : to;
}

// Check if one data type (dt1) is a subset of another (dt2).
static bool isSubsetOf(DataType dt1, DataType dt2)
{
    if (dt1 == DataType::invalid || dt2 == DataType::invalid) return false;
    if (dt1 == dt2) return true;
    if (isFP(dt1) && isInt(dt2)) return false;
    if (isW(dt1) && dt2 == DataType::tf32) return false;
    if (is4(dt1) && (isB(dt2) || dt2 == DataType::hf8)) return true;
    if (dt1 == DataType::s4 && dt2 == DataType::bf8) return true;
    return getBytes(dt1) < getBytes(dt2);
}


/***********************/
/* CopyOperand methods */
/***********************/

CopyOperand::CopyOperand(RegData rd)
        : grf(rd.getBase()), offset(rd.getLogicalOffset()),
          stride(rd.getHS()), type(rd.getType()), kind(GRF),
          overwrite(false), overwriteStride(false), neg(rd.getNeg()), abs(rd.getAbs()), inv(false)
{
    if (rd.getAbs()) stub("Unsupported modifier");
    if (rd.getVS() != 0 || rd.getWidth() != 0)
        if (rd.getVS() != rd.getWidth() * stride)
            inVS = rd.getVS(), inW = rd.getWidth();
}

CopyOperand CopyOperand::operator-() const
{
    auto clone = *this;
    clone.neg = !clone.neg;
    return clone;
}

CopyOperand CopyOperand::operator~() const
{
    auto clone = *this;
    clone.inv = !clone.inv;
    return clone;
}

// Convert a GRF CopyOperand to an nGEN object.
RegData CopyOperand::ngen() const
{
    if (kind == Null)
        return ngen::NullRegister().retype(type);
    if (kind != GRF || temp) stub("Invalid operation");

    RegData rd = ngen::GRF(grf).sub(offset, type)(stride);
    if (abs) rd = ngen::abs(rd);
    if (neg) rd = -rd;
    if (inv) rd = ~rd;

    return rd;
}

// Convert an immediate CopyOperand to an nGEN object.
Immediate CopyOperand::ngenImmediate() const
{
    if (kind != Immediate) stub("Invalid operation");
    ngen::Immediate imm = value;
    imm.setType(type);
    return imm;
}

// Convert a flag CopyOperand to an nGEN object.
FlagRegister CopyOperand::ngenFlag() const
{
    if (kind != Flag || temp) stub("Invalid operation");
    auto flag = FlagRegister::createFromIndex(grf + (offset >> 4));
    flag.setType(type);
    return flag;
}

/***************************/
/* CopyInstruction methods */
/***************************/

// Move an instruction to the integer pipe if possible.
void CopyInstruction::moveToIntegerPipe()
{
    auto &st = src0.type, &dt = dst.type;

    if (op != Opcode::mov) return;
    if (asSigned(st) != asSigned(dt)) return;

    switch (getBytes(st)) {
        case 1: st = dt = isInt4(st) ? DataType::u4 : DataType::ub; break;
        case 2: st = dt = DataType::uw; break;
        case 4: st = dt = DataType::ud; break;
        case 8:
            if (src0.stride == 1 && dst.stride == 1) {
                st = dt = DataType::ud;
                simd *= 2;
            } else
                st = dt = DataType::uq;
            break;
        default: break;
    }
}

// Retrieve nGEN instruction modifiers for an instruction.
InstructionModifier CopyInstruction::ngenModifiers() const
{
    InstructionModifier mod = simd;
    mod |= cmod;
    if (flag) {
        mod |= flag.ngenFlag();
        mod |= InstructionModifier::createChanOff(flag.offset & 0xF);
    }
    if (atomic) mod |= ThreadCtrl::Atomic;
    if (sat) mod |= InstructionModifier::createSaturate();
    return mod;
}

/********************/
/* CopyPlan methods */
/********************/

// Run all transformation passes on a CopyPlan.
void CopyPlan::transform()
{
    distributePhases();
    planEarlyInt4Upconversions();
    split2DRegions();

    sort(SortType::Register);

    optimizeIntegerDownconvert();
    optimizeZip();
    optimizeZipAdjacent();
    optimizeWidenIntegers();
    optimizeConcatenate(true);

    legalizeSIMD(true);
    planTypeConversions();

    sort(SortType::Register);

    optimizeZip();
    optimizeZipAdjacent();
    optimizeWidenIntegers();
    optimizeConcatenate();

    legalizeSIMD();

    sort(SortType::Register);     /* for nicer temporary numbering; not required */

    legalizeRegions();
    legalizeNegation();
    optimizeSaturate();

    sort(SortType::SourceOrder);

    optimizeWriteCombine();
    optimizeWriteSpread();

    sort(SortType::PhaseOnly);

    legalizeImmediateTypes();
}



/* Basic operations on copy plans. */
CopyInstruction &CopyPlan::append(CopyInstruction &&i)
{
    i.cnumMin = i.cnumMax = int16_t(insns.size());
    insns.push_back(std::move(i));
    return insns.back();
}

CopyInstruction &CopyPlan::append(int phase, Opcode op, int simd, InstructionModifier mod, const CopyOperand &dst, const CopyOperand &src0, const CopyOperand &src1, const CopyOperand &src2)
{
    bool sat = mod.isSaturate();

    InstructionModifier mmod{};
    if (sat) mmod |= InstructionModifier::createSaturate();

    if (mod.getAll() != mmod.getAll()) stub("Unsupported instruction modifiers");

    CopyInstruction i;
    i.op = op;
    i.simd = simd;
    i.dst = dst;
    i.src0 = src0;
    i.src1 = src1;
    i.src2 = src2;
    i.sat = sat;
    i.phase = phase;
    return append(std::move(i));
}

CopyInstruction &CopyPlan::append(int phase, Opcode op, int simd, const CopyOperand &dst, const CopyOperand &src0, const CopyOperand &src1, const CopyOperand &src2)
{
    return append(phase, op, simd, InstructionModifier{}, dst, src0, src1, src2);
}

CopyInstruction &CopyPlan::append(Opcode op, int simd, const CopyOperand &dst, const CopyOperand &src0, const CopyOperand &src1, const CopyOperand &src2)
{
    return append(0, op, simd, dst, src0, src1, src2);
}

CopyInstruction &CopyPlan::append(Opcode op, int simd, InstructionModifier mod, const CopyOperand &dst, const CopyOperand &src0, const CopyOperand &src1, const CopyOperand &src2)
{
    return append(0, op, simd, mod, dst, src0, src1, src2);
}

CopyInstruction &CopyPlan::appendDestructiveMov(int simd, const CopyOperand &dst, const CopyOperand &src0, bool overwriteStride)
{
    return appendDestructiveMov(simd, InstructionModifier{}, dst, src0, overwriteStride);
}

CopyInstruction &CopyPlan::appendDestructiveMov(int simd, InstructionModifier mod, const CopyOperand &dst, const CopyOperand &src0, bool overwriteStride)
{
    auto &i = append(Opcode::mov, simd, mod, dst, src0);
    i.src0.overwrite = true;
    i.src0.overwriteStride = overwriteStride;
    return i;
}

CopyOperand CopyPlan::newTemp(DataType type, int elems, int stride, int align, int offset)
{
    int grf = GRF::bytes(hw);
    auto bytes = elementsToBytes(elems * stride, type);

    if (align == 0) align = grf;
    int soffset = (offset / stride) * stride;

    if (soffset > 0 && soffset + bytes > grf)
        stub("Misaligned multi-GRF temporary");

    CopyOperand op{};
    op.grf = 0;
    op.type = type;
    op.offset = offset;
    op.stride = stride;
    op.kind = op.GRF;
    op.temp = true;
    op.value = temps.size();

    temps.emplace_back(bytes, align, offset);

    return op;
}

CopyOperand CopyPlan::newFlag(int bits)
{
    CopyOperand op{};
    op.grf = 0;
    op.kind = op.Flag;
    op.value = temps.size();
    op.temp = true;

    temps.push_back(CopyTemporary::createFlag(bits));

    return op;
}

CopyTemporary CopyTemporary::createFlag(int bits)
{
    CopyTemporary temp;
    if (bits > 32) stub();
    temp.bytes = (bits > 16) ? 4 : 2;
    temp.flag = true;
    return temp;
}

int CopyPlan::tempFlagBytes() const
{
    int bytes = 0;
    for (const auto &t: temps)
        if (t.flag)
            bytes += t.bytes;
    return bytes;
}

// Split an instruction into two.
//   If sequenced is true (default), the two instructions depend on each other
//   and should be spaced apart.
// After splitting, mergeChanges must be applied to incorporate the new instruction.
CopyInstruction &CopyPlan::split(CopyInstruction &i, bool sequenced)
{
    newInsns.emplace_back(i);
    auto &clone = newInsns.back();

    if (sequenced) {
        if (i.spread == 0) stub("Too many splits");
        i.spread >>= 1;
        clone.spread >>= 1;
        i.phase -= i.spread;
        clone.phase += i.spread;
    }

    return clone;
}

// Split an instruction into n instructions.
// After splitting, mergeChanges must be applied to incorporate the new instruction(s).
template <int n>
std::array<CopyInstruction*, n> CopyPlan::splitMultiple(CopyInstruction &i)
{
    std::array<CopyInstruction*, n> result;

    i.phase -= i.spread;
    i.spread /= n;
    i.phase += i.spread;

    newInsns.reserve(newInsns.size() + n - 1);
    result[0] = &i;
    for (int j = 1; j < n; j++) {
        newInsns.emplace_back(i);
        result[j] = &newInsns.back();
        result[j]->phase += 2 * j * i.spread;
    }

    return result;
}

// Join two instructions.
// The second instruction will be marked for removal, but not removed until
//   a call to mergeChanges.
CopyInstruction &CopyPlan::join(CopyInstruction &i1, CopyInstruction &i2)
{
    i1.cnumMin = std::min(i1.cnumMin, i2.cnumMin);
    i1.cnumMax = std::max(i1.cnumMax, i2.cnumMax);
    i2.invalidate();

    return i1;
}

// Update all pending instruction insertions/removals.
void CopyPlan::mergeChanges()
{
    insns.insert(insns.end(), newInsns.begin(), newInsns.end());
    newInsns.clear();

    for (auto iter = insns.begin(); iter != insns.end(); ) {
        if (iter->isInvalid())
            iter = insns.erase(iter);
        else
            iter++;
    }
}

// Add an intermediate copy through the given type.
//   If stride != 0, require the given stride for the intermediate result.
//   If strideOff0 == true, require the intermediate result to have offset % stride = 0.
void CopyPlan::copyThrough(CopyInstruction &i, DataType type, int stride, bool strideOff0)
{
    auto st = i.src0.type, dt = i.dst.type;
    auto sstride = i.src0.stride, dstride = i.dst.stride;
    auto ssize = getBytes(st), dsize = getBytes(dt), isize = getBytes(type);

    auto &i0 = i, &i1 = split(i);

    auto inplaceSrc = (stride == 0) ? (ssize >= isize && i.src0.overwrite)
                                        || (ssize * sstride >= isize && i.src0.overwriteStride)
                                    : (ssize * sstride == isize * stride && i.src0.overwrite)
                                        && (isize <= ssize || i.src0.overwriteStride);
    auto inplaceDst = (stride == 0) ? (dsize >= isize)
                                        || (dsize * dstride >= isize && i.dst.overwriteStride)
                                    : (dsize * dstride == isize * stride)
                                        && (isize <= dsize || i.dst.overwriteStride);

    if (strideOff0) {
        inplaceSrc &= i.src0.stride > 0 && (i.src0.offset % i.src0.stride) == 0;
        inplaceDst &= i.dst.stride > 0  && (i.dst.offset  % i.dst.stride)  == 0;
    }

    if (i.src1) inplaceDst &= !mayOverlap(hw, i, i.src1, i.dst);
    if (i.src2) inplaceDst &= !mayOverlap(hw, i, i.src2, i.dst);

    if (inplaceSrc && inplaceDst)
        inplaceSrc = isFP(st) && !isFP(dt);     /* prioritize in-place on floating point types */

    if (inplaceSrc) {
        // Convert src0 in place
        i0.op = Opcode::mov;
        i0.dst = i0.src0;
        i0.dst.offset = (i0.dst.offset * ssize) / isize;
        i0.dst.stride = (i0.dst.stride * ssize) / isize;
        i0.src1 = i0.src2 = CopyOperand();
        i1.src0 = i0.dst;
    } else if (inplaceDst) {
        // Convert dst in place
        i1.op = Opcode::mov;
        i1.src0 = i1.dst;
        i1.src0.offset = (i1.src0.offset * dsize) / isize;
        i1.src0.stride = (i1.src0.stride * dsize) / isize;
        i1.src1 = i1.src2 = CopyOperand();
        i0.dst = i1.src0;
    } else {
        // No space for in-place conversion -- create temporary.
        if (stride == 0)
            stride = std::max(1, ssize * sstride / isize);
        i0.op = Opcode::mov;
        int offset = 0;
        auto tryOffset = [&](const CopyOperand &op) {
            if (op.byteStride() == isize * stride) {
                auto bo = op.byteOffset();
                if (bo < GRF::bytes(hw) - stride * getBytes(type) * (i.simd - 1))
                    offset = bo / isize;
            }
        };
        if (isize <= dsize) tryOffset(i1.dst.byteOffset());
        if (isize <= ssize) tryOffset(i0.src0.byteOffset());
        if (type == DataType::hf)
            offset &= ~1;
        i0.dst = newTemp(type, i.simd, stride, 0, offset);
        i0.src1 = i0.src2 = CopyOperand();
        i1.src0 = i0.dst;
        i1.src0.overwriteStride = true;
    }
    i1.src0.overwrite = true;
    i0.dst.type = i1.src0.type = type;
    i0.moveToIntegerPipe();
    i1.moveToIntegerPipe();
    if (i0.op == Opcode::mov) {
        auto srange = i0.src0.range;
        if (srange == DataType::invalid)
            srange = i0.src0.type;
        i1.src0.range = conversionRange(srange, i0.dst.type);
    }

    if (isInt(st) && isInt(type) && isFP(dt) && getBytes(st) > getBytes(type))
        i0.sat = true;

    i0.cmod = ConditionModifier::none;
}

// Adjust stride on src0.
void CopyPlan::restrideSrc0(CopyInstruction &i, int stride, bool strideOff0)
{
    copyThrough(i, i.src0.type, stride, strideOff0);
}

// Adjust stride on dst.
void CopyPlan::restrideDst(CopyInstruction &i, int stride, bool strideOff0)
{
    copyThrough(i, i.dst.type, stride, strideOff0);
}

// Change src0/1/2 region.
void CopyPlan::repositionSrc(CopyInstruction &i, int n, int stride, int offset)
{
    if (n < 0 || n > 2) stub();
    auto CopyInstruction::* src = (n == 0) ? &CopyInstruction::src0 :
                                  (n == 1) ? &CopyInstruction::src1 :
                                             &CopyInstruction::src2;
    auto type = (i.*src).type;
    auto bytes = getBytes(type);

    bool inplaceDst = stride * bytes == i.dst.byteStride()
                   && offset * bytes == i.dst.byteOffset()
                   && (bytes <= getBytes(i.dst.type) || i.dst.overwriteStride);

    if (n != 0) inplaceDst &= !mayOverlap(hw, i, i.dst, i.src0);
    if (n != 1) inplaceDst &= !mayOverlap(hw, i, i.dst, i.src1);
    if (n != 2) inplaceDst &= !mayOverlap(hw, i, i.dst, i.src2);

    auto &i0 = i, &i1 = split(i);

    i0.op = Opcode::mov;
    i0.src0 = i0.*src;
    if (inplaceDst) {
        i0.dst.type = type;
        i0.dst.stride = stride;
        i0.dst.offset = offset;
    } else {
        i0.dst = newTemp(type, i.simd, stride, 0, offset);
        i0.dst.overwriteStride = true;
    }
    i0.dst.neg = i0.src0.neg;
    i0.src1 = i0.src2 = i0.flag = CopyOperand{};

    i1.*src = i0.dst;
    (i1.*src).overwrite = true;

    i0.cmod = ConditionModifier::none;
    i0.moveToIntegerPipe();
}

// Change dst region.
void CopyPlan::repositionDst(CopyInstruction &i, int stride, int offset)
{
    auto &i0 = i, &i1 = split(i);

    i0.dst = newTemp(i.dst.type, i.simd, stride, 0, offset);

    i1.op = Opcode::mov;
    i1.src0 = i0.dst;
    i1.src0.overwrite = true;
    i1.src0.overwriteStride = true;
    i1.src1 = i1.src2 = i1.flag = CopyOperand{};
    i1.cmod = ConditionModifier::none;
    i1.moveToIntegerPipe();
}

// Pass to split 2D regioned instructions into 1D regions.
void CopyPlan::split2DRegions()
{
    auto is2D = [](const CopyOperand &op) { return op.inVS || op.inW; };

    for (auto &i: insns) {
        if ((is2D(i.dst) && !is4Bit(i.dst.type)) || is2D(i.src1) || is2D(i.src2))
            stub("Unsupported 2D region");
        if (is2D(i.src0)){
            if (i.flag) stub("Unsupported predication");
            int w = i.src0.inW, vs = i.src0.inVS, hs = i.src0.stride;
            bool splitH = (w * w >= i.simd);
            int nsplit = splitH ? (i.simd / w) : w;
            i.simd /= nsplit;
            i.src0.stride = splitH ? hs : vs;
            i.src0.inVS = i.src0.inW = 0;
            i.src0.overwriteStride = false;
            for (int isplit = 1; isplit < nsplit; isplit++) {
                newInsns.emplace_back(i);
                auto &inew = newInsns.back();
                inew.src0.offset += (splitH ? vs : hs) * isplit;
                for (auto *op: {&inew.dst, &inew.src1, &inew.src2}) {
                    if (op->kind == CopyOperand::GRF) {
                        op->offset += op->stride * (splitH ? w : 1) * isplit;
                        if (!splitH) op->stride *= w;
                    }
                }
            }
            if (!splitH) for (auto *op: {&i.dst, &i.src1, &i.src2})
                op->stride *= w;
        }
    }

    mergeChanges();
}

// Pass to spread phases through phase space.
// Instructions with the same phase are assumed to be logically independent.
void CopyPlan::distributePhases()
{
    uint16_t nphase = 0;
    for (const auto &i: insns)
        nphase = std::max(nphase, i.phase);
    nphase++;

    uint16_t spread = 0x8000 / nphase;
    for (auto &i: insns) {
        i.spread = spread;
        i.phase = (2*i.phase + 1) * spread;
    }
}

// Pass to legalize type conversions.
void CopyPlan::planTypeConversions()
{
    bool rerun = false;

    for (auto &i: insns) {
        if (i.op != Opcode::mov) continue;

        auto &st = i.src0.type, &dt = i.dst.type;
        auto &srange = i.src0.range;

        if (asSigned(st) == asSigned(dt) && st != dt)
            dt = st;

        if (isInt4(st) && isInt(dt)) {
            planInt4Upconversion(i);
            rerun = true;
        } else if (st == Type::ngen_f4_e2m1() && dt == DataType::hf) {
            planEmulatedF4E2M1ToHF(i);
            rerun = true;
        } else if (dt == Type::ngen_f4_e2m1() && st == DataType::hf) {
            planEmulatedHFToF4E2M1(i);
            rerun = true;
        } else if (st == Type::ngen_f4_e3m0() && dt == DataType::hf) {
            planEmulatedE3M0ToHF(i);
            rerun = true;
        } else if (dt == Type::ngen_f4_e2m1()) {
            copyThrough(i, DataType::hf);
            rerun = true;
        } else if (st == Type::ngen_f4_e2m1()) {
            copyThrough(i, DataType::hf);
            rerun = true;
        } else if (st == DataType::u4 && dt == DataType::hf) {
            copyThrough(i, DataType::uw);
            rerun = true;
        } else if (st == DataType::s4 && dt == DataType::hf) {
            planS4ToHF(i);
            rerun = true;
        } else if (isInt4(st) && isFP(dt)) {
            copyThrough(i, DataType::hf, 1);
            rerun = true;
        } else if (isInt4(dt))
            stub("Unsupported move to int4");
        else if (isFP4(dt))
            stub("Unsupported move to FP4");
        else if (isB(st) && getBytes(dt) == 8)
            copyThrough(i, DataType::w);
        else if (getBytes(st) == 8 && isB(dt))
            copyThrough(i, DataType::w);
        else if (st == DataType::hf && dt == DataType::df)
            copyThrough(i, DataType::f);
        else if (st == DataType::df && dt == DataType::hf)
            copyThrough(i, DataType::f, 1);
        else if (st == DataType::hf && isQ(dt))
            copyThrough(i, DataType::d);
        else if (isQ(st) && dt == DataType::hf)
            copyThrough(i, DataType::d);
        else if (st == DataType::uw && dt == DataType::hf) {
            if (one_of(srange, DataType::u4, DataType::ub))
                planSmallUWToHF(i);
        } else if (st == DataType::ub && dt == DataType::hf) {
            copyThrough(i, DataType::uw);
            rerun = true;
        } else if (st == DataType::b && dt == DataType::hf)
            planBToHF(i);
        else if (st == DataType::f && dt == DataType::tf32) {
            if (hw < HW::XeHPC)
                stub("No emulation for tf32 rounding");
        } else if (st != DataType::tf32 && dt == DataType::tf32) {
            if (isSubsetOf(st, dt))
                dt = DataType::f;
            else
                copyThrough(i, DataType::f);
            rerun = true;
        } else if (st == DataType::tf32) {
            st = DataType::f;
            if (dt == DataType::tf32)
                dt = DataType::f;
            rerun = true;
        } else if (st == DataType::bf && dt == DataType::f) {
            i.op = Opcode::shl;
            i.dst.type = DataType::ud;
            i.src0.type = DataType::uw;
            i.src1 = 16;
        } else if (st == DataType::f && dt == DataType::bf) {
            if (isSubsetOf(i.src0.range, dt)) {
                i.op = Opcode::mov;
                i.src0.type = i.dst.type = DataType::uw;
                i.src0.offset *= 2;
                i.src0.stride *= 2;
                i.src0.offset++;
            } else if (!systolicAvailable)
                planEmulatedHalveFloat(i);
        } else if (st == DataType::bf && dt != DataType::bf) {
            copyThrough(i, DataType::f);
            rerun = true;
        } else if (st != DataType::bf && dt == DataType::bf) {
            copyThrough(i, DataType::f);
            rerun = true;
        } else if (st == DataType::bf8 && dt == DataType::hf) {
            i.op = Opcode::shl;
            i.dst.type = DataType::uw;
            i.src0.type = DataType::ub;
            i.src1 = 8;
        } else if (st == DataType::hf && dt == DataType::bf8) {
            if (isSubsetOf(i.src0.range, dt)) {
                i.op = Opcode::mov;
                i.src0.type = i.dst.type = DataType::ub;
                i.src0.offset *= 2;
                i.src0.stride *= 2;
                i.src0.offset++;
            } else if (hw < HW::XeHPC) {
                if (i.dst.stride == 1) {
                    restrideDst(i, 2);
                    rerun = true;
                } else
                    planEmulatedHalveFloat(i);
            }
        } else if (st == DataType::hf8 && dt == DataType::hf) {
            if (hw < HW::Xe3)
                planEmulatedHF8ToHF(i);
        } else if (st == DataType::hf && dt == DataType::hf8) {
            if (hw < HW::Xe3)
                planEmulatedHFToHF8(i);
        } else if (st == Type::ngen_f8_e8m0() && dt == DataType::hf) {
                planEmulatedFP8E8M0ToHF(i);
        } else if (st != dt && (isFP8(st) || isFP8(dt))) {
            copyThrough(i, DataType::hf, 1);
            rerun = true;
        } else if (st == dt)
            i.moveToIntegerPipe();
    }

    mergeChanges();
    if (rerun)
        planTypeConversions();
}

// uw->hf sequence when source range is uint10 or smaller.
void CopyPlan::planSmallUWToHF(CopyInstruction &i)
{
    if (i.src0.neg || i.sat || i.hasCMod()) return;

    auto ie = splitMultiple<2>(i);

    // Reinterpret as f16 denormal and multiply by 2^24 to scale to correct range.
    // Alternatively, second mul may be replaced by integer add.
    for (auto &inew: ie) {
        inew->op = Opcode::mul;
        inew->src1 = Immediate::hf(0x6C00);       // f16(2^12)
    }

    ie[0]->src0.type = DataType::hf;
    ie[1]->src0 = ie[0]->dst;
}

// b->hf sequence.
void CopyPlan::planBToHF(CopyInstruction &i)
{
    if (i.src0.neg || i.sat || i.hasCMod()) return;

    auto ie = splitMultiple<3>(i);

    // Copy to u16 and shift by 128.
    ie[0]->op = Opcode::add;
    ie[0]->dst.type = DataType::uw;
    ie[0]->src1 = 0x80;

    // Reinterpret as f16 denormal and scale to correct range,
    //   then undo offset.
    ie[1]->op = Opcode::mul;
    ie[1]->src0 = ie[1]->dst;
    ie[1]->src1 = Immediate::hf(0x6C00);       // f16(2^12)

    ie[2]->op = Opcode::mad;
    ie[2]->src0 = Immediate::hf(0xD800);       // -128
    ie[2]->src1 = ie[2]->dst;
    ie[2]->src2 = Immediate::hf(0x6C00);
    ie[2]->dst.range = ie[0]->src0.type;
}

// s4->hf sequence.
void CopyPlan::planS4ToHF(CopyInstruction &i)
{
    if (i.src0.neg || i.sat || i.hasCMod()) stub("Unsupported modifier");

    auto ie = splitMultiple<4>(i);

    bool preserveSrc = !i.src0.overwrite;
    auto ssrc = i.src0;
    if (!i.src0.overwrite)
        ssrc = newTemp(DataType::s4, i.simd, 1);

    // Shift by 8.
    ie[0]->op = Opcode::xor_;
    ie[0]->src0.type = DataType::ub;

    auto &ss = ie[0]->src0.stride;
    auto &so = ie[0]->src0.offset;
    if (ss == 1) {
        if (so & 1) stub("Misaligned s4 region");
        ie[0]->src1 = 0x88;
        ie[0]->simd /= 2;
    } else {
        ie[0]->src1 = (so & 1) ? 0x80 : 0x08;
        ss /= 2;
    }
    so /= 2;
    ie[0]->src1.type = DataType::ub;
    if (preserveSrc) {
        ie[0]->dst = ssrc;
        ie[0]->dst.type = DataType::ub;
        ie[0]->dst.offset /= 2;
    } else
        ie[0]->dst = ie[0]->src0;

    // Copy to u16.
    ie[1]->src0 = ssrc;
    ie[1]->src0.type = DataType::u4;
    ie[1]->dst.type = DataType::uw;

    // Reinterpret as f16 denormal and scale to correct range,
    //   then undo offset.
    ie[2]->op = Opcode::mul;
    ie[2]->src0 = ie[2]->dst;
    ie[2]->src1 = Immediate::hf(0x6C00);       // f16(2^12)

    ie[3]->op = Opcode::mad;
    ie[3]->src0 = Immediate::hf(0xC800);       // -8
    ie[3]->src1 = ie[3]->dst;
    ie[3]->src2 = Immediate::hf(0x6C00);
    ie[3]->dst.range = ie[0]->src0.type;
}

// hf->f4 sequence.
void CopyPlan::planEmulatedHFToF4E2M1(CopyInstruction &i)
{
    // Emulation sequence for mov y:f4 x:hf:
    //   and (nz)fN.N   x:uw                0x7C00        /* NaN/inf check */
    //   sel fN.N       t1:uw    x:uw       0x0
    //   sel (lt)fN.N   t2:hf    t1:uw      0x4600:hf     /* Clamp/round */
    //   mul t2:hf      t2:hf               0x400:hf
    //   and (nz)fN.N   null     t2:uw      0x200
    //   add            t2:uw    t2:uw      0x100
    //   shr            t1:uw    x:uw       0x8
    //   shr            t2:uw    t2:uw      0x5
    //   bfn.0xCA       t2:uw    t2:uw      t1:uw  0x8000 /* copy sign */
    //   and            t2:uw    t2:uw      0x00f0:uw     /* byte pack */
    //   shr            t2:uw<2> t2:uw<2>   0x4:uw
    //   or             t2:uw    t2.1:uw<2> t2:uw<2>
    //   mov            t2:ub    t2:ub<2>
    //   mov            y:uw     t2:ub
    //

    if (i.src0.neg || i.sat || i.hasCMod()) stub("Unsupported modifier");
    int simd = i.simd;

    auto ie = splitMultiple<14>(i);
    auto tmp = newTemp(DataType::ud, simd, 1);
    auto tmp2 = newTemp(DataType::ud, simd, 1);

    auto convFlag = newFlag(simd);
    auto ddst = CopyOperand(i.dst);
    auto invSrc = CopyOperand(i.src0);
    auto ssrc = CopyOperand(i.src0);
    invSrc.inv = true;

    // Check for NaN/infs.
    ie[0]->op = Opcode::and_;
    ie[0]->simd = simd;
    ie[0]->flag = convFlag;
    ie[0]->cmod = ConditionModifier::nz;
    ie[0]->dst = CopyOperand();
    ie[0]->dst.type = DataType::uw;
    ie[0]->src0 = invSrc;
    ie[0]->src0.type = DataType::uw;
    ie[0]->src1 = Immediate::uw(0x7c00);

    ie[1]->op = Opcode::sel;
    ie[1]->flag = convFlag;
    ie[1]->simd = simd;
    ie[1]->dst = tmp;
    ie[1]->dst.type = DataType::uw;
    ie[1]->src0 = ssrc;
    ie[1]->src0.type = DataType::uw;
    ie[1]->src1 = Immediate::uw(0x0);

    // Clamp and round.
    ie[2]->op = Opcode::sel;
    ie[2]->flag = convFlag;
    ie[2]->cmod = ConditionModifier::lt;
    ie[2]->simd = simd;
    ie[2]->dst = tmp2;
    ie[2]->dst.type = DataType::hf;
    ie[2]->src0 = tmp;
    ie[2]->src0.type = DataType::hf;
    ie[2]->src0.abs = true;
    ie[2]->src1 = Immediate::hf(0x4600);

    ie[3]->op = Opcode::mul;
    ie[3]->simd = simd;
    ie[3]->dst = tmp2;
    ie[3]->dst.type = DataType::hf;
    ie[3]->src0 = tmp2;
    ie[3]->src0.type = DataType::hf;
    ie[3]->src1 = Immediate::hf(0x0400);

    ie[4]->op = Opcode::and_;
    ie[4]->simd = simd;
    ie[4]->flag = convFlag;
    ie[4]->cmod = ConditionModifier::nz;
    ie[4]->dst = CopyOperand();
    ie[4]->dst.type = DataType::uw;
    ie[4]->src0 = tmp2;
    ie[4]->src0.type = DataType::uw;
    ie[4]->src1 = Immediate::uw(0x0200);

    ie[5]->op = Opcode::add;
    ie[5]->simd = simd;
    ie[5]->flag = convFlag;
    ie[5]->dst = tmp2;
    ie[5]->dst.type = DataType::uw;
    ie[5]->src0 = tmp2;
    ie[5]->src0.type = DataType::uw;
    ie[5]->src1 = Immediate::uw(0x0100);

    ie[6]->op = Opcode::shr;
    ie[6]->simd = simd;
    ie[6]->dst = tmp;
    ie[6]->dst.type = DataType::uw;
    ie[6]->src0 = ssrc;
    ie[6]->src0.type = DataType::uw;
    ie[6]->src1 = Immediate::uw(8);

    ie[7]->op = Opcode::shr;
    ie[7]->simd = simd;
    ie[7]->dst = tmp2;
    ie[7]->dst.type = DataType::uw;
    ie[7]->src0 = tmp2;
    ie[7]->src0.type = DataType::uw;
    ie[7]->src1 = Immediate::uw(5);

    // Restore sign.
    ie[8]->op = Opcode::bfn;
    ie[8]->dst = tmp2;
    ie[8]->dst.stride = 1;
    ie[8]->dst.type = DataType::uw;
    ie[8]->src0 = tmp2;
    ie[8]->src0.type = DataType::uw;
    ie[8]->src1 = tmp;
    ie[8]->src1.type = DataType::uw;
    ie[8]->src2 = 0x80;
    ie[8]->ctrl = 0xCA;

    // Pack into byte.
    ie[9]->op = Opcode::and_;
    ie[9]->simd = simd;
    ie[9]->dst = tmp2;
    ie[9]->dst.type = DataType::uw;
    ie[9]->src0 = tmp2;
    ie[9]->src0.type = DataType::uw;
    ie[9]->src1 = Immediate(0x00f0);

    ie[10]->op = Opcode::shr;
    ie[10]->simd = simd/2;
    ie[10]->dst = tmp2;
    ie[10]->dst.type = DataType::uw;
    ie[10]->dst.stride = 2;
    ie[10]->src0 = tmp2;
    ie[10]->src0.type = DataType::uw;
    ie[10]->src0.stride = 2;
    ie[10]->src1 = Immediate::uw(4);

    ie[11]->op = Opcode::or_;
    ie[11]->simd = simd/2;
    ie[11]->dst = tmp2;
    ie[11]->dst.type = DataType::uw;
    ie[11]->src0 = tmp2;
    ie[11]->src0.offset = 1;
    ie[11]->src0.type = DataType::uw;
    ie[11]->src0.stride = 2;
    ie[11]->src1 = tmp2;
    ie[11]->src1.type = DataType::uw;
    ie[11]->src1.stride = 2;

    ie[12]->op = Opcode::mov;
    ie[12]->simd = simd/2;
    ie[12]->dst = tmp2;
    ie[12]->dst.stride = 1;
    ie[12]->dst.type = DataType::ub;
    ie[12]->src0 = tmp2;
    ie[12]->src0.stride = 2;
    ie[12]->src0.type = DataType::ub;

    ie[13]->op = Opcode::mov;
    ie[13]->simd = simd/2;
    ie[13]->dst = ddst;
    if (ddst.inVS != 0)
    ie[13]->dst.stride = ddst.inVS / ddst.inW;
    ie[13]->dst.type = DataType::ub;
    if (ie[13]->dst.offset != 0)
            ie[13]->dst.offset /= 2;
    ie[13]->src0 = tmp2;
    ie[13]->src0.type = DataType::ub;

}

// Emulated f->bf or hf->bf8 sequence.
void CopyPlan::planEmulatedHalveFloat(CopyInstruction &i)
{
    if (i.src0.neg || i.sat || i.hasCMod()) stub("Unsupported modifier");

    auto ie = splitMultiple<4>(i);

    bool toBF = (i.src0.type == DataType::f);
    if (!toBF && i.src0.type != DataType::hf) stub();

    auto T_large = toBF ? DataType::ud : DataType::uw;
    auto T_small = toBF ? DataType::uw : DataType::ub;

    auto esrc0 = i.src0;
    if (esrc0.overwrite && !multiGRF(hw, i, i.src0))
        esrc0.type = T_large;
    else
        esrc0 = newTemp(T_large, i.simd, 1);

    // Emulation sequence for mov y:bf x:f:
    //   add            x:ud x:ud -0x8000
    //   and (nz)fN.N   x:ud       0x1FFFF
    //   mov            y:uw       x_hi:uw
    //   (fN.N) add     y:uw       x_hi:uw     1
    //
    // hf->bf8 is similar but half as wide.

    ie[0]->op = Opcode::add;
    ie[0]->src0.type = T_large;
    ie[0]->src1 = toBF ? -0x8000 : -0x80;
    ie[0]->dst = esrc0;

    ie[1]->op = Opcode::and_;
    ie[1]->src0 = esrc0;
    ie[1]->src1 = toBF ? 0x1FFFF : 0x1FF;
    ie[1]->dst = CopyOperand();
    ie[1]->dst.type = T_large;
    ie[1]->cmod = ConditionModifier::nz;
    ie[1]->flag = newFlag(ie[1]->simd);

    ie[2]->op = Opcode::mov;
    ie[2]->src0 = esrc0;
    ie[2]->src0.type = ie[2]->dst.type = T_small;
    ie[2]->src0.stride *= 2;
    ie[2]->src0.offset++;

    ie[3]->op = Opcode::add;
    ie[3]->src0 = esrc0;
    ie[3]->src0.type = ie[3]->dst.type = T_small;
    ie[3]->src0.stride *= 2;
    ie[3]->src0.offset++;
    ie[3]->src1 = 1;
    ie[3]->flag = ie[1]->flag;
}

// Pass to perform early int4 upconversion transformations before 2D
//   regions are split into 1D regions.
void CopyPlan::planEarlyInt4Upconversions()
{
    for (auto &i: insns) {
        if (i.op == Opcode::mov && isInt4(i.src0.type) && isB(i.dst.type)) {
            bool s4 = (i.src0.type == DataType::s4);
            if (i.src0.inW == 2 && i.src0.stride == 1 && i.dst.stride >= (s4 ? 2 : 4)) {
                planInt4Upconversion(i);
            }
        }
    }

    mergeChanges();
}

// Rewrite int4 -> int upconversion using byte operations.
// May need to be run twice.
//
// Example input:
//    mov (16)   r0.0<1>:ub   r1.0<1>:s4
// Output:
//    and (16)   r0.0<1>:ub   r1.0<1>:ub   0xF:uw
//    shr (16)   r0.0<1>:ub   r1.0<1>:ub   4:uw
//
void CopyPlan::planInt4Upconversion(CopyInstruction &i)
{
    if (i.src0.neg || i.hasCMod()) stub("Unsupported modifier");
    i.sat = false;

    bool s4 = (i.src0.type == DataType::s4);

    if (i.src0.stride == 1) {
        // Split into high and low nybble conversions first, if needed.
        // If dst stride is too large, copy through uw<1>.
        //   This path allows 2D regions.
        if (i.dst.stride >= (s4 ? 2 : 4)) {
            auto ie = splitMultiple<3>(i);
            auto t = newTemp(DataType::uw, i.simd, 1);
            ie[0]->simd /= 2;
            ie[0]->op = s4 ? Opcode::shl : Opcode::and_;
            ie[0]->src0.type = DataType::ub;
            ie[0]->src0.offset /= 2;
            if (ie[0]->src0.inW > 1) {
                ie[0]->src0.inW /= 2;
                if (ie[0]->src0.inW == 1) {
                    ie[0]->src0.stride = ie[0]->src0.inVS / 2;
                    ie[0]->src0.inVS = ie[0]->src0.inW = 0;
                }
            }
            ie[0]->src1 = s4 ? 4 : 0xF;
            ie[0]->dst = t;
            ie[0]->dst.stride *= 2;

            ie[1]->simd /= 2;
            ie[1]->op = s4 ? Opcode::mov : Opcode::shr;
            ie[1]->src0 = ie[0]->src0;
            if (!s4) ie[1]->src1 = 4;
            ie[1]->dst = ie[0]->dst;
            ie[1]->dst.offset++;

            ie[2]->op = s4 ? Opcode::asr : Opcode::mov;
            ie[2]->src0 = t;
            ie[2]->src0.type = s4 ? DataType::b : DataType::ub;
            ie[2]->src0.stride *= 2;
            ie[2]->src0.offset *= 2;
            if (s4) ie[2]->src1 = 4;
        } else {
            auto &i0 = i;
            i0.dst.stride *= 2;
            i0.src0.stride *= 2;
            i0.simd /= 2;
            auto &i1 = split(i, false);
            i1.dst.offset += i1.dst.stride / 2;
            i1.src0.offset += i1.src0.stride / 2;
        }
    } else {
        bool even = (i.src0.offset % 2 == 0);
        i.src0.stride /= 2;
        i.src0.offset /= 2;

        if (even) {
            // Low nybbles
            if (s4) {
                auto &i0 = i, &i1 = split(i);
                if (getBytes(i0.dst.type) == 1)
                    i0.dst = newTemp(DataType::uw, i0.simd, (i0.src0.stride > 2) ? 2 : 1);
                i1.src0 = i0.dst;
                auto shift = getBytes(i0.dst.type) * 8 - 4;

                i0.op = Opcode::shl;
                i0.src0.type = DataType::ub;
                i0.src1 = shift;

                i1.op = Opcode::asr;
                i1.src0.type = asSigned(i1.src0.type);
                i1.src1 = shift;
            } else {
                i.op = Opcode::and_;
                i.src0.type = DataType::ub;
                i.src1 = 0xF;
            }
        } else {
            // High nybbles
            i.op = s4 ? Opcode::asr : Opcode::shr;
            i.src0.type = s4 ? DataType::b : DataType::ub;
            i.src1 = 4;
        }
    }
}


// Emulation sequence for fp8 e8m0->hf conversion.
void CopyPlan::planEmulatedFP8E8M0ToHF(CopyInstruction &i) {
    if (i.src0.neg || i.sat || i.hasCMod()) stub("Unsupported modifier");
    if (hw < HW::XeHP) stub("Unsupported HW");


    // Emulation sequence for mov y:hf y:e8m0
    // mov                 y:uw   x:u8                    /* emulated separately
    // xor                 t1:uw, 0xFF, y:uw
    // add                 y:w, y:w, -112
    // csel (ze)           y:w,  0 , y:w, y:w
    // cmp  (ge)           t0:w, y:w, 31
    // shr                 y:uw, 10
    // csel (ge)           y:fp16,  0x7bff, y:fp16, t0:fp16
    // csel (ze)           y:fp16, NaN:fp16, y:fp16, t1:fp16

    auto ie = splitMultiple<9>(i);

    auto yOrig = i.dst, y = yOrig;

    bool tempY = (y.stride > 1 && multiGRF(hw, i, y));
    if (tempY)  /* Replace y by temporary if nonunit stride hurts performance */
        y = newTemp(DataType::uw, i.simd, 1);

    auto yW = y;
    yW.type = DataType::w;

    auto t0 = i.src0;
    if (t0.overwrite && t0.overwriteStride
        && t0.stride == y.stride * 2 && t0.offset == y.offset * 2) {
        t0.type = DataType::hf;
        t0.stride /= 2;
        t0.offset /= 2;
    } else
        t0 = newTemp(DataType::hf, i.simd, y.stride, 0, y.offset);


    auto t1 = i.src0;
    t1 = newTemp(DataType::hf, i.simd, y.stride, 0, y.offset);

    auto t0UW = t0;
    t0UW.type = DataType::uw;

    auto t1W = t1;
    t1W.type = DataType::w;

    // Copy to u16.
    ie[0]->src0.type = DataType::ub;
    ie[0]->dst.type = DataType::uw;
    ie[0]->dst = yW;

    ie[1]->op = Opcode::xor_;
    ie[1]->dst =  t0UW;
    ie[1]->src0 = yW;
    ie[1]->src1 = 0xFF;

    ie[2]->op = Opcode::add;
    ie[2]->dst = ie[2]->src0 =yW;
    ie[2]->src1 = -112;

    ie[3]->op = Opcode::csel;
    ie[3]->dst = yW;
    ie[3]->src0 = Immediate::w(0x0);
    ie[3]->src1 = yW;
    ie[3]->src2 = yW;
    ie[3]->cmod = ConditionModifier::le;

    ie[4]->op = Opcode::cmp;
    ie[4]->dst = t1W;
    ie[4]->src0 = yW;
    ie[4]->src1 = Immediate::uw(31);;
    ie[4]->cmod = ConditionModifier::le;
    ie[4]->flag = newFlag(ie[4]->simd);


    ie[5]->op = Opcode::shl;
    ie[5]->dst = yW;
    ie[5]->src0 = yW;
    ie[5]->src0.type = DataType::uw;
    ie[5]->dst.type  = DataType::uw;
    ie[5]->src1 = 10;

    ie[6]->op = Opcode::csel;
    ie[6]->dst = y;
    ie[6]->src0 = Immediate::hf(0x7BFF);
    ie[6]->src1 = y;
    ie[6]->src2 = t1;
    ie[6]->cmod = ConditionModifier::ze;

    ie[7]->op = Opcode::csel;
    ie[7]->dst = y;
    ie[7]->src0 = Immediate::hf(0x7C01);
    ie[7]->src1 = y;
    ie[7]->src2 = t0;
    ie[7]->cmod = ConditionModifier::ze;

    if (tempY) {
        ie[8]->op = Opcode::mov;
        ie[8]->dst = yOrig;
        ie[8]->dst.type = DataType::uw;
        ie[8]->src0 = yW;
    } else
        ie[8]->invalidate();
}


// Emulation sequence for hf8->hf conversion.
void CopyPlan::planEmulatedHF8ToHF(CopyInstruction &i)
{
    if (i.src0.neg || i.sat || i.hasCMod()) stub("Unsupported modifier");
    if (hw < HW::XeHP) stub("Unsupported HW");

    auto ie = splitMultiple<8>(i);

    // Emulation sequence for mov y:hf x:hf8:
    // shl          y:uw    x:ub    7
    // shl          t0:uw   x:ub    8
    // and          y:uw    y:uw    0x3F80
    // xor          t0:uw   t0:uw   0x7F00
    // mul          y:hf    y:hf    256:hf
    // csel (ze)    y:hf    nan:hf  y:hf   t0:hf
    // bfn.0xCA     y:uw    y:uw    t0:uw  0x8000   /* copy sign */

    auto yOrig = i.dst, y = yOrig;

    bool tempY = (y.stride > 1 && multiGRF(hw, i, y));
    if (tempY)  /* Replace y by temporary if nonunit stride hurts performance */
        y = newTemp(DataType::uw, i.simd, 1);

    auto yUW = y;
    yUW.type = DataType::uw;

    auto t0 = i.src0;
    if (t0.overwrite && t0.overwriteStride
            && t0.stride == y.stride * 2 && t0.offset == y.offset * 2) {
        t0.type = DataType::hf;
        t0.stride /= 2;
        t0.offset /= 2;
    } else
        t0 = newTemp(DataType::hf, i.simd, y.stride, 0, y.offset);

    auto t0UW = t0;
    t0UW.type = DataType::uw;

    ie[0]->op = Opcode::shl;
    ie[0]->dst = yUW;
    ie[0]->src0.type = DataType::ub;
    ie[0]->src1 = 7;

    ie[1]->op = Opcode::shl;
    ie[1]->dst = t0UW;
    ie[1]->src0 = ie[0]->src0;
    ie[1]->src1 = 8;

    ie[2]->op = Opcode::and_;
    ie[2]->dst = ie[2]->src0 = yUW;
    ie[2]->src1 = 0x3F80;

    ie[3]->op = Opcode::xor_;
    ie[3]->dst = ie[3]->src0 = t0UW;
    ie[3]->src1 = 0x7F00;

    ie[4]->op = Opcode::mul;
    ie[4]->dst = ie[4]->src0 = y;
    ie[4]->src1 = Immediate::hf(0x5C00);

    ie[5]->op = Opcode::csel;
    ie[5]->dst = y;
    ie[5]->src0 = Immediate::hf(0x7C01);
    ie[5]->src1 = y;
    ie[5]->src2 = t0;
    ie[5]->cmod = ConditionModifier::ze;

    ie[6]->op = Opcode::bfn;
    ie[6]->dst = ie[6]->src0 = yUW;
    ie[6]->src1 = t0UW;
    ie[6]->src2 = 0x8000;
    ie[6]->ctrl = 0xCA;

    if (tempY) {
        ie[7]->op = Opcode::mov;
        ie[7]->dst = yOrig;
        ie[7]->dst.type = DataType::uw;
        ie[7]->src0 = yUW;
    } else
        ie[7]->invalidate();
}

// Emulation sequence for f4_e2m1->hf conversion.
void CopyPlan::planEmulatedF4E2M1ToHF(CopyInstruction &i) {

    // Emulation sequence for mov y:hf x:hf4_E2M1
    // mov                 t0:uw   x:u4                    /* emulated separately */
    // shl                 y:uw    t0:uw    9
    // shl                 t0:uw   t0:uw    12
    // and                 y:uw    y:uw    0x0E00
    // mul                 y:hf    y:hf    16384:hf
    // bfn.0xCA            y:uw    y:uw    t0:uw  0x8000   /* copy sign */


    if (i.src0.neg || i.sat || i.hasCMod()) stub("Unsupported modifier");

    auto ie = splitMultiple<8>(i);

    auto yOrig = i.dst, y = yOrig;

    bool tempY = (y.stride > 1 && multiGRF(hw, i, y));
    if (tempY)  /* Replace y by temporary if nonunit stride hurts performance */
        y = newTemp(DataType::uw, i.simd, 1);

    auto yUW = y;
    yUW.type = DataType::uw;

    auto t0 = newTemp(DataType::hf, i.simd, y.stride, 0, y.offset);

    auto t0UW = t0;
    t0UW.type = DataType::uw;
    t0UW.stride = 1;
    t0UW.offset = 0;

    // Copy to u16.
    ie[0]->src0.type = DataType::u4;
    ie[0]->dst.type = DataType::uw;
    ie[0]->dst = t0UW;

    ie[1]->src0 = ie[0]->dst;
    ie[1]->op = Opcode::shl;
    ie[1]->dst = yUW;
    ie[1]->src1 = 9;

    ie[2]->op = Opcode::shl;
    ie[2]->dst = t0UW;
    ie[2]->src0 = ie[0]->dst;
    ie[2]->src1 = 12;

    ie[3]->op = Opcode::and_;
    ie[3]->dst = ie[3]->src0 = yUW;
    ie[3]->src1 = 0x0E00;

    ie[4]->op = Opcode::mul;
    ie[4]->dst = ie[4]->src0 = y;
    ie[4]->src1 = Immediate::hf(0x7400);

    ie[5]->op = Opcode::and_;
    ie[5]->dst = ie[5]->src0 = t0UW;
    ie[5]->src1 = 0x8000;

    ie[6]->op = Opcode::or_;
    ie[6]->dst = ie[6]->src0 = yUW;
    ie[6]->src1 = t0UW;

    if (tempY) {
        ie[7]->op = Opcode::mov;
        ie[7]->dst = yOrig;
        ie[7]->dst.type = DataType::uw;
        ie[7]->src0 = yUW;
    } else
        ie[7]->invalidate();
}

// Emulation sequence for e3m0->hf conversion.
void CopyPlan::planEmulatedE3M0ToHF(CopyInstruction &i)
{
    // Emulation sequence for mov y:hf x:e3m0: play only on the exponent bits
    // mov                 y:uw   x:u4                    /* emulated separately */
    // shl                 t0:uw   t0:uw   12
    // and                 y:uw    y:uw    0x7
    // (f1)add             y:uw    y:uw    12
    // shl                 y:uw    y:uw    10
    // bfn.0xCA            y:uw    y:uw    t0:uw  0x8000   /* copy sign */

    if (i.src0.neg || i.sat || i.hasCMod()) stub("Unsupported modifier");

    auto ie = splitMultiple<8>(i);

    auto yOrig = i.dst, y = yOrig;

    bool tempY = (y.stride > 1 && multiGRF(hw, i, y));
    if (tempY)  /* Replace y by temporary if nonunit stride hurts performance */
        y = newTemp(DataType::uw, i.simd, 1);

    auto yUW = y;
    yUW.type = DataType::uw;

    auto t0 = newTemp(DataType::hf, i.simd, y.stride, 0, y.offset);

    auto t0UW = t0;
    t0UW.type = DataType::uw;

    // Copy to u16.
    ie[0]->src0.type = DataType::u4;
    ie[0]->dst.type = DataType::uw;
    ie[0]->dst = yUW;

    ie[1]->op = Opcode::shl;
    ie[1]->dst = t0UW;
    ie[1]->src0 = ie[0]->dst;
    ie[1]->src1 = 12;

    ie[2]->op = Opcode::and_;
    ie[2]->dst = ie[2]->src0 = yUW;
    ie[2]->src1 = 0x7;
    ie[2]->cmod = ConditionModifier::nz;
    ie[2]->flag = newFlag(ie[2]->simd);

    ie[3]->op = Opcode::add;
    ie[3]->dst = ie[3]->src0 = yUW;
    ie[3]->src1 = 0xc;
    ie[3]->dst.type = DataType::uw;
    ie[3]->src0.type = DataType::uw;
    ie[3]->src1.type = DataType::uw;
    ie[3]->flag = ie[2]->flag;

    ie[4]->src0 = ie[3]->dst;
    ie[4]->op = Opcode::shl;
    ie[4]->dst = yUW;
    ie[4]->src1 = 10;

    ie[5]->op = Opcode::and_;
    ie[5]->dst = ie[5]->src0 = t0UW;
    ie[5]->src1 = 0x8000;

    ie[6]->op = Opcode::or_;
    ie[6]->dst = ie[6]->src0 = yUW;
    ie[6]->src1 = t0UW;

    if (tempY) {
        ie[7]->op = Opcode::mov;
        ie[7]->dst = yOrig;
        ie[7]->dst.type = DataType::uw;
        ie[7]->src0 = yUW;
    } else
        ie[7]->invalidate();
}

// Emulation sequence for hf->hf8 conversion.
void CopyPlan::planEmulatedHFToHF8(CopyInstruction &i)
{
    if (i.src0.neg || i.sat || i.hasCMod()) stub("Unsupported modifier");
    if (hw < HW::XeHP) stub("Unsupported HW");

    auto ie = splitMultiple<10>(i);

    // Emulation sequence for mov y:hf8 x:hf:
    // mul          t0:hf   x:hf    128:hf          /* hf8 overflow */
    // mul          t0:hf   t0:hf   2^(-15):hf      /* hf8 underflow */
    // and (ze)f1   null    ~t0:uw  0x7C00          /* nan/inf check */
    // add          t0:uw   t0:uw   -0x40           /* round */
    // and (nz)f2   null    t0:uw   0x0FF
    // shl          t0:uw   t0:uw   1               /* move to high byte */
    // (f2) add     t0:uw   t0:uw   0x100
    // (f1) mov     t0:uw   0x7F00
    // bfn.0xCA     t0:uw   t0:uw   x:uw   0x8000   /* copy sign */
    // mov          y:ub    t0_hi:ub

    auto x = i.src0;
    auto t0 = newTemp(DataType::hf, i.simd, x.stride);
    auto t0UW = t0;
    t0UW.type = DataType::uw;

    ie[0]->op = Opcode::mul;
    ie[0]->dst = t0;
    ie[0]->src1 = Immediate::hf(0x5800);

    ie[1]->op = Opcode::mul;
    ie[1]->dst = ie[1]->src0 = t0;
    ie[1]->src1 = Immediate::hf(0x0200);

    ie[2]->op = Opcode::and_;
    ie[2]->dst = CopyOperand();
    ie[2]->dst.type = DataType::uw;
    ie[2]->src0 = t0UW;
    ie[2]->src0.neg = true;
    ie[2]->src1 = 0x7C00;
    ie[2]->cmod = ConditionModifier::ze;
    ie[2]->flag = newFlag(ie[2]->simd);

    ie[3]->op = Opcode::add;
    ie[3]->dst = ie[3]->src0 = t0UW;
    ie[3]->src1 = -0x40;

    ie[4]->op = Opcode::and_;
    ie[4]->dst = CopyOperand();
    ie[4]->dst.type = DataType::uw;
    ie[4]->src0 = t0UW;
    ie[4]->src1 = 0x0FF;
    ie[4]->cmod = ConditionModifier::nz;
    ie[4]->flag = newFlag(ie[4]->simd);

    ie[5]->op = Opcode::shl;
    ie[5]->dst = ie[5]->src0 = t0UW;
    ie[5]->src1 = 1;

    ie[6]->op = Opcode::add;
    ie[6]->dst = ie[6]->src0 = t0UW;
    ie[6]->src1 = 0x100;
    ie[6]->flag = ie[4]->flag;

    ie[7]->op = Opcode::mov;
    ie[7]->dst = t0UW;
    ie[7]->src0 = 0x7F00;
    ie[7]->flag = ie[2]->flag;

    ie[8]->op = Opcode::bfn;
    ie[8]->dst = ie[8]->src0 = t0UW;
    ie[8]->src1 = x;
    ie[8]->src1.type = DataType::uw;
    ie[8]->src2 = 0x8000;
    ie[8]->ctrl = 0xCA;

    ie[9]->op = Opcode::mov;
    ie[9]->src0 = t0;
    ie[9]->src0.type = ie[9]->dst.type = DataType::ub;
    ie[9]->src0.stride *= 2;
    ie[9]->src0.offset++;
}

// Check that no types smaller than a byte are present.
void CopyPlan::checkNoSubbytes()
{
    for (auto &i: insns)
        if (is4(i.dst.type) || is4(i.src0.type) || is4(i.src1.type) || is4(i.src2.type))
            stub("Unexpected 4-bit type");
}

// Collapse multiple-cnum instructions into single-cnum instructions if possible.
void CopyPlan::collapseCNums()
{
    int ncnum = 0;
    for (auto &i: insns)
        ncnum = std::max(ncnum, i.cnumMax + 1);

    std::vector<int16_t> snapDown(ncnum);
    for (auto &i: insns)
        for (auto cnum = i.cnumMin; cnum <= i.cnumMax; cnum++)
            snapDown[cnum] = std::max(snapDown[cnum], i.cnumMin);

    for (auto &i: insns) {
        i.cnumMin = snapDown[i.cnumMin];
        i.cnumMax = snapDown[i.cnumMax];
    }
}

// Pass to legalize SIMD lengths.
// If initial = true, does not perform complete legalization,
//   only SIMD32 limits for complex conversion sequences.
void CopyPlan::legalizeSIMD(bool initial)
{
    int grf = GRF::bytes(hw);
    bool splitting = false;
    int16_t maxCNumSub = 1;

    if (!initial)
        checkNoSubbytes();

    collapseCNums();
    for (auto &i: insns)
        i.cnumSub = 0;

    auto ninsn = insns.size();
    for (size_t n = 0; n < ninsn; ) {
        auto &i = insns[n];

        int simdMax = 32;

        if (!initial) {
            // Basic rule: maximum of 2 registers per operand.
            auto opSimdMax = [=](const CopyOperand &op) {
                int bstride = getBytes(op.type) * op.stride;
                if (bstride == 0)
                    return 256;
                else if (op.offset >= op.stride)
                    return grf / bstride;
                else
                    return 2 * grf / bstride;
            };

            simdMax = std::min({simdMax, opSimdMax(i.dst), opSimdMax(i.src0), opSimdMax(i.src1)});

            // Special handling for mixed mode (f16/bf16 with f32) instructions.
            bool hasF  = one_of(DataType::f,  i.dst.type, i.src0.type, i.src1.type, i.src2.type);
            bool hasHF = one_of(DataType::hf, i.dst.type, i.src0.type, i.src1.type, i.src2.type);
            bool hasBF = one_of(DataType::bf, i.dst.type, i.src0.type, i.src1.type, i.src2.type);
            bool dstHF = (i.dst.type == DataType::hf);
            bool bfException = (i.op == Opcode::mov && i.dst.type == DataType::bf && i.dst.stride == 2);

            if (hasF && ((hasBF && !bfException) || (hasHF && hw <= HW::XeLP) || dstHF))
                simdMax = std::min(simdMax, grf >> 2);
        }

        if (initial) {
            bool skip = isInt(i.dst.type) && isInt(i.src0.type);
            skip |= is4(i.dst.type) || is4(i.src0.type) || is4(i.src1.type) || is4(i.src2.type);
            if (skip) {
                n++; continue;
            }
        }

        // Fracture instruction into legal SIMD lengths.
        int simd0 = std::min<int>(rounddown_pow2(i.simd), simdMax);
        if (simd0 < i.simd || splitting) {
            if (i.dst.offset >= i.dst.stride && i.dst.stride > 0) {   /* align dst to GRF boundary */
                int remaining = div_up(bytesToElements(grf, i.dst.type) - i.dst.offset, i.dst.stride);
                simd0 = std::min(simd0, rounddown_pow2(remaining));
            }

            auto &isplit = split(i, false);
            isplit.simd = simd0;

            auto advance = [grf](CopyOperand &op, int n) {
                if (op.kind == CopyOperand::Flag)
                    op.offset += n;
                if (op.kind != CopyOperand::GRF) return;
                int ne = bytesToElements(grf, op.type);
                op.offset += n * op.stride;
                int grfOffset = op.offset / ne;
                op.grf += grfOffset;
                op.offset -= grfOffset * ne;
            };

            i.cnumSub++;
            maxCNumSub = std::max(maxCNumSub, i.cnumSub);

            i.simd -= simd0;
            advance(i.dst, simd0);
            advance(i.src0, simd0);
            advance(i.src1, simd0);
            advance(i.src2, simd0);
            advance(i.flag, simd0);
            splitting = (i.simd > 0);
        } else
            n++;    /* done with this instruction */
    }

    mergeChanges();

    /* Split apart cnums */
    for (auto &i: insns) {
        i.cnumMin *= maxCNumSub;
        i.cnumMax *= maxCNumSub;
        if (i.cnumMin == i.cnumMax)
            i.cnumMin = i.cnumMax += i.cnumSub;
    }
}

// Check if an operand is a legal packed bfloat16 region.
inline bool legalPackedBF(HW hw, const CopyOperand &op)
{
    if (op.kind != op.GRF) return true;

    int align = GRF::bytes(hw) / 4;
    return (op.stride == 1 && (op.offset & (align - 1)) == 0);
}

void   CopyPlan::planFP8SIMD1Mov(CopyInstruction &i){
    /* Simd 1 not allowed, use following sequence instead:
       hf8->hf (analagous sequence will be generated for hf->hf8)
       mov(2, t_dst.hf, src<2,2,1>.hf8)
       mov(1, dst.uw, t_dst<1,1,1>.uw) */

     auto dt = i.dst.type;
     auto ie = splitMultiple<2>(i);
     auto src = i.src0;
     auto dst = i.dst;
     auto t_dst = newTemp(dt, 2, 1);
     t_dst.stride = 1;

     ie[0]->op = Opcode::mov;
     ie[0]->dst = t_dst;
     ie[0]->src0 = src;
     ie[0]->src0.stride = 1;
     ie[0]->simd = 2;

     ie[1]->op = Opcode::mov;
     ie[1]->dst = dst;
     ie[1]->src0 = t_dst;
     ie[1]->moveToIntegerPipe();
}

// Pass to legalize regions.
void CopyPlan::legalizeRegions()
{
    bool rerun = false;

    checkNoSubbytes();

    for (auto &i: insns) {
        auto s0t = i.src0.type;
        auto s1t = i.src1.type;
        auto s2t = i.src2.type;
        auto dt = i.dst.type;

        if (!i.dst) continue;

        /* Check for special packed conversion cases */
        if (i.op == Opcode::mov && ((s0t == DataType::hf && isFP8(dt))
                                 || (dt == DataType::hf && isFP8(s0t)))) {
            // hf <-> bf8/hf8: src0/dst must be packed unit stride, zero offset
            if (i.simd == 1 && i.src0.offset == 0 && i.src0.stride == 1){
                planFP8SIMD1Mov(i);
                rerun = true;
            } else if (i.src0.offset != 0 || i.src0.stride != 1) {
                repositionSrc(i, 0, 1, 0);
                rerun = true;
            } else if (i.dst.offset != 0 || i.dst.stride != 1)
                repositionDst(i, 1, 0);
            continue;
        }

        if (dt == DataType::bf || s0t == DataType::bf || s1t == DataType::bf) {
            // bf/f mixed mode: src/dst may be packed unit stride
            if (legalPackedBF(hw, i.dst) && legalPackedBF(hw, i.src0) && legalPackedBF(hw, i.src1))
                continue;
        }

        if (i.op == Opcode::mov) {
            if (dt == DataType::hf || s0t == DataType::hf) {
                if (dt == DataType::f || s0t == DataType::f) {
                    // hf/f mixed mode: src/dst may be packed unit stride
                    if (i.dst.stride == 1 && i.src0.stride == 1) {
                        int dstBO  = (i.dst.offset  * 4) & (GRF::bytes(hw) - 1);
                        int src0BO = (i.src0.offset * 4) & (GRF::bytes(hw) - 1);
                        if (dstBO == src0BO)
                            continue;
                    }
                }
            }
        }

        bool hfIntConvert = (dt  == DataType::hf && isInt(s0t))
                         || (s0t == DataType::hf && isInt(dt));
        hfIntConvert &= (i.op == Opcode::mov);

        /* Check destination stride against execution channels */
        int channelSize = 1;
        for (auto &op: {i.dst, i.src0, i.src1, i.src2})
            if (op.kind == op.GRF)
                channelSize = std::max(channelSize, getBytes(op.type));

        if (channelSize == 1 && i.op != Opcode::mov)
            channelSize = 2;
        if (hfIntConvert)
            channelSize = 4;

        int dstMinStride = channelSize >> getLog2Bytes(dt);
        bool doRestrideDst = (i.dst.stride < dstMinStride);

        /* Check destination offset */
        int channelOffset = (i.dst.offset * getBytes(dt)) & (channelSize - 1);
        int maxChanOff = 4 / getBytes(dt);
        if (getBytes(dt) == 1 && hw < HW::XeHPC)
            maxChanOff = 2;     /* special case: pre-PVC only allows .{0,1}:b */
        if (hfIntConvert)
            maxChanOff = 1;     /* special case: integer<->hf only allows .0:hf */

        bool badChanOff = (channelOffset >= maxChanOff);
        doRestrideDst |= badChanOff;

        /* For illegal dst, copy through temporary dst */
        if (doRestrideDst) {
            if (i.simd == 1)
                i.dst.stride = dstMinStride;
            else {
                restrideDst(i, dstMinStride, badChanOff);
                rerun = true;
                continue;
            }
        }

        /* Check for swizzling */
        bool canSwizzle = true;
        if (hw >= HW::XeHP) {
            if (isQ(dt) || isQ(s0t) || isQ(s1t))
                if (!(i.op == Opcode::mov && i.dst.stride == 1))
                    canSwizzle = false;
            if (isFP(dt))
                canSwizzle = false;
        }

        if (!canSwizzle) {
            int dstBO  = i.dst.byteOffset();
            int src0BO = i.src0.byteOffset();
            int src1BO = i.src1.byteOffset();
            int src2BO = i.src2.byteOffset();
            int dstBS  = i.dst.byteStride();
            int src0BS = i.src0.byteStride();
            int src1BS = i.src1.byteStride();
            int src2BS = i.src2.byteStride();
            int dboMask = GRF::bytes(hw) - (isFP(dt) ? 1 : 4);

            auto matchesDstBO = [=](int bo) -> bool {
                return (dstBO & dboMask) == (bo & dboMask);
            };

            auto doRepositionSrc = [&](int n, DataType st) -> bool {
                int stride = dstBS >> getLog2Bytes(st);
                int offset = dstBO >> getLog2Bytes(st);
                if (stride * getBytes(st) != dstBS || offset * getBytes(st) != dstBO)
                    return false;
                repositionSrc(i, n, stride, offset);
                return true;
            };

            /* Check src0 */
            if (i.src0 && !isBroadcast(i.src0)) {
                if (!matchesDstBO(src0BO)) {
                    if (!doRepositionSrc(0, s0t)) {
                        int stride = src0BS >> getLog2Bytes(dt);
                        int offset = src0BO >> getLog2Bytes(dt);
                        if (stride * getBytes(dt) != src0BS || offset * getBytes(dt) != src0BO)
                            stub("Cannot legalize src0/dst regions");
                        repositionDst(i, stride, offset);
                    }
                    continue;
                } else if (src0BS < dstBS)
                    restrideSrc0(i, dstBS >> getLog2Bytes(s0t));
                else if (src0BS > dstBS)
                    restrideDst(i, src0BS >> getLog2Bytes(dt));
            }

            /* Check src1 */
            if (i.src1 && !isBroadcast(i.src1) && (!matchesDstBO(src1BO) || dstBS != src1BS)) {
                if (!doRepositionSrc(1, s1t))
                    stub("Cannot legalize src1 region");
                continue;
            }
            /* Check src2 */
            if (i.src2 && !isBroadcast(i.src2) && (!matchesDstBO(src2BO) || dstBS != src2BS)) {
                if (!doRepositionSrc(2, s2t))
                    stub("Cannot legalize src2 region");
                continue;
            }
        }

        /* PVC limitations on packing multiple execution channels into a DWord */
        if (canSwizzle && hw >= HW::XeHPC && channelSize < 4 && i.dst.stride * getBytes(dt) < 4) {
            int d0s = i.dst.stride;
            int d0o = i.dst.offset;
            int s0s = i.src0.stride;
            int s0o = i.src0.offset;

            if (!isW(dt)  && !isB(dt))  stub();
            if (!isW(s0t) && !isB(s0t)) stub();

            if (isW(s0t)) {
                canSwizzle &= (s0s <= 2);
                if (s0s == 2) {
                    if (isW(dt))
                        canSwizzle &= (s0o / 2 == d0o % 16);
                    else
                        canSwizzle &= (s0o == d0o % 32);
                }
            } else {
                if (isW(dt) || d0s > 1)
                    s0s /= 2;
                if (isW(dt))
                    d0o *= 2;
                canSwizzle &= (s0s <= 4);
                if (s0s >= 2)
                    canSwizzle &= (d0o % (64 / s0s) == s0o / s0s);
            }

            if (!canSwizzle) {
                int istride = 4 / getBytes(dt);
                (i.src0.byteStride() < i.dst.byteStride()) ? restrideSrc0(i, istride)
                                                           : restrideDst(i, istride);
            }
        }
    }

    mergeChanges();
    if (rerun)
        legalizeRegions();
}

// Pass to legalize negation use.
void CopyPlan::legalizeNegation()
{
    for (auto &i: insns) if (i.dst.neg) {
        i.src0.neg = !i.src0.neg;
        i.src1.neg = !i.src1.neg;
        i.src2.neg = !i.src2.neg;
    }
}

// Pass to legalize immediate types.
void CopyPlan::legalizeImmediateTypes()
{
    for (auto &i: insns) {
        for (auto *op: {&i.src0, &i.src1, &i.src2}) {
            if (op->kind != CopyOperand::Immediate)
                continue;
            if (one_of(op->type, DataType::ub, DataType::u4))
                op->type = DataType::uw;
            else if (one_of(op->type, DataType::b, DataType::s4))
                op->type = DataType::w;
        }
    }
}

// Pass to sort instructions by phase and dst.
void CopyPlan::sort(SortType type)
{
    auto sortOrder = [type](const CopyInstruction &i) {
        switch (type) {
            case SortType::PhaseOnly:
                return std::make_tuple(i.phase, 0, 0);
            case SortType::SourceOrder:
                return std::make_tuple(i.phase, int(i.cnumMin), int(i.cnumMax));
            case SortType::Register:
            default:
                auto &op = i.dst.temp ? i.src0 : i.dst;
                return std::make_tuple(i.phase, int(op.grf), op.byteOffset());
        };
    };

    std::stable_sort(insns.begin(), insns.end(), [=](const CopyInstruction &i1, const CopyInstruction &i2) {
        return sortOrder(i1) < sortOrder(i2);
    });
}

// Optimization pass: zip together interleaved operations.
// Requires a sorted plan.
//
// Example input:
//    mov (8)  r0.0<4>:uw   r10.0<2>:uw
//    mov (8)  r0.2<4>:uw   r10.1<2>:uw
// Output:
//    mov (16) r0.0<2>:uw   r10.0<1>:uw
void CopyPlan::optimizeZip()
{
    auto ninsn = insns.size();
    for (size_t n1 = 0; n1 < ninsn; n1++) {
        for (size_t n2 = n1 + 1; n2 < ninsn; n2++) {
            auto &i1 = insns[n1];
            auto &i2 = insns[n2];

            if (i1.op != i2.op || i1.phase != i2.phase || i1.dst.grf != i2.dst.grf || i1.flag) break;
            if (i1.simd != i2.simd) continue;

            auto zippable = [](const CopyOperand &o1, const CopyOperand &o2) {
                if (o1.kind != o2.kind) return false;
                if (o1.kind != CopyOperand::GRF) return true;
                if (o1.type != o2.type || o1.stride != o2.stride || o1.grf != o2.grf) return false;
                if (o1.temp != o2.temp) return false;
                if (o1.temp && o1.value != o2.value) return false;
                if (o1.stride & 1) return false;
                if (o1.neg != o2.neg) return false;
                if (o1.abs != o2.abs) return false;
                if (o1.inv != o2.inv) return false;
                return (o1.offset + (o1.stride >> 1) == o2.offset);
            };

            bool zip = zippable(i1.dst, i2.dst) && zippable(i1.src0, i2.src0);
            if (i1.src1) zip = zip && zippable(i1.src1, i2.src1);
            if (i1.src2) zip = zip && zippable(i1.src2, i2.src2);

            if (zip) {
                auto &i = join(i1, i2);
                i.simd *= 2;
                i.dst.stride /= 2;
                i.src0.stride /= 2;
                i.src1.stride /= 2;
                i.src2.stride /= 2;
                std::swap(i1, i2);      /* move joined entry to end for further processing */
                break;
            }
        }
    }

    mergeChanges();
}

// Make an integer operand twice as wide.
static void widen(CopyOperand &op, bool zipping = false)
{
    switch (op.kind) {
        case CopyOperand::GRF:
            op.offset /= 2;
            if (zipping)
                op.stride /= 2;
            break;
        case CopyOperand::Immediate:
            op.value |= op.value << getBits(op.type);
            break;
        case CopyOperand::Flag: stub();
        case CopyOperand::Null: return;
    }

    if (isInt4(op.type))      op.type = DataType::ub;
    else if (isB(op.type)) op.type = DataType::uw;
    else if (isW(op.type)) op.type = DataType::ud;
    else stub();
    op.range = op.type;
}

// Check if an integer operand can be widened.
static bool widenable(const CopyOperand &op, bool zipping = false)
{
    if (op.kind == CopyOperand::Flag) return false;
    if (op.kind != CopyOperand::GRF) return true;
    if (isFP(op.type) || getBytes(op.type) >= 4) return false;
    if (zipping && (op.stride & 1)) return false;
    if (!zipping && (op.stride != 1)) return false;
    if (op.offset & 1) return false;
    return true;
}

// Optimization pass: join adjacent integer operations into larger ones.
// Requires a sorted plan.
//
// Example input:
//    or (16)   r0.0<4>:uw    r10.0<4>:uw   0x1111:uw
//    or (16)   r0.1<4>:uw    r10.1<4>:uw   0x2222:uw
// Output:
//    or (16)   r0.0<2>:ud    r10.0<2>:ud   0x22221111:ud
//
void CopyPlan::optimizeZipAdjacent()
{
    bool changed = false;

    auto ninsn = insns.size();
    for (size_t n2 = 1; n2 < ninsn; n2++) {
        auto &i1 = insns[n2 - 1];
        auto &i2 = insns[n2];

        if (i1.isInvalid() || i1.op != i2.op || i1.simd != i2.simd || i1.phase != i2.phase || i1.dst.grf != i2.dst.grf) continue;
        if (i1.flag || i2.flag || i1.sat || i2.sat) continue;
        if (!isBitwise(i1.op)) continue;

        auto zippable = [](const CopyOperand &o1, const CopyOperand &o2) {
            if (o1.kind != o2.kind) return false;
            if (o1.kind != CopyOperand::GRF) return true;
            if (o1.type != o2.type || o1.stride != o2.stride || o1.grf != o2.grf) return false;
            if (o1.temp != o2.temp) return false;
            if (o1.temp && o1.value != o2.value) return false;
            if (!widenable(o1, true)) return false;
            if (o1.neg != o2.neg) return false;
            if (o1.abs != o2.abs) return false;
            if (o1.inv != o2.inv) return false;
            return (o1.offset + 1 == o2.offset);
        };

        bool zip = zippable(i1.dst, i2.dst) && zippable(i1.src0, i2.src0)
                && asSigned(i1.dst.type) == asSigned(i1.src0.type);
        if (i1.src1)
            zip = zip && zippable(i1.src1, i2.src1) && (asSigned(i1.src1.type) == asSigned(i1.dst.type));
        if (i1.src2)
            zip = zip && zippable(i1.src2, i2.src2) && (asSigned(i1.src2.type) == asSigned(i1.dst.type));

        if (zip) {
            auto &i = join(i1, i2);
            widen(i.dst, true);
            if (i.src0) widen(i.src0, true);
            if (i.src1) widen(i.src1, true);
            if (i.src2) widen(i.src2, true);
            changed = true;
            break;
        }
    }

    if (changed) {
        mergeChanges();
        optimizeZipAdjacent();
    }
}

// Optimization pass: use larger integer types for contiguous operands if possible.
//
// Example input:
//    mov (16)  r0.0<1>:ub   r1.0<1>:ub
// Output:
//    mov (4)   r0.0<1>:ud   r1.0<1>:ud
//
void CopyPlan::optimizeWidenIntegers()
{
    for (auto &i: insns) {
        if (!isBitwise(i.op) || i.flag || i.sat || i.src2) continue;

        while (true) {
            bool doWiden = widenable(i.dst) && widenable(i.src0)
                        && asSigned(i.dst.type) == asSigned(i.src0.type)
                        && !i.src0.neg && !i.src0.inv && !i.src0.abs && i.simd % 2 == 0;

            for (auto op: {&i.src1, &i.src2}) if (*op) {
                doWiden = doWiden && widenable(*op)
                        && (asSigned(op->type) == asSigned(i.dst.type));
            }

            if (!doWiden) break;

            i.simd /= 2;
            widen(i.dst);
            widen(i.src0);
            widen(i.src1);
        }
    }
}

// Optimization pass: concatenate instructions.
//   On the initial pass (initial = true), there is no limit on the SIMD width.
//   Otherwise, do not concatenate beyond SIMD32.
//
// Example input:
//    mov (8)  r0.0<1>:uw  r10.0<1>:uw
//    mov (8)  r0.8<1>:uw  r10.8<1>:uw
// Output:
//    mov (16) r0.0<1>:uw  r10.0<1>:uw
//
void CopyPlan::optimizeConcatenate(bool initial)
{
    auto ninsn = insns.size();
    for (size_t n1 = 0; n1 < ninsn; n1++) {
        for (size_t n2 = n1 + 1; n2 < ninsn; n2++) {
            auto &i1 = insns[n1];
            auto &i2 = insns[n2];

            if (i1.op != i2.op || i1.phase != i2.phase || i1.flag || i2.flag) break;

            auto joinable = [&](const CopyOperand &o1, const CopyOperand &o2, bool *outTooFar = nullptr) {
                if (o1.kind != o2.kind) return false;
                if (o1.kind == CopyOperand::Null) return true;
                if (o1.type != o2.type || o1.stride != o2.stride) return false;
                if (o1.kind == CopyOperand::Immediate) return (o1.value == o2.value);
                if (o1.temp != o2.temp) return false;
                if (o1.temp && (o1.value != o2.value)) return false;
                if (o1.neg != o2.neg) return false;
                if (o1.abs != o2.abs) return false;
                if (o1.inv != o2.inv) return false;
                auto gap = (o2.absByteOffset(hw) - o1.absByteOffset(hw))
                         - elementsToBytes(o1.stride * i1.simd, o1.type);
                if (outTooFar)
                    *outTooFar = (gap > 0);
                return (gap == 0);
            };

            bool tooFar = false;
            bool doJoin = joinable(i1.dst, i2.dst, &tooFar) && joinable(i1.src0, i2.src0)
                       && joinable(i1.src1, i2.src1) && joinable(i1.src2, i2.src2);

            if (!initial)
                doJoin &= (i1.simd + i2.simd <= 32);

            if (tooFar) break;

            if (doJoin) {
                i1.simd += i2.simd;
                (void) join(i1, i2);
            }
        }
    }

    mergeChanges();
}

// Optimization pass: enable write combining for byte writes (XeHPC+).
// Requires a sorted plan.
//
// Example input:
//    mov (8)  r0.0<4>:ub  r10.0<1>:ub
//    mov (8)  r0.1<4>:ub  r20.0<1>:ub
//    mov (8)  r0.2<4>:ub  r30.0<1>:ub
//    mov (8)  r0.3<4>:ub  r40.0<1>:ub
// Output:
//    mov (8)  r0.0<4>:ub  r10.0<1>:ub  {Atomic}
//    mov (8)  r0.1<4>:ub  r20.0<1>:ub  {Atomic}
//    mov (8)  r0.2<4>:ub  r30.0<1>:ub  {Atomic}
//    mov (8)  r0.3<4>:ub  r40.0<1>:ub
//
void CopyPlan::optimizeWriteCombine()
{
    auto ninsn = insns.size();

    if (hw < HW::XeHPC) return;

    for (size_t n = 0; n + 1 < ninsn; ) {
        auto &i0 = insns[n];

        auto canWC = [](HW hw, CopyInstruction &i) {
            auto st = i.src0.type;
            if (i.op != Opcode::mov || i.flag) return false;
            if (!isB(i.dst.type)) return false;
            if (!(isB(st) || isW(st) || isD(st) || st == DataType::f)) return false;
            if (multiGRF(hw, i, i.dst)) return false;
            return true;
        };

        if (!canWC(hw, i0)) {
            n++; continue;
        }

        auto cnumMin = i0.cnumMin, cnumMax = i0.cnumMax;
        size_t n1;
        for (n1 = n + 1; n1 < ninsn; n1++) {
            auto &i1 = insns[n1];
            if (!canWC(hw, i1)) break;
            if (i1.dst.grf != i0.dst.grf) break;
            if (i1.dst.offset + n != i0.dst.offset + n1) break;
            cnumMin = std::min(cnumMin, i1.cnumMin);
            cnumMax = std::max(cnumMax, i1.cnumMax);
        }

        auto length = int(rounddown_pow2(n1 - n));
        for (n1 = n; n1 + 1 < n + length; n1++)
            insns[n1].atomic = true;
        for (n1 = n; n1 < n + length; n1++) {
            insns[n1].cnumMin = cnumMin;
            insns[n1].cnumMax = cnumMax;
        }

        n += length;
    }
}

// Optimization pass: spread writes to bytes in the same word (XeHPC+).
//   This reduces false WAW dependencies between the instructions.
// Requires a sorted plan.
//
// Example input:
//    shr (8)  r0.0<4>:ub  r10.0<1>:ub  4:uw
//    shr (8)  r0.1<4>:ub  r20.0<1>:ub  4:uw    // would be {@1}
//    shr (8)  r0.2<4>:ub  r30.0<1>:ub  4:uw
//    shr (8)  r0.3<4>:ub  r40.0<1>:ub  4:uw    // would be {@1}
// Output:
//    shr (8)  r0.0<4>:ub  r10.0<1>:ub  4:uw
//    shr (8)  r0.2<4>:ub  r30.0<1>:ub  4:uw
//  ...
//    shr (8)  r0.1<4>:ub  r20.0<1>:ub  4:uw
//    shr (8)  r0.3<4>:ub  r40.0<1>:ub  4:uw
//
void CopyPlan::optimizeWriteSpread()
{
    if (hw < HW::XeHPC) return;

    auto ninsn = insns.size();
    for (size_t n = 1; n < ninsn; n++) {
        auto &iprev = insns[n - 1];
        auto &i = insns[n];

        if (isB(i.dst.type) && i.dst.stride > 1 && i.dst.offset & 1 && !iprev.atomic) {
            (void) split(i, false);
            i.invalidate();
        }
    }

    mergeChanges();
}

// Optimization pass: reduce excess source bits in integer downconversions.
//
// Example input:
//    mov (8)  r0.0<1>:ub  r10.0<1>:ud
// Output:
//    mov (8)  r0.0<1>:ub  r10.0<4>:ub
//
void CopyPlan::optimizeIntegerDownconvert()
{
    for (auto &i: insns) {
        if (i.op != Opcode::mov || i.sat) continue;
        if (!isInt(i.dst.type) || !isInt(i.src0.type)) continue;

        int expand = getBytes(i.src0.type) / getBytes(i.dst.type);
        if (expand > 1) {
            i.src0.type = i.dst.type;
            i.src0.offset *= expand;
            i.src0.stride *= expand;
        }
    }
}

// Optimization/cleanup pass: remove unneeded saturation modifiers.
void CopyPlan::optimizeSaturate()
{
    for (auto &i: insns) {
        if (i.op != Opcode::mov) continue;
        i.sat &= !isSubsetOf(i.src0.type, i.dst.type);
    }
}

// Materialize temporary GRF and flag registers in a copy plan, replacing
//   them by physical GRF and flag registers.
// Instructions will be reordered as needed if there are not enough temporary
//   resources to give each temporary a distinct physical resource.
void CopyPlan::materializeTemps(const GRFAllocator &grfAllocator, const FlagAllocator &flagAllocator)
{
    std::vector<CopyInstruction> sortedInsns;
    std::vector<GRFRange> grfAllocs;
    std::vector<FlagRegister> flagAllocs;
    int ncnum = 0;

    sortedInsns.reserve(insns.size());
    grfAllocs.reserve(temps.size());
    flagAllocs.reserve(temps.size());

    /* Round up instruction usage by each temporary */
    for (auto &i: insns) {
        for (auto o: {&i.dst, &i.src0, &i.src1, &i.src2, &i.flag})
            if (*o && o->temp) temps[o->value].usedBy(i);
        ncnum = std::max(ncnum, i.cnumMax + 1);
    }

    /* Check which instruction groups must be issued together */
    std::vector<bool> joined(ncnum);
    for (auto &i: insns)
        for (int cnum = i.cnumMin; cnum < i.cnumMax; cnum++)
            joined[cnum] = true;

    /* Sort instructions and temporaries by parent instruction (cnum) */
    std::vector<std::pair<int, int>> cnumOrder;
    cnumOrder.reserve(temps.size());

    for (size_t t = 0; t < temps.size(); t++)
        cnumOrder.push_back(std::make_pair(temps[t].cnumMin, int(t)));
    std::sort(cnumOrder.begin(), cnumOrder.end());

    for (int cnum0 = 0; cnum0 < ncnum; ) {
        int cnum1 = ncnum;

        /* Allocate temporaries until we run out of space */
        for (size_t ti = 0; ti < temps.size(); ti++) {
            auto &temp = temps[cnumOrder[ti].second];
            bool ok = false;

            if (temp.cnumMax < cnum0) continue;

            if (temp.flag) {
                FlagRegister flag;
                flagAllocator(temp.bytes, flag);
                ok = flag.isValid();
                if (ok) {
                    flagAllocs.push_back(flag);
                    temp.assignment = flag.index();
                }
            } else {
                GRFRange range;
                grfAllocator(div_up(temp.bytes, GRF::bytes(hw)), range);
                ok = range.isValid();
                if (ok) {
                    grfAllocs.push_back(range);
                    temp.assignment = range.getBase();
                }
            }

            if (!ok) {
                cnum1 = temp.cnumMin; break;
            }
        }

        /* Back off to the nearest instruction group boundary */
        while (cnum1 > 0 && joined[cnum1 - 1])
            cnum1--;
        if (cnum1 <= cnum0)
            throw out_of_registers_exception();

        /* Issue instructions for this batch of instruction groups */
        for (const auto &i: insns)
            if (i.cnumMin >= cnum0 && i.cnumMax < cnum1)
                sortedInsns.push_back(i);

        /* Release temporaries for next round. */
        for (auto &range: grfAllocs) grfAllocator(0, range);
        for (auto &flag: flagAllocs) flagAllocator(0, flag);

        grfAllocs.clear();
        flagAllocs.clear();

        cnum0 = cnum1;
    }

    std::swap(insns, sortedInsns);

    /* Update operands with assignments */
    for (auto &i: insns) {
        for (auto o: {&i.dst, &i.src0, &i.src1, &i.src2, &i.flag}) {
            if (o->temp) {
                o->temp = false;
                o->grf += temps[o->value].assignment;
            }
        }
    }

    temps.clear();
}


#include "internal/namespace_end.hxx"
