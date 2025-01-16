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

/*
 * Do not #include this file directly; ngen uses it internally.
 */


// Pseudo-instructions and macros.
template <typename DT = void>
void min_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
    sel<DT>(mod | lt | f0[0], dst, src0, src1, loc);
}
template <typename DT = void>
void min_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
    sel<DT>(mod | lt | f0[0], dst, src0, src1, loc);
}
#ifndef NGEN_WINDOWS_COMPAT
template <typename DT = void>
void min(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
    sel<DT>(mod | lt | f0[0], dst, src0, src1, loc);
}
template <typename DT = void>
void min(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
    sel<DT>(mod | lt | f0[0], dst, src0, src1, loc);
}
#endif
template <typename DT = void>
void max_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
    sel<DT>(mod | ge | f0[0], dst, src0, src1, loc);
}
template <typename DT = void>
void max_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
    sel<DT>(mod | ge | f0[0], dst, src0, src1, loc);
}
#ifndef NGEN_WINDOWS_COMPAT
template <typename DT = void>
void max(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
    sel<DT>(mod | ge | f0[0], dst, src0, src1, loc);
}
template <typename DT = void>
void max(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
    sel<DT>(mod | ge | f0[0], dst, src0, src1, loc);
}
#endif

template <typename DT = void>
void bfi(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, const RegData &src3, SourceLocation loc = {}) {
    bfi1<DT>(mod, dst, src0, src1, loc);
    bfi2<DT>(mod, dst, dst, src2, src3, loc);
}

// Brief compare instructions.
template <typename DT = void>
void cmp(const InstructionModifier &mod, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
    auto dt = getDataType<DT>();
    if (dt == DataType::invalid)
        dt = src0.getType();
    cmp<DT>(mod, null.retype(dt), src0, src1, loc);
}
template <typename DT = void>
void cmp(const InstructionModifier &mod, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
    auto dt = getDataType<DT>();
    if (dt == DataType::invalid)
        dt = src0.getType();
    cmp<DT>(mod, null.retype(dt), src0, src1, loc);
}

// Brief math instructions.
template <typename DT = void>
void cos(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::cos, dst, src0, loc);
}
template <typename DT = void>
void exp(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::exp, dst, src0, loc);
}
template <typename DT = void>
void fdiv(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::fdiv, dst, src0, src1, loc);
}
template <typename DT = void>
void fdiv(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::fdiv, dst, src0, src1, loc);
}
template <typename DT = void>
void idiv(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::idiv, dst, src0, src1, loc);
}
template <typename DT = void>
void idiv(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::idiv, dst, src0, src1, loc);
}
template <typename DT = void>
void inv(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::inv, dst, src0, loc);
}
template <typename DT = void>
void invm(const InstructionModifier &mod, const ExtendedReg &dst, const ExtendedReg &src0, const ExtendedReg &src1, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::invm, dst, src0, src1, loc);
}
template <typename DT = void>
void iqot(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::iqot, dst, src0, src1, loc);
}
template <typename DT = void>
void iqot(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::iqot, dst, src0, src1, loc);
}
template <typename DT = void>
void irem(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::irem, dst, src0, src1, loc);
}
template <typename DT = void>
void irem(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::irem, dst, src0, src1, loc);
}
template <typename DT = void>
void log(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::log, dst, src0, loc);
}
template <typename DT = void>
void pow(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::pow, dst, src0, src1, loc);
}
template <typename DT = void>
void pow(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::pow, dst, src0, src1, loc);
}
template <typename DT = void>
void rsqt(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::rsqt, dst, src0, loc);
}
template <typename DT = void>
void rsqtm(const InstructionModifier &mod, const ExtendedReg &dst, const ExtendedReg &src0, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::rsqtm, dst, src0, loc);
}
template <typename DT = void>
void sin(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::sin, dst, src0, loc);
}
template <typename DT = void>
void sqt(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::sqt, dst, src0, loc);
}

#define TMP(n) tmp[n].retype(dst.getType())

// IEEE 754-compliant divide math macro sequence.
//   Requires GRFs initialized with 0.0 and 1.0, as well as temporary GRFs (4 for single precision, 5 for double precision).
//   dst, num, denom must be distinct GRFs.
template <typename DT = void, typename A>
void fdiv_ieee(const InstructionModifier &mod, FlagRegister flag, RegData dst, RegData num, RegData denom,
               RegData zero, RegData one, const A &tmp, InstructionModifier cfmod = InstructionModifier(),
               SourceLocation loc = {}) {
    DataType dt = getDataType<DT>();
    if (dt == DataType::invalid)
        dt = dst.getType();
    if (cfmod.getExecSize() == 0)
        cfmod = mod;

    Label labelSkip;

    switch (dt) {
        case DataType::hf:
            fdiv<DT>(mod, dst, num, denom, loc);
            break;
        case DataType::f:
            invm<DT>(mod | eo | flag,         dst | mme0,      num | nomme,   denom | nomme, loc);
            if_(cfmod | ~flag, labelSkip, loc);

            madm<DT>(mod, TMP(0) | mme1,     zero | nomme,     num | nomme,     dst | mme0, loc);
            madm<DT>(mod, TMP(1) | mme2,      one | nomme,  -denom | nomme,     dst | mme0, loc);
            madm<DT>(mod, TMP(2) | mme3,      dst | mme0,   TMP(1) | mme2,      dst | mme0, loc);
            madm<DT>(mod, TMP(3) | mme4,      num | nomme,  -denom | nomme,  TMP(0) | mme1, loc);
            madm<DT>(mod, TMP(0) | mme5,   TMP(0) | mme1,   TMP(3) | mme4,   TMP(2) | mme3, loc);
            madm<DT>(mod, TMP(1) | mme6,      num | nomme,  -denom | nomme,  TMP(0) | mme5, loc);
            madm<DT>(mod,    dst | nomme,  TMP(0) | mme5,   TMP(1) | mme6,   TMP(2) | mme3, loc);

            mark(labelSkip);
            endif(cfmod, loc);
            break;
        case DataType::df:
            invm<DT>(mod | eo | flag,         dst | mme0,      num | nomme,   denom | nomme, loc);
            if_(cfmod | ~flag, labelSkip, loc);

            madm<DT>(mod, TMP(0) | mme1,     zero | nomme,     num | nomme,     dst | mme0, loc);
            madm<DT>(mod, TMP(1) | mme2,      one | nomme,  -denom | nomme,     dst | mme0, loc);
            madm<DT>(mod, TMP(2) | mme3,      num | nomme,  -denom | nomme,  TMP(0) | mme1, loc);
            madm<DT>(mod, TMP(3) | mme4,      dst | mme0,   TMP(1) | mme2,      dst | mme0, loc);
            madm<DT>(mod, TMP(4) | mme5,      one | nomme,  -denom | nomme,  TMP(3) | mme4, loc);
            madm<DT>(mod,    dst | mme6,      dst | mme0,   TMP(1) | mme2,   TMP(3) | mme4, loc);
            madm<DT>(mod, TMP(0) | mme7,   TMP(0) | mme1,   TMP(2) | mme3,   TMP(3) | mme4, loc);
            madm<DT>(mod, TMP(3) | mme0,   TMP(3) | mme4,      dst | mme6,   TMP(4) | mme5, loc);
            madm<DT>(mod, TMP(2) | mme1,      num | nomme,  -denom | nomme,  TMP(0) | mme7, loc);
            madm<DT>(mod,    dst | nomme,  TMP(0) | mme7,   TMP(2) | mme1,   TMP(3) | mme0, loc);

            mark(labelSkip);
            endif(cfmod, loc);
            break;
        default:
#ifdef NGEN_SAFE
            throw invalid_type_exception();
#endif
            break;
    }
}

// IEEE 754-compliant reciprocal math macro sequence.
//   Requires GRF initialized with 1.0, as well as 3 temporary GRFs.
//   dst and src must be distinct GRFs.
template <typename DT = void, typename A>
void inv_ieee(const InstructionModifier &mod, FlagRegister flag, RegData dst, RegData src, RegData one,
              const A &tmp, InstructionModifier cfmod = InstructionModifier(), SourceLocation loc = {}) {
    DataType dt = getDataType<DT>();
    if (dt == DataType::invalid)
        dt = dst.getType();
    if (cfmod.getExecSize() == 0)
        cfmod = mod;

    Label labelSkip;

    switch (dt) {
        case DataType::hf:
            inv<DT>(mod, dst, src, loc);
            break;
        case DataType::f:
            invm<DT>(mod | eo | flag,         dst | mme0,      one | nomme,     src | nomme, loc);
            if_(cfmod | ~flag, labelSkip, loc);

            madm<DT>(mod, TMP(1) | mme2,      one | nomme,    -src | nomme,     dst | mme0, loc);
            madm<DT>(mod, TMP(2) | mme3,      dst | mme0,   TMP(1) | mme2,      dst | mme0, loc);
            madm<DT>(mod, TMP(0) | mme5,      dst | mme0,   TMP(1) | mme2,   TMP(2) | mme3, loc);
            madm<DT>(mod, TMP(1) | mme6,      one | nomme,    -src | nomme,  TMP(0) | mme5, loc);
            madm<DT>(mod,    dst | nomme,  TMP(0) | mme5,   TMP(1) | mme6,   TMP(2) | mme3, loc);

            mark(labelSkip);
            endif(cfmod, loc);
            break;
        case DataType::df:
            invm<DT>(mod | eo | flag,        dst | mme0,      one | nomme,     src | nomme, loc);
            if_(cfmod | ~flag, labelSkip, loc);

            madm<DT>(mod, TMP(0) | mme2,     one | nomme,    -src | nomme,     dst | mme0, loc);
            madm<DT>(mod, TMP(1) | mme4,     dst | mme0,   TMP(0) | mme2,      dst | mme0, loc);
            madm<DT>(mod, TMP(2) | mme5,     one | nomme,    -src | nomme,  TMP(1) | mme4, loc);
            madm<DT>(mod,    dst | mme6,     dst | mme0,   TMP(0) | mme2,   TMP(1) | mme4, loc);
            madm<DT>(mod, TMP(1) | mme0,  TMP(1) | mme4,      dst | mme6,   TMP(2) | mme5, loc);
            madm<DT>(mod, TMP(0) | mme1,     one | nomme,    -src | nomme,     dst | mme6, loc);
            madm<DT>(mod,    dst | nomme,    dst | mme6,   TMP(0) | mme1,   TMP(1) | mme0, loc);

            mark(labelSkip);
            endif(cfmod, loc);
            break;
        default:
#ifdef NGEN_SAFE
            throw invalid_type_exception();
#endif
            break;
    }
}

// IEEE 754-compliant square root macro sequence.
//   Requires GRFs initialized with 0.0 and 0.5 (also 1.0 for double precision),
//     and temporary GRFs (3 for single precision, 4 for double precision).
//   dst and src must be distinct GRFs.
template <typename DT = void, typename A>
void sqt_ieee(const InstructionModifier &mod, FlagRegister flag, RegData dst, RegData src,
              RegData zero, RegData oneHalf, RegData one, const A &tmp, InstructionModifier cfmod = InstructionModifier(),
              SourceLocation loc = {}) {
    DataType dt = getDataType<DT>();
    if (dt == DataType::invalid)
        dt = dst.getType();
    if (cfmod.getExecSize() == 0)
        cfmod = mod;

    Label labelSkip;

    switch (dt) {
        case DataType::hf:
            sqt<DT>(mod, dst, src, loc);
            break;
        case DataType::f:
            rsqtm<DT>(mod | eo | flag,        dst | mme0,       src | nomme, loc);
            if_(cfmod | ~flag, labelSkip, loc);

            madm<DT>(mod, TMP(0) | mme1,     zero | nomme,  oneHalf | nomme,     dst | mme0, loc);
            madm<DT>(mod, TMP(1) | mme2,     zero | nomme,      src | nomme,     dst | mme0, loc);
            madm<DT>(mod, TMP(2) | mme3,  oneHalf | nomme,  -TMP(1) | mme2,   TMP(0) | mme1, loc);
            madm<DT>(mod, TMP(0) | mme4,   TMP(0) | mme1,    TMP(2) | mme3,   TMP(0) | mme1, loc);
            madm<DT>(mod,    dst | mme5,   TMP(1) | mme2,    TMP(2) | mme3,   TMP(1) | mme2, loc);
            madm<DT>(mod, TMP(2) | mme6,      src | nomme,     -dst | mme5,      dst | mme5, loc);
            madm<DT>(mod,    dst | nomme,     dst | mme5,    TMP(0) | mme4,   TMP(2) | mme6, loc);

            mark(labelSkip);
            endif(cfmod, loc);
            break;
        case DataType::df:
            rsqtm<DT>(mod | eo | flag,        dst | mme0,       src | nomme, loc);
            if_(cfmod | ~flag, labelSkip, loc);

            madm<DT>(mod, TMP(0) | mme1,     zero | mme0,   oneHalf | nomme,     dst | mme0, loc);
            madm<DT>(mod, TMP(1) | mme2,     zero | mme0,       src | nomme,     dst | mme0, loc);
            madm<DT>(mod, TMP(2) | mme3,  oneHalf | nomme,  -TMP(1) | mme2,   TMP(0) | mme1, loc);
            madm<DT>(mod, TMP(3) | mme4,      one | nomme,  oneHalf | nomme,     dst | nomme, loc);
            madm<DT>(mod, TMP(3) | mme5,      one | nomme,   TMP(3) | mme4,   TMP(2) | mme3, loc);
            madm<DT>(mod,    dst | mme6,     zero | mme0,    TMP(2) | mme3,   TMP(1) | mme2, loc);
            madm<DT>(mod, TMP(2) | mme7,     zero | mme0,    TMP(2) | mme3,   TMP(0) | mme1, loc);
            madm<DT>(mod,    dst | mme6,   TMP(1) | mme2,    TMP(3) | mme5,      dst | mme6, loc);
            madm<DT>(mod, TMP(3) | mme5,   TMP(0) | mme1,    TMP(3) | mme5,   TMP(2) | mme7, loc);
            madm<DT>(mod, TMP(0) | mme1,      src | nomme,     -dst | mme6,      dst | mme6, loc);
            madm<DT>(mod,    dst | nomme,     dst | mme6,    TMP(0) | mme1,   TMP(3) | mme5, loc);

            mark(labelSkip);
            endif(cfmod, loc);
            break;
        default:
#ifdef NGEN_SAFE
            throw invalid_type_exception();
#endif
            break;
    }
}

#undef TMP

// Thread spawner messages.
void threadend(const InstructionModifier &mod, const RegData &r0_info, SourceLocation loc = {}) {
    {
        auto sf = (hardware <= HW::XeHP) ? SharedFunction::ts
                                         : SharedFunction::gtwy;
        uint32_t exdesc = 0x20 | (static_cast<int>(sf) & 0xF);
        send(8 | EOT | mod | NoMask, null, r0_info, exdesc, 0x2000010, loc);
    }
}

void threadend(const RegData &r0_info, SourceLocation loc = {}) {
    threadend(InstructionModifier(), r0_info, loc);
}

// Gateway messages.
void barriermsg(const InstructionModifier &mod, const GRF &header, SourceLocation loc = {}) {
    {
        uint32_t exdesc = static_cast<int>(SharedFunction::gtwy) & 0xF;
        send(1 | mod | NoMask, null, header, exdesc, 0x2000004, loc);
    }
}

void barriermsg(const GRF &header, SourceLocation loc = {}) { barriermsg(InstructionModifier(), header, loc); }

// Prepare barrier header.
void barrierheader(const GRF &header, const GRF &r0_info = r0, SourceLocation loc = {}) {
    if (hardware >= HW::XeHPG) {
        mov(1 | NoMask, header.hf(4), Immediate::hf(0), loc);
        mov(2 | NoMask, header.ub(10)(1), r0_info.ub(11)(0), loc);
    } else
        and_(8 | NoMask, header.ud(), r0_info.ud(2), uint32_t((hardware >= HW::Gen11) ? 0x7F000000 : 0x8F000000), loc);
}

void barrierheader(const GRF &header, uint32_t threadCount, const GRF &r0_info = r0, SourceLocation loc = {}) {
    if (hardware >= HW::XeHPG)
        mov(1 | NoMask, header.ud(2), (threadCount << 24) | (threadCount << 16), loc);
    else {
        and_(8 | NoMask, header.ud(), r0_info.ud(2), uint32_t((hardware >= HW::Gen11) ? 0x7F000000 : 0x8F000000), loc);
        mov(1 | NoMask, header.ub(9), 0x80 | (threadCount & 0x7F), loc);
    }
}

void barriersignal(const InstructionModifier &mod, const GRF &temp, const GRF &r0_info = r0, SourceLocation loc = {}) {
    {
        barrierheader(temp, r0_info, loc);
        barriermsg(mod, temp, loc);
    }
}

void barriersignal(const InstructionModifier &mod, const GRF &temp, uint32_t threadCount, const GRF &r0_info = r0, SourceLocation loc = {}) {
    barrierheader(temp, threadCount, r0_info, loc);
    barriermsg(mod, temp, loc);
}

void barriersignal(const GRF &temp, const GRF &r0_info = r0, SourceLocation loc = {}) { barriersignal(InstructionModifier(), temp, r0_info, loc); }
void barriersignal(const GRF &temp, uint32_t threadCount, const GRF &r0_info = r0, SourceLocation loc = {}) { barriersignal(InstructionModifier(), temp, threadCount, r0_info, loc); }

// Named barriers.
void nbarriermsg(const InstructionModifier &mod, const GRF &header, SourceLocation loc = {}) {
        barriermsg(mod, header, loc);
}

void nbarriermsg(const GRF &header, SourceLocation loc = {}) { nbarriermsg(InstructionModifier(), header, loc); }

void barriersignal(const InstructionModifier &mod, uint32_t barrierID, const GRF &temp, const GRF &r0_info = r0, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
    if (hardware < HW::XeHPC)
        throw unsupported_message();
#endif
    mov(1 | NoMask, temp.uw(4), uint8_t(barrierID), loc);
    mov(2 | NoMask, temp.ub(10)(1), r0_info.ub(11)(0), loc);
    nbarriermsg(mod, temp, loc);
}

void barriersignal(const InstructionModifier &mod, uint32_t barrierID, const GRF &temp, BarrierType barrierType, uint32_t producers, uint32_t consumers, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
    if (hardware < HW::XeHPC)
        throw unsupported_message();
#endif
    mov(1 | NoMask, temp.ud(2), (barrierID & 0xFF) | (static_cast<uint32_t>(barrierType) << 14) | ((producers & 0xFF) << 16) | ((consumers & 0xFF) << 24), loc);
    nbarriermsg(mod, temp, loc);
}

void barriersignal(uint32_t barrierID, const GRF &temp, const GRF &r0_info = r0, SourceLocation loc = {}) { barriersignal(InstructionModifier(), barrierID, temp, r0_info, loc); }
void barriersignal(uint32_t barrierID, const GRF &temp, BarrierType barrierType, uint32_t producers, uint32_t consumers, SourceLocation loc = {}) { barriersignal(InstructionModifier(), barrierID, temp, barrierType, producers, consumers, loc); }

void barrierwait(SourceLocation loc = {}) {
    if (isGen12)
        sync.bar(NoMask, loc);
    else
        wait(NoMask, n0[0], loc);
}

void barrier(const InstructionModifier &mod, const GRF &temp, const GRF &r0_info = r0,
             SourceLocation loc = {}) {
    barriersignal(mod, temp, r0_info, loc);
    barrierwait(loc);
}

void barrier(const InstructionModifier &mod, const GRF &temp, uint32_t threadCount,
             const GRF &r0_info = r0, SourceLocation loc = {}) {
    barriersignal(mod, temp, threadCount, r0_info, loc);
    barrierwait(loc);
}

void barrier(const GRF &temp, const GRF &r0_info = r0, SourceLocation loc = {}) {
    barriersignal(InstructionModifier(), temp, r0_info, loc);
    barrierwait(loc);
}

void barrier(const GRF &temp, uint32_t threadCount, const GRF &r0_info = r0,
             SourceLocation loc = {}) {
    barriersignal(temp, threadCount, r0_info, loc);
    barrierwait(loc);
}

void barrier(const InstructionModifier &mod, uint32_t barrierID,
             const GRF &temp, const GRF &r0_info = r0,
             SourceLocation loc = {}) {
    barriersignal(mod, barrierID, temp, r0_info, loc);
    barrierwait(loc);
}

void barrier(const InstructionModifier &mod, uint32_t barrierID,
             const GRF &temp, BarrierType barrierType, uint32_t producers,
             uint32_t consumers, SourceLocation loc = {}) {
    barriersignal(mod, barrierID, temp, barrierType, producers, consumers, loc);
    barrierwait(loc);
}

void barrier(uint32_t barrierID, const GRF &temp, const GRF &r0_info = r0,
             SourceLocation loc = {}) {
    barriersignal(barrierID, temp, r0_info, loc);
    barrierwait(loc);
}

void barrier(uint32_t barrierID, const GRF &temp, BarrierType barrierType,
             uint32_t producers, uint32_t consumers, SourceLocation loc = {}) {
    barriersignal(barrierID, temp, barrierType, producers, consumers, loc);
    barrierwait(loc);
}

void registerfence(const RegData &dst, SourceLocation loc = {}) {
    _lastFenceDst = dst;
    if (isGen12) {
        _lastFenceLabel = Label();
        mark(_lastFenceLabel);
    }
}

// Global memory fence.
void memfence(const InstructionModifier &mod, FenceScopeLSC scope, FlushTypeLSC flushing, const RegData &dst = NullRegister(), const RegData &header = GRF(0), SourceLocation loc = {}) {
    registerfence(dst, loc);

    if (hardware >= HW::XeHPG) {
        if (flushing == FlushTypeLSC::None && hardware == HW::XeHPG && scope > FenceScopeLSC::Subslice)
            flushing = static_cast<FlushTypeLSC>(6);    /* workaround for DG2 bug */

        uint32_t desc = 0x0210011F;
        desc |= static_cast<uint32_t>(scope) << 9;
        desc |= static_cast<uint32_t>(flushing) << 12;
        send(1 | mod | NoMask, SharedFunction::ugm, dst, header, null, 0, desc, loc);
    } else {
        const uint32_t exdesc = static_cast<int>(SharedFunction::dc0) & 0xF;
        send(8 | mod | NoMask, dst, header, exdesc, 0x219E000, loc);
    }
}

void memfence(const InstructionModifier &mod, const RegData &dst = NullRegister(), const RegData &header = GRF(0), SourceLocation loc = {}) {
    memfence(mod, FenceScopeLSC::GPU, FlushTypeLSC::None, dst, header, loc);
}

void memfence(FenceScopeLSC scope, FlushTypeLSC flushing, const RegData &dst = NullRegister(), const RegData &header = GRF(0), SourceLocation loc = {}) {
    memfence(InstructionModifier(), scope, flushing, dst, header, loc);
}

void memfence(const RegData &dst = NullRegister(), const RegData &header = GRF(0), SourceLocation loc = {}) {
    memfence(InstructionModifier(), dst, header, loc);
}

// SLM-only memory fence.
void slmfence(const InstructionModifier &mod, const RegData &dst = NullRegister(), const RegData &header = GRF(0), SourceLocation loc = {}) {
    registerfence(dst, loc);

    if (hardware >= HW::XeHPG)
        send(1 | mod | NoMask, SharedFunction::slm, dst, header, null, 0, 0x210011F, loc);
    else {
        const uint32_t exdesc = static_cast<int>(SharedFunction::dc0) & 0xF;
        send(8 | mod | NoMask, dst, header, exdesc, 0x219E0FE, loc);
    }
}

void slmfence(const RegData &dst = NullRegister(), const RegData &header = GRF(0), SourceLocation loc = {}) { slmfence(InstructionModifier(), dst, header, loc); }

// Wait on the last global memory or SLM fence.
void fencewait(SourceLocation loc = {}) {
    if (isGen12)
        fencedep(_lastFenceLabel, loc);
    else
        mov<uint32_t>(8 | NoMask, null, _lastFenceDst, loc);
}

// XeHP+ prologues.
void loadlid(int argBytes, int dims = 3, int simd = 8, const GRF &temp = GRF(127), int paddedSize = 0, SourceLocation loc = {}) {
    if (hardware >= HW::XeHP) {
        if (paddedSize < 0)
            paddedSize = 12*16;
        const int grfSize = GRF::bytes(hardware);
        const int grfOW = grfSize / 16;
        int simdGRFs = (simd > 16 && grfSize < 64) ? 2 : 1;
        int insns = 0;
        const bool lsc = (hardware >= HW::XeHPG);
        auto tempAddr = temp[lsc ? 0 : 2];

        if (dims > 0) {
            auto dmSave = defaultModifier;
            defaultModifier |= NoMask | AutoSWSB;


            {
                insns = lsc ? 5 : 6;
                if (!lsc)
                    mov<uint32_t>(8, temp, uint16_t(0), loc);
                and_<uint32_t>(1, temp[2], r0[0], uint32_t(~0x1F), loc);
                and_<uint16_t>(1, temp[0], r0[4], uint16_t(0xFF), loc);
                add<uint32_t>(1, temp[2], temp[2], uint16_t(argBytes), loc);
                if (simd == 1) {
                    mad<uint32_t>(1, tempAddr, temp[2], temp.uw(0), uint16_t(grfSize), loc);
                    lsc ? load(1, r1, D32T(4) | L1C_L3C,      A32,   temp, loc)
                        : load(8, r1, aligned_block_oword(1), A32NC, temp, loc);
                } else {
                    mad<uint32_t>(1, tempAddr, temp[2], temp.uw(0), uint16_t(3 * simdGRFs * grfSize), loc);
                    lsc ? load(1, r1, D32T(simdGRFs * ((dims == 1) ? 1 : 2) * grfOW * 4) | L1C_L3C,  A32,   temp, loc)
                        : load(8, r1, aligned_block_oword(simdGRFs * ((dims == 1) ? 1 : 2) * grfOW), A32NC, temp, loc);
                    if (dims == 3) {
                        add<uint32_t>(1, tempAddr, tempAddr, uint16_t(2 * simdGRFs * grfSize), loc);
                        lsc ? load(1, GRF(1 + 2 * simdGRFs), D32T(grfOW * 4 * simdGRFs) | L1C_L3C,  A32,   temp, loc)
                            : load(8, GRF(1 + 2 * simdGRFs), aligned_block_oword(grfOW * simdGRFs), A32NC, temp, loc);
                        insns += 2;
                    }
                }
            }

            defaultModifier = dmSave;
        }

        if (paddedSize > 0) {
            int nops = (paddedSize >> 4) - insns;
#ifdef NGEN_SAFE
            if (paddedSize & 0xF) throw invalid_operand_exception();
            if (nops < 0)         throw invalid_operand_exception();
#endif
            for (int i = 0; i < nops; i++)
                nop(loc);
        }

        if (!_labelLocalIDsLoaded.defined(labelManager))
            mark(_labelLocalIDsLoaded);

    }
}

void loadargs(const GRF &base, int argGRFs, const GRF &temp = GRF(127), bool inPrologue = true, SourceLocation loc = {}) {
    if (hardware >= HW::XeHP) {
        if (argGRFs > 0) {
            const bool lsc = (hardware >= HW::XeHPG);
            auto tempAddr = temp[lsc ? 0 : 2];
            auto dst = base;
            auto dmSave = defaultModifier;
            defaultModifier |= NoMask | AutoSWSB;

            {
                if (!lsc)
                    mov<uint32_t>(8, temp, uint16_t(0), loc);
                and_<uint32_t>(1, tempAddr, r0[0], uint32_t(~0x1F), loc);
                while (argGRFs > 0) {
                    int nload = std::min(utils::rounddown_pow2(argGRFs), lsc ? 8 : 4);
                    int loadBytes = nload * GRF::bytes(hardware);
                    lsc ? load(1, dst, D64T(loadBytes >> 3) | L1C_L3C,      A32,   temp, loc)
                        : load(8, dst, aligned_block_oword(loadBytes >> 4), A32NC, temp, loc);
                    argGRFs -= nload;
                    dst += nload;
                    if (argGRFs > 0)
                        add<uint32_t>(1, tempAddr, tempAddr, uint32_t(loadBytes), loc);
                }
            }

            defaultModifier = dmSave;
        }

        if (!_labelArgsLoaded.defined(labelManager))
            mark(_labelArgsLoaded);
    }
}

void epilogue(int GRFCount, bool hasSLM, const RegData &r0_info, SourceLocation loc = {}) {
    GRF tmp0(GRFCount - 3);
    GRF tmp1(GRFCount - 2);
    GRF r0_copy(GRFCount - 4);

    bool doMemFence = false;
    bool doSLMFence = false;
    bool setAccToZero = false;

    switch (hardware) {
        case HW::XeLP:
        case HW::XeHP:
            doMemFence = true;
            doSLMFence = true;
            setAccToZero = true;
            break;
        case HW::XeHPG:
            setAccToZero = true;
            break;
        default: break;
    }

    if (!hasSLM) doSLMFence = false;

    int dwordsPerReg = GRF::bytes(hardware) / sizeof(uint32_t);
    mov<uint32_t>(dwordsPerReg, r0_copy, r0_info, loc);

    if (doMemFence) memfence(tmp0, r0_info, loc);
    if (doSLMFence) slmfence(tmp1, r0_info, loc);

    if (setAccToZero) {
        mov(16, acc0.f(), 0.f, loc);
        if (hardware == HW::XeHP) mov(16, acc2.f(), 0.f, loc);
    }

    if (doMemFence) wrdep(tmp0, loc);
    if (doSLMFence) wrdep(tmp1, loc);

    threadend(r0_copy, loc);
}


private:

struct Load {
    _self &parent;

    Load(_self *parent_) : parent(*parent_) {}

    template <typename DataSpec>
    void operator()(const InstructionModifier &mod, const RegData &dst, const DataSpec &spec, AddressBase base, const RegData &addr, SourceLocation loc = {})
    {
        this->operator()(SharedFunction::automatic, mod, dst, spec, base, GRFDisp(addr), loc);
    }

    template <typename DataSpec>
    void operator()(const InstructionModifier &mod, const RegData &dst, const DataSpec &spec, AddressBase base, const GRFDisp &addr, SourceLocation loc = {})
    {
        this->operator()(SharedFunction::automatic, mod, dst, spec, base, addr, loc);
    }

    template <typename DataSpec>
    void operator()(SharedFunction sfid, const InstructionModifier &mod, const RegData &dst, const DataSpec &spec, AddressBase base, const GRFDisp &addr, SourceLocation loc = {})
    {
        {
            MessageDescriptor desc;
            ExtendedMessageDescriptor exdesc;

            if (sfid != SharedFunction::automatic)
                exdesc.parts.sfid = static_cast<unsigned>(sfid);
            encodeLoadDescriptors(parent.hardware, desc, exdesc, mod, dst, spec, base, addr);
            if (sfid != SharedFunction::automatic)
                exdesc.parts.sfid = static_cast<unsigned>(sfid);
            parent.send(mod, dst, addr.getBase(), exdesc.all, desc.all, loc);
        }
    }

    void ugm(const InstructionModifier &mod, const RegData &dst, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, SourceLocation loc = {})
    {
        this->operator()(SharedFunction::ugm, mod, dst, spec, base, addr, loc);
    }
    void ugml(const InstructionModifier &mod, const RegData &dst, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, SourceLocation loc = {})
    {
        this->operator()(SharedFunction::ugml, mod, dst, spec, base, addr, loc);
    }
    void tgm(const InstructionModifier &mod, const RegData &dst, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, SourceLocation loc = {})
    {
        this->operator()(SharedFunction::tgm, mod, dst, spec, base, addr, loc);
    }
    void slm(const InstructionModifier &mod, const RegData &dst, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, SourceLocation loc = {})
    {
        this->operator()(SharedFunction::slm, mod, dst, spec, base, addr, loc);
    }
};

struct Store {
    _self &parent;

    Store(_self *parent_) : parent(*parent_) {}

    template <typename DataSpec>
    void operator()(const InstructionModifier &mod, const DataSpec &spec, AddressBase base, const RegData &addr, const RegData &data, SourceLocation loc = {})
    {
        this->operator()(SharedFunction::automatic, mod, spec, base, GRFDisp(addr), data, loc);
    }

    template <typename DataSpec>
    void operator()(const InstructionModifier &mod, const DataSpec &spec, AddressBase base, const GRFDisp &addr, const RegData &data, SourceLocation loc = {})
    {
        this->operator()(SharedFunction::automatic, mod, spec, base, addr, data, {});
    }

    template <typename DataSpec>
    void operator()(SharedFunction sfid, const InstructionModifier &mod, const DataSpec &spec, AddressBase base, const GRFDisp &addr, const RegData &data, SourceLocation loc = {})
    {
        {
            MessageDescriptor desc;
            ExtendedMessageDescriptor exdesc;

            if (sfid != SharedFunction::automatic)
                exdesc.parts.sfid = static_cast<unsigned>(sfid);
            encodeStoreDescriptors(parent.hardware, desc, exdesc, mod, spec, base, addr);
            if (sfid != SharedFunction::automatic)
                exdesc.parts.sfid = static_cast<unsigned>(sfid);
            parent.sends(mod, NullRegister(), addr.getBase(), data, exdesc.all, desc.all, loc);
        }
    }

    void ugm(const InstructionModifier &mod, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, const RegData &data, SourceLocation loc = {})
    {
        this->operator()(SharedFunction::ugm, mod, spec, base, addr, data, loc);
    }
    void ugml(const InstructionModifier &mod, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, const RegData &data, SourceLocation loc = {})
    {
        this->operator()(SharedFunction::ugml, mod, spec, base, addr, data, loc);
    }
    void tgm(const InstructionModifier &mod, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, const RegData &data, SourceLocation loc = {})
    {
        this->operator()(SharedFunction::tgm, mod, spec, base, addr, data, loc);
    }
    void slm(const InstructionModifier &mod, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, const RegData &data, SourceLocation loc = {})
    {
        this->operator()(SharedFunction::slm, mod, spec, base, addr, data, loc);
    }
};

struct Atomic_ {
    _self &parent;

    Atomic_(_self *parent_) : parent(*parent_) {}

    template <typename DataSpec>
    void operator()(AtomicOp op, const InstructionModifier &mod, const RegData &dst, const DataSpec &spec, AddressBase base, const RegData &addr, const RegData &data = NullRegister(), SourceLocation loc = {})
    {
        this->operator()(SharedFunction::automatic, op, mod, dst, spec, base, GRFDisp(addr), data, loc);
    }
    template <typename DataSpec>
    void operator()(AtomicOp op, const InstructionModifier &mod, const DataSpec &spec, AddressBase base, const RegData &addr, const RegData &data = NullRegister(), SourceLocation loc = {})
    {
        this->operator()(SharedFunction::automatic, op, mod, NullRegister(), spec, base, GRFDisp(addr), data, loc);
    }

    template <typename DataSpec>
    void operator()(AtomicOp op, const InstructionModifier &mod, const RegData &dst, const DataSpec &spec, AddressBase base, const GRFDisp &addr, const RegData &data = NullRegister(), SourceLocation loc = {})
    {
        this->operator()(SharedFunction::automatic, op, mod, dst, spec, base, addr, data, loc);
    }
    template <typename DataSpec>
    void operator()(AtomicOp op, const InstructionModifier &mod, const DataSpec &spec, AddressBase base, const GRFDisp &addr, const RegData &data = NullRegister(), SourceLocation loc = {})
    {
        this->operator()(SharedFunction::automatic, op, mod, NullRegister(), spec, base, addr, data, loc);
    }
    template <typename DataSpec>
    void operator()(SharedFunction sfid, AtomicOp op, const InstructionModifier &mod, const RegData &dst, const DataSpec &spec, AddressBase base, const GRFDisp &addr, const RegData &data, SourceLocation loc = {})
    {
        {
            MessageDescriptor desc;
            ExtendedMessageDescriptor exdesc;

            if (sfid != SharedFunction::automatic)
                exdesc.parts.sfid = static_cast<unsigned>(sfid);
            encodeAtomicDescriptors(parent.hardware, desc, exdesc, op, mod, dst, spec, base, addr);
            if (sfid != SharedFunction::automatic)
                exdesc.parts.sfid = static_cast<unsigned>(sfid);
            if (data.isNull())
                parent.send(mod, dst, addr.getBase(), exdesc.all, desc.all, loc);
            else
                parent.sends(mod, dst, addr.getBase(), data, exdesc.all, desc.all, loc);
        }
    }

    void ugm(AtomicOp op, const InstructionModifier &mod, const RegData &dst, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, const RegData &data = NullRegister(), SourceLocation loc = {})
    {
        this->operator()(SharedFunction::ugm, op, mod, dst, spec, base, addr, data, loc);
    }
    void ugm(AtomicOp op, const InstructionModifier &mod, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, const RegData &data = NullRegister(), SourceLocation loc = {})
    {
        this->operator()(SharedFunction::ugm, op, mod, NullRegister(), spec, base, addr, data, loc);
    }
    void ugml(AtomicOp op, const InstructionModifier &mod, const RegData &dst, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, const RegData &data = NullRegister(), SourceLocation loc = {})
    {
        this->operator()(SharedFunction::ugml, op, mod, dst, spec, base, addr, data, loc);
    }
    void ugml(AtomicOp op, const InstructionModifier &mod, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, const RegData &data = NullRegister(), SourceLocation loc = {})
    {
        this->operator()(SharedFunction::ugml, op, mod, NullRegister(), spec, base, addr, data, loc);
    }
    void tgm(AtomicOp op, const InstructionModifier &mod, const RegData &dst, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, const RegData &data = NullRegister(), SourceLocation loc = {})
    {
        this->operator()(SharedFunction::tgm, op, mod, dst, spec, base, addr, data, loc);
    }
    void tgm(AtomicOp op, const InstructionModifier &mod, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, const RegData &data = NullRegister(), SourceLocation loc = {})
    {
        this->operator()(SharedFunction::tgm, op, mod, NullRegister(), spec, base, addr, data, loc);
    }
    void slm(AtomicOp op, const InstructionModifier &mod, const RegData &dst, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, const RegData &data = NullRegister(), SourceLocation loc = {})
    {
        this->operator()(SharedFunction::slm, op, mod, dst, spec, base, addr, data, loc);
    }
    void slm(AtomicOp op, const InstructionModifier &mod, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, const RegData &data = NullRegister(), SourceLocation loc = {})
    {
        this->operator()(SharedFunction::slm, op, mod, NullRegister(), spec, base, addr, data, loc);
    }
};

public:

Load load;
Store store;
Atomic_ atomic;
