/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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
void min_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
    sel(mod | lt | f0[0], dst, src0, src1);
}
template <typename DT = void>
void min_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
    sel(mod | lt | f0[0], dst, src0, src1);
}
#ifndef NGEN_WINDOWS_COMPAT
template <typename DT = void>
void min(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
    sel(mod | lt | f0[0], dst, src0, src1);
}
template <typename DT = void>
void min(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
    sel(mod | lt | f0[0], dst, src0, src1);
}
#endif
template <typename DT = void>
void max_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
    sel(mod | ge | f0[0], dst, src0, src1);
}
template <typename DT = void>
void max_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
    sel(mod | ge | f0[0], dst, src0, src1);
}
#ifndef NGEN_WINDOWS_COMPAT
template <typename DT = void>
void max(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
    sel(mod | ge | f0[0], dst, src0, src1);
}
template <typename DT = void>
void max(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
    sel(mod | ge | f0[0], dst, src0, src1);
}
#endif

template <typename DT = void>
void bfi(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, const RegData &src3) {
    bfi1(mod, dst, src0, src1);
    bfi2(mod, dst, dst, src2, src3);
}

// Brief compare instructions.
template <typename DT = void>
void cmp(const InstructionModifier &mod, const RegData &src0, const RegData &src1) {
    auto dt = getDataType<DT>();
    if (dt == DataType::invalid)
        dt = src0.getType();
    cmp<DT>(mod, null.retype(dt), src0, src1);
}
template <typename DT = void>
void cmp(const InstructionModifier &mod, const RegData &src0, const Immediate &src1) {
    auto dt = getDataType<DT>();
    if (dt == DataType::invalid)
        dt = src0.getType();
    cmp<DT>(mod, null.retype(dt), src0, src1);
}

// Brief math instructions.
template <typename DT = void>
void cos(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
    math<DT>(mod, MathFunction::cos, dst, src0);
}
template <typename DT = void>
void exp(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
    math<DT>(mod, MathFunction::exp, dst, src0);
}
template <typename DT = void>
void fdiv(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
    math<DT>(mod, MathFunction::fdiv, dst, src0, src1);
}
template <typename DT = void>
void fdiv(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
    math<DT>(mod, MathFunction::fdiv, dst, src0, src1);
}
template <typename DT = void>
void idiv(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
    math<DT>(mod, MathFunction::idiv, dst, src0, src1);
}
template <typename DT = void>
void idiv(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
    math<DT>(mod, MathFunction::idiv, dst, src0, src1);
}
template <typename DT = void>
void inv(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
    math<DT>(mod, MathFunction::inv, dst, src0);
}
template <typename DT = void>
void invm(const InstructionModifier &mod, const ExtendedReg &dst, const ExtendedReg &src0, const ExtendedReg &src1) {
    math<DT>(mod, MathFunction::invm, dst, src0, src1);
}
template <typename DT = void>
void iqot(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
    math<DT>(mod, MathFunction::iqot, dst, src0, src1);
}
template <typename DT = void>
void iqot(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
    math<DT>(mod, MathFunction::iqot, dst, src0, src1);
}
template <typename DT = void>
void irem(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
    math<DT>(mod, MathFunction::irem, dst, src0, src1);
}
template <typename DT = void>
void irem(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
    math<DT>(mod, MathFunction::irem, dst, src0, src1);
}
template <typename DT = void>
void log(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
    math<DT>(mod, MathFunction::log, dst, src0);
}
template <typename DT = void>
void pow(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
    math<DT>(mod, MathFunction::pow, dst, src0, src1);
}
template <typename DT = void>
void pow(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
    math<DT>(mod, MathFunction::pow, dst, src0, src1);
}
template <typename DT = void>
void rsqt(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
    math<DT>(mod, MathFunction::rsqt, dst, src0);
}
template <typename DT = void>
void rsqtm(const InstructionModifier &mod, const ExtendedReg &dst, const ExtendedReg &src0) {
    math<DT>(mod, MathFunction::rsqtm, dst, src0);
}
template <typename DT = void>
void sin(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
    math<DT>(mod, MathFunction::sin, dst, src0);
}
template <typename DT = void>
void sqt(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
    math<DT>(mod, MathFunction::sqt, dst, src0);
}

#define TMP(n) tmp[n].retype(dst.getType())

// IEEE 754-compliant divide math macro sequence.
//   Requires GRFs initialized with 0.0 and 1.0, as well as temporary GRFs (4 for single precision, 5 for double precision).
//   dst, num, denom must be distinct GRFs.
template <typename DT = void, typename A>
void fdiv_ieee(const InstructionModifier &mod, FlagRegister flag, RegData dst, RegData num, RegData denom,
               RegData zero, RegData one, const A &tmp, InstructionModifier cfmod = InstructionModifier())
{
    DataType dt = getDataType<DT>();
    if (dt == DataType::invalid)
        dt = dst.getType();
    if (cfmod.getExecSize() == 0)
        cfmod = mod;

    Label labelSkip;

    switch (dt) {
        case DataType::hf:
            fdiv<DT>(mod, dst, num, denom);
            break;
        case DataType::f:
            invm<DT>(mod | eo | flag,         dst | mme0,      num | nomme,   denom | nomme);
            if_(cfmod | ~flag, labelSkip);

            madm<DT>(mod, TMP(0) | mme1,     zero | nomme,     num | nomme,     dst | mme0);
            madm<DT>(mod, TMP(1) | mme2,      one | nomme,  -denom | nomme,     dst | mme0);
            madm<DT>(mod, TMP(2) | mme3,      dst | mme0,   TMP(1) | mme2,      dst | mme0);
            madm<DT>(mod, TMP(3) | mme4,      num | nomme,  -denom | nomme,  TMP(0) | mme1);
            madm<DT>(mod, TMP(0) | mme5,   TMP(0) | mme1,   TMP(3) | mme4,   TMP(2) | mme3);
            madm<DT>(mod, TMP(1) | mme6,      num | nomme,  -denom | nomme,  TMP(0) | mme5);
            madm<DT>(mod,    dst | nomme,  TMP(0) | mme5,   TMP(1) | mme6,   TMP(2) | mme3);

            mark(labelSkip);
            endif(cfmod);
            break;
        case DataType::df:
            invm<DT>(mod | eo | flag,         dst | mme0,      num | nomme,   denom | nomme);
            if_(cfmod | ~flag, labelSkip);

            madm<DT>(mod, TMP(0) | mme1,     zero | nomme,     num | nomme,     dst | mme0);
            madm<DT>(mod, TMP(1) | mme2,      one | nomme,  -denom | nomme,     dst | mme0);
            madm<DT>(mod, TMP(2) | mme3,      num | nomme,  -denom | nomme,  TMP(0) | mme1);
            madm<DT>(mod, TMP(3) | mme4,      dst | mme0,   TMP(1) | mme2,      dst | mme0);
            madm<DT>(mod, TMP(4) | mme5,      one | nomme,  -denom | nomme,  TMP(3) | mme4);
            madm<DT>(mod,    dst | mme6,      dst | mme0,   TMP(1) | mme2,   TMP(3) | mme4);
            madm<DT>(mod, TMP(0) | mme7,   TMP(0) | mme1,   TMP(2) | mme3,   TMP(3) | mme4);
            madm<DT>(mod, TMP(3) | mme0,   TMP(3) | mme4,      dst | mme6,   TMP(4) | mme5);
            madm<DT>(mod, TMP(2) | mme1,      num | nomme,  -denom | nomme,  TMP(0) | mme7);
            madm<DT>(mod,    dst | nomme,  TMP(0) | mme7,   TMP(2) | mme1,   TMP(3) | mme0);

            mark(labelSkip);
            endif(cfmod);
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
              const A &tmp, InstructionModifier cfmod = InstructionModifier())
{
    DataType dt = getDataType<DT>();
    if (dt == DataType::invalid)
        dt = dst.getType();
    if (cfmod.getExecSize() == 0)
        cfmod = mod;

    Label labelSkip;

    switch (dt) {
        case DataType::hf:
            inv<DT>(mod, dst, src);
            break;
        case DataType::f:
            invm<DT>(mod | eo | flag,         dst | mme0,      one | nomme,     src | nomme);
            if_(cfmod | ~flag, labelSkip);

            madm<DT>(mod, TMP(1) | mme2,      one | nomme,    -src | nomme,     dst | mme0);
            madm<DT>(mod, TMP(2) | mme3,      dst | mme0,   TMP(1) | mme2,      dst | mme0);
            madm<DT>(mod, TMP(0) | mme5,      dst | mme0,   TMP(1) | mme2,   TMP(2) | mme3);
            madm<DT>(mod, TMP(1) | mme6,      one | nomme,    -src | nomme,  TMP(0) | mme5);
            madm<DT>(mod,    dst | nomme,  TMP(0) | mme5,   TMP(1) | mme6,   TMP(2) | mme3);

            mark(labelSkip);
            endif(cfmod);
            break;
        case DataType::df:
            invm<DT>(mod | eo | flag,        dst | mme0,      one | nomme,     src | nomme);
            if_(cfmod | ~flag, labelSkip);

            madm<DT>(mod, TMP(0) | mme2,     one | nomme,    -src | nomme,     dst | mme0);
            madm<DT>(mod, TMP(1) | mme4,     dst | mme0,   TMP(0) | mme2,      dst | mme0);
            madm<DT>(mod, TMP(2) | mme5,     one | nomme,    -src | nomme,  TMP(1) | mme4);
            madm<DT>(mod,    dst | mme6,     dst | mme0,   TMP(0) | mme2,   TMP(1) | mme4);
            madm<DT>(mod, TMP(1) | mme0,  TMP(1) | mme4,      dst | mme6,   TMP(2) | mme5);
            madm<DT>(mod, TMP(0) | mme1,     one | nomme,    -src | nomme,     dst | mme6);
            madm<DT>(mod,    dst | nomme,    dst | mme6,   TMP(0) | mme1,   TMP(1) | mme0);

            mark(labelSkip);
            endif(cfmod);
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
               RegData zero, RegData oneHalf, RegData one, const A &tmp, InstructionModifier cfmod = InstructionModifier())
{
    DataType dt = getDataType<DT>();
    if (dt == DataType::invalid)
        dt = dst.getType();
    if (cfmod.getExecSize() == 0)
        cfmod = mod;

    Label labelSkip;

    switch (dt) {
        case DataType::hf:
            sqt<DT>(mod, dst, src);
            break;
        case DataType::f:
            rsqtm<DT>(mod | eo | flag,        dst | mme0,       src | nomme);
            if_(cfmod | ~flag, labelSkip);

            madm<DT>(mod, TMP(0) | mme1,     zero | nomme,  oneHalf | nomme,     dst | mme0);
            madm<DT>(mod, TMP(1) | mme2,     zero | nomme,      src | nomme,     dst | mme0);
            madm<DT>(mod, TMP(2) | mme3,  oneHalf | nomme,  -TMP(1) | mme2,   TMP(0) | mme1);
            madm<DT>(mod, TMP(0) | mme4,   TMP(0) | mme1,    TMP(2) | mme3,   TMP(0) | mme1);
            madm<DT>(mod,    dst | mme5,   TMP(1) | mme2,    TMP(2) | mme3,   TMP(1) | mme2);
            madm<DT>(mod, TMP(2) | mme6,      src | nomme,     -dst | mme5,      dst | mme5);
            madm<DT>(mod,    dst | nomme,     dst | mme5,    TMP(0) | mme4,   TMP(2) | mme6);

            mark(labelSkip);
            endif(cfmod);
            break;
        case DataType::df:
            rsqtm<DT>(mod | eo | flag,        dst | mme0,       src | nomme);
            if_(cfmod | ~flag, labelSkip);

            madm<DT>(mod, TMP(0) | mme1,     zero | mme0,   oneHalf | nomme,     dst | mme0);
            madm<DT>(mod, TMP(1) | mme2,     zero | mme0,       src | nomme,     dst | mme0);
            madm<DT>(mod, TMP(2) | mme3,  oneHalf | nomme,  -TMP(1) | mme2,   TMP(0) | mme1);
            madm<DT>(mod, TMP(3) | mme4,      one | nomme,  oneHalf | nomme,     dst | nomme);
            madm<DT>(mod, TMP(3) | mme5,      one | nomme,   TMP(3) | mme4,   TMP(2) | mme3);
            madm<DT>(mod,    dst | mme6,     zero | mme0,    TMP(2) | mme3,   TMP(1) | mme2);
            madm<DT>(mod, TMP(2) | mme7,     zero | mme0,    TMP(2) | mme3,   TMP(0) | mme1);
            madm<DT>(mod,    dst | mme6,   TMP(1) | mme2,    TMP(3) | mme5,      dst | mme6);
            madm<DT>(mod, TMP(3) | mme5,   TMP(0) | mme1,    TMP(3) | mme5,   TMP(2) | mme7);
            madm<DT>(mod, TMP(0) | mme1,      src | nomme,     -dst | mme6,      dst | mme6);
            madm<DT>(mod,    dst | nomme,     dst | mme6,    TMP(0) | mme1,   TMP(3) | mme5);

            mark(labelSkip);
            endif(cfmod);
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
void threadend(const InstructionModifier &mod, const RegData &r0_info) {
    auto sf = (hardware <= HW::XeHP) ? SharedFunction::ts
                                        : SharedFunction::gtwy;
    uint32_t exdesc = 0x20 | (static_cast<int>(sf) & 0xF);
    send(8 | EOT | mod | NoMask, null, r0_info, exdesc, 0x2000010);
}

void threadend(const RegData &r0_info) { threadend(InstructionModifier(), r0_info); }

// Gateway messages.
void barriermsg(const InstructionModifier &mod, const GRF &header)
{
    uint32_t exdesc = static_cast<int>(SharedFunction::gtwy) & 0xF;
    send(1 | mod | NoMask, null, header, exdesc, 0x2000004);
}

void barriermsg(const GRF &header) { barriermsg(InstructionModifier(), header); }

// Prepare barrier header.
void barrierheader(const GRF &header, const GRF &r0_info = r0) {
    if (hardware >= HW::XeHPG) {
        mov(1 | NoMask, header.hf(4), Immediate::hf(0));
        mov(2 | NoMask, header.ub(10)(1), r0_info.ub(11)(0));
    } else
        and_(8 | NoMask, header.ud(), r0_info.ud(2), uint32_t((hardware >= HW::Gen11) ? 0x7F000000 : 0x8F000000));
}

void barriersignal(const InstructionModifier &mod, const GRF &temp, const GRF &r0_info = r0)
{
    barrierheader(temp, r0_info);
    barriermsg(mod, temp);
}

void barriersignal(const InstructionModifier &mod, const GRF &temp, uint32_t threadCount, const GRF &r0_info = r0)
{
    if (hardware >= HW::XeHPG)
        mov(1 | NoMask, temp.ud(2), (threadCount << 24) | (threadCount << 16));
    else
    {
        and_(8 | NoMask, temp.ud(), r0_info.ud(2), uint32_t((hardware >= HW::Gen11) ? 0x7F000000 : 0x8F000000));
        mov(1 | NoMask, temp.ub(9), 0x80 | (threadCount & 0x7F));
    }
    barriermsg(mod, temp);
}

void barriersignal(const GRF &temp, const GRF &r0_info = r0) { barriersignal(InstructionModifier(), temp, r0_info); }
void barriersignal(const GRF &temp, uint32_t threadCount, const GRF &r0_info = r0) { barriersignal(InstructionModifier(), temp, threadCount, r0_info); }

// Named barriers.
void barriersignal(const InstructionModifier &mod, uint32_t barrierID, const GRF &temp, const GRF &r0_info = r0)
{
#ifdef NGEN_SAFE
    if (hardware < HW::XeHPC)
        throw unsupported_message();
#endif
    mov(1 | NoMask, temp.uw(4), uint8_t(barrierID));
    mov(2 | NoMask, temp.ub(10)(1), r0_info.ub(11)(0));
    barriermsg(mod, temp);
}

void barriersignal(const InstructionModifier &mod, uint32_t barrierID, const GRF &temp, BarrierType barrierType, uint32_t producers, uint32_t consumers)
{
#ifdef NGEN_SAFE
    if (hardware < HW::XeHPC)
        throw unsupported_message();
#endif
    mov(1 | NoMask, temp.ud(2), (barrierID & 0xFF) | (static_cast<uint32_t>(barrierType) << 14) | ((producers & 0xFF) << 16) | ((consumers & 0xFF) << 24));
    barriermsg(mod, temp);
}

void barriersignal(uint32_t barrierID, const GRF &temp, const GRF &r0_info = r0) { barriersignal(InstructionModifier(), barrierID, temp, r0_info); }
void barriersignal(uint32_t barrierID, const GRF &temp, BarrierType barrierType, uint32_t producers, uint32_t consumers) { barriersignal(InstructionModifier(), barrierID, temp, barrierType, producers, consumers); }

void barrierwait()
{
    if (isGen12)
        sync.bar(NoMask);
    else
        wait(NoMask, n0[0]);
}

template <typename... Targs>
void barrier(const Targs &...barrierArgs)
{
    barriersignal(barrierArgs...);
    barrierwait();
}

// Global memory fence.
void memfence(const InstructionModifier &mod, const RegData &dst, const RegData &header = GRF(0))
{
    if (hardware <= HW::XeHP) {
        const uint32_t exdesc = static_cast<int>(SharedFunction::dc0) & 0xF;
        send(8 | mod | NoMask, dst, header, exdesc, 0x219E000);
    } else
        memfence(mod, FenceScopeLSC::GPU, FlushTypeLSC::None, dst, header);
}

void memfence(const InstructionModifier &mod, FenceScopeLSC scope, FlushTypeLSC flushing, const RegData &dst, const RegData &header)
{
    if (hardware < HW::XeHPG) {
        memfence(mod, dst, header);
        return;
    }

    if (flushing == FlushTypeLSC::None && hardware == HW::XeHPG && scope > FenceScopeLSC::Subslice)
        flushing = static_cast<FlushTypeLSC>(6);    /* workaround for DG2 bug */

    uint32_t desc = 0x0210011F;
    desc |= static_cast<uint32_t>(scope) << 9;
    desc |= static_cast<uint32_t>(flushing) << 12;
    send(1 | mod | NoMask, SharedFunction::ugm, dst, header, null, 0, desc);
}

void memfence(const RegData &dst, const RegData &header = GRF(0))
{
    memfence(InstructionModifier(), dst, header);
}

void memfence(FenceScopeLSC scope, FlushTypeLSC flushing, const RegData &dst, const RegData &header = GRF(0))
{
    memfence(InstructionModifier(), scope, flushing, dst, header);
}

// SLM-only memory fence.
void slmfence(const InstructionModifier &mod, const RegData &dst, const RegData &header = GRF(0))
{
    if (hardware <= HW::XeHP) {
        const uint32_t exdesc = static_cast<int>(SharedFunction::dc0) & 0xF;
        send(8 | mod | NoMask, dst, header, exdesc, 0x219E0FE);
    } else
        send(1 | mod | NoMask, SharedFunction::slm, dst, header, null, 0, 0x210011F);
}

void slmfence(const RegData &dst, const RegData &header = GRF(0)) { slmfence(InstructionModifier(), dst, header); }

// XeHP+ prologues.
void loadlid(int argBytes, int dims = 3, int simd = 8, const GRF &temp = GRF(127), int paddedSize = 0)
{
    if (hardware >= HW::XeHP) {
        if (paddedSize < 0)
            paddedSize = 12*16;
        const int grfSize = GRF::bytes(hardware);
        const int grfOW = grfSize / 16;
        int simdGRFs = (simd > 16 && grfSize < 64) ? 2 : 1;
        int insns = 0;
        bool lsc = (hardware >= HW::XeHPG);

        if (dims > 0) {
            auto dmSave = defaultModifier;
            defaultModifier |= NoMask | AutoSWSB;

            mov<uint32_t>(8, temp, uint16_t(0));
            and_<uint32_t>(1, temp[2], r0[0], uint32_t(~0x1F));
            and_<uint16_t>(1, temp[0], r0[4], uint16_t(0xFF));
            add<uint32_t>(1, temp[2], temp[2], uint16_t(argBytes));
            if (simd == 1) {
                mad<uint32_t>(1, temp[2], temp[2], temp.uw(0), uint16_t(grfSize));
                lsc ? load(1, r1, D32T(4) | L1C_L3C,      A32,   temp)
                    : load(8, r1, aligned_block_oword(1), A32NC, temp);
            } else {
                mad<uint32_t>(1, temp[2], temp[2], temp.uw(0), uint16_t(3 * simdGRFs * grfSize));
                lsc ? load(1, r1, D32T(simdGRFs * ((dims == 1) ? 1 : 2) * grfOW * 4) | L1C_L3C,  A32,   temp)
                    : load(8, r1, aligned_block_oword(simdGRFs * ((dims == 1) ? 1 : 2) * grfOW), A32NC, temp);
                insns += 6;
                if (dims == 3) {
                    add<uint32_t>(1, temp[2], temp[2], uint16_t(2 * simdGRFs * grfSize));
                    lsc ? load(1, GRF(1 + 2 * simdGRFs), D32T(grfOW * 4 * simdGRFs) | L1C_L3C,  A32,   temp)
                        : load(8, GRF(1 + 2 * simdGRFs), aligned_block_oword(grfOW * simdGRFs), A32NC, temp);
                    insns += 2;
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
                nop();
        }

        if (!_labelLocalIDsLoaded.defined(labelManager))
            mark(_labelLocalIDsLoaded);
    }
}

void loadargs(const GRF &base, int argGRFs, const GRF &temp = GRF(127))
{
    if (hardware >= HW::XeHP) {
        if (argGRFs > 0) {
            bool lsc = (hardware >= HW::XeHPG);
            auto tempAddr = temp[lsc ? 0 : 2];
            auto dst = base;
            auto dmSave = defaultModifier;
            defaultModifier |= NoMask | AutoSWSB;

            if (!lsc)
                mov<uint32_t>(8, temp, uint16_t(0));
            and_<uint32_t>(1, tempAddr, r0[0], uint32_t(~0x1F));
            while (argGRFs > 0) {
                int nload = std::min(utils::rounddown_pow2(argGRFs), lsc ? 8 : 4);
                int loadBytes = nload * GRF::bytes(hardware);
                lsc ? load(1, dst, D64T(loadBytes >> 3) | L1C_L3C,      A32,   temp)
                    : load(8, dst, aligned_block_oword(loadBytes >> 4), A32NC, temp);
                argGRFs -= nload;
                dst += nload;
                if (argGRFs > 0)
                    add<uint32_t>(1, tempAddr, tempAddr, uint32_t(loadBytes));
            }

            defaultModifier = dmSave;
        }

        if (!_labelArgsLoaded.defined(labelManager))
            mark(_labelArgsLoaded);
    }
}

void epilogue(int GRFCount, bool hasSLM, const RegData &r0_info)
{
    GRF tmp0(GRFCount - 3);
    GRF tmp1(GRFCount - 2);
    GRF lastReg(GRFCount - 1);

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
    mov<uint32_t>(dwordsPerReg, lastReg, r0_info);

    if (doMemFence) memfence(tmp0, r0_info);
    if (doSLMFence) slmfence(tmp1, r0_info);

    if (setAccToZero) {
        mov(16, acc0.f(), 0.f);
        if (hardware == HW::XeHP) mov(16, acc2.f(), 0.f);
    }

    if (doMemFence) wrdep(tmp0);
    if (doSLMFence) wrdep(tmp1);

    threadend(lastReg);
}
