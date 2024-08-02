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
#include "hw_utils.hpp"
#include "layout_utils.hpp"
#include "map.hpp"
#include "ngen_object_helpers.hpp"
#include "quantization.hpp"

using namespace ngen;
using namespace ngen::utils;
using std::vector;

#include "internal/namespace_start.hxx"


// Register-to-register copy of a single block, ignoring register offsets in the block.
template <HW hw>
bool BLASKernelGenerator<hw>::copyRegisterBlock(Type Ts, Type Td, const RegisterBlock &blockSrc, const RegisterBlock &blockDst,
                                                const GRFMultirange &src, const GRFMultirange &dst, int dOffR, int dOffC,
                                                const CommonStrategy &strategy, CommonState &state, bool preserveSrc)
{
    std::vector<RegisterBlock> modSrc{1, blockSrc}, modDst{1, blockDst};
    modSrc[0].offsetBytes %= GRF::bytes(hw);
    modDst[0].offsetBytes %= GRF::bytes(hw);
    return copyRegisters(Ts, Td, modSrc, modDst, src, dst, dOffR, dOffC, false, strategy, state, preserveSrc);
}

// Register-to-register copy, with no scaling.
template <HW hw>
bool BLASKernelGenerator<hw>::copyRegisters(Type Ts, Type Td, const vector<RegisterBlock> &layoutSrc, const vector<RegisterBlock> &layoutDst,
                                            const GRFMultirange &src, const GRFMultirange &dst,
                                            int dOffR, int dOffC,
                                            bool conjugate, const CommonStrategy &strategy, CommonState &state, bool preserveSrc, bool s4Shift)
{
    return copyRegisters(Ts, Td, layoutSrc, layoutDst, src, dst, dOffR, dOffC, Scalar{1},
                         SubregisterPair(), SubregisterPair(), conjugate, strategy, state, preserveSrc, s4Shift);
}

// Register-to-register copy, with scaling.
template <HW hw>
bool BLASKernelGenerator<hw>::copyRegisters(Type Ts, Type Td, const vector<RegisterBlock> &layoutSrc, const vector<RegisterBlock> &layoutDst,
                                            const GRFMultirange &src, const GRFMultirange &dst,
                                            int dOffR, int dOffC, const Scalar &alpha, const SubregisterPair &alpha_real, const SubregisterPair &alpha_imag,
                                            bool conjugate, const CommonStrategy &strategy, CommonState &state, bool preserveSrc, bool s4Shift)
{
    auto Ts_real = Ts.real(), Td_real = Td.real();
    auto nes_real = elementsPerGRF(hw, Ts_real);
    auto ned_real = elementsPerGRF(hw, Td_real);

    const int nphases = 2, qCXMin = -1, qCXMax = -1;

    // Special int4 upconversion path.
    if (alpha == 1 && !conjugate && !preserveSrc) {
        vector<RegisterBlock> emptyLayout;
        GRFMultirange emptyRegs;
        if (canDequantizeInt4(Ts, Td, layoutSrc, layoutDst, emptyLayout, emptyLayout)) {
            dequantizeInt4(true, Ts, Td, layoutSrc, layoutDst, emptyLayout, emptyLayout,
                           src, dst, emptyRegs, emptyRegs, Td, dOffR, dOffC, nullptr, strategy, state, s4Shift);
            return true;
        }
    }

    Subregister saveF0, saveF1, saveF2;
    bool releaseEmuFlag = false;
    bool preswizzle = (hw >= HW::XeHP);
    GRFRange copyTemp;

    // Prepare for f->bf emulation.
    if (!strategy.systolicAvailable && Td_real == Type::bf16 && Ts_real == Type::f32) {
        if (state.emulate.flag.isInvalid()) {
            int nflag = (GRF::bytes(hw) == 64) ? 2 : 1;
            state.emulate.flag = state.raVFlag.tryAlloc(nflag);
            state.emulate.flagOffset = 0;
            if (state.emulate.flag.isValid())
                releaseEmuFlag = true;
            else {
                state.emulate.flag = f0;
                saveF0 = state.ra.alloc_sub<uint32_t>();
                mov(1, saveF0, f0);
            }
        }
    }

    if (Ts_real == Type::hf8) {
        saveF0 = state.ra.alloc_sub<uint32_t>();
        mov(1, saveF0, f0);
    }
    if (Td_real == Type::hf8) {
        if (FlagRegister::count(hw) < 4) stub();
        saveF0 = state.ra.alloc_sub<uint32_t>();
        mov(1, saveF0, f0);
        saveF1 = state.ra.alloc_sub<uint32_t>();
        mov(1, saveF1, f1);
        saveF2 = state.ra.alloc_sub<uint32_t>();
        mov(1, saveF2, f2);
    }

    auto allocTemp = [&]() {
        if (copyTemp.isInvalid())
            copyTemp = state.ra.alloc_range(2);
    };

    int srcM, srcN;
    getLayoutDims(layoutSrc, srcM, srcN);
    bool vectorCopy = (srcM == 1 || srcN == 1);
    bool int4Zip = false;
    int periodY = 1, phasesY = 1;

    if (GRF::bytes(hw) == 64 && Td_real.paddedSize() == 1 && layoutDst[0].crosspack > 1)
        periodY = 2, phasesY = 2;
    if (Ts_real.isInt4() && layoutSrc[0].crosspack > 1)
        if ((layoutSrc[0].colMajor ? layoutSrc[0].nc : layoutSrc[0].nr) > 1)
            int4Zip = true, periodY = 2, phasesY = 1;

    for (int phase = -1; phase < nphases; phase++) {
    for (int phaseY = 0; phaseY < phasesY; phaseY++) {
    for (auto &sblock : layoutSrc) {
        auto RegisterBlock::*nx = sblock.colMajor ? &RegisterBlock::nr : &RegisterBlock::nc;
        auto RegisterBlock::*ny = sblock.colMajor ? &RegisterBlock::nc : &RegisterBlock::nr;
        auto RegisterBlock::*offsetY = sblock.colMajor ? &RegisterBlock::offsetC : &RegisterBlock::offsetR;

        for (int eoffY = 0; eoffY < sblock.*ny; eoffY++) {
            if (((eoffY + sblock.*offsetY) & (periodY - 1)) != phaseY) continue;
            for (int qCX = qCXMin; qCX <= qCXMax; qCX++) {

            for (int eoffX = 0; eoffX < sblock.*nx;) {
                auto eoffR = sblock.colMajor ? eoffX : eoffY;
                auto eoffC = sblock.colMajor ? eoffY : eoffX;

                int selems, delems;
                const RegisterBlock *sblockPtr, *dblockPtr;

                // Locate source and destination register.
                auto sreg = findBlockReg(Ts, layoutSrc, sblock.offsetR + eoffR,         sblock.offsetC + eoffC,         src, selems, sblockPtr, qCX);
                auto dreg = findBlockReg(Td, layoutDst, sblock.offsetR + eoffR + dOffR, sblock.offsetC + eoffC + dOffC, dst, delems, dblockPtr, qCX);

                // Limit due to powers of 2 instruction exec size
                selems = rounddown_pow2(selems);
                delems = rounddown_pow2(delems);

                auto scrosspack = sblock.crosspack;
                auto dcrosspack = dblockPtr->crosspack;

                bool skip = false;

                if (sblock.colMajor != dblockPtr->colMajor) {
                    bool sLargeCP = isLargeCrosspack(Ts, scrosspack);
                    bool dLargeCP = isLargeCrosspack(Td, dcrosspack);
                    bool sEffCM = sblock.colMajor     ^ sLargeCP;
                    bool dEffCM = dblockPtr->colMajor ^ dLargeCP;
                    if (sEffCM == dEffCM) {
                        if (sLargeCP) selems = std::min<int>(selems, scrosspack);
                        if (dLargeCP) delems = std::min<int>(delems, dcrosspack);
                    } else {
                        if (!vectorCopy) stub();         // No in-register matrix transposes.
                        selems = delems = 1;
                    }
                }

                // Find out how many consecutive elements we can copy.
                auto selems_real = selems * Ts.complexComponents();
                auto delems_real = delems * Td.complexComponents();
                auto nGRFs = (strategy.dualGRF ? 2 : 1);
                auto nGRFs_d = (dreg.getOffset() >= dcrosspack) ? 1 : nGRFs;  // Don't cross destination GRF boundaries for efficiency.
                auto selems_limit = div_up(nGRFs   * nes_real - sreg.getOffset(), scrosspack);
                auto delems_limit = div_up(nGRFs_d * ned_real - dreg.getOffset(), dcrosspack);
                selems_real = std::min({selems_real, selems_limit});
                delems_real = std::min({delems_real, delems_limit});
                auto nelems_real = std::min(selems_real, delems_real);
                if (phase != 0)
                    nelems_real = std::min(rounddown_pow2(nelems_real), 32);

                if (Ts_real == Type::f32 && Td_real != Type::f32 && dcrosspack == 1)
                    nelems_real = std::min(nelems_real, nes_real);     // Special case: mixed mode packed downconversion limited to 1 GRF.

                bool src_f8 = Ts.isF8();
                bool dst_f8 = Td.isF8();
                bool f8_align = src_f8 ^ dst_f8;

                // Check if separate conversions are needed due to size changes.
                auto sconvertCP = Ts_real.isInt4() ? 1 : (Ts_real.size() / Td_real) * scrosspack;
                bool b_to_bf = Ts_real.isInt8() && Td_real == Type::bf16;
                bool allInt4 = (Td_real.isInt4() && Ts_real.isInt4());
                bool sconvert = !allInt4 && (Ts_real.isInt4()
                                         || (Td_real.size() == 1 && Ts_real.size() > 1 && dcrosspack != sconvertCP)
                                         || (Td_real.size() == 2 && Ts_real.size() > 2 && dcrosspack != sconvertCP && scrosspack > 1)
                                         || (Td_real.size() == 1 && Td_real.isInteger() && Ts_real.size() == 4 && (dreg.getOffset() & 2))
                                         || (Td_real.size() == 2 && Td_real.isFP() && !Ts_real.isFP() && dcrosspack != sconvertCP && hw > HW::Gen9));
                sconvert &= !f8_align;
                if (sconvert && preserveSrc) stub();
                bool byteAlign = sconvert && (Ts_real.isInt4() || Td_real.isInt4() || (Ts_real.bits() < Td_real.bits()))
                                          && one_of(Td_real, Type::u8, Type::s8, Type::u16, Type::s16, Type::f16, Type::bf16, Type::f32);
                byteAlign |= allInt4;
                if (Ts_real.isInt4() && (!byteAlign && Td_real != Ts_real))
                    stub();

                auto sregConverted = sconvert ? sreg.reinterpret(0, Td_real.ngen())(sconvertCP)
                                              : sreg(scrosspack);

                auto dconvertCP = Td_real.isInt4() ? 1 : (Td_real.size() / Ts_real) * dcrosspack;
                bool dconvert = !Ts_real.isInt4() && !Td_real.isInt4() &&
                            ((Ts_real.size() == 1 && Td_real.size() > 1
                                    && (scrosspack != dconvertCP || ((sreg.getOffset() & 3) && hw >= HW::XeHP))
                                    && !byteAlign
                                    && !b_to_bf)
                         || (Ts_real == Type::f16 && Td_real.size() > 2 && (sreg.getOffset() & 1) && hw >= HW::XeHP));
                dconvert &= !f8_align;
                bool bfHfCvt = (!sconvert && !dconvert && !byteAlign
                                       && !f8_align)
                        && one_of(Ts_real, Type::bf16, Type::f16)
                        && one_of(Td_real, Type::f16, Type::bf16)
                        && Ts_real != Td_real;
                if (bfHfCvt)
                    sregConverted = sreg.reinterpret( 0, ngen::DataType::f)(scrosspack);

                if (f8_align || bfHfCvt) allocTemp();
                if ((byteAlign || bfHfCvt) && Td_real != Type::u16)
                    nelems_real = std::min(nelems_real, elementsPerGRF<float>(hw));
                auto dregConverted = dconvert ? dreg.reinterpret(0, Ts_real.ngen())(dconvertCP)
                                              : dreg(dcrosspack);

                InstructionModifier modMov, mmodMov;
                if (Ts_real != Td_real && Td_real.isInteger() && Td_real.bits() <= Ts_real.bits()) {
                    modMov = modMov | sat;
                    if (!sconvert && !dconvert)
                        mmodMov = mmodMov | sat;
                }

                auto cvt_f8_to_x = [&]() {
                   if (Ts_real == Type::bf8) {
                       mov(nelems_real | modMov, copyTemp[0].ub(),sreg.ub()(scrosspack));
                       shl(nelems_real, copyTemp[1].w(), copyTemp[0].b(), 8);
                       if (Td_real == Type::f16) {
                           mov(nelems_real | modMov, dreg.uw()(dcrosspack), copyTemp[1].uw());
                       } else if (Td_real == Type::f32) {
                           mov(nelems_real | modMov, copyTemp[0].sub(0, Td_real.ngen())(1), copyTemp[1].hf());
                           mov(nelems_real | modMov, dreg.ud()(dcrosspack), copyTemp[0].ud());
                       } else
                           stub();
                   } else if (Ts_real == Type::hf8) {
                       if (!one_of(Td_real, Type::f16, Type::f32))
                           stub();
                       if (scrosspack > 2) {
                           mov(nelems_real | modMov, copyTemp[1].ub(), sreg.ub()(scrosspack));
                           eshl(nelems_real, copyTemp[0].uw(), copyTemp[1].ub(), 8, strategy, state);
                           eshl(nelems_real, copyTemp[1].uw(), copyTemp[1].ub(), 7, strategy, state);
                       } else {
                           eshl(nelems_real, copyTemp[0].uw(), sreg.ub()(scrosspack), 8, strategy, state);
                           eshl(nelems_real, copyTemp[1].uw(), sreg.ub()(scrosspack), 7, strategy, state);
                       }
                       and_(nelems_real, copyTemp[1].uw(), copyTemp[1].uw(), 0x3F80);
                       cmp(nelems_real | eq | f0[0], null.uw(), copyTemp[1].uw(), Immediate(0x3F80));
                       mul(nelems_real, copyTemp[1].hf(), copyTemp[1].hf(), Immediate::hf(0x5c00));
                       mov(nelems_real | modMov | f0[0], copyTemp[1].uw(), Immediate(0x7C01));
                       csel(nelems_real | gt, copyTemp[1].hf(), copyTemp[1].hf(), -copyTemp[1].hf(), copyTemp[0].hf());
                       if (Td_real == Type::f16) {
                           mov(nelems_real | modMov, dreg.uw()(dcrosspack), copyTemp[1].uw());
                       } else {
                           mov(nelems_real | modMov, copyTemp[1].sub(0, ngen::DataType::uw)(2), copyTemp[1].uw());
                           mov(nelems_real | modMov, copyTemp[1].f(), copyTemp[1].sub(0, ngen::DataType::hf)(2));
                           mov(nelems_real | modMov, dreg.ud()(dcrosspack), copyTemp[1].ud());
                       }
                   } else {
                        stub();
                   }
                };

                auto cvt_x_to_f8 = [&]() {
                    if (Td_real == Type::bf8) {
                        auto tmp0 = copyTemp[0].sub( 0, Ts_real.ngen());
                        moveToIntPipe(tmp0);
                        moveToIntPipe(sreg);
                        mov(nelems_real | modMov, tmp0(1), sreg(scrosspack));
                        if (Ts_real.ngen() == ngen::DataType::hf) {
                            auto tmp1 = copyTemp[1].bf8();
                            mov(nelems_real | modMov, tmp1, tmp0.hf());
                            mov(nelems_real | modMov, dreg.ub()(dcrosspack), tmp1.ub());
                        } else if (Ts_real.ngen() == ngen::DataType::f) {
                            auto tmp1 = copyTemp[1].sub( 0, ngen::DataType::hf);
                            movePipes(tmp0);
                            movePipes(sreg);
                            mov(nelems_real | modMov, tmp1(1), tmp0(1));
                            mov(nelems_real | modMov, tmp0.bf8()(1), tmp1(1));
                            mov(nelems_real | modMov, dreg.ub()(dcrosspack), tmp0.ub()(1));
                        } else
                            stub();
                    } else if (Td_real == Type::hf8) {
                        auto tmp0 = copyTemp[0].sub( 0, ngen::DataType::ub);
                        moveToIntPipe(tmp0);
                        moveToIntPipe(sreg);
                        if (!one_of(Ts_real, Type::f16, Type::f32))
                            stub();
                        auto tmp1 = copyTemp[1].sub( 0, ngen::DataType::uw);
                        if (Ts_real == Type::f32) {
                            mov(nelems_real | modMov, tmp1.ud()(1), sreg.ud()(scrosspack));
                            mov(nelems_real | modMov, tmp1.hf()(2), tmp1.f()(1));
                            mov(nelems_real | modMov, tmp1(1), tmp1(2));
                        } else {
                            mov(nelems_real | modMov, tmp1(1), sreg(scrosspack));
                        }
                        // get sign bits
                        and_(nelems_real | nz | f2[0], null.uw(), tmp1(1), Immediate(0x8000));
                        // multiply by hf 128 to force overflow of exponent
                        mul(nelems_real, tmp1.hf()(1), tmp1.hf()(1), Immediate::hf(0x5800));
                        // multiply by 2^(-15) to undo mul, preserving overflows,
                        // shift and underflow for hf8
                        mul(nelems_real, tmp1.hf()(1), tmp1.hf()(1), Immediate::hf(0x0200));
                        // check for NaN, inf.
                        and_(nelems_real | ze | f0[0], null.uw(), ~tmp1(1), 0x7C00);
                        // round.
                        add(nelems_real, tmp1(1), tmp1(1), Immediate(-0x40));
                        // check for zero mantissa.
                        and_(nelems_real | nz | f1[0], null.uw(), tmp1(1), 0x3FF);
                        eshr(nelems_real, tmp1(1), tmp1(1), 7, strategy, state);
                        add(nelems_real | f1[0], tmp1(1), tmp1(1), Immediate(1));
                        mov(nelems_real | modMov | f0[0], tmp1(1), Immediate(0x7F));
                        or_(nelems_real | f2[0], tmp1(1), tmp1(1), Immediate(0x80));
                        mov(nelems_real | modMov, tmp0(2), tmp1(1));
                        mov(nelems_real | modMov, tmp0(1), tmp0(2));
                        mov(nelems_real | modMov, dreg.ub()(dcrosspack), tmp0.ub()(1));
                    } else {
                         stub();
                    }
                };

                auto doByteAlign = [&]() {
                    allocTemp();
                    if (Ts_real.isInt4()) {
                        auto tmp0 = copyTemp[0].w(0);
                        auto tmp1 = copyTemp[1].f(0);
                        int n_bytes = nelems_real;

                        int scrosspack_byte = scrosspack;
                        if (int4Zip) scrosspack_byte >>= 1;

                        auto dreg1 = dreg;
                        int effDCP = dcrosspack;
                        InstructionModifier writeCombine;
                        if (int4Zip) {
                            int delems1;
                            const RegisterBlock *dblockPtr1;
                            int di1 = sblock.offsetR + eoffR + dOffR;
                            int dj1 = sblock.offsetC + eoffC + dOffC;
                            (sblock.colMajor ? dj1 : di1)++;
                            dreg1 = findBlockReg(Td, layoutDst, di1, dj1, dst, delems1, dblockPtr1, qCX);
                            nelems_real *= 2;
                        } else {
                            dreg1.setOffset(dreg1.getOffset() + dcrosspack);
                            effDCP *= 2;
                            n_bytes /= 2;
                        }

                        if (hw >= HW::XeHPC && Td.isInt8() && elementDiff(hw, dreg1, dreg) == 1)
                            writeCombine |= Atomic;

                        if (Td.isInteger()) {
                            if (Ts_real.isSigned()) {
                                if (Td_real.isInt16()) {
                                    shl(n_bytes | modMov,  dreg(effDCP), sreg.ub()(scrosspack_byte), 12);
                                    shl(n_bytes | modMov, dreg1(effDCP), sreg.ub()(scrosspack_byte), 8);
                                    asr(n_bytes | modMov,  dreg.w()(effDCP), dreg.w()(effDCP), 12);
                                    asr(n_bytes | modMov,  dreg1.w()(effDCP), dreg1.w()(effDCP), 12);
                                } else if (!Td_real.isInt8())
                                    stub();
                                else if (effDCP > 2 && !int4Zip) {
                                    shl(n_bytes | modMov, tmp0.w(0)(2), sreg.b()(scrosspack_byte), 4);
                                    mov(n_bytes | modMov, tmp0.w(1)(2), sreg.b()(scrosspack_byte));
                                    asr(nelems_real | modMov, dreg(effDCP / 2), tmp0.b()(2), 4);
                                } else {
                                    shl(n_bytes | modMov,  dreg(effDCP), sreg.ub()(scrosspack_byte), 4);
                                    asr(n_bytes | modMov | writeCombine, dreg1(effDCP), sreg.b()(scrosspack_byte), 4);
                                    asr(n_bytes | modMov,                 dreg(effDCP), dreg.b()(effDCP), 4);
                                }
                            } else {
                                if (Td_real.size() == 1 && effDCP > 4) {
                                    and_(n_bytes | modMov, tmp0.uw(0)(2), sreg.ub()(scrosspack_byte), 0x0F);
                                    shr (n_bytes | modMov, tmp0.uw(1)(2), sreg.ub()(scrosspack_byte), 4);
                                    mov (nelems_real | modMov, dreg(effDCP / 2), tmp0.ub()(2));
                                } else {
                                    and_(n_bytes | modMov | writeCombine,  dreg(effDCP), sreg.ub()(scrosspack_byte), 0x0F);
                                    shr (n_bytes | modMov,                dreg1(effDCP), sreg.ub()(scrosspack_byte), 4);
                                }
                            }
                        } else {
                            if (scrosspack_byte > 2) {
                                mov(n_bytes | modMov, tmp0.ub()(scrosspack_byte), sreg.ub()(scrosspack_byte));
                                mov(n_bytes | modMov, tmp0.ub()(2),               tmp0.ub()(scrosspack_byte));
                            } else
                                mov(n_bytes | modMov, tmp0.ub()(2),               sreg.ub()(scrosspack_byte));

                            and_(n_bytes | modMov, tmp1.w()(4),  tmp0.w()(1), 0x0F);
                            shr (n_bytes | modMov, tmp0.w()(1),  tmp0.w()(1), 4);
                            and_(n_bytes | modMov, tmp1.w(2)(4), tmp0.w()(1), 0x0F);
                            // if signed, do sign extension
                            if (Ts == Type::s4) {
                                shl(nelems_real | modMov, tmp1.w()(2), tmp1.w()(2), 12);
                                asr(nelems_real | modMov, tmp1.w()(2), tmp1.w()(2), 12);
                            }

                            mov(nelems_real | modMov, tmp1(1), tmp1.w()(2));
                            if (Td != Type::f32) {
                                mov(nelems_real | modMov, tmp1.reinterpret(0, Td_real.ngen())(2), tmp1.f()(1));
                                mov(nelems_real | modMov, tmp1.w()(1),                            tmp1.w()(2));
                                if (!int4Zip)
                                    mov(nelems_real | modMov, dreg.w()(dcrosspack),               tmp1.w()(1));
                                else {
                                    mov(n_bytes | modMov, dreg.w()(effDCP),                       tmp1.w(0)(2));
                                    mov(n_bytes | modMov, dreg1.w()(effDCP),                      tmp1.w(1)(2));
                                }
                            } else if (!int4Zip)
                                mov(nelems_real | modMov, dreg.d()(dcrosspack),                   tmp1.d()(1));
                            else {
                                mov(n_bytes | modMov, dreg.d()(effDCP),                           tmp1.d(0)(2));
                                mov(n_bytes | modMov, dreg1.d()(effDCP),                          tmp1.d(1)(2));
                            }
                        }
                    } else {
                        auto tmp0 = copyTemp[0].w(0);
                        auto tmp1 = copyTemp[1].f(0);
                        mov(nelems_real | modMov, tmp0(2),                                sreg(scrosspack));
                        mov(nelems_real | modMov, tmp1(1),                                tmp0.w()(2));
                        mov(nelems_real | modMov, tmp0.reinterpret(0, Td_real.ngen())(2), tmp1(1));
                        mov(nelems_real | modMov, dreg.w()(dcrosspack),                   tmp0.w()(2));
                    }
                };

                auto doBfHfCvt = [&]() {
                    if (Ts_real == Type::bf16) {
                        if (scrosspack != 1) {
                            mov(nelems_real, copyTemp[0].uw(0)(2), sreg.uw()(scrosspack));
                            shl(nelems_real, copyTemp[0].ud(0)(1), copyTemp[0].uw(0)(2), 16);
                            mov(nelems_real, copyTemp[0].hf(0)(2), copyTemp[0].f(0)(1));
                            emov(nelems_real | mmodMov, dreg.uw(0)(dcrosspack), copyTemp[0].uw(0)(2), strategy, state);
                        } else {
                            shl(nelems_real, copyTemp[0].ud(sreg.getOffset())(scrosspack), sreg.uw()(scrosspack), 16);
                            emov(nelems_real | mmodMov, dregConverted, copyTemp[0].f(sreg.getOffset())(scrosspack), strategy, state);
                        }
                    } else {
                        mov(nelems_real, copyTemp[0].uw(0)(2), sreg.uw()(scrosspack));
                        mov(nelems_real, copyTemp[0].f(0)(1), copyTemp[0].hf(0)(2));
                        shr(nelems_real, copyTemp[0].uw(0)(2), copyTemp[0].ud(0)(1), 16);
                        emov(nelems_real | mmodMov, dreg.uw(0)(dcrosspack), copyTemp[0].uw(0)(2), strategy, state);
                    }
                };

                // Finally, copy, with any necessary conjugation and scaling. If doing a raw copy, use another pipe.
                if (!skip) switch (phase) {
                    case -1:
                        if (hw == HW::Gen9 && Ts_real == Type::f32 && !Td.isFP()) {
                            // Gen9: round to nearest before downconvert (not done by mov).
                            rnde(nelems_real, sreg(scrosspack), sreg(scrosspack));
                        }
                        if (f8_align) {
                            if (src_f8)
                                cvt_f8_to_x();
                            else if (dst_f8)
                                cvt_x_to_f8();
                        } else if (sconvert) {
                            if (byteAlign)
                                doByteAlign();
                            else {
                                mov(nelems_real | modMov,
                                        sregConverted,
                                        sreg(scrosspack));
                            }
                        } else if (bfHfCvt) {
                            doBfHfCvt();
                        }
                        break;
                    case 0:
                        if (alpha == 1 || alpha == -1) {
                            if (Ts_real == Td_real && !Ts_real.isInteger()) {
                                movePipes(sreg, scrosspack == 1 && dcrosspack == 1);
                                movePipes(dreg, scrosspack == 1 && dcrosspack == 1);
                                if (!sconvert) sregConverted = sreg(scrosspack);
                                if (!dconvert) dregConverted = dreg(dcrosspack);
                                if (hw >= HW::XeHP && scrosspack != dcrosspack) {
                                    moveToIntPipe(nelems_real, sregConverted);
                                    moveToIntPipe(nelems_real, dregConverted);
                                    sreg = sreg.reinterpret(0, sregConverted.getType());
                                    dreg = dreg.reinterpret(0, dregConverted.getType());
                                }
                            }
                            int telems = nelems_real * Ts_real / sreg.getBytes();
                            if (telems > 32) {
                                nelems_real = (nelems_real * 32) / telems;
                                telems = 32;
                            }
                            if (alpha == -1) {
                                auto wd = elementsPerGRF(hw, sreg.getType());
                                auto base = state.signChange.sub(0, dreg.getType());
                                xor_(telems, dreg(1), sreg(1), (wd >= telems) ? base(1) : base(0, wd, 1));
                            } else if (!byteAlign && !f8_align && !bfHfCvt)
                                emov(telems | mmodMov, dregConverted, sregConverted, strategy, state);
                        } else {
                            auto realDst = dreg(dcrosspack);
                            auto effDst = realDst;
                            if (preswizzle && (Ts.isFP() || Td.isFP())) {
                                allocTemp();
                                if ((sreg.getOffset() != dreg.getOffset()) || (scrosspack != dcrosspack))
                                    effDst = copyTemp[0].sub(sreg.getOffset(), sreg.getType())(scrosspack);
                            }

                            if (alpha.fixed())
                                mul(nelems_real, effDst, sregConverted, cast(Ts_real, alpha));
                            else
                                mul(nelems_real, effDst, sregConverted, alpha_real.getRegAvoiding(hw, sreg));

                            if (effDst != realDst) {
                                moveToIntPipe(nelems_real, realDst);
                                moveToIntPipe(nelems_real, effDst);
                                int nelems_real_int = nelems_real * Td / getBytes(effDst.getType());
                                emov(nelems_real_int, realDst, effDst, strategy, state);
                                dconvert = false;
                            }
                        }
                        break;
                    case 1:
                        if (dconvert)
                            mov(nelems_real | modMov, dreg(dcrosspack), dregConverted);
                        break;
                } /* switch phase */

                int nelems = nelems_real;
                eoffX += nelems;
            } /* eoffX loop */
            } /* qCX loop */
        } /* eoffY loop */
        } /* sblock loop */
        } /* phaseY loop */

    } /* phase loop */

    if (releaseEmuFlag)
        state.raVFlag.safeRelease(state.emulate.flag);

    if (saveF2.isValid()) {
        mov(1, f2, saveF2);
        state.ra.safeRelease(saveF2);
    }
    if (saveF1.isValid()) {
        mov(1, f1, saveF1);
        state.ra.safeRelease(saveF1);
        state.emulate.flag = invalid;
    }
    if (saveF0.isValid()) {
        mov(1, f0, saveF0);
        state.ra.safeRelease(saveF0);
        state.emulate.flag = invalid;
    }
    state.ra.safeRelease(copyTemp);
    return true;            // Success
}

// Copy one GRFMultirange to another, allowing overlap between the two.
template <HW hw>
void BLASKernelGenerator<hw>::overlappedCopy(const GRFMultirange &src, const GRFMultirange &dst, CommonState &state)
{
    constexpr int regs = GRF::maxRegs();
    std::array<int16_t, regs> map;

    std::vector<int16_t> temps;
    temps.reserve(src.getLen());

    std::vector<GRF> alloced;

    for (auto &m: map) m = -1;

    for (int i = 0; i < src.getLen(); i++)
        if (src[i].getBase() != dst[i].getBase())
            map[src[i].getBase()] = dst[i].getBase();

    int n = 1, ne = elementsPerGRF<uint32_t>(hw);
    bool done = false;
    bool useFloat = false;

    while (!done) {
        bool progress = false;
        done = true;

        // Copy registers where dst doesn't overlap src, then clear associated entries.
        for (int i = 0; i < regs; i += n) {
            n = 1;
            if (map[i] >= 0)
                done = false;

            if (map[i] >= 0 && map[map[i]] < 0) {
                temps.push_back(i);
                if (i + 1 < regs && map[i + 1] == map[i] + 1) {
                    /* copy 2 consecutive registers at once */
                    temps.push_back(map[i + 1]);
                    map[i + 1] = -1;
                    n++;
                }

                auto dt = useFloat ? DataType::f : DataType::ud;
                useFloat = !useFloat;

                mov(n * ne, GRF(map[i]).retype(dt), GRF(i).retype(dt));
                map[i] = -1;
                progress = true;
            }
        }

        if (!progress && !done) {
            // Get a few temporaries to break cycles, copy and continue.
            // Grab temporaries from already-moved src registers if available.
            int unstuck = 0;
            constexpr int maxUnstuck = 8;
            std::array<int16_t, maxUnstuck> from, to;

            for (int i = 0; i < regs; i++) if (map[i] >= 0) {
                GRF temp;
                if (temps.empty()) {
                    temp = state.ra.tryAlloc();
                    if (temp.isInvalid()) {
                        if (unstuck == 0) throw out_of_registers_exception();
                        break;
                    }
                    alloced.push_back(temp);
                } else {
                    temp = GRF(temps.back());
                    temps.pop_back();
                }

                mov<int32_t>(ne, temp, GRF(i));
                from[unstuck] = temp.getBase();
                to[unstuck] = map[i];
                map[i] = -1;
                if (++unstuck >= maxUnstuck) break;  /* that's enough for now */
            }

            for (int j = 0; j < unstuck; j++)
                map[from[j]] = to[j];
        }
    }

    for (auto &r: alloced)
        state.ra.release(r);
}

#include "internal/namespace_end.hxx"
