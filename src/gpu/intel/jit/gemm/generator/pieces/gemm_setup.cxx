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


#include "alloc_utils.hpp"
#include "compute_utils.hpp"
#include "cooperative_split.hpp"
#include "generator.hpp"
#include "hw_utils.hpp"
#include "kernel_queries.hpp"
#include "layout_utils.hpp"
#include "ngen_object_helpers.hpp"
#include "state_utils.hpp"
#include "token_alloc_utils.hpp"

using namespace ngen;
using namespace ngen::utils;
using std::vector;

#include "internal/namespace_start.hxx"

// Generate code for checking whether 32-bit address arithmetic can be used inside k loop.
// Assumes leading dimensions have not been shifted yet.
template <HW hw>
void BLASKernelGenerator<hw>::gemmCheck32(const GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state)
{
    if (!strategy.checkAdd32)
        return;

    bool checkA = (strategy.A.base.getModel() == ModelA64);
    bool checkB = (strategy.B.base.getModel() == ModelA64);
    if (!checkA && !checkB)
        return;

    bool emulate = strategy.emulate.emulate64_mul;
    auto &m = state.inputs.m;
    auto &n = state.inputs.n;
    auto &k = state.fullK.isValid() ? state.fullK : state.inputs.k;
    auto &lda = state.inputs.lda;
    auto &ldb = state.inputs.ldb;
    auto temp1GRF = state.ra.alloc();
    auto temp2GRF = state.ra.alloc();
    auto temp1 = temp1GRF.uq(0);
    auto temp1Hi = temp1GRF.ud(emulate ? 0 : 1);
    auto temp2 = temp2GRF.ud(0);
    auto temp3 = temp2GRF.ud(4);
    auto flag = state.raVFlag.alloc();

    auto mulHigh = [&](Subregister dst, Subregister src0, Subregister src1) {
        if (emulate)
            emul32High(1, dst.ud(), src0, src1);
        else
            mul(1, dst, src0, src1);
    };

    if (checkA) {
        state.offsetA.isValid() ? add(1, temp2, state.effA.ud(), state.offsetA.ud())
                                : mov(1, temp2, state.effA.ud());
        switch (problem.A.layout) {                 // Conservatively estimate upper bound for size of A.
            case MatrixLayout::N:  mulHigh(temp1, lda, k); break;
            case MatrixLayout::T:  mulHigh(temp1, lda, m); break;
            case MatrixLayout::Pc: {
                if (strategy.fixedWG(problem))
                    add(1, temp3, m, uint16_t(strategy.wg[LoopM] * strategy.unroll[LoopM] - 1));
                else
                    emad(1, temp3, m, state.inputs.localSizeM, strategy.unroll[LoopM], strategy, state);
                mulHigh(temp1, lda, temp3); break;
            }
            default: stub();
        }
        add(1 | ov | flag, temp2, acc0.ud(0), temp2);
        cmp(1 | ~flag | ne | flag, temp1Hi, uint16_t(0));
    }

    if (checkB) {
        state.offsetB.isValid() ? add(1, temp2, state.effB.ud(), state.offsetB.ud())
                                : mov(1, temp2, state.effB.ud());
        switch (problem.B.layout) {
            case MatrixLayout::T:  mulHigh(temp1, ldb, k); break;
            case MatrixLayout::N:  mulHigh(temp1, ldb, n); break;
            case MatrixLayout::Pr: {
                if (strategy.fixedWG(problem))
                    add(1, temp3, n, uint16_t(strategy.wg[LoopN] * strategy.unroll[LoopN] - 1));
                else
                    emad(1, temp3, n, state.inputs.localSizeN, strategy.unroll[LoopN], strategy, state);
                mulHigh(temp1, ldb, temp3); break;
            }
            default: stub();
        }
        InstructionModifier mod = 1;
        if (checkA)
            mod |= ~flag;
        add(mod | ov | flag, temp2, acc0.ud(0), temp2);
        cmp(1 | ~flag | ne | flag, temp1Hi, uint16_t(0));
    }

    state.add64 = state.ra.alloc_sub<uint16_t>();
    and_(1, state.add64, flag, 1u);
    state.raVFlag.safeRelease(flag);

    state.ra.safeRelease(temp1GRF); temp1 = invalid;
    state.ra.safeRelease(temp2GRF); temp2 = invalid; temp3 = invalid;
}

// Calculate A offset for SLM copies or cooperative prefetches for this local ID.
template <HW hw>
void BLASKernelGenerator<hw>::gemmCalcWorkshareAOffset(Subregister &off, Subregister &offR, Subregister &offC,
                                                       const MatrixAddressing &A, const MatrixAddressingStrategy &A_strategy, int ma, int ka,
                                                       const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    bool splitM = (state.effCoopA == CoopSplit::MN);
    bool splitLinear = (state.effCoopA == CoopSplit::Linear);
    bool splitKFull = (state.effCoopA == CoopSplit::FullK);

    auto lid = splitKFull ? gemmMNLinearID(strategy, state)
                          : state.lidN;

    if (A_strategy.address2D) {
        if (splitLinear) stub();
        if (splitM) {
            offR = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp0, strategy));
            mulConstant(1, offR, lid, ma);
        } else {
            offC = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp0, strategy));
            mulConstant(1, offC, lid, ka);
        }
    } else {
        auto Ta_ext = problem.Ta_ext;
        off = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp0, strategy));

        switch (A.layout) {
            case MatrixLayout::Pc:
                mulConstant(1, off, lid, ma * ka * Ta_ext);
                break;
            case MatrixLayout::T:
                if (splitLinear) stub();
                if (splitM) {
                    mul(1, off, state.inputs.lda, lid);
                    mulConstant(1, off, off, ma);
                } else
                    mulConstant(1, off, lid, ka * Ta_ext);
                break;
            case MatrixLayout::N:
                if (splitLinear) stub();
                if (splitM)
                    mulConstant(1, off, lid, ma * Ta_ext);
                else {
                    mul(1, off, state.inputs.lda, lid);
                    mulConstant(1, off, off, ka);
                }
                break;
            default: stub();
        }
    }

    if (splitKFull) state.ra.safeRelease(lid);
}

// Calculate B offset for SLM copies or cooperative prefetches for this local ID.
template <HW hw>
void BLASKernelGenerator<hw>::gemmCalcWorkshareBOffset(Subregister &off, Subregister &offR, Subregister &offC,
                                                       const MatrixAddressing &B, const MatrixAddressingStrategy &B_strategy, int kb, int nb,
                                                       const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    bool splitN = (state.effCoopB == CoopSplit::MN);
    bool splitLinear = (state.effCoopB == CoopSplit::Linear);
    bool splitKFull = (state.effCoopB == CoopSplit::FullK);

    auto lid = splitKFull ? gemmMNLinearID(strategy, state)
                          : state.lidM;

    if (B_strategy.address2D) {
        if (splitLinear) stub();
        if (splitN) {
            offC = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp0, strategy));
            mulConstant(1, offC, lid, nb);
        } else {
            offR = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp0, strategy));
            mulConstant(1, offR, lid, kb);
        }
    } else {
        auto Tb_ext = problem.Tb_ext;
        off = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp0, strategy));

        switch (B.layout) {
            case MatrixLayout::Pr:
                mulConstant(1, off, lid, nb * kb * Tb_ext);
                break;
            case MatrixLayout::N:
                if (splitLinear) stub();
                if (splitN) {
                    mul(1, off, state.inputs.ldb, lid);
                    mulConstant(1, off, off, nb);
                } else
                    mulConstant(1, off, lid, kb * Tb_ext);
                break;
            case MatrixLayout::T:
                if (splitLinear) stub();
                if (splitN)
                    mulConstant(1, off, lid, nb * Tb_ext);
                else {
                    mul(1, off, state.inputs.ldb, lid);
                    mulConstant(1, off, off, kb);
                }
                break;
            default: stub();
        }
    }

    if (splitKFull) state.ra.safeRelease(lid);
}

// Calculate m,n joint linear local ID.
template <HW hw>
Subregister BLASKernelGenerator<hw>::gemmMNLinearID(const GEMMStrategy &strategy, GEMMState &state)
{
    auto lid = state.ra.alloc_sub<uint16_t>();
    if (strategy.loopOrder[0] == LoopM)
        emad(1, lid, state.lidM, state.lidN, strategy.wg[LoopM], strategy, state);
    else
        emad(1, lid, state.lidN, state.lidM, strategy.wg[LoopN], strategy, state);
    return lid;
}

// TODO: move me
template <HW hw>
CoopSplit BLASKernelGenerator<hw>::effCoopSplitA(const GEMMProblem &problem, const GEMMStrategy &strategy)
{
    if (isPacked(problem.A.layout))
        return CoopSplit::Linear;
    else if (!isRegisterColMajor(problem.Ta_ext, problem.A, strategy.A)
                && (strategy.unroll[LoopM] % strategy.wg[LoopN] == 0)
                && !isBlock2D(strategy.A.accessType)
                && strategy.coopA != CoopSplit::FullK)
        return CoopSplit::MN;
    else
        return strategy.coopA;
}

template <HW hw>
CoopSplit BLASKernelGenerator<hw>::effCoopSplitB(const GEMMProblem &problem, const GEMMStrategy &strategy)
{
    if (isPacked(problem.B.layout))
        return CoopSplit::Linear;
    else if (isRegisterColMajor(problem.Tb_ext, problem.B, strategy.B)
                && (strategy.unroll[LoopN] % strategy.wg[LoopM] == 0)
                && !isBlock2D(strategy.B.accessType)
                && strategy.coopB != CoopSplit::FullK)
        return CoopSplit::MN;
    else
        return strategy.coopB;
}

// Offset A pointer in k dimension by a constant value.
template <HW hw>
void BLASKernelGenerator<hw>::gemmOffsetAk(int h, const Subregister &effA, const MatrixAddressing &globalA, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto Ta_ext = problem.Ta_ext;
    if (h) switch (globalA.layout) {
        case MatrixLayout::N:  emad(1, effA, effA, state.inputs.lda, Immediate::w(h), strategy, state); break;
        case MatrixLayout::T:  eadd(1, effA, effA, h * Ta_ext, strategy, state); break;
        case MatrixLayout::Pc: eadd(1, effA, effA, h * globalA.packSize * Ta_ext, strategy, state); break;
        default: stub();
    }
}

// Offset A pointer in k dimension by a variable value.
template <HW hw>
void BLASKernelGenerator<hw>::gemmOffsetAk(const Subregister &h, const Subregister &effA, const MatrixAddressing &globalA, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto Ta_ext = problem.Ta_ext;
    switch (globalA.layout) {
        case MatrixLayout::N:  emad(1, effA, effA, state.inputs.lda, h, strategy, state); break;
        case MatrixLayout::T:  eaddScaled(1, effA, effA, h, Ta_ext, strategy, state); break;
        case MatrixLayout::Pc: emad(1, effA, effA, h, globalA.packSize * Ta_ext, strategy, state); break;
        default: stub();
    }
}

// Offset B pointer in k dimension by a constant value.
template <HW hw>
void BLASKernelGenerator<hw>::gemmOffsetBk(int h, const Subregister &effB, const MatrixAddressing &globalB, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto Tb_ext = problem.Tb_ext;
    if (h) switch (globalB.layout) {
        case MatrixLayout::N:  eadd(1, effB, effB, h * Tb_ext, strategy, state); break;
        case MatrixLayout::Pr: eadd(1, effB, effB, h * globalB.packSize * Tb_ext, strategy, state); break;
        case MatrixLayout::T:  emad(1, effB, effB, state.inputs.ldb, Immediate::w(h), strategy, state); break;
        default: stub();
    }
}

// Offset B pointer in k dimension by a variable value.
template <HW hw>
void BLASKernelGenerator<hw>::gemmOffsetBk(const Subregister &h, const Subregister &effB, const MatrixAddressing &globalB, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto Tb_ext = problem.Tb_ext;
    switch (globalB.layout) {
        case MatrixLayout::T:  emad(1, effB, effB, state.inputs.ldb, h, strategy, state); break;
        case MatrixLayout::N:  eaddScaled(1, effB, effB, h, Tb_ext, strategy, state); break;
        case MatrixLayout::Pr: emad(1, effB, effB, h, globalB.packSize * Tb_ext, strategy, state); break;
        default: stub();
    }
}

// Adjust A, B, C to start at (i0, j0).
//  initial is true to adjust offset_{A,B,C}, false to adjust A,B,C pointers.
template <HW hw>
void BLASKernelGenerator<hw>::gemmOffsetABC(bool initial, Subregister i0, Subregister j0, Subregister h0, Subregister i0p, Subregister j0p,
                                            const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state,
                                            bool doA, bool doB, bool doC, bool doBinary)
{
    auto Ta_ext = problem.Ta_ext, Tb_ext = problem.Tb_ext, Tc_ext = problem.Tc_ext, Tco = problem.Tco;
    auto &offsetA  = initial ? state.offsetA    : state.effA;
    auto &offsetB  = initial ? state.offsetB    : state.effB;
    auto &offsetC0 = initial ? state.offsetC[0] : state.effC[0];
    auto &offsetAp = initial ? state.offsetAp   : state.effAp;
    auto &offsetBp = initial ? state.offsetBp   : state.effBp;
    auto &offsetCp = initial ? state.offsetCp   : state.effCp;
    auto  offsetCO = initial ? state.offsetCO   : state.effCO;
    bool doCO = doC && (problem.cOffset != COffset::None);
    bool doAp = doA, doBp = doB;

    Subregister tempQ0 = state.ra.alloc_sub<int64_t>(getHint(HintType::TempComp0, strategy));
    Subregister tempQ1 = state.ra.alloc_sub<int64_t>(getHint(HintType::TempComp1, strategy));

    bool a2D = strategy.A.address2D;
    bool b2D = strategy.B.address2D;
    bool c2D = strategy.C.address2D;
    bool ap2D = strategy.prefetchA ? strategy.A_prefetch.address2D : a2D;
    bool bp2D = strategy.prefetchB ? strategy.B_prefetch.address2D : b2D;
    bool cp2D = strategy.prefetchC ? strategy.C_prefetch.address2D : c2D;

    if (a2D)         doA  = false;
    if (ap2D)        doAp = false;
    if (b2D)         doB  = false;
    if (bp2D)        doBp = false;
    if (c2D && cp2D) doC  = false;

    if (doA && doAp && i0 == i0p) doAp = false;
    if (doB && doBp && j0 == j0p) doBp = false;

    bool splitAp = (a2D != ap2D) || (doA && doAp);
    bool splitBp = (b2D != bp2D) || (doB && doBp);

    if ((doA || doAp) && splitAp && (offsetAp.isInvalid() || offsetA == offsetAp)) {
        offsetAp = state.ra.alloc_sub(offsetA.getType(), getHint(HintType::LongTerm, strategy));
        emov(1, offsetAp, offsetA, strategy, state);
    }
    if ((doB || doBp) && splitBp && (offsetBp.isInvalid() || offsetB == offsetBp)) {
        offsetBp = state.ra.alloc_sub(offsetB.getType(), getHint(HintType::LongTerm, strategy));
        emov(1, offsetBp, offsetB, strategy, state);
    }
    if (doC && (c2D != cp2D)) {
        if (offsetCp.isInvalid() || offsetC0 == offsetCp) {
            offsetCp = state.ra.alloc_sub(offsetC0.getType(), getHint(HintType::LongTerm, strategy));
            emov(1, offsetCp, offsetC0, strategy, state);
        } else if (c2D && !cp2D)
            std::swap(offsetC0, offsetCp);
    }

    // To do: interleave code.
    // A += i0 (N) i0 * lda (T, Pc)
    // B += j0 * ldb (N, Pr) j0 (T)
    // C += i0 + j0 * ldc (N, Pr) j0 + i0 * ldc (T, Pc)
    // CO += i0 (row offsets) j0 (col offsets)
    auto doAOffset = [&](Subregister offsetAx, Subregister i0x) {
        if (problem.A.layout == MatrixLayout::Nontranspose)
            eaddScaled(1, offsetAx, offsetAx, i0x, Ta_ext, strategy, state);
        else {
            emul(1, tempQ1, i0x, state.inputs.lda, strategy, state);
            eadd(1, offsetAx, offsetAx, tempQ1.reinterpret(0, offsetAx.getType()), strategy, state);
        }
    };

    auto doBOffset = [&](Subregister offsetBx, Subregister j0x) {
        if (problem.B.layout == MatrixLayout::Transpose)
            eaddScaled(1, offsetBx, offsetBx, j0x, Tb_ext, strategy, state);
        else {
            emul(1, tempQ0, j0x, state.inputs.ldb, strategy, state);
            eadd(1, offsetBx, offsetBx, tempQ0.reinterpret(0, offsetBx.getType()), strategy, state);
        }
    };

    if (doA  && i0.isValid())  doAOffset(offsetA,  i0);
    if (doAp && i0p.isValid()) doAOffset(offsetAp, i0p);
    if (doB  && j0.isValid())  doBOffset(offsetB,  j0);
    if (doBp && j0p.isValid()) doBOffset(offsetBp, j0p);

    FlagRegister flagCOR, flagCOC;
    if (doCO) {
        flagCOR = state.raVFlag.alloc();
        flagCOC = state.raVFlag.alloc();
        and_(1 | nz | flagCOC, null.ud(), state.inputs.flags, FlagCOColumn);
        and_(1 | nz | flagCOR, null.ud(), state.inputs.flags, FlagCORow);
    }
    if (doC) {
        for (int q = 0; q < state.C_count; q++) {
            auto offsetC = initial ? state.offsetC[q] : state.effC[q];

            Subregister x, y;
            int xstride = Tc_ext.size();
            switch (problem.C.layout) {
                case MatrixLayout::Pr:  xstride *= strategy.unroll[LoopN];   /* fall through */
                case MatrixLayout::N:   x = i0; y = j0;             break;
                case MatrixLayout::Pc:  xstride *= strategy.unroll[LoopM];   /* fall through */
                case MatrixLayout::T:   x = j0; y = i0;             break;
            }
            emad(1, offsetC, offsetC, x, xstride, strategy, state);
            emul(1, tempQ0, y, state.inputs.ldc[q], strategy, state);
            eadd(1, offsetC, offsetC, tempQ0.reinterpret(0, offsetC.getType()), strategy, state);       // Gen12: Use add3.
        }
    }
    if (doCO) {
        Label lNoMatrixOffset, lDone;
        if (problem.allowMatrixOffset()) {
            auto x0 = isColMajor(problem.CO.layout) ? i0 : j0;
            auto y0 = isColMajor(problem.CO.layout) ? j0 : i0;
            jmpi(1 | ~flagCOC, lNoMatrixOffset);
            jmpi(1 | ~flagCOR, lNoMatrixOffset);
            eaddScaled(1, offsetCO, offsetCO, x0, Tco, strategy, state);
            emul(1, tempQ0, y0, state.inputs.ldco, strategy, state);
            eadd(1, offsetCO, offsetCO, tempQ0.reinterpret(0, offsetCO.getType()), strategy, state);
            jmpi(1, lDone);
            mark(lNoMatrixOffset);
        }
        eaddScaled(1 | flagCOC, offsetCO, offsetCO, j0, Tco, strategy, state);
        eaddScaled(1 | flagCOR, offsetCO, offsetCO, i0, Tco, strategy, state);
        state.raVFlag.safeRelease(flagCOR);
        state.raVFlag.safeRelease(flagCOC);
        if (problem.allowMatrixOffset())
            mark(lDone);
    }
    if (doBinary) for (size_t i = 0; i < problem.postOps.len(); i++) {
        if (!problem.postOps[i].is_binary()) continue;
        bool row = problem.binaryRow[i], col = problem.binaryCol[i];
        auto T = problem.Tbinary[i];
        auto &ld = state.inputs.binaryLDs[i];
        auto offset = initial ? state.inputs.binaryOffsets[i] : state.effBinary[i];
        if (row && col) {
            auto x0 = isColMajor(problem.binary[i].layout) ? i0 : j0;
            auto y0 = isColMajor(problem.binary[i].layout) ? j0 : i0;
            eaddScaled(1, offset, offset, x0, T, strategy, state);
            emul(1, tempQ0, y0, ld, strategy, state);
            eadd(1, offset, offset, tempQ0.reinterpret(0, offset.getType()), strategy, state);
        } else if (row)
            eaddScaled(1, offset, offset, i0, T, strategy, state);
        else if (col)
            eaddScaled(1, offset, offset, j0, T, strategy, state);
    }
    if (doC && problem.sumA) eaddScaled(1, offsetCO, offsetCO, i0, Tco, strategy, state);
    if (doC && problem.sumB) eaddScaled(1, offsetCO, offsetCO, j0, Tco, strategy, state);

    // When k blocking (or certain triangular source kernels)
    //   A += h0 * lda (N) h0 (T) h0 * mb (Pc)
    //   B += h0 (N) h0 * ldb (T) h0 * nb (Pr)
    if (!h0.isInvalid()) {
        if (doA || doAp) switch (problem.A.layout) {
            case MatrixLayout::Nontranspose:
                emul(1, tempQ1, h0, state.inputs.lda, strategy, state);
                if (doA)  eadd(1, offsetA,  offsetA,  tempQ1.reinterpret(0, offsetA.getType()), strategy, state);
                if (doAp) eadd(1, offsetAp, offsetAp, tempQ1.reinterpret(0, offsetAp.getType()), strategy, state);
                break;
            case MatrixLayout::Transpose:
                if (doA)  eaddScaled(1, offsetA,  offsetA,  h0, Ta_ext, strategy, state);
                if (doAp) eaddScaled(1, offsetAp, offsetAp, h0, Ta_ext, strategy, state);
                break;
            case MatrixLayout::PackedColumns:
                if (doA)  emad(1, offsetA,  offsetA,  h0, strategy.unroll[LoopM] * Ta_ext, strategy, state);
                if (doAp) emad(1, offsetAp, offsetAp, h0, strategy.unroll[LoopM] * Ta_ext, strategy, state);
                break;
            default: stub();
        }
        if (doB || doBp) switch (problem.B.layout) {
            case MatrixLayout::Nontranspose:
                if (doB)  eaddScaled(1, offsetB,  offsetB,  h0, Tb_ext, strategy, state);
                if (doBp) eaddScaled(1, offsetBp, offsetBp, h0, Tb_ext, strategy, state);
                break;
            case MatrixLayout::Transpose:
                emul(1, tempQ0, h0, state.inputs.ldb, strategy, state);
                if (doB)  eadd(1, offsetB,  offsetB,  tempQ0.reinterpret(0, offsetB.getType()), strategy, state);
                if (doBp) eadd(1, offsetBp, offsetBp, tempQ0.reinterpret(0, offsetBp.getType()), strategy, state);
                break;
            case MatrixLayout::PackedRows:
                if (doB)  emad(1, offsetB,  offsetB,  h0, strategy.unroll[LoopN] * Tb_ext, strategy, state);
                if (doBp) emad(1, offsetBp, offsetBp, h0, strategy.unroll[LoopN] * Tb_ext, strategy, state);
                break;
            default: stub();
        }
    }

    state.ra.safeRelease(tempQ0);
    state.ra.safeRelease(tempQ1);

    if (doC && c2D && !cp2D) std::swap(offsetC0, offsetCp);
}

template <ngen::HW hw>
void BLASKernelGenerator<hw>::gemmOffsetBatchABC(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{

    // Strided batch support.
    if (problem.batch == BatchMode::Strided) {
        Subregister bOffsetA[4], bOffsetB[4], bOffsetC[4];

        for (int b = 0; b < problem.batchDims; b++) {
            bOffsetA[b] = state.inputs.strideA[b];
            bOffsetB[b] = state.inputs.strideB[b];
            bOffsetC[b] = state.inputs.strideC[b];
            if (strategy.A.base.isStateless()) bOffsetA[b] = state.ra.alloc_sub<uint64_t>();
            if (strategy.B.base.isStateless()) bOffsetB[b] = state.ra.alloc_sub<uint64_t>();
            if (strategy.C.base.isStateless()) bOffsetC[b] = state.ra.alloc_sub<uint64_t>();
        }

        for (int b = 0; b < problem.batchDims; b++) {
            emul(1, bOffsetA[b], state.inputs.strideA[b], state.batchID[b], strategy, state);
            emul(1, bOffsetB[b], state.inputs.strideB[b], state.batchID[b], strategy, state);
            emul(1, bOffsetC[b], state.inputs.strideC[b], state.batchID[b], strategy, state);
        }

        for (int b = 0; b < problem.batchDims; b++) {
            eadd(1, state.offsetA, state.offsetA, bOffsetA[b], strategy, state);
            eadd(1, state.offsetB, state.offsetB, bOffsetB[b], strategy, state);
            for (int q = 0; q < state.C_count; q++) {
                auto offsetC = state.offsetC[q];
                eadd(1, offsetC, offsetC, bOffsetC[b], strategy, state);
            }
            if (!strategy.persistent) {
                state.ra.safeRelease(state.inputs.strideA[b]);
                state.ra.safeRelease(state.inputs.strideB[b]);
                state.ra.safeRelease(state.inputs.strideC[b]);
            }
            if (strategy.A.base.isStateless()) state.ra.safeRelease(bOffsetA[b]);
            if (strategy.B.base.isStateless()) state.ra.safeRelease(bOffsetB[b]);
            if (strategy.C.base.isStateless()) state.ra.safeRelease(bOffsetC[b]);
        }
    }

}

// Prepare for persistent GEMM by folding offsets into A/B/C pointers (if stateless),
//  or saving offsets (if stateful)
template <HW hw>
void BLASKernelGenerator<hw>::gemmFoldOffsets(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto foldOrSave = [&](const MatrixAddressingStrategy &sX, Subregister &inputX, Subregister &offsetX, const Subregister &inputOffsetX, Subregister &saveOffsetX, bool newInput = false) {
        if (sX.base.isStateless()) {
            auto oldInputX = inputX;
            if (newInput)
                inputX = state.ra.alloc_sub(DataType::uq, getHint(HintType::LongTerm, strategy));
            eadd(1, inputX, oldInputX, offsetX, strategy, state);
            if (getBytes(offsetX.getType()) < 8) {
                state.ra.safeRelease(offsetX);
                offsetX = state.ra.alloc_sub(DataType::uq, getHint(HintType::LongTerm, strategy));
            }
            emov(1, offsetX, 0, strategy, state);
        } else {
            offsetX = state.ra.alloc_sub(offsetX.getType(), getHint(HintType::LongTerm, strategy));
            mov(1, offsetX, inputOffsetX);
        }
        saveOffsetX = offsetX;
    };

    bool deduplicateAB = (state.inputs.A == state.inputs.B);

    foldOrSave(strategy.A, state.inputs.A, state.offsetA, state.inputs.offsetA, state.saveOffsetA, deduplicateAB);
    foldOrSave(strategy.B, state.inputs.B, state.offsetB, state.inputs.offsetB, state.saveOffsetB);
    for (int q = 0; q < state.C_count; q++)
        foldOrSave(strategy.C, state.inputs.C[q], state.offsetC[q], state.inputs.offsetC[q], state.saveOffsetC[q]); // todo init for hpl
    if (problem.usesCO())
        foldOrSave(strategy.CO, state.inputs.CO, state.offsetCO, state.inputs.offsetCO, state.saveOffsetCO);

    if (deduplicateAB)
        state.effA = state.inputs.A;
}

// Restore input offsets from saved copies, for persistent GEMM.
template <HW hw>
void BLASKernelGenerator<hw>::gemmRestoreOffsets(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto zeroOrRestore = [&](const MatrixAddressingStrategy &sX, const Subregister &offsetX, const Subregister &inputOffsetX) {
        if (sX.base.isStateless())
            emov(1, offsetX, 0, strategy, state);
        else
            mov(1, offsetX, inputOffsetX);
    };

    zeroOrRestore(strategy.A, state.saveOffsetA, state.inputs.offsetA);
    zeroOrRestore(strategy.B, state.saveOffsetB, state.inputs.offsetB);
    for (int q = 0; q < state.C_count; q++)
        zeroOrRestore(strategy.C, state.saveOffsetC[q], state.inputs.offsetC[q]);
    if (problem.usesCO())
        zeroOrRestore(strategy.CO, state.saveOffsetCO, state.inputs.offsetCO);
}

// Prepare final A/B/C pointers for a GEMM-like inner loop.
template <HW hw>
void BLASKernelGenerator<hw>::gemmSetupABC(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    if (strategy.persistent) {
        state.effA = state.offsetA;
        state.effB = state.offsetB;
        for (int q = 0; q < state.C_count; q++)
            state.effC[q] = state.offsetC[q];
        state.effCO = state.offsetCO;
    }

    if (strategy.C.base.isStateless()) {
        for (int q = 0; q < state.C_count; q++) {
            auto Csrc = state.inputs.C[q];
            if ((q > 0) && strategy.C.base.isStateless() && state.inputs.base.isValid())
                state.effC[q] = state.inputs.C[q] = state.ra.alloc_sub<uint64_t>(getHint(HintType::LongTerm, strategy));

            eadd(1, state.effC[q], Csrc, state.offsetC[q], strategy, state);
            if (strategy.persistent)
                state.offsetC[q] = invalid;
            else
                state.ra.safeRelease(state.offsetC[q]);
        }
    }

    if (problem.usesCO() && strategy.CO.base.isStateless()) {
        eadd(1, state.effCO, state.inputs.CO, state.offsetCO, strategy, state);
        if (strategy.persistent)
            state.offsetCO = invalid;
        else
            state.ra.safeRelease(state.offsetCO);
    }

    if (state.offsetAp.isValid()) {
        if (strategy.A.base.isStateless()) {
            state.effAp = state.ra.alloc_sub<uint64_t>(getHint(HintType::LongTerm, strategy));
            eadd(1, state.effAp, state.inputs.A, state.offsetAp, strategy, state);
            state.ra.safeRelease(state.offsetAp);
        } else
            state.effAp = state.offsetAp;
    }

    if (state.offsetBp.isValid()) {
        if (strategy.B.base.isStateless()) {
            state.effBp = state.ra.alloc_sub<uint64_t>(getHint(HintType::LongTerm, strategy));
            eadd(1, state.effBp, state.inputs.B, state.offsetBp, strategy, state);
            state.ra.safeRelease(state.offsetBp);
        } else
            state.effBp = state.offsetBp;
    }

    if (state.offsetCp.isValid()) {
        if (strategy.C.base.isStateless()) {
            state.effCp = state.ra.alloc_sub<uint64_t>(getHint(HintType::LongTerm, strategy));
            eadd(1, state.effCp, state.inputs.C[0], state.offsetCp, strategy, state);
            state.ra.safeRelease(state.offsetCp);
        } else
            state.effCp = state.offsetCp;
    }

    if (strategy.A.base.isStateless()) {
        auto Asrc = state.inputs.A;
        if (strategy.B.base.isStateless() && (state.effA == state.effB))
            state.effA = state.inputs.A = state.ra.alloc_sub<uint64_t>(getHint(HintType::LongTerm, strategy));

        eadd(1, state.effA, Asrc, state.offsetA, strategy, state);
        if (strategy.persistent)
            state.offsetA = invalid;
        else
            state.ra.safeRelease(state.offsetA);
    }

    if (strategy.B.base.isStateless()) {
        eadd(1, state.effB, state.inputs.B, state.offsetB, strategy, state);
        if (strategy.persistent)
            state.offsetB = invalid;
        else
            state.ra.safeRelease(state.offsetB);
    }

    if (strategy.prefetchA && state.effAp.isInvalid()) state.effAp = state.effA;
    if (strategy.prefetchB && state.effBp.isInvalid()) state.effBp = state.effB;
    if (strategy.prefetchC && state.effCp.isInvalid()) state.effCp = state.effC[0];
}

// Get (possibly multidimensional) batch IDs.
template <HW hw>
void BLASKernelGenerator<hw>::gemmGetBatchIDs(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    if (problem.batchDims == 0) return;

    if (problem.batchDims == 1) {
        state.batchID[0] = state.inputs.groupIDK;
        return;
    }

    for (int i = 0; i < problem.batchDims; i++)
        state.batchID[i] = state.ra.alloc_sub<uint32_t>();

    auto div = state.ra.alloc_sub<uint32_t>();
    mov(1, div, state.inputs.groupIDK);
    for (int i = problem.batchDims - 1; i >= 0; i--) {
        auto idx = problem.batchDims - 1 - i;
        mov(1, state.batchID[idx], div);
        if (i > 0) {
            divDown(div, div, state.inputs.batchSize[i - 1], state.inputs.recipBatchSize[i - 1], state.flagAP, strategy, state);
            emad(1, state.batchID[idx], state.batchID[idx], -div, state.inputs.batchSize[i - 1], strategy, state);
            if (!strategy.persistent) {
                state.ra.safeRelease(state.inputs.batchSize[i - 1]);
                state.ra.safeRelease(state.inputs.recipBatchSize[i - 1]);
            }
        }
    }
    state.ra.safeRelease(div);
}

template <HW hw>
void BLASKernelGenerator<hw>::gemmReleaseBatchIDs(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    if (problem.batch != BatchMode::Strided) return;
    if (problem.batchDims == 1 && state.r0_info == r0) return;
    if (problem.hasBinaryPostOp()) return;
    for (int b = 0; b < problem.batchDims; b++)
        state.ra.safeRelease(state.batchID[b]);
}


// Convert leading dimension and offset inputs to bytes.
template <ngen::HW hw>
void BLASKernelGenerator<hw>::gemmScaleInputs(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto Ta_ext = problem.Ta_ext, Tb_ext = problem.Tb_ext, Tc_ext = problem.Tc_ext, Tco = problem.Tco;
    auto &inputs = state.inputs;

    auto scale = [&](Type T, Subregister &s, Subregister defaultSrc = Subregister()) {
        if (s.isValid())
            emulConstant(1, s, s, T, strategy, state);
        else if (defaultSrc.isValid()) {
            s = state.ra.alloc_sub(defaultSrc.getType(), getHint(HintType::LongTerm, strategy));
            emulConstant(1, s, defaultSrc, T, strategy, state);
        }
    };

    scale(Ta_ext, inputs.lda);
    if (state.inputs.ldb != state.inputs.lda)
        scale(Tb_ext, inputs.ldb);
    for (int q = 0; q < state.C_count; q++)
        scale(Tc_ext, inputs.ldc[q]);
    scale(Tco, inputs.ldco);

    {
        if (strategy.A.base.getModel() != ModelSLM)
            scale(Ta_ext, inputs.offsetA);
        if (strategy.B.base.getModel() != ModelSLM)
            scale(Tb_ext, inputs.offsetB);
        for (int q = 0; q < state.C_count; q++)
            scale(Tc_ext, inputs.offsetC[q]);
        if (problem.usesCO())
            scale(Tco, inputs.offsetCO);
    }

    if (problem.batch == BatchMode::Strided) for (int b = 0; b < problem.batchDims; b++) {
        scale(Ta_ext, inputs.strideA[b]);
        scale(Tb_ext, inputs.strideB[b]);
        scale(Tc_ext, inputs.strideC[b]);
    }

    auto ldaq = inputs.ldaq, ldbq = inputs.ldbq;
    if (ldaq.isInvalid()) ldaq = inputs.m;
    if (ldbq.isInvalid()) ldbq = inputs.n;

    if (problem.aoPtrDims == 2)
        scale(problem.Tao, inputs.ldao, ldaq);
    if (problem.aoPtrDims >= 0)
        scale(problem.Tao, inputs.offsetAO, inputs.offsetAq);
    if (problem.boPtrDims == 2)
        scale(problem.Tbo, inputs.ldbo, ldbq);
    if (problem.boPtrDims >= 0)
        scale(problem.Tbo, inputs.offsetBO, inputs.offsetBq);
    if (problem.aScale2D) {
        scale(problem.Ta_scale, inputs.ldaScale, ldaq);
        scale(problem.Ta_scale, inputs.offsetAScale, inputs.offsetAq);
    }
    if (problem.bScale2D) {
        scale(problem.Tb_scale, inputs.ldbScale, ldbq);
        scale(problem.Tb_scale, inputs.offsetBScale, inputs.offsetBq);
    }

    state.ldao = inputs.ldao;
    state.ldbo = inputs.ldbo;
    state.ldaScale = inputs.ldaScale;
    state.ldbScale = inputs.ldbScale;

    state.ra.safeRelease(inputs.ldaq);
    state.ra.safeRelease(inputs.ldbq);
    state.ra.safeRelease(inputs.offsetAq);
    state.ra.safeRelease(inputs.offsetBq);
}

// Calculate workgroup m/n remainders.
template <HW hw>
void BLASKernelGenerator<hw>::gemmCalcWGRemainders(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    if (wgRemCheck(problem, strategy)) {
        state.remaindersWG[LoopM] = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp1, strategy));
        state.remaindersWG[LoopN] = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp0, strategy));
        add(1 | sat, state.remaindersWG[LoopM], -state.wgI0, state.inputs.m);
        add(1 | sat, state.remaindersWG[LoopN], -state.wgJ0, state.inputs.n);
    }
    if (strategy.coopA != CoopSplit::FullK) state.ra.safeRelease(state.wgI0);
    if (strategy.coopB != CoopSplit::FullK) state.ra.safeRelease(state.wgJ0);
}

// Cache multiples of lda/ldb for later address calculations.
template <HW hw>
void BLASKernelGenerator<hw>::gemmCacheLDABMultiples(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, bool doA, bool doB)
{
    int na = 0, nb = 0;

    if (doA && !strategy.A.address2D) switch (problem.A.layout) {
        case MatrixLayout::N: na = std::max(strategy.ka_load, strategy.ka_prefetch); break;
        case MatrixLayout::T:
            na = strategy.unroll[LoopM];
            if (isTransposing(strategy.A.accessType)) na = std::min(na, maxScatteredSIMD(hw, strategy.A));
            break;
        default: break;
    }

    if (doB && !strategy.B.address2D) switch (problem.B.layout) {
        case MatrixLayout::T: nb = std::max(strategy.kb_load, strategy.kb_prefetch); break;
        case MatrixLayout::N:
            nb = strategy.unroll[LoopN];
            if (isTransposing(strategy.B.accessType)) nb = std::min(nb, maxScatteredSIMD(hw, strategy.B));
            break;
        default: break;
    }

    if (na <= 2) na = 0;
    if (nb <= 2) nb = 0;

    if (na || nb)
        extendIndexVec(std::max(na, nb), state);

    if (na) {
        bool a64 = (strategy.A.base.getModel() == ModelA64);
        state.ldaMultiples = createLDMultiples(a64, na, state.lda, strategy, state);
    }

    if (nb) {
        bool a64 = (strategy.B.base.getModel() == ModelA64);
        state.ldbMultiples = createLDMultiples(a64, nb, state.ldb, strategy, state);
    }
}

// Cache multiples of ldc for later address calculations.
template <HW hw>
void BLASKernelGenerator<hw>::gemmCacheLDCMultiples(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, bool prefetch)
{
    if ((prefetch ? strategy.C_prefetch : strategy.C).address2D) return;

    int nc = 0;
    switch (problem.C.layout) {
        case MatrixLayout::N: nc = strategy.unroll[LoopN]; break;
        case MatrixLayout::T: nc = strategy.unroll[LoopM]; break;
        default: break;
    }

    if (nc <= 2) return;

    bool a64 = (strategy.C.base.getModel() == ModelA64);
    int C_count = prefetch ? 1 : state.C_count;
    for (int q = 0; q < C_count; q++)
        state.ldcMultiples[q] = createLDMultiples(a64, nc, state.inputs.ldc[q], strategy, state);
}

// Calculate actual amount of k-padding for variable k-slicing.
template <HW hw>
Subregister BLASKernelGenerator<hw>::gemmCalcKPadding(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    // kpad = min(strategy.kPadding, 2*k0).
    auto effKPad = state.ra.alloc_sub<uint32_t>();
    shl(1, effKPad, state.inputs.k0, 1);
    min_(1, effKPad, effKPad, strategy.kPadding);
    return effKPad;
}

template <HW hw>
void BLASKernelGenerator<hw>::gemmDowngradeAccess(const GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state)
{
    bool pfA = strategy.prefetchA, pfB = strategy.prefetchB;
    bool a2D = strategy.A.address2D, ap2D = strategy.A_prefetch.address2D;
    bool b2D = strategy.B.address2D, bp2D = strategy.B_prefetch.address2D;

    bool oldA2D = a2D, oldAp2D = ap2D, oldB2D = b2D, oldBp2D = bp2D;

    strategy.A.accessType = strategy.unalignedAccA;
    strategy.B.accessType = strategy.unalignedAccB;

    a2D &= isBlock2D(strategy.A.accessType);
    b2D &= isBlock2D(strategy.B.accessType);

    int minAlignA = block2DMinAlignment(hw, problem.A, strategy.A_prefetch);
    int minAlignB = block2DMinAlignment(hw, problem.B, strategy.B_prefetch);

    if (pfA && isBlock2D(strategy.A_prefetch.accessType) && problem.A.alignment < minAlignA) {
        downgradeAPFAccess(problem, strategy);
        ap2D = false;
    }

    if (pfB && isBlock2D(strategy.B_prefetch.accessType) && problem.B.alignment < minAlignB) {
        downgradeBPFAccess(problem, strategy);
        bp2D = false;
    }

    if (pfA && !a2D && !ap2D) {
        if (oldAp2D && !oldA2D) state.effAp = state.effA;
        if (oldA2D && !oldAp2D) state.effA = state.effAp;
    }
    if (pfB && !b2D && !bp2D) {
        if (oldBp2D && !oldB2D) state.effBp = state.effB;
        if (oldB2D && !oldBp2D) state.effB = state.effBp;
    }

    bool applyOffsetA = !pfA ? (oldA2D && !a2D) : (oldA2D && oldAp2D && (!a2D || !ap2D));
    bool applyOffsetB = !pfB ? (oldB2D && !b2D) : (oldB2D && oldBp2D && (!b2D || !bp2D));

    strategy.A.address2D = a2D;
    strategy.B.address2D = b2D;
    strategy.A_prefetch.address2D = ap2D;
    strategy.B_prefetch.address2D = bp2D;

    if (applyOffsetA || applyOffsetB)
        gemmOffsetABC(false, state.i0, state.j0, state.h0, Subregister(), Subregister(), problem, strategy, state, applyOffsetA, applyOffsetB, false);
}


static inline bool needsKLoopReset(const GEMMProblem &problem)
{
    return false;
}

// Setup for C accumulation.
// NOTE: modifies problem/strategy/state.
template <HW hw>
bool BLASKernelGenerator<hw>::gemmAccumulateCSetup(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state)
{
    auto &Ta = problem.Ta, &Tb = problem.Tb, Tc = problem.Tc, Tc_compute = problem.Tc_compute();
    auto Ta_ext = problem.Ta_ext, Tb_ext = problem.Tb_ext, Tc_ext = problem.Tc_ext;
    auto Tao = problem.Tao, Tbo = problem.Tbo;
    auto Ta_scale = problem.Ta_scale, Tb_scale = problem.Tb_scale;
    auto &Ta_load = state.Ta_load, &Tb_load = state.Tb_load;
    bool slmA = strategy.slmA, slmB = strategy.slmB;

    bool cLoadAhead = strategy.cLoadAhead;
    auto unrollM = strategy.unroll[LoopM];
    auto unrollN = strategy.unroll[LoopN];
    auto unrollK = strategy.unroll[LoopK];
    auto unrollKSLM = strategy.unrollKSLM;

    // Grab a whole flag register now to do emulated SIMD16 64-bit adds, if needed.
    FlagRegister simd16EmulationFlag;
    if (state.emulate.temp[0].isValid() && GRF::bytes(hw) == 64
            && strategy.emulate.emulate64 && !strategy.emulate.emulate64_add32)
        simd16EmulationFlag = state.raVFlag.alloc();

    // Decide what remainder handling needs to be done.
    bool remainderM = strategy.remHandling[LoopM] != RemainderHandling::Ignore;
    bool remainderN = strategy.remHandling[LoopN] != RemainderHandling::Ignore;
    bool remainderK = strategy.remHandling[LoopK] != RemainderHandling::Ignore;
    bool remM_A = remainderM && !strategy.A.padded;
    bool remK_A = false;
    bool remK_B = false;
    bool remN_B = remainderN && !strategy.B.padded;
    bool remM_C, remN_C;
    getCRemainders(hw, problem, strategy, remM_C, remN_C);
    bool remM_Ce = remM_C;
    bool remN_Ce = remN_C;

    if (state.copyC)
        remM_C = remN_C = false;

    auto globalA = problem.A;
    auto globalB = problem.B;

    // 2D addressing parameters.
    auto &A_params = state.A_params, &B_params = state.B_params;
    auto &Ai_params = state.Ai_params, &Bi_params = state.Bi_params;
    auto &Ap_params = state.Ap_params, &Bp_params = state.Bp_params;
    A_params.rows = state.inputs.m;
    A_params.cols = state.fullK;
    A_params.offR = state.i0;
    A_params.offC = state.h0;
    A_params.remR = state.remainders[LoopM];
    B_params.rows = state.fullK;
    B_params.cols = state.inputs.n;
    B_params.offR = state.h0;
    B_params.offC = state.j0;
    B_params.remC = state.remainders[LoopN];
    Ai_params = A_params, Bi_params = B_params;
    Ap_params = A_params, Bp_params = B_params;

    // Decide which dimensions to split for WG-cooperative operations (SLM copy, cooperative PF).
    state.effCoopA = effCoopSplitA(problem, strategy);
    state.effCoopB = effCoopSplitB(problem, strategy);

    if (state.effCoopA == CoopSplit::FullK) Ap_params.offR = Ai_params.offR = state.wgI0;
    if (state.effCoopB == CoopSplit::FullK) Bp_params.offC = Bi_params.offC = state.wgJ0;

    // Prepare remainders for cooperative operations.
    for (LoopType loop: {LoopM, LoopN, LoopK})
        state.remaindersCoop[loop] = state.remainders[loop];

    auto calcMNRemCoop = [&](CoopSplit split, bool isM) {
        auto loopX = isM ? LoopM : LoopN;
        auto loopY = isM ? LoopN : LoopM;
        switch (split) {
            default:
                return state.remainders[loopX];
            case CoopSplit::FullK:
                return state.remaindersWG[loopX];
            case CoopSplit::MN: {
                auto rem = state.ra.alloc_sub<uint16_t>();
                int32_t chunk = strategy.unroll[loopX] / strategy.wg[loopY];
                auto lid = isM ? state.lidN : state.lidM;
                emad(1 | sat, rem, state.remainders[loopX], -lid.w(), chunk, strategy, state);
                return rem;
            }
        }
    };

    if ((slmA || (strategy.prefetchA && strategy.cooperativePF)) && remM_A) {
        state.remaindersCoop[LoopM] = calcMNRemCoop(state.effCoopA, true);
    }

    if ((slmB || (strategy.prefetchB && strategy.cooperativePF)) && remN_B) {
          state.remaindersCoop[LoopN] = calcMNRemCoop(state.effCoopB, false);
    }

    // Prepare layouts for prefetch.
    bool remM_Cp = remM_C && strategy.C.base.isStateless();
    bool remN_Cp = remN_C && strategy.C.base.isStateless();

    state.ma_prefetch = state.ka_prefetch = state.kb_prefetch = state.nb_prefetch = 0;
    if (strategy.prefetchA) coopSplit(true,  state.ma_prefetch, state.ka_prefetch, unrollM, strategy.ka_prefetch, strategy.wgTile(LoopM), state.effCoopA, problem.A, strategy);
    if (strategy.prefetchB) coopSplit(false, state.kb_prefetch, state.nb_prefetch, strategy.kb_prefetch, unrollN, strategy.wgTile(LoopN), state.effCoopB, problem.B, strategy);

    if (strategy.prefetchA && !getRegLayout(Ta_ext, state.Ap_layout, state.ma_prefetch, state.ka_prefetch, remM_A,  remK_A,  false, AvoidFragment, 0, 0, problem.A, strategy.A_prefetch)) return false;
    if (strategy.prefetchB && !getRegLayout(Tb_ext, state.Bp_layout, state.kb_prefetch, state.nb_prefetch, remK_B,  remN_B,  false, AvoidFragment, 0, 0, problem.B, strategy.B_prefetch)) return false;
    if (strategy.prefetchC && !getRegLayout(Tc_ext, state.Cp_layout, unrollM,           unrollN,           remM_Cp, remN_Cp, false, AvoidFragment, 0, 0, problem.C, strategy.C_prefetch)) return false;

    if (hasMasking(state.Cp_layout) || hasFragmenting(state.Cp_layout))
        stub();

    gemmABPrefetchAddrSetup(problem, strategy, state);

    // Prepare layouts and starting addresses for SLM copies and adjust problem.
    if (strategy.slmBuffers > 0) {
        int A_slmCP, B_slmCP;
        int A_tileR, A_tileC, B_tileR, B_tileC;
        std::tie(A_slmCP, B_slmCP) = targetSLMCrosspack(hw, problem, strategy);
        std::tie(A_tileR, A_tileC, B_tileR, B_tileC) = targetKernelTiling(hw, problem, strategy);
        auto opCount = outerProductCount(hw, problem, strategy);

        if (slmA) {
            coopSplit(true, state.ma_slm, state.ka_slm, unrollM, unrollKSLM, strategy.wgTile(LoopM), state.effCoopA, problem.A, strategy);

            if (state.effCoopA == CoopSplit::MN)
                remK_A = remainderK && strategy.slmEarlyKMask;

            if (strategy.slmATrans) {
                A_slmCP = state.ka_slm;
                if (strategy.ka_load % A_slmCP)
                    stub("ka_load must be a multiple of ka_slm");
            }
            if ((state.ka_slm < A_slmCP) && (unrollKSLM != A_slmCP) && (A_tileC != A_slmCP))
                stub("ka_slm must be a multiple of crosspack, or unrollKSLM = crosspack.");
            if (isPacked(problem.A.layout) && problem.A.packSize != unrollM)
                stub("A panel height must match unroll");

            // Layout in from memory...
            state.Ai = problem.A;
            state.Ai_strategy = strategy.A;
            if (state.Ai_strategy.dpasw) {
                state.Ai_strategy.dpasw = false;
                state.Ai_strategy.tileR = 0;
            }

            // ... layout out to SLM.
            state.Ao.layout = MatrixLayout::Pc;
            state.Ao.packSize = unrollM;
            state.Ao.panelLength = unrollKSLM;
            state.Ao.crosspack = A_slmCP;
            state.Ao.setAlignment(state.Ao.packSize * Ta);
            state.Ao.tileR = A_tileR;
            state.Ao.tileC = (A_tileC || !A_tileR) ? A_tileC : std::max(opCount, strategy.ka_load);

            bool colMajorIn = isRegisterColMajor(Ta_ext, state.Ai, state.Ai_strategy);
            bool colMajorSLM = !isLargeCrosspack(Ta, A_slmCP);
            state.Ao_strategy.base = SLM;
            state.Ao_strategy.accessType = (colMajorIn == colMajorSLM) ? AccessType::Block
                                                                       : AccessType::Scattered;
            state.Ao_strategy.smode = ScatterSIMD::Default;

            if (state.Ai.layout == MatrixLayout::N && one_of(state.Ai_strategy.accessType, AccessType::Block2D, AccessType::Block2DVNNI) && isLargeCrosspack(Ta, A_slmCP)) {
                state.Ao_strategy.accessType = AccessType::ChannelScattered;
                state.Ao_strategy.smode = ScatterSIMD::Narrow;
            }
            state.Ao_strategy.padded = true;
            state.Ao_strategy.atomic = false;
            state.Ao_strategy.address2D = false;
            state.Ao_strategy.newDP = (hw >= HW::XeHPG);
            state.Ao_strategy.cachingW = CacheSettingsLSC::Default;

            // Layout in from memory...
            if (!getRegLayout(Ta_ext, state.Ai_layout, state.ma_slm, state.ka_slm, remM_A, remK_A, false, AvoidFragment, 0, 0, state.Ai, state.Ai_strategy)) return false;

            if (hasFragmenting(state.Ai_layout, false, true)) {
                status << "Can't fragment in m dimension." << status_stream::endl;
                return false;
            }

            // ... layout out to SLM...
            remM_A = remK_A = false;
            if (!getRegLayout(Ta, state.Ao_layout, state.ma_slm, state.ka_slm, remM_A, remK_A, true, AvoidFragment, 0, 0, state.Ao, state.Ao_strategy)) return false;

            // ... and layout back from SLM.
            problem.A = state.Ao;
            strategy.A.base = SLM;
            strategy.A.accessType = AccessType::Block;
            strategy.A.address2D = false;
            strategy.A.newDP = (hw >= HW::XeHPG);
            strategy.A.cachingR = CacheSettingsLSC::Default;
            Ta_load = Ta;
            state.aioShare = Ta.bits() == Ta_ext.bits()
                          && Ta.components() == Ta_ext.components()
                          && matchLayoutsBidirectional(Ta, state.Ai_layout, state.Ao_layout);

            // If we will add k-masking later, check if extra registers are needed.
            state.Ai_regCount = getRegCount(state.Ai_layout);
            if (!remK_A && remainderK && !state.Ai_strategy.address2D && !isRegisterColMajor(Ta_ext, state.Ai, state.Ai_strategy)) {
                std::vector<RegisterBlock> Ai_layoutKMasked;
                if (getRegLayout(Ta_ext, Ai_layoutKMasked, state.ma_slm, state.ka_slm, remM_A, true, false, AvoidFragment, 0, 0, state.Ai, state.Ai_strategy))
                    state.Ai_regCount = std::max(state.Ai_regCount, getRegCount(Ai_layoutKMasked));
            }

            // Offset A addresses in and out.
            state.effAi = state.effA;
            state.effA = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
            state.effAo = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));

            auto temp = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp0, strategy));
            Subregister temp2;

            uint32_t noff, noffTile, tileSplit = 1;

            switch (state.effCoopA) {
                case CoopSplit::Linear:
                    // FIXME: assumes compatible tiling between global and SLM layouts.
                    noff = state.ma_slm * state.ka_slm;
                    break;
                case CoopSplit::MN:
                    noff = untile(Ta, state.Ao, 0, state.ma_slm, 0, state.Ao.packSize, unrollKSLM);
                    if (state.ma_slm < state.Ao.tileR && state.Ao.tileR < state.Ao.packSize) {
                        // m division splits tiles -- starting offsets no longer a linear sequence.
                        if (state.Ao.tileR % state.ma_slm) stub();
                        tileSplit = state.Ao.tileR / state.ma_slm;
                        noffTile = untile(Ta, state.Ao, 0, state.Ao.tileR, 0, state.Ao.packSize, unrollKSLM);
                    }
                    break;
                case CoopSplit::K:
                    noff = untile(Ta, state.Ao, 0, 0, state.ka_slm, state.Ao.packSize, unrollKSLM);
                    if (state.ka_slm < state.Ao.tileC && state.Ao.tileC < unrollKSLM) {
                        // k division splits tiles -- starting offsets no longer a linear sequence.
                        if (state.Ao.tileC % state.ka_slm) stub();
                        tileSplit = state.Ao.tileC / state.ka_slm;
                        noffTile = untile(Ta, state.Ao, 0, 0, state.Ao.tileC, state.Ao.packSize, unrollKSLM);
                    }
                    break;
                default: stub();
            }

            int32_t A_slmStride = strategy.slmABufBlockSize(problem) * strategy.slmBuffers;

            if (tileSplit > 1) {
                if (!is_zero_or_pow2(tileSplit)) stub();
                shr(1, temp, state.lidN, ilog2(tileSplit));
            }
            gemmCalcWorkshareAOffset(temp2, Ai_params.offR, Ai_params.offC, state.Ai, state.Ai_strategy, state.ma_slm, state.ka_slm, problem, strategy, state);
            if (tileSplit > 1) {
                mulConstant(1, temp, temp, (noffTile - noff * tileSplit) * Ta);
                emad(1, temp, temp, state.lidN, noff * Ta, strategy, state);
            } else
                mulConstant(1, temp, state.lidN, noff * Ta);
            mulConstant(1, state.effA, state.lidM, A_slmStride);
            if (strategy.wg[LoopK] > 1)
                emad(1, state.effA, state.effA, state.lidK, A_slmStride * strategy.wg[LoopM], strategy, state);
            if (state.Ai_strategy.address2D) {
                if (Ai_params.offR != A_params.offR && A_params.offR.isValid())
                    add(1, Ai_params.offR, Ai_params.offR, A_params.offR);
                if (Ai_params.offC != A_params.offC && A_params.offC.isValid())
                    add(1, Ai_params.offC, Ai_params.offC, A_params.offC);
            } else
                eadd(1, state.effAi, state.effAi, temp2, strategy, state);
            makeSLMBaseRelative(state.effA, state);
            add(1, state.effAo, state.effA, temp);
            if (problem.backward())
                add(1, state.effA, state.effA, (unrollKSLM - strategy.ka_load) * unrollM * Ta);

            state.ra.safeRelease(temp2);
            state.ra.safeRelease(temp);
        }
        if (slmB) {
            coopSplit(false, state.kb_slm, state.nb_slm, unrollKSLM, unrollN, strategy.wgTile(LoopN), state.effCoopB, problem.B, strategy);

            if (state.effCoopB == CoopSplit::MN)
                remK_B = remainderK && strategy.slmEarlyKMask;

            if (strategy.slmBTrans) {
                B_slmCP = state.kb_slm;
                if (strategy.kb_load % B_slmCP)
                    stub("kb_load must be a multiple of kb_slm");
            }
            if ((state.kb_slm < B_slmCP) && (unrollKSLM != B_slmCP) && (B_tileR != B_slmCP))
                stub("kb_slm must be a multiple of crosspack, or unrollKSLM = crosspack.");
            if (isPacked(problem.B.layout) && problem.B.packSize != unrollN)
                stub("B panel height must match unroll");

            // Layout in from memory...
            state.Bi = problem.B;
            state.Bi_strategy = strategy.B;
            if (state.Bi_strategy.dpasw) {
                state.Bi_strategy.dpasw = false;
                state.Bi_strategy.tileC = 0;
            }

            // ... layout out to SLM.
            state.Bo.layout = MatrixLayout::Pr;
            state.Bo.packSize = unrollN;
            state.Bo.panelLength = unrollKSLM;
            state.Bo.crosspack = B_slmCP;
            state.Bo.setAlignment(state.Bo.packSize * Tb);
            state.Bo.tileR = (B_tileR || !B_tileC) ? B_tileR : std::max(opCount, strategy.kb_load);
            state.Bo.tileC = B_tileC;

            bool colMajorIn = isRegisterColMajor(Tb_ext, state.Bi, state.Bi_strategy);
            bool colMajorSLM = isLargeCrosspack(Tb, B_slmCP);
            state.Bo_strategy.base = SLM;
            state.Bo_strategy.accessType = (colMajorIn == colMajorSLM) ? AccessType::Block
                                                                       : AccessType::Scattered;
            state.Bo_strategy.smode = ScatterSIMD::Default;

            if (state.Bi.layout == MatrixLayout::T && one_of(state.Bi_strategy.accessType, AccessType::Block2D, AccessType::Block2DVNNI) && isLargeCrosspack(Tb, B_slmCP)) {
                state.Bo_strategy.accessType = AccessType::ChannelScattered;
                state.Bo_strategy.smode = ScatterSIMD::Narrow;
            }
            state.Bo_strategy.padded = true;
            state.Bo_strategy.atomic = false;
            state.Bo_strategy.address2D = false;
            state.Bo_strategy.newDP = (hw >= HW::XeHPG);
            state.Bo_strategy.cachingW = CacheSettingsLSC::Default;

            // Layout in from memory...
            if (!getRegLayout(Tb_ext, state.Bi_layout, state.kb_slm, state.nb_slm, remK_B, remN_B, false, AvoidFragment, 0, 0, state.Bi, state.Bi_strategy)) return false;

            if (hasFragmenting(state.Bi_layout, true, false)) {
                status << "Can't fragment in n dimension." << status_stream::endl;
                return false;
            }

            // ... layout out to SLM...
            remK_B = remN_B = false;
            if (!getRegLayout(Tb, state.Bo_layout, state.kb_slm, state.nb_slm, remK_B, remN_B, true, AvoidFragment, 0, 0, state.Bo, state.Bo_strategy)) return false;

            // ... and layout back from SLM.
            problem.B = state.Bo;
            strategy.B.base = SLM;
            strategy.B.accessType = AccessType::Block;
            strategy.B.address2D = false;
            strategy.B.newDP = (hw >= HW::XeHPG);
            strategy.B.cachingR = CacheSettingsLSC::Default;
            Tb_load = Tb;
            state.bioShare = Tb.bits() == Tb_ext.bits()
                          && Tb.components() == Tb_ext.components()
                          && matchLayoutsBidirectional(Tb, state.Bi_layout, state.Bo_layout);

            // If we will add k-masking later, check if extra registers are needed.
            state.Bi_regCount = getRegCount(state.Bi_layout);
            if (!remK_B && remainderK && !state.Bi_strategy.address2D && isRegisterColMajor(Tb_ext, state.Bi, state.Bi_strategy)) {
                std::vector<RegisterBlock> Bi_layoutKMasked;
                if (getRegLayout(Tb_ext, Bi_layoutKMasked, state.kb_slm, state.nb_slm, true, remN_B, false, AvoidFragment, 0, 0, state.Bi, state.Bi_strategy))
                    state.Bi_regCount = std::max(state.Bi_regCount, getRegCount(Bi_layoutKMasked));
            }

            // Offset B addresses in and out.
            state.effBi = state.effB;
            state.effB = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
            state.effBo = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));

            auto temp = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp0, strategy));
            Subregister temp2;

            uint32_t moff, moffTile, tileSplit = 1;

            switch (state.effCoopB) {
                case CoopSplit::Linear:
                    moff = state.kb_slm * state.nb_slm;
                    break;
                case CoopSplit::MN:
                    moff = untile(Tb, state.Bo, 0, 0, state.nb_slm, unrollKSLM, state.Bo.packSize);
                    if (state.nb_slm < state.Bo.tileC && state.Bo.tileC < state.Bo.packSize) {
                        if (state.Bo.tileC % state.nb_slm) stub();
                        tileSplit = state.Bo.tileC / state.nb_slm;
                        moffTile = untile(Tb, state.Bo, 0, 0, state.Bo.tileC, unrollKSLM, state.Bo.packSize);
                    }
                    break;
                case CoopSplit::K:
                    moff = untile(Tb, state.Bo, 0, state.kb_slm, 0, unrollKSLM, state.Bo.packSize);
                    if (state.kb_slm < state.Bo.tileR) {
                        if (state.Bo.tileR % state.kb_slm) stub();
                        tileSplit = state.Bo.tileR / state.kb_slm;
                        moffTile = untile(Tb, state.Bo, 0, state.Bo.tileR, 0, unrollKSLM, state.Bo.packSize);
                    }
                    break;
                default: stub();
            }

            int32_t B_slmStride = strategy.slmBBufBlockSize(problem) * strategy.slmBuffers;

            if (tileSplit > 1) {
                if (!is_zero_or_pow2(tileSplit)) stub();
                shr(1, temp, state.lidM, ilog2(tileSplit));
            }
            gemmCalcWorkshareBOffset(temp2, Bi_params.offR, Bi_params.offC, state.Bi, state.Bi_strategy, state.kb_slm, state.nb_slm, problem, strategy, state);
            if (tileSplit > 1) {
                mulConstant(1, temp, temp, (moffTile - moff * tileSplit) * Tb);
                emad(1, temp, temp, state.lidM, moff * Tb, strategy, state);
            } else
                mulConstant(1, temp, state.lidM, moff * Tb);
            mulConstant(1, state.effB, state.lidN, B_slmStride);
            if (strategy.wg[LoopK] > 1)
                emad(1, state.effB, state.effB, state.lidK, B_slmStride * strategy.wg[LoopN], strategy, state);
            if (state.Bi_strategy.address2D) {
                if (Bi_params.offR != B_params.offR && B_params.offR.isValid())
                    add(1, Bi_params.offR, Bi_params.offR, B_params.offR);
                if (Bi_params.offC != B_params.offC && B_params.offC.isValid())
                    add(1, Bi_params.offC, Bi_params.offC, B_params.offC);
            } else
                eadd(1, state.effBi, state.effBi, temp2, strategy, state);
            if (strategy.slmABufSize(problem) > 0) {
                if (strategy.kParallelLocal) {
                    int32_t A_perKSLM = strategy.slmABufBlockSize(problem) * strategy.wg[LoopM] * strategy.slmBuffers;
                    emad(1, state.effB, state.effB, state.lszK, A_perKSLM, strategy, state);
                } else
                    add(1, state.effB, state.effB, strategy.slmABufSize(problem));
            }
            makeSLMBaseRelative(state.effB, state);
            add(1, state.effBo, state.effB, temp);
            if (problem.backward())
                add(1, state.effB, state.effB, (unrollKSLM - strategy.kb_load) * unrollN * Tb);

            state.ra.safeRelease(temp2);
            state.ra.safeRelease(temp);
        }
    }

    // Starting address and remainder adjustments for DPASW.
    bool cColMajor = isRegisterColMajor(Tc_ext, problem.C, strategy.C);
    Subregister saveRemM = state.remainders[LoopM];
    Subregister saveRemN = state.remainders[LoopN];
    if (strategy.dpasw) {
        if (cColMajor) {
            int t = strategy.B.tileC;
            and_(1 | nz | state.flagAP, null.uw(), state.lidM, 1);
            switch (problem.B.layout) {
                case MatrixLayout::N:  emad(1 | state.flagAP, state.effB, state.effB, state.inputs.ldb, Immediate::w(t), strategy, state);                               break;
                case MatrixLayout::T:  eadd(1 | state.flagAP, state.effB, state.effB, t * Tb_load, strategy, state);                                                     break;
                case MatrixLayout::Pr: eadd(1 | state.flagAP, state.effB, state.effB, untile(Tb_load, problem.B, 0, 0, t, unrollK, unrollN) * Tb_load, strategy, state); break;
                default: stub();
            }
            if (!slmB && remN_B) {
                state.remainders[LoopN] = state.ra.alloc_sub<int16_t>();
                mov(1, state.remainders[LoopN], saveRemN);
                add(1 | state.flagAP, state.remainders[LoopN], saveRemN, -t);
            }
        } else {
            int t = strategy.A.tileR;
            and_(1 | nz | state.flagAP, null.uw(), state.lidN, 1);
            switch (problem.A.layout) {
                case MatrixLayout::T:  emad(1 | state.flagAP, state.effA, state.effA, state.inputs.lda, Immediate::w(t), strategy, state);                               break;
                case MatrixLayout::N:  eadd(1 | state.flagAP, state.effA, state.effA, t * Ta_load, strategy, state);                                                     break;
                case MatrixLayout::Pc: eadd(1 | state.flagAP, state.effA, state.effA, untile(Ta_load, problem.A, 0, t, 0, unrollM, unrollK) * Ta_load, strategy, state); break;
                default: stub();
            }
            if (!slmA && remM_A) {
                state.remainders[LoopM] = state.ra.alloc_sub<int16_t>();
                mov(1, state.remainders[LoopM], saveRemM);
                add(1 | state.flagAP, state.remainders[LoopM], saveRemM, -t);
            }
        }
    }

    // Get register layouts for A/B/C.
    if (!getRegLayout(Ta_load, state.A_layout, unrollM,          strategy.ka_load, remM_A, remK_A, false, AvoidFragment, 0, 0, problem.A, strategy.A)) return false;
    if (!getRegLayout(Tb_load, state.B_layout, strategy.kb_load, unrollN,          remK_B, remN_B, false, AvoidFragment, 0, 0, problem.B, strategy.B)) return false;

    if (state.copyC) {
        if (state.useTempC) {
            if (!getRegLayout(Tc, state.C_layout, unrollM, unrollN, false, false, true, AvoidFragment, 0, 0, state.tempC, state.tempCStrategy)) return false;
        } else {
            makeUnbackedRegLayout(Tc, state.C_layout, unrollM, unrollN, cColMajor, 1, strategy.C.tileR, strategy.C.tileC, true);
        }
        if (!getRegLayout(Tc_ext, state.C_layoutExt, unrollM, unrollN, remM_Ce, remN_Ce, true, AllowFragDescNFM, 0, 0, problem.C, state.Cext_strategy)) return false;
    } else {
        if (!getRegLayout(Tc, state.C_layout, unrollM, unrollN, remM_C, remN_C, true, AllowFragDescNFM, 0, 0, problem.C, strategy.C)) return false;
    }

    auto &layoutExt = state.copyC ? state.C_layoutExt : state.C_layout;
    if (!strategy.altCRemainder && (remM_Ce || remN_Ce)) {
        // Try preparing C layout without masking (may reduce memory accesses).
        // Only use it if compatible with the masked layout, and saves on send instructions (or avoids pseudoblock/slow scattered byte accesses).
        (void) getRegLayout(Tc_ext, state.C_layoutExtUnmasked, unrollM, unrollN, false, false, true, AllowFragDescNFM, 0, 0, problem.C, state.Cext_strategy);

        bool useUnmasked = true;
        if (!state.copyC)
            useUnmasked &= matchLayouts(Tc, layoutExt, state.C_layoutExtUnmasked);
        if (state.C_layoutExtUnmasked.size() == layoutExt.size())
            useUnmasked &= (Tc_ext.size() < 4) || (needsPseudoblock(hw, Tc_ext, unrollM, unrollN, problem.C, state.Cext_strategy, true, false)
                                                != needsPseudoblock(hw, Tc_ext, unrollM, unrollN, problem.C, state.Cext_strategy, true, true));
        if (!useUnmasked)
            state.C_layoutExtUnmasked.clear();
    }

    if (state.Cext_strategy.atomic && (!problem.beta1() || strategy.fuseBeta)) {
        auto nonatomicCExt = state.Cext_strategy;
        nonatomicCExt.atomic = false;
        (void) getRegLayout(Tc_ext, state.C_layoutExtNonatomicUnmasked, unrollM, unrollN, false, false, true, AllowFragDescNFM, 0, 0, problem.C, nonatomicCExt);
        bool ok = true;
        if (!state.C_layoutExtUnmasked.empty()) {
            ok = matchLayouts(Tc, state.C_layoutExtUnmasked, state.C_layoutExtNonatomicUnmasked);
            if (ok && !matchLayouts(Tc, layoutExt, state.C_layoutExtNonatomicUnmasked)) stub();
        } else
            ok = matchLayouts(Tc, layoutExt, state.C_layoutExtNonatomicUnmasked);

        if (!ok) state.C_layoutExtNonatomicUnmasked.clear();
    }

    if (!state.copyC)
        state.C_layoutExt = state.C_layout;

    if (strategy.dotVL) {
        int mx = std::max(1, strategy.dotVL *  cColMajor);
        int nx = std::max(1, strategy.dotVL * !cColMajor);
        makeUnbackedRegLayout(Tc, state.C_layoutReduced, unrollM * mx, unrollN * nx, cColMajor, 1, 0, 0, true);
        std::swap(state.C_layout, state.C_layoutReduced);
    }

    if (hasFragmenting(state.A_layout, false, true) || hasFragmenting(state.B_layout, true, false)) {
        status << "Can't fragment in m/n dimensions." << status_stream::endl;
        return false;
    }

    bool globalCM = isLayoutColMajor(state.C_layout);

    // Prepare to repack A/B if needed.
    int crosspackA, crosspackB, tileM_A, tileK_A, tileK_B, tileN_B;
    std::tie(crosspackA, crosspackB) = targetKernelCrosspack(hw, problem, strategy);
    std::tie(tileM_A, tileK_A, tileK_B, tileN_B) = targetKernelTiling(hw, problem, strategy);

    state.repackA |= (crosspackA && !hasFullCrosspack(state.A_layout, crosspackA))
                        || !hasTiling(state.A_layout, tileM_A, tileK_A);
    state.repackB |= (crosspackB && !hasFullCrosspack(state.B_layout, crosspackB))
                        || !hasTiling(state.B_layout, tileK_B, tileN_B);

    state.repackA |= (Ta.bits() != Ta_ext.bits() || Ta.components() != Ta_ext.components()) && !slmA;
    state.repackB |= (Tb.bits() != Tb_ext.bits() || Tb.components() != Tb_ext.components()) && !slmB;

    if (crosspackA == 0) crosspackA = 1;
    if (crosspackB == 0) crosspackB = 1;

    bool splitA = false, splitB = false;

    state.ka_repack = strategy.ka_load;
    if (state.repackA) {
        // Repacked data can use significantly more registers than the loaded
        // data. Lazy repacking can reduce register utilization and improve load
        // pipelining at (in some cases) the expense of more work.
        bool lazyRepack = state.Ta_load.isInt4() && one_of(Ta, Type::f16, Type::bf16, Type::f32);    // Other cases are unimplemented
        if (lazyRepack)
            state.ka_repack = std::min(state.ka_repack, strategy.kb_load);
        makeUnbackedRegLayout(Ta, state.Ar_layout, unrollM, state.ka_repack, isLayoutColMajor(state.A_layout), crosspackA, tileM_A, tileK_A, true, splitA);
    }

    if (state.repackB) makeUnbackedRegLayout(Tb, state.Br_layout, strategy.kb_load, unrollN, isLayoutColMajor(state.B_layout), crosspackB, tileK_B, tileN_B, true, splitB);

    // Prepare to repack C if needed, and choose repack tile size.
    if (Tc != Tc_compute) {
        int panel = strategy.cRepackPanel;
        if (panel == 0)
            panel = 2 * elementsPerGRF(hw, Tc_compute);

        int Cr_unrollM = unrollM, Cr_unrollN = unrollN;
        auto &Cr_unrollX = globalCM ? Cr_unrollM : Cr_unrollN;
        Cr_unrollX = std::min(Cr_unrollX, panel);

        makeUnbackedRegLayout(Tc_compute, state.Cr_layout, Cr_unrollM, Cr_unrollN, globalCM, 1, strategy.C.tileR, strategy.C.tileC, true);
    }

    // Prepare layouts for row/column sum calculation.
    if (problem.needsASums()) {
        state.systolicSumA = strategy.systolic && globalCM;
        state.slmASums = slmA && !state.systolicSumA;

        if (!state.slmASums && !globalCM && strategy.dpasw) stub();  /* don't have full A data */

        auto As_srcLayout = state.slmASums ? state.Ao_layout :
                             state.repackA ? state.Ar_layout :
                                             state.A_layout;
        makeSumLayout(false, Ta, As_srcLayout, Tc, state.As_layout, strategy, state);
        if (state.systolicSumA)
            setupTeardownAccumulateSumSystolic(true, Tb, problem, strategy, state);
    }
    if (problem.needsBSums()) {
        state.systolicSumB = strategy.systolic && !globalCM;
        state.slmBSums = slmB && !state.systolicSumB;

        if (!state.slmBSums && globalCM && strategy.dpasw) stub();

        auto Bs_srcLayout = state.slmBSums ? state.Bo_layout :
                             state.repackB ? state.Br_layout :
                                             state.B_layout;
        makeSumLayout(true,  Tb, Bs_srcLayout, Tc, state.Bs_layout, strategy, state);
        if (state.systolicSumB)
            setupTeardownAccumulateSumSystolic(true, Ta, problem, strategy, state);
    }

    // Prepare strategies and layouts for 2D A/B grouped quantization parameters (offsets and scales).
    bool as2D = problem.aScale2D;
    bool bs2D = problem.bScale2D;
    bool ao2D = (problem.aoPtrDims == 2);
    bool bo2D = (problem.boPtrDims == 2);
    bool aoTo2D = problem.aOffset == ABOffset::Calc && !ao2D && problem.earlyDequantizeA();
    bool boTo2D = problem.bOffset == ABOffset::Calc && !bo2D && problem.earlyDequantizeB();

    for (bool isA : {true, false})
        gemmMake2DQuantizationLayouts(isA, problem, strategy, state);

    gemmCalcQuantizationIncrements(problem, strategy, state);

    // Grab flag registers now for named barriers. TODO: unlock these.
    if (strategy.needsNamedBarriersM(problem))
        state.barrierM = state.raVFlag.allocSubreg0();
    if (strategy.needsNamedBarriersN(problem))
        state.barrierN = state.raVFlag.allocSubreg0();

    // Round up needed A/B flag registers; hold off on C.
    // Try first without virtual flags and retry if needed.
    // m/n cooperative SLM copies may use k masking; skip those masks for now.
    auto &masks     = state.AB_masks;
    auto &masksCoop = state.AB_masksCoop;
    auto &A_cmasks = (state.effCoopA == CoopSplit::K) ? masks : masksCoop;
    auto &B_cmasks = (state.effCoopB == CoopSplit::K) ? masks : masksCoop;

    auto assignAllMasks = [&]() {
        return assignMasks(state.A_layout,       LoopM,    LoopK,       masks, strategy, state)
            && assignMasks(state.A_offsetLayout, LoopM,    LoopK,       masks, strategy, state)
            && assignMasks(state.A_scaleLayout,  LoopM,    LoopK,       masks, strategy, state)
            && assignMasks(state.Ap_layout,      LoopM,    LoopK,    A_cmasks, strategy, state)
            && assignMasks(state.Ai_layout,      LoopM,    LoopNone, A_cmasks, strategy, state)
            && assignMasks(state.B_layout,       LoopK,    LoopN,       masks, strategy, state)
            && assignMasks(state.B_offsetLayout, LoopK,    LoopN,       masks, strategy, state)
            && assignMasks(state.B_scaleLayout,  LoopK,    LoopN,       masks, strategy, state)
            && assignMasks(state.Bp_layout,      LoopK,    LoopN,    B_cmasks, strategy, state)
            && assignMasks(state.Bi_layout,      LoopNone, LoopN,    B_cmasks, strategy, state)
        ;
    };

    state.lateKLoopCheck = false;
    bool success = assignAllMasks();
    if (!success && !state.vflagsEnabled()) {
        status << "Retrying with virtual flags." << status_stream::endl;
        allocVFlagStorage(strategy, state, false);
        success = assignAllMasks();
        state.lateKLoopCheck = true;
    }

    if (!success) return false;

    loadMasks(masks,        state.remainders,        strategy, state);
    loadMasks(masksCoop,    state.remaindersCoop,    strategy, state);

    state.remainders[LoopM] = saveRemM;
    state.remainders[LoopN] = saveRemN;

    if (!state.simd32KMasks)
        releaseCoopRemainders(state);     /* may need SLM m/n remainders for k masking later */

    // Apply panel masks, if defined, to all A/B blocks.
    if (state.panelMaskA.isValid()) {
        assignUniformMask(slmA ? state.Ai_layout : state.A_layout, state.panelMaskA);
        assignUniformMask(state.Ap_layout, state.panelMaskA);
    }
    if (state.panelMaskB.isValid()) {
        assignUniformMask(slmB ? state.Bi_layout : state.B_layout, state.panelMaskB);
        assignUniformMask(state.Bp_layout, state.panelMaskB);
    }

    // Temporary: move add64 out of the way (later: general cramming).
    if (state.add64.isValid()) {
        auto oldAdd64 = state.add64;
        state.ra.safeRelease(state.add64);
        state.add64 = state.ra.alloc_sub<uint32_t>();
        if (oldAdd64 != state.add64)
            mov(1, state.add64, oldAdd64);
    }

    // Allocate data registers.
    gemmAllocRegs(problem, strategy, state);
    gemmAllocAoBoRegs(strategy, state);

    // Allocate address registers for A/B loads. We don't need C addresses yet.
    allocAddrRegs(state.A_addrs, state.A_layout, problem.A, strategy.A, state);
    allocAddrRegs(state.B_addrs, state.B_layout, problem.B, strategy.B, state);
    allocAddrRegs(state.Ap_addrs, state.Ap_layout, globalA, strategy.A_prefetch, state);
    allocAddrRegs(state.Bp_addrs, state.Bp_layout, globalB, strategy.B_prefetch, state);
    allocAddrRegs(state.Ai_addrs, state.Ai_layout, state.Ai, state.Ai_strategy, state);
    allocAddrRegs(state.Bi_addrs, state.Bi_layout, state.Bi, state.Bi_strategy, state);
    allocAddrRegs(state.Ao_addrs, state.Ao_layout, state.Ao, state.Ao_strategy, state);
    allocAddrRegs(state.Bo_addrs, state.Bo_layout, state.Bo, state.Bo_strategy, state);
    allocAddrRegs(state.A_offsetAddrs, state.A_offsetLayout, problem.AO, strategy.AO, state);
    allocAddrRegs(state.B_offsetAddrs, state.B_offsetLayout, problem.BO, strategy.BO, state);
    allocAddrRegs(state.A_scaleAddrs, state.A_scaleLayout, problem.A_scale, strategy.A_scale, state);
    allocAddrRegs(state.B_scaleAddrs, state.B_scaleLayout, problem.B_scale, strategy.B_scale, state);

    // Free up some C registers temporarily for use in address calculations.
    releaseRanges(state.C_regs, state);

    // Set up address registers.
    gemmCacheLDABMultiples(problem, strategy, state);
    setupAddr(Ta_ext,  state.Ap_addrs, state.effAp, state.Ap_layout, state.inputs.lda, globalA,   strategy.A_prefetch, strategy, state, Ap_params, state.ldaMultiples);
    setupAddr(Tb_ext,  state.Bp_addrs, state.effBp, state.Bp_layout, state.inputs.ldb, globalB,   strategy.B_prefetch, strategy, state, Bp_params, state.ldbMultiples);
    setupAddr(Ta_ext,  state.Ai_addrs, state.effAi, state.Ai_layout, state.inputs.lda, state.Ai,  state.Ai_strategy,   strategy, state, Ai_params, state.ldaMultiples);
    setupAddr(Tb_ext,  state.Bi_addrs, state.effBi, state.Bi_layout, state.inputs.ldb, state.Bi,  state.Bi_strategy,   strategy, state, Bi_params, state.ldbMultiples);
    setupAddr(Ta,      state.Ao_addrs, state.effAo, state.Ao_layout, Subregister(),    state.Ao,  state.Ao_strategy,   strategy, state);
    setupAddr(Tb,      state.Bo_addrs, state.effBo, state.Bo_layout, Subregister(),    state.Bo,  state.Bo_strategy,   strategy, state);
    setupAddr(Ta_load, state.A_addrs,  state.effA,  state.A_layout,  state.inputs.lda, problem.A, strategy.A,          strategy, state, A_params,  state.ldaMultiples);
    setupAddr(Tb_load, state.B_addrs,  state.effB,  state.B_layout,  state.inputs.ldb, problem.B, strategy.B,          strategy, state, B_params,  state.ldbMultiples);

    // 2D quantization address setup.
    auto i0q = state.i0, j0q = state.j0;
    Subregister A_h0q, B_h0q;

    if (state.h0.isValid()) {
        if (ao2D || as2D) {
            A_h0q = state.ra.alloc_sub<uint32_t>();
            divDown(A_h0q, state.h0, problem.aqGroupK, strategy, state);
        }
        if (bo2D || bs2D) {
            B_h0q = state.ra.alloc_sub<uint32_t>();
            divDown(B_h0q, state.h0, problem.bqGroupK, strategy, state);
        }
    }

    auto i0s = i0q, j0s = j0q;
    auto A_h0s = A_h0q, B_h0s = B_h0q;

    if (slmA && (ao2D || aoTo2D || (as2D && !state.lateScale2DA))) {
        if (state.ma_slm < unrollM) {
            if (state.ma_slm * strategy.wg[LoopN] != unrollM) stub();
            i0q = state.ra.alloc_sub<uint32_t>();
            emad(1, i0q, state.i0, state.lidN, state.ma_slm, strategy, state);
        }
        if ((ao2D || (as2D && !state.lateScale2DA)) && state.ka_slm < strategy.unrollKSLM && problem.aqGroupK < strategy.unrollKSLM) {
            if (state.lateScale2DA)
                A_h0s = copySubregister(A_h0q, state);
            if (A_h0q.isInvalid()) {
                A_h0q = state.ra.alloc_sub<uint32_t>();
                mov(1, A_h0q, 0);
            }
            addScaled(1, A_h0q, A_h0q, state.lidN, state.ka_slm, problem.aqGroupK, state, true);
        }
    }
    if (slmB && (bo2D || boTo2D || (bs2D && !state.lateScale2DB))) {
        if (state.nb_slm < unrollN) {
            if (state.nb_slm * strategy.wg[LoopM] != unrollN) stub();
            j0q = state.ra.alloc_sub<uint32_t>();
            emad(1, j0q, state.j0, state.lidM, state.nb_slm, strategy, state);
        }
        if ((bo2D || (bs2D && !state.lateScale2DB)) && state.kb_slm < strategy.unrollKSLM && problem.bqGroupK < strategy.unrollKSLM) {
            if (state.lateScale2DB)
                B_h0s = copySubregister(B_h0q, state);
            if (B_h0q.isInvalid()) {
                B_h0q = state.ra.alloc_sub<uint32_t>();
                mov(1, B_h0q, 0);
            }
            addScaled(1, B_h0q, B_h0q, state.lidM, state.kb_slm, problem.bqGroupK, state, true);
        }
    }

    if (problem.aqGroupM > 1 && (ao2D || as2D)) {
        auto inI0Q = i0q, inI0S = i0s;
        if (i0q == state.i0) i0q = state.ra.alloc_sub<uint32_t>();
        divDown(i0q, inI0Q, problem.aqGroupM, strategy, state);
        if (inI0S == inI0Q)
            i0s = i0q;
        else
            divDown(i0s, i0s, problem.aqGroupM, strategy, state);
    }

    if (problem.bqGroupN > 1 && (bo2D || bs2D)) {
        auto inJ0Q = j0q, inJ0S = j0s;
        if (j0q == state.j0) j0q = state.ra.alloc_sub<uint32_t>();
        divDown(j0q, inJ0Q, problem.bqGroupN, strategy, state);
        if (inJ0S == inJ0Q)
            j0s = j0q;
        else
            divDown(j0s, j0s, problem.bqGroupN, strategy, state);
    }

    auto setupQAddr = [&](Type T, vector<GRFRange> &addrs, const vector<RegisterBlock> &layout,
                          Subregister ptr, Subregister r0, Subregister c0, Subregister ld,
                          const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy)
    {
        auto base = state.ra.alloc_sub(ptr.getType());
        if (!isColMajor(atype.layout)) std::swap(r0, c0);
        if (r0.isValid()) eaddScaled(1, base, ptr, r0, T, strategy, state);
        if (c0.isValid()) emad(1, base, r0.isValid() ? base : ptr, c0, ld, strategy, state);
        if (r0.isInvalid() && c0.isInvalid()) emov(1, base, ptr, strategy, state);
        setupAddr(T, addrs, base, layout, ld, atype, astrategy, strategy, state);
        state.ra.safeRelease(base);
    };

    if (ao2D) {
        setupQAddr(Tao, state.A_offsetAddrs, state.A_offsetLayout, state.inputs.aoPtr,
                   i0q, A_h0q, state.inputs.ldao, problem.AO, strategy.AO);
    }
    if (as2D) {
        if (!state.lateScale2DA)
            i0s = i0q, A_h0s = A_h0q;
        setupQAddr(Ta_scale, state.A_scaleAddrs, state.A_scaleLayout, state.inputs.aScalePtr,
                   i0s, A_h0s, state.inputs.ldaScale, problem.A_scale, strategy.A_scale);
    }
    if (bo2D) {
        setupQAddr(Tbo, state.B_offsetAddrs, state.B_offsetLayout, state.inputs.boPtr,
                   B_h0q, j0q, state.inputs.ldbo, problem.BO, strategy.BO);
    }
    if (bs2D) {
        if (!state.lateScale2DB)
            j0s = j0q, B_h0s = B_h0q;
        setupQAddr(Tb_scale, state.B_scaleAddrs, state.B_scaleLayout, state.inputs.bScalePtr,
                   B_h0s, j0s, state.inputs.ldbScale, problem.B_scale, strategy.B_scale);
    }

    state.ra.safeRelease(A_h0q);
    state.ra.safeRelease(B_h0q);
    state.ra.safeRelease(A_h0s);
    state.ra.safeRelease(B_h0s);

    // Load and convert 0D/1D offsets for 2D dequantization.
    if (aoTo2D) {
        if (!strategy.AO.base.isStateless()) stub();
        std::vector<RegisterBlock> A_offsetLayout;
        GRFRange aoLoad;
        if (problem.aoPtrDims == 1) {
            reclaimRanges(state.C_regs, state);
            auto aoBase = state.ra.alloc_sub<uint64_t>();
            eaddScaled(1, aoBase, state.inputs.aoPtr, slmA ? i0q : state.i0, problem.Tao, strategy, state);
            auto rem = slmA ? state.remaindersCoop[LoopM] : state.remainders[LoopM];
            auto r = slmA ? state.ma_slm : strategy.unroll[LoopM];
            aoLoad = loadVector(problem.Tao, problem.Tao, aoBase, r, rem, strategy, state);
            makeUnbackedRegLayout(problem.Tao, A_offsetLayout, r, 1, true);
            state.ra.safeRelease(aoBase);
        } else {
            auto grf = loadScalars(problem.Tao, {state.inputs.aoPtr}, strategy, state);
            aoLoad = grf-grf;
            A_offsetLayout = state.Ar_offsetLayout;
        }
        gemmRepack2DOffsetData(problem.Ta_ext, problem.Tao, state.Tao_int, A_offsetLayout, state.Ar_offsetLayout, aoLoad, state.Ar_offsetRegs, problem, strategy, state);
        state.ra.safeRelease(aoLoad);
        if (!strategy.persistent) state.ra.safeRelease(state.inputs.aoPtr);
    }
    if (boTo2D) {
        if (!strategy.BO.base.isStateless()) stub();
        std::vector<RegisterBlock> B_offsetLayout;
        GRFRange boLoad;
        if (problem.boPtrDims == 1) {
            reclaimRanges(state.C_regs, state);
            auto boBase = state.ra.alloc_sub<uint64_t>();
            auto rem = slmB ? state.remaindersCoop[LoopN] : state.remainders[LoopN];
            auto c = slmB ? state.nb_slm : strategy.unroll[LoopN];
            eaddScaled(1, boBase, state.inputs.boPtr, slmB ? j0q : state.j0, Tbo, strategy, state);
            boLoad = loadVector(problem.Tbo, problem.Tbo, boBase, c, rem, strategy, state);
            makeUnbackedRegLayout(problem.Tbo, B_offsetLayout, 1, c, false);
            state.ra.safeRelease(boBase);
        } else {
            auto grf = loadScalars(problem.Tbo, {state.inputs.boPtr}, strategy, state);
            boLoad = grf-grf;
            B_offsetLayout = state.Br_offsetLayout;
        }
        gemmRepack2DOffsetData(problem.Tb_ext, problem.Tbo, state.Tbo_int, B_offsetLayout, state.Br_offsetLayout, boLoad, state.Br_offsetRegs, problem, strategy, state);
        state.ra.safeRelease(boLoad);
        if (!strategy.persistent) state.ra.safeRelease(state.inputs.boPtr);
    }

    if (i0q != state.i0) state.ra.safeRelease(i0q);
    if (j0q != state.j0) state.ra.safeRelease(j0q);

    // Free unneeded registers after address setup.
    if (!state.isNested) {
        if (strategy.A.address2D && (!strategy.prefetchA || strategy.A_prefetch.address2D))
            state.ra.safeRelease(state.inputs.lda);
        if (strategy.B.address2D && (!strategy.prefetchB || strategy.B_prefetch.address2D))
            state.ra.safeRelease(state.inputs.ldb);
        if (!strategy.C.address2D && (!strategy.prefetchC || !strategy.C_prefetch.address2D) && !keepIJ0(problem, strategy)) {
            state.ra.safeRelease(state.i0);
            state.ra.safeRelease(state.j0);
        }
        if (!keepH0(problem, strategy))
            state.ra.safeRelease(state.h0);
    }

    if (!needsKLoopReset(problem)) {
        if (state.Ai_strategy.address2D) {
            if (Ai_params.offR != A_params.offR) state.ra.safeRelease(Ai_params.offR);
            if (Ai_params.offC != A_params.offC) state.ra.safeRelease(Ai_params.offC);
        }
        if (state.Bi_strategy.address2D) {
            if (Bi_params.offR != B_params.offR) state.ra.safeRelease(Bi_params.offR);
            if (Bi_params.offC != B_params.offC) state.ra.safeRelease(Bi_params.offC);
        }
        if (strategy.A_prefetch.address2D) {
            if (Ap_params.offR != A_params.offR) state.ra.safeRelease(Ap_params.offR);
            if (Ap_params.offC != A_params.offC) state.ra.safeRelease(Ap_params.offC);
        }
        if (strategy.B_prefetch.address2D) {
            if (Bp_params.offR != B_params.offR) state.ra.safeRelease(Bp_params.offR);
            if (Bp_params.offC != B_params.offC) state.ra.safeRelease(Bp_params.offC);
        }

        if (!one_of(state.effAp, state.effA, state.effAi)) state.ra.safeRelease(state.effAp);
        if (!one_of(state.effBp, state.effB, state.effBi)) state.ra.safeRelease(state.effBp);
    }

    releaseLDMultiples(state.ldaMultiples, state);
    releaseLDMultiples(state.ldbMultiples, state);
    releaseIndexVec(state);

    reclaimRanges(state.C_regs, state);

    // Allocate tokens.
    gemmAllocateTokens(problem, strategy, state);

    // Preloading C and fused beta scaling need some extra registers for C headers.
    // Temporarily free up A/B data registers for that purpose.
    releaseRanges(state.A_regs, state);
    releaseRanges(state.B_regs, state);
    releaseRanges(state.Ar_regs, state);
    releaseRanges(state.Br_regs, state);

    // Load C now if configured (and perform beta scaling).
    if (cLoadAhead) {
        if (problem.checkBeta0 && !problem.beta.fixed())   stub();
        if (state.C_accCount > 0)                          stub();
        if (strategy.kParallelLocal)                       stub();

        status << "Loading C" << status_stream::endl;
        gemmAccessC(COperation::Load, problem, strategy, state);

        gemmBetaScale(problem, strategy, state);
    }

    // Beta prescaling for atomic kernels.
    if (strategy.fuseBeta)
        gemmFusedBetaScale(problem, strategy, state);

    // Reclaim A/B data registers.
    reclaimRanges(state.Br_regs, state);
    reclaimRanges(state.Ar_regs, state);
    reclaimRanges(state.B_regs, state);
    reclaimRanges(state.A_regs, state);

    for (int q = 0; q < state.C_count; q++)
        releaseLDMultiples(state.ldcMultiples[q], state);
    releaseIndexVec(state);

    // Release 64-bit emulation registers as they aren't needed in the inner loop.
    // Could also move r0 to acc here.
    if (state.emulate.temp[0].isValid()) {
        for (int q = 0; q < 2; q++) {
            state.emulate64TempSave[q] = state.emulate.temp[q];
            state.ra.safeRelease(state.emulate.temp[q]);
        }
        if (strategy.emulate.emulate64 && !strategy.emulate.emulate64_add32) {
            if (GRF::bytes(hw) == 64) {
                state.emulate.flag = simd16EmulationFlag;
                state.emulate.flagOffset = 0;
            } else {
                state.emulate.flag = state.flagAP;
                state.emulate.flagOffset = 8;
                state.lateKLoopCheck = false;
            }
        }
    }

    return true;
}

template <HW hw>
void BLASKernelGenerator<hw>::gemmAccumulateCTeardown(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state)
{
    // We're done with A and B. Free their address, data, and flag registers.
    // Also done with loop counter.
    safeReleaseMaskAssignments(state.AB_masks, state);
    safeReleaseMaskAssignments(state.AB_masksCoop, state);
    safeReleaseRanges(state.A_addrs, state);
    safeReleaseRanges(state.B_addrs, state);
    safeReleaseRanges(state.A_addrsAlt, state);
    safeReleaseRanges(state.B_addrsAlt, state);
    safeReleaseRanges(state.Ai_addrs, state);
    safeReleaseRanges(state.Bi_addrs, state);
    safeReleaseRanges(state.Ao_addrs, state);
    safeReleaseRanges(state.Bo_addrs, state);
    safeReleaseRanges(state.Ap_addrs, state);
    safeReleaseRanges(state.Bp_addrs, state);
    safeReleaseRanges(state.Ap_addrsAlt, state);
    safeReleaseRanges(state.Bp_addrsAlt, state);
    safeReleaseRanges(state.A_offsetAddrs, state);
    safeReleaseRanges(state.B_offsetAddrs, state);
    safeReleaseRanges(state.A_scaleAddrs, state);
    safeReleaseRanges(state.B_scaleAddrs, state);

    safeReleaseRanges(state.A_regs, state);
    safeReleaseRanges(state.Ar_regs, state);
    safeReleaseRanges(state.Ai_regs, state);
    safeReleaseRanges(state.Ao_regs, state);
    safeReleaseRanges(state.Ap_regs, state);
    safeReleaseRanges(state.A_offsetRegs, state);
    safeReleaseRanges(state.A_scaleRegs, state);
    safeReleaseRanges(state.Ar_offsetRegs, state);
    safeReleaseRanges(state.Ar_scaleRegs, state);
    safeReleaseRanges(state.B_regs, state);
    safeReleaseRanges(state.Br_regs, state);
    safeReleaseRanges(state.Bi_regs, state);
    safeReleaseRanges(state.Bo_regs, state);
    safeReleaseRanges(state.Bp_regs, state);
    safeReleaseRanges(state.B_offsetRegs, state);
    safeReleaseRanges(state.B_scaleRegs, state);
    safeReleaseRanges(state.Br_offsetRegs, state);
    safeReleaseRanges(state.Br_scaleRegs, state);
    state.ra.safeRelease(state.broadcast_regs);
    safeReleaseRanges(state.tempMul_regs, state);
    clearTokenAllocations(hw, state);
    releaseCoopRemainders(state);

    deduplicateScalar(state.lda, state);
    deduplicateScalar(state.ldb, state);

    state.raVFlag.safeRelease(state.barrierM);
    state.raVFlag.safeRelease(state.barrierN);

    if (state.effCp != state.effC[0])
        state.ra.safeRelease(state.effCp);

    state.A_layout.clear();
    state.B_layout.clear();
    state.A_layoutAlt.clear();
    state.B_layoutAlt.clear();
    state.Ai_layout.clear();
    state.Bi_layout.clear();
    state.Ao_layout.clear();
    state.Bo_layout.clear();
    state.Ar_layout.clear();
    state.Br_layout.clear();
    state.Ap_layout.clear();
    state.Bp_layout.clear();
    state.Cp_layout.clear();
    state.Ap_layoutAlt.clear();
    state.Bp_layoutAlt.clear();
    state.A_offsetLayout.clear();
    state.B_offsetLayout.clear();
    state.A_scaleLayout.clear();
    state.B_scaleLayout.clear();

    if (state.systolicSumA || state.systolicSumB)
        setupTeardownAccumulateSumSystolic(false, Type::invalid, problem, strategy, state);

    // Restore effA/B if needed.
    bool restoreEffAB = false;

    restoreEffAB |= (problem.aOffset == ABOffset::Load);
    restoreEffAB |= (problem.bOffset == ABOffset::Load);

    if (restoreEffAB) {
        Subregister aiOff, biOff;
        Subregister aiOffR, biOffR;
        Subregister aiOffC, biOffC;
        if (strategy.slmA)
            gemmCalcWorkshareAOffset(aiOff, aiOffR, aiOffC, state.Ai, state.Ai_strategy, state.ma_slm, state.ka_slm, problem, strategy, state);
        if (strategy.slmB)
            gemmCalcWorkshareBOffset(biOff, biOffR, biOffC, state.Bi, state.Bi_strategy, state.kb_slm, state.nb_slm, problem, strategy, state);
        if (strategy.slmA) eadd(1, state.effAi, state.effAi, -aiOff, strategy, state);
        if (strategy.slmB) eadd(1, state.effBi, state.effBi, -biOff, strategy, state);

        state.ra.safeRelease(aiOff);
        state.ra.safeRelease(biOff);
        state.ra.safeRelease(aiOffR);
        state.ra.safeRelease(biOffR);
        state.ra.safeRelease(aiOffC);
        state.ra.safeRelease(biOffC);
    }

    // Restore A/B addresses and strategies that were modified by SLM copies.

    if (strategy.slmA) {
        state.ra.safeRelease(state.effA);
        state.ra.safeRelease(state.effAo);
        state.effA = state.effAi;
        state.effAi = invalid;
        state.Ta_load = problem.Ta_ext;
        problem.A = state.Ai;
        strategy.A = state.Ai_strategy;
    }
    if (strategy.slmB) {
        state.ra.safeRelease(state.effB);
        state.ra.safeRelease(state.effBo);
        state.effB = state.effBi;
        state.effBi = invalid;
        state.Tb_load = problem.Tb_ext;
        problem.B = state.Bi;
        strategy.B = state.Bi_strategy;
    }

    // Put accumulators with the rest of C.
    if (state.C_accCount > 0) {
        // Reclaim the bottom registers of C.
        reclaimRanges(state.C_regs[0], state);

        auto e = elementsPerGRF<uint32_t>(hw);
        for (int i = 0; i < state.C_accCount; i += 2)
            mov<uint32_t>(2 * e, state.C_regs[0][i], AccumulatorRegister(i));
    }

    // Restore emulation registers.
    if (state.emulate64TempSave[0].isValid()) {
        for (int q = 0; q < 2; q++) {
            state.emulate.temp[q] = state.emulate64TempSave[q];
            if (state.emulate64TempSave[q].isValid())
                state.ra.claim(state.emulate64TempSave[q]);
        }
        if (GRF::bytes(hw) == 64)
            state.raVFlag.release(state.emulate.flag);
        state.emulate.flag = invalid;
        state.emulate.flagOffset = 0;
    }
}

// Calculate and cache multiples of lda/ldb used for address increments in the inner loop.
template <HW hw>
void BLASKernelGenerator<hw>::gemmCalcIncrements(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state,
                                                 int ka_load, int kb_load, bool doA, bool doB)
{
    gemmFreeIncrements(problem, strategy, state, doA, doB);

    doA &= (problem.A.layout == MatrixLayout::N);
    doB &= (problem.B.layout == MatrixLayout::T);

    if (ka_load == 0) ka_load = strategy.ka_inc();
    if (kb_load == 0) kb_load = strategy.kb_inc();

    auto calcInterleavedIncrement = [&](bool isA, int inc) {
        auto &increments = isA ? state.ldaIncrements : state.ldbIncrements;
        auto &base       = isA ? state.lda           : state.ldb;
        if (strategy.kInterleave) {
            int chunk = strategy.kInterleaveChunk;
            if (inc < chunk)
                calcIncrement(increments, base, inc, strategy, state);
            calcIncrement(increments, base, inc + chunk * (strategy.wg[LoopK] - 1), strategy, state);
        } else
            calcIncrement(increments, base, inc, strategy, state);
    };

    if (doA) {
        if (!strategy.A.address2D)
            calcInterleavedIncrement(true, ka_load);
        if (strategy.prefetchA && !strategy.A_prefetch.address2D)
            calcInterleavedIncrement(true, strategy.ka_pfStride);
    }

    if (doB) {
        if (!strategy.B.address2D)
            calcInterleavedIncrement(false, kb_load);
        if (strategy.prefetchB && !strategy.B_prefetch.address2D)
            calcInterleavedIncrement(false, strategy.kb_pfStride);
    }
}

template <HW hw>
void BLASKernelGenerator<hw>::gemmCalcQuantizationIncrements(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    bool ao2D = (problem.aoPtrDims == 2);
    bool bo2D = (problem.boPtrDims == 2);
    bool as2D = problem.aScale2D;
    bool bs2D = problem.bScale2D;

    auto calcInterleavedQIncrement = [&](bool isA, SubregisterPair &base, LDIncrements &increments) {
        auto inc   = isA ? state.kaqStride  : state.kbqStride;
        auto group = isA ? problem.aqGroupK : problem.bqGroupK;
        if (strategy.kInterleave) {
            int chunk = strategy.kInterleaveChunk;
            if (group < chunk) {
                calcIncrement(increments, base, inc, strategy, state);
                calcIncrement(increments, base, (inc * group + chunk * (strategy.wg[LoopK] - 1)) / group, strategy, state);
            } else
                calcIncrement(increments, base, chunk * strategy.wg[LoopK] / group, strategy, state);
        } else
            calcIncrement(increments, base, inc, strategy, state);
    };

    if (ao2D && problem.AO.layout == MatrixLayout::N)
        calcInterleavedQIncrement(true,  state.ldao,     state.ldaoIncrements);
    if (as2D && problem.A_scale.layout == MatrixLayout::N)
        calcInterleavedQIncrement(true,  state.ldaScale, state.ldasIncrements);
    if (bo2D && problem.BO.layout == MatrixLayout::T)
        calcInterleavedQIncrement(false, state.ldbo,     state.ldboIncrements);
    if (bs2D && problem.B_scale.layout == MatrixLayout::T)
        calcInterleavedQIncrement(false, state.ldbScale, state.ldbsIncrements);
}

// Free cached multiples of lda/ldb.
template <HW hw>
void BLASKernelGenerator<hw>::gemmFreeIncrements(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, bool doA, bool doB)
{
    auto freeIncrements = [&](SubregisterPair &base, LDIncrements &increments) {
        for (auto &inc: increments)
            safeRelease(inc.second, state);
        deduplicateScalar(base, state);
        state.ra.claim(base.getReg(0));
        increments.clear();
    };

    if (doA) {
        freeIncrements(state.lda, state.ldaIncrements);
        freeIncrements(state.ldao, state.ldaoIncrements);
        freeIncrements(state.ldaScale, state.ldasIncrements);
    }

    if (doB) {
        freeIncrements(state.ldb, state.ldbIncrements);
        freeIncrements(state.ldbo, state.ldboIncrements);
        freeIncrements(state.ldbScale, state.ldbsIncrements);
    }
}

// Prepare A/B prefetch addresses.
template <HW hw>
void BLASKernelGenerator<hw>::gemmABPrefetchAddrSetup(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, bool doA, bool doB)
{
    if (doA && strategy.cooperativePF && strategy.prefetchA) {
        Subregister offAp;
        auto &A_offR = state.A_params.offR, &Ap_offR = state.Ap_params.offR;
        auto &A_offC = state.A_params.offC, &Ap_offC = state.Ap_params.offC;
        gemmCalcWorkshareAOffset(offAp, Ap_offR, Ap_offC, problem.A, strategy.A_prefetch, state.ma_prefetch, state.ka_prefetch, problem, strategy, state);
        if (strategy.A_prefetch.address2D) {
            if (A_offR.isValid() && A_offR != Ap_offR) add(1, Ap_offR, Ap_offR, A_offR);
            if (A_offC.isValid() && A_offC != Ap_offC) add(1, Ap_offC, Ap_offC, A_offC);
        } else {
            auto inEffAp = state.effAp;
            if (state.effA == state.effAp)
                state.effAp = state.ra.alloc_sub(state.effA.getType());
            eadd(1, state.effAp, inEffAp, offAp, strategy, state);
        }
        state.ra.safeRelease(offAp);
    }

    if (doB && strategy.cooperativePF && strategy.prefetchB) {
        Subregister offBp;
        auto &B_offR = state.B_params.offR, &Bp_offR = state.Bp_params.offR;
        auto &B_offC = state.B_params.offC, &Bp_offC = state.Bp_params.offC;
        gemmCalcWorkshareBOffset(offBp, Bp_offR, Bp_offC, problem.B, strategy.B_prefetch, state.kb_prefetch, state.nb_prefetch, problem, strategy, state);
        if (strategy.B_prefetch.address2D) {
            if (B_offR.isValid() && B_offR != Bp_offR) add(1, Bp_offR, Bp_offR, B_offR);
            if (B_offC.isValid() && B_offC != Bp_offC) add(1, Bp_offC, Bp_offC, B_offC);
        } else {
            auto inEffBp = state.effBp;
            if (state.effB == state.effBp)
                state.effBp = state.ra.alloc_sub(state.effB.getType());
            eadd(1, state.effBp, inEffBp, offBp, strategy, state);
        }
        state.ra.safeRelease(offBp);
    }

    if (problem.backward()) {
        if (doA && strategy.prefetchA)
            gemmOffsetAk(strategy.ka_load - strategy.ka_prefetch, state.effAp, problem.A, problem, strategy, state);
        if (doB && strategy.prefetchB)
            gemmOffsetBk(strategy.kb_load - strategy.kb_prefetch, state.effBp, problem.B, problem, strategy, state);
    }
}

// Allocate tokens for k loop loads/stores.
template <HW hw>
void BLASKernelGenerator<hw>::gemmAllocateTokens(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    if (hw < HW::Gen12LP) return;

    bool success = true;
    for (int q = 0; q < strategy.A_copies; q++)
        success &= allocateTokens(state.A_layout, state.A_regs[q], state);
    for (int q = 0; q < strategy.B_copies; q++)
        success &= allocateTokens(state.B_layout, state.B_regs[q], state);
    for (int q = 0; q < strategy.slmCopies; q++) {
        if (strategy.slmA) success &= allocateTokens(state.Ai_layout, state.Ai_regs[q], state);
        if (strategy.slmB) success &= allocateTokens(state.Bi_layout, state.Bi_regs[q], state);
    }
    if (strategy.slmA && !state.aioShare)
        success &= allocateTokens(state.Ao_layout, state.Ao_regs, state);
    if (strategy.slmB && !state.bioShare)
        success &= allocateTokens(state.Bo_layout, state.Bo_regs, state);
    success &= allocateTokens(state.Ap_layout, state.Ap_regs, state, state.Ap_addrs);
    success &= allocateTokens(state.Bp_layout, state.Bp_regs, state, state.Bp_addrs);

    success &= allocateTokens(state.A_offsetLayout, state.A_offsetRegs, state);
    success &= allocateTokens(state.B_offsetLayout, state.B_offsetRegs, state);
    success &= allocateTokens(state.A_scaleLayout, state.A_scaleRegs, state);
    success &= allocateTokens(state.B_scaleLayout, state.B_scaleRegs, state);

    if (!success) {
        status << "Not enough tokens for k loop." << status_stream::endl;
        clearMappedTokenAllocations(hw, state);
    }
}

// Offset an SLM address by the base SLM address if present.
template <HW hw>
void BLASKernelGenerator<hw>::makeSLMBaseRelative(Subregister addr, const GEMMState &state)
{
    if (state.inputs.slmBase.isValid())
        add(1, addr, addr, state.inputs.slmBase);
}

// Initialize the interface.
template <HW hw>
void BLASKernelGenerator<hw>::gemmInitInterface(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state, bool inSK)
{
    Subregister localSize[3];
    GRF localID[3];
    Subregister tgids[3] = {r0.ud(1), r0.ud(6), r0.ud(7)};   // X, Y, Z threadgroup IDs

    initInterface(problem, strategy, state);
    if (strategy.systolic)
        interface.requireDPAS();
    if (strategy.C.atomic)
        interface.requireGlobalAtomics();
    if (strategy.barrierFreq > 0 || strategy.fuseBeta || strategy.fusePostOps)
        interface.requireBarrier();

    auto slmSize = gemmSLMSize(hw, problem, strategy);
    auto slmPerK = gemmPerKSLMSize(hw, problem, strategy);
    if (slmSize > 0 || slmPerK > 0) {
        status << "SLM usage: " << slmSize / 1024. << 'k';
        if (slmPerK)
            status << " (" << slmPerK / 1024. << "k per-k)";
        status << status_stream::endl;
        if (!slmPerK)
            interface.requireSLM(slmSize);
        interface.requireBarrier();
    }

    size_t slm = maxSLMPerWG(hw, strategy.GRFs);
    if (slmSize > slm || slmPerK * strategy.wg[LoopK] > slm)
        stub("Strategy requests more SLM than available");

    if (strategy.fixedWG(problem)) {
        auto wgX = strategy.wg[strategy.loopOrder[0]];
        auto wgY = strategy.wg[strategy.loopOrder[1]];
        auto wgZ = strategy.wg[strategy.loopOrder[2]];
        if (strategy.splitCopy)
            wgY *= 2;
        wgY *= strategy.wgPadFactor;
        if (wgZ <= 1)
            interface.requireWorkgroup(strategy.subgroupSize * wgX, wgY, wgZ);
    }
    interface.requireWalkOrder(0, 1, 2);

    bool needStatelessWrites = strategy.C.base.isStateless();
    if (problem.sumA || problem.sumB)
        needStatelessWrites |= strategy.CO.base.isStateless();

    interface.requireStatelessWrites(needStatelessWrites);

    int nb = int(strategy.needsNamedBarriersM(problem)) * strategy.namedBarriers[LoopM] * strategy.wg[LoopK]
           + int(strategy.needsNamedBarriersN(problem)) * strategy.namedBarriers[LoopN] * strategy.wg[LoopK];
    if (nb) {
        if (strategy.needsUnnamedBarrier(problem)) nb++;
        if (nb > 32) hw_unsupported();                    /* also causes IGC/NEO crashes */
        interface.requireBarriers(nb);
    }

    interface.finalize();

    for (int dim = 0; dim < 3; dim++) {
        localID[dim] = interface.getLocalID(dim);
        localSize[dim] = interface.getLocalSize(dim);
    }

    // Get input arguments.
    state.inputs.base = interface.getArgumentIfExists("base");
    auto baseSurface = interface.getArgumentSurfaceIfExists("base");
    if (state.inputs.base.isInvalid() && baseSurface == InterfaceHandler::noSurface) {
        state.inputs.A = interface.getArgumentIfExists("A");
        state.inputs.B = interface.getArgumentIfExists("B");
        state.inputs.C[1] = interface.getArgumentIfExists("P");
        state.inputs.surfaceA = interface.getArgumentSurfaceIfExists("A");
        state.inputs.surfaceB = interface.getArgumentSurfaceIfExists("B");
        state.inputs.surfaceC[1] = interface.getArgumentSurfaceIfExists("P");
    } else {
        state.inputs.A = state.inputs.B = state.inputs.base;
        state.inputs.surfaceA = state.inputs.surfaceB = baseSurface;
        if (interface.getArgumentIfExists("offset_P").isValid()) {
            state.inputs.C[1] = state.inputs.base;
            state.inputs.surfaceC[1] = state.inputs.surfaceA;
        }
    }

    state.inputs.C[0] = interface.getArgumentIfExists("C");
    state.inputs.surfaceC[0] = interface.getArgumentSurfaceIfExists("C");
    state.C_count = state.inputs.C[1].isValid() ? 2 : 1;
    if (problem.usesCO()) {
        state.inputs.CO = interface.getArgumentIfExists("CO");
        state.inputs.surfaceCO = interface.getArgumentSurfaceIfExists("CO");
    }
    if (state.useTempC) {
        state.inputs.tempC = interface.getArgumentIfExists("temp_C");
        state.inputs.surfaceTempC = interface.getArgumentSurfaceIfExists("temp_C");
    }

    bool aOffset = (problem.aOffset != ABOffset::None);
    bool bOffset = (problem.bOffset != ABOffset::None);
    if (aOffset || bOffset) {
        state.inputs.abo = interface.getArgumentIfExists("abo");
        if (state.inputs.abo.isValid()) {
            // A/B offset are two words packed into a single dword argument.
            state.inputs.ao = state.inputs.abo.w(0);
            state.inputs.bo = state.inputs.abo.w(1);
        } else {
            state.inputs.ao = interface.getArgumentIfExists("ao");
            state.inputs.bo = interface.getArgumentIfExists("bo");
        }
        state.inputs.aoPtr = interface.getArgumentIfExists("ao_ptr");
        state.inputs.boPtr = interface.getArgumentIfExists("bo_ptr");
        state.inputs.surfaceAO = interface.getArgumentSurfaceIfExists("ao_ptr");
        state.inputs.surfaceBO = interface.getArgumentSurfaceIfExists("bo_ptr");
    }
    if (problem.aScale2D) {
        state.inputs.aScalePtr = interface.getArgumentIfExists("a_scale_ptr");
        state.inputs.surfaceAScale = interface.getArgumentSurfaceIfExists("a_scale_ptr");
    }
    if (problem.bScale2D) {
        state.inputs.bScalePtr = interface.getArgumentIfExists("b_scale_ptr");
        state.inputs.surfaceBScale = interface.getArgumentSurfaceIfExists("b_scale_ptr");
    }
    if (problem.cStochasticRound) 
        state.inputs.sroundSeedPtr = interface.getArgument("sround_seed");
    state.inputs.offsetA = interface.getArgumentIfExists("offset_A");
    state.inputs.offsetB = interface.getArgumentIfExists("offset_B");
    state.inputs.offsetC[0] = interface.getArgumentIfExists("offset_C");
    state.inputs.offsetC[1] = interface.getArgumentIfExists("offset_P");
    state.inputs.offsetAO = interface.getArgumentIfExists("offset_AO");
    state.inputs.offsetBO = interface.getArgumentIfExists("offset_BO");
    state.inputs.offsetCO = interface.getArgumentIfExists("offset_CO");
    state.inputs.offsetAScale = interface.getArgumentIfExists("offset_A_scale");
    state.inputs.offsetBScale = interface.getArgumentIfExists("offset_B_scale");
    state.inputs.offsetAq = interface.getArgumentIfExists("offset_Aq");
    state.inputs.offsetBq = interface.getArgumentIfExists("offset_Bq");
    if (problem.batch == BatchMode::Strided) {
        for (int i = 0; i < problem.batchDims; i++) {
            auto istr = std::to_string(i);
            state.inputs.strideA.push_back(interface.getArgument("stride_A" + istr));
            state.inputs.strideB.push_back(interface.getArgument("stride_B" + istr));
            state.inputs.strideC.push_back(interface.getArgument("stride_C" + istr));
            if (i < problem.batchDims - 1) {
                state.inputs.batchSize.push_back(interface.getArgument("batch_size" + istr));
                state.inputs.recipBatchSize.push_back(interface.getArgument("recip_batch_size" + istr));
            }
        }
    } else if (problem.batch == BatchMode::Nonstrided)
        state.inputs.offsetBatch = interface.getArgumentIfExists("offset_batch");
    else if (problem.batch == BatchMode::Variable) {
        state.inputs.incr_a_array = interface.getArgumentIfExists("incr_a_array");
        state.inputs.incr_b_array = interface.getArgumentIfExists("incr_b_array");
    }
    state.inputs.lda = interface.getArgumentIfExists("lda");
    state.inputs.ldb = interface.getArgumentIfExists("ldb");
    state.inputs.ldc[0] = interface.getArgumentIfExists("ldc");
    state.inputs.ldc[1] = interface.getArgumentIfExists("ldp");
    state.inputs.ldao = interface.getArgumentIfExists("ldao");
    state.inputs.ldbo = interface.getArgumentIfExists("ldbo");
    state.inputs.ldco = interface.getArgumentIfExists("ldco");
    state.inputs.ldaScale = interface.getArgumentIfExists("lda_scale");
    state.inputs.ldbScale = interface.getArgumentIfExists("ldb_scale");
    state.inputs.ldaq = interface.getArgumentIfExists("ldaq");
    state.inputs.ldbq = interface.getArgumentIfExists("ldbq");
    state.inputs.m = interface.getArgumentIfExists("m");
    state.inputs.n = interface.getArgumentIfExists("n");
    state.inputs.k = interface.getArgumentIfExists("k");
    state.inputs.k0 = interface.getArgumentIfExists("k0");
    state.inputs.alpha_real = interface.getArgumentIfExists("alpha_real");
    state.inputs.beta_real = interface.getArgumentIfExists("beta_real");
    if (problem.batch == BatchMode::Variable) {
        state.inputs.alpha_array = interface.getArgumentIfExists("alpha_array");
        state.inputs.beta_array = interface.getArgumentIfExists("beta_array");
        state.inputs.incr_alpha = interface.getArgumentIfExists("incr_alpha");
        state.inputs.incr_beta = interface.getArgumentIfExists("incr_beta");
    }
    state.inputs.diagA = interface.getArgumentIfExists("diag_A");
    state.inputs.diagB = interface.getArgumentIfExists("diag_B");
    state.inputs.diagC = interface.getArgumentIfExists("diag_C");
    state.inputs.flags = interface.getArgumentIfExists("flags");
    state.inputs.slmBase = interface.getArgumentIfExists("slm_base");

    if (strategy.linearOrder()) {
        state.inputs.groupCountM = interface.getArgument("group_count_m");
        state.inputs.groupCountN = interface.getArgument("group_count_n");
    }
    if (strategy.cWalkOrder == WalkOrder::SimpleLinear)
        state.inputs.gcMNRecip = interface.getArgument("group_count_recip");
    else if (strategy.cWalkOrder == WalkOrder::Hilbertlike) {
        state.inputs.hilbertVD = interface.getArgumentIfExists("hilbert_vd");
        state.inputs.hilbertUVDRecip = interface.getArgumentIfExists("hilbert_uvd_recip");
        state.inputs.hilbertBail = interface.getArgumentIfExists("hilbert_bail");
    } else if (strategy.cWalkOrder == WalkOrder::Boustrophedon) {
        state.inputs.bslice = interface.getArgument("bslice");
        state.inputs.bthresh = interface.getArgument("bthresh");
    }
    if (strategy.persistent) {
        state.inputs.groupCountMN = interface.getArgumentIfExists("group_count");
        state.inputs.groupStride = interface.getArgument("group_stride");
        if (strategy.kParallelVariable) {
            state.inputs.kParallelStart = interface.getArgument("k_parallel_start");
            state.inputs.kRecip = interface.getArgument("k_recip");
            if (strategy.fuseBeta)
                state.inputs.k0Recip = interface.getArgument("k0_recip");
        }
    }
    if (strategy.kParallel && strategy.fuseBeta)
        state.inputs.groupCountK = interface.getArgument("group_count_k");
    if (strategy.fuseBeta || strategy.fusePostOps)
        state.inputs.statusBuffer = interface.getArgumentIfExists("status");

    size_t poCount = problem.postOps.len();
    state.inputs.binarySrcs.resize(poCount);
    state.inputs.binaryOffsets.resize(poCount);
    state.inputs.binaryLDs.resize(poCount);
    state.inputs.binaryStrides.resize(poCount);
    state.inputs.binarySurfaces.resize(poCount);
    for (size_t i = 0; i < poCount; i++) {
        std::string srcName = "binary" + std::to_string(i);
        state.inputs.binarySrcs[i] = interface.getArgumentIfExists(srcName);
        state.inputs.binarySurfaces[i] = interface.getArgumentSurfaceIfExists(srcName);
        state.inputs.binaryOffsets[i] = interface.getArgumentIfExists("offset_" + srcName);
        state.inputs.binaryLDs[i] = interface.getArgumentIfExists("ld" + srcName);
        if (problem.batch == BatchMode::Strided)
            for (int b = 0; b < problem.batchDims; b++)
                state.inputs.binaryStrides[i].push_back(interface.getArgumentIfExists("stride" + std::to_string(b) + srcName));
    }

    Subregister tgids_reordered[3];
    GRF lids_reordered[3];
    Subregister lszs_reordered[3];

    for (int l = 0; l < 3; l++) {
        int i = static_cast<int>(strategy.loopOrder[l]);
        tgids_reordered[i] = tgids[l];
        lids_reordered[i] = localID[l];
        lszs_reordered[i] = localSize[l];
    }
    state.inputs.groupIDM = tgids_reordered[0];
    state.inputs.groupIDN = tgids_reordered[1];
    state.inputs.groupIDK = tgids_reordered[2];
    state.inputs.localIDM = lids_reordered[0];
    state.inputs.localIDN = lids_reordered[1];
    state.inputs.localIDK = lids_reordered[2];
    state.inputs.localSizeM = lszs_reordered[0];
    state.inputs.localSizeN = lszs_reordered[1];
    state.inputs.localSizeK = lszs_reordered[2];

    if (strategy.linearOrder()) {
        state.inputs.groupIDMN = tgids[0];
        state.inputs.groupIDM = invalid;
        state.inputs.groupIDN = invalid;
    }

    // Move SLM pointers to offset arguments.
    if (strategy.A.base.getModel() == ModelSLM) std::swap(state.inputs.A, state.inputs.offsetA);
    if (strategy.B.base.getModel() == ModelSLM) std::swap(state.inputs.B, state.inputs.offsetB);
    if (strategy.C.base.getModel() == ModelSLM) std::swap(state.inputs.C[0], state.inputs.offsetC[0]);

    // Downgrade offsets to 32 bits for non-A64 accesses.
    if (strategy.A.base.getModel() != ModelA64)
        state.inputs.offsetA = state.inputs.offsetA.d();
    if (strategy.B.base.getModel() != ModelA64)
        state.inputs.offsetB = state.inputs.offsetB.d();
    if (strategy.C.base.getModel() != ModelA64)
        for (int q = 0; q < state.C_count; q++)
            state.inputs.offsetC[q] = state.inputs.offsetC[q].d();
    if (problem.usesCO() && strategy.CO.base.getModel() != ModelA64)
        state.inputs.offsetCO = state.inputs.offsetCO.d();
    for (auto &off: state.inputs.binaryOffsets)
        off = off.d();

    // For now, reinterpret m/n/k/ld/diag variables to 32-bit if they are 64-bit.
    state.inputs.m = state.inputs.m.d();
    state.inputs.n = state.inputs.n.d();
    state.inputs.k = state.inputs.k.d();
    state.inputs.lda = state.inputs.lda.ud();
    state.inputs.ldb = state.inputs.ldb.ud();
    for (int q = 0; q < state.C_count; q++)
        state.inputs.ldc[q] = state.inputs.ldc[q].ud();
    state.inputs.ldao = state.inputs.ldao.ud();
    state.inputs.ldbo = state.inputs.ldbo.ud();
    state.inputs.ldco = state.inputs.ldco.ud();
    state.inputs.diagA = state.inputs.diagA.d();
    state.inputs.diagB = state.inputs.diagB.d();
    state.inputs.diagC = state.inputs.diagC.d();

    // Claim registers.
    for (int i = 0; i < r0DWords(hw); i++)
        state.ra.claim(r0.ud(i));

    if (strategy.A.base.isStateless()) state.ra.claim(state.inputs.A);
    if (strategy.B.base.isStateless()) state.ra.claim(state.inputs.B);
    if (strategy.C.base.isStateless())
        for (int q = 0; q < state.C_count; q++)
            state.ra.claim(state.inputs.C[q]);

    if (aOffset) {
        if (state.inputs.ao.isValid())
            state.ra.claim(state.inputs.ao);
        if (state.inputs.aoPtr.isValid())
            state.ra.claim(state.inputs.aoPtr);
        if (state.inputs.offsetAO.isValid())
            state.ra.claim(state.inputs.offsetAO);
    }

    if (bOffset) {
        if (state.inputs.bo.isValid())
            state.ra.claim(state.inputs.bo);
        if (state.inputs.boPtr.isValid())
            state.ra.claim(state.inputs.boPtr);
        if (state.inputs.offsetBO.isValid())
            state.ra.claim(state.inputs.offsetBO);
    }

    if (problem.aScale2D) {
        state.ra.claim(state.inputs.aScalePtr);
        if (state.inputs.offsetAScale.isValid())
            state.ra.claim(state.inputs.offsetAScale);
    }

    if (problem.bScale2D) {
        state.ra.claim(state.inputs.bScalePtr);
        if (state.inputs.offsetBScale.isValid())
            state.ra.claim(state.inputs.offsetBScale);
    }

    if (state.inputs.ldaq.isValid())
        state.ra.claim(state.inputs.ldaq);
    if (state.inputs.ldbq.isValid())
        state.ra.claim(state.inputs.ldbq);
    if (state.inputs.offsetAq.isValid())
        state.ra.claim(state.inputs.offsetAq);
    if (state.inputs.offsetBq.isValid())
        state.ra.claim(state.inputs.offsetBq);

    if (problem.usesCO()) {
        if (strategy.CO.base.isStateless())
            state.ra.claim(state.inputs.CO);
        state.ra.claim(state.inputs.offsetCO);
    }

    if (state.useTempC)
        if (strategy.C.base.isStateless())
            state.ra.claim(state.inputs.tempC);

    state.ra.claim(state.inputs.offsetA);
    state.ra.claim(state.inputs.offsetB);
    for (int q = 0; q < state.C_count; q++)
        state.ra.claim(state.inputs.offsetC[q]);
    state.ra.claim(state.inputs.lda);
    state.ra.claim(state.inputs.ldb);
    for (int q = 0; q < state.C_count; q++)
        state.ra.claim(state.inputs.ldc[q]);
    if (state.inputs.ldao.isValid())
        state.ra.claim(state.inputs.ldao);
    if (state.inputs.ldbo.isValid())
        state.ra.claim(state.inputs.ldbo);
    if (problem.allowMatrixOffset())
        state.ra.claim(state.inputs.ldco);
    if (state.inputs.ldaScale.isValid())
        state.ra.claim(state.inputs.ldaScale);
    if (state.inputs.ldbScale.isValid())
        state.ra.claim(state.inputs.ldbScale);
    state.ra.claim(state.inputs.m);
    state.ra.claim(state.inputs.n);
    state.ra.claim(state.inputs.k);
    if (strategy.kParallel || strategy.kParallelLocal || strategy.kParallelVariable)
        state.ra.claim(state.inputs.k0);

    if (problem.alpha == Scalar::Variable)
        state.ra.claim(state.inputs.alpha_real);
    else if (problem.alpha.pointer())
        state.ra.claim(state.inputs.alphaPtr);

    if (problem.beta == Scalar::Variable)
        state.ra.claim(state.inputs.beta_real);
    else if (problem.beta.pointer())
        state.ra.claim(state.inputs.betaPtr);

    if (!inSK) {
        state.ra.claim(state.inputs.localIDM);
        state.ra.claim(state.inputs.localIDN);
        if (!strategy.fixedWG(problem)) {
            state.ra.claim(state.inputs.localSizeM);
            state.ra.claim(state.inputs.localSizeN);
        } else
            state.inputs.localSizeM = state.inputs.localSizeN = invalid;
        if (strategy.kParallel || strategy.kParallelLocal) {
            state.ra.claim(state.inputs.localIDK);
            state.ra.claim(state.inputs.localSizeK);
        }
    }

    if (state.inputs.flags.isValid())
        state.ra.claim(state.inputs.flags);
    if (problem.cStochasticRound)
        state.ra.claim(state.inputs.sroundSeedPtr);
    if (state.inputs.slmBase.isValid())
        state.ra.claim(state.inputs.slmBase);

    if (problem.batch == BatchMode::Strided) {
        for (int i = 0; i < problem.batchDims; i++) {
            state.ra.claim(state.inputs.strideA[i]);
            state.ra.claim(state.inputs.strideB[i]);
            state.ra.claim(state.inputs.strideC[i]);
        }
        for (int i = 0; i < problem.batchDims - 1; i++) {
            state.ra.claim(state.inputs.batchSize[i]);
            state.ra.claim(state.inputs.recipBatchSize[i]);
        }
        state.ra.claim(state.inputs.groupIDK);
    } else if (problem.batch == BatchMode::Nonstrided) {
        state.ra.claim(state.inputs.offsetBatch);
        state.ra.claim(state.inputs.groupIDK);
    }
    else if (problem.batch == BatchMode::Variable) {
        state.ra.claim(state.inputs.incr_a_array);
        state.ra.claim(state.inputs.incr_b_array);
        state.ra.claim(state.inputs.alpha_array);
        state.ra.claim(state.inputs.beta_array);
        state.ra.claim(state.inputs.incr_alpha);
        state.ra.claim(state.inputs.incr_beta);
        state.ra.claim(state.inputs.groupIDK);
    }

    if (strategy.linearOrder()) {
        state.ra.claim(state.inputs.groupCountM);
        state.ra.claim(state.inputs.groupCountN);
    }

    if (strategy.cWalkOrder == WalkOrder::SimpleLinear)
        state.ra.claim(state.inputs.gcMNRecip);
    else if (strategy.cWalkOrder == WalkOrder::Hilbertlike) {
        {
            state.ra.claim(state.inputs.hilbertVD);
            state.ra.claim(state.inputs.hilbertUVDRecip);
        }
        state.ra.claim(state.inputs.hilbertBail);
    } else if (strategy.cWalkOrder == WalkOrder::Boustrophedon) {
        state.ra.claim(state.inputs.bslice);
        state.ra.claim(state.inputs.bthresh);
    }

    if (strategy.persistent) {
        state.ra.claim(state.inputs.groupStride);
        if (state.inputs.groupCountMN.isValid())
            state.ra.claim(state.inputs.groupCountMN);
    }

    if (strategy.kParallelVariable) {
        state.ra.claim(state.inputs.kParallelStart);
        state.ra.claim(state.inputs.kRecip);
        if (strategy.fuseBeta)
            state.ra.claim(state.inputs.k0Recip);
    }

    if (strategy.kParallel && strategy.fuseBeta)
        state.ra.claim(state.inputs.groupCountK);

    if (strategy.fuseBeta || strategy.fusePostOps)
        state.ra.claim(state.inputs.statusBuffer);

    // Binary-related arguments are not claimed here, but instead
    //  are reloaded later in the kernel when needed.

}

// Initialize the state structure.
template <HW hw>
void BLASKernelGenerator<hw>::gemmInitState(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state, bool inSK)
{
    auto Ta = problem.Ta, Tb = problem.Tb, Tc = problem.Tc;
    auto Ta_ext = problem.Ta_ext, Tb_ext = problem.Tb_ext;

    state.useTempC = strategy.needsTempC(problem);

    {
        initState(problem, strategy, state);
        gemmInitInterface(problem, strategy, state, inSK);
        state.isNested |= strategy.fused;
        state.isNested |= strategy.persistent;
    }

    state.effA = strategy.A.base.isStateless() ? state.inputs.A
                                               : state.inputs.offsetA.d();
    state.effB = strategy.B.base.isStateless() ? state.inputs.B
                                               : state.inputs.offsetB.d();
    for (int q = 0; q < state.C_count; q++) {
        state.effC[q] = strategy.C.base.isStateless() ? state.inputs.C[q]
                                                      : state.inputs.offsetC[q].d();
    }
    if (problem.usesCO()) {
        state.effCO = strategy.CO.base.isStateless() ? state.inputs.CO
                                                     : state.inputs.offsetCO.d();
    }
    if (state.useTempC) {
        state.effTempC = strategy.C.base.isStateless() ? state.inputs.tempC
                                                       : state.ra.alloc_sub<uint32_t>();
    }

    state.offsetA = state.inputs.offsetA;
    state.offsetB = state.inputs.offsetB;
    for (int q = 0; q < state.C_count; q++)
        state.offsetC[q] = state.inputs.offsetC[q];
    state.offsetCO = state.inputs.offsetCO;

    state.flagAP = state.raVFlag.alloc();

    state.allocEmulate64Temp(strategy.emulate);

    state.Ta_load = problem.Ta_ext;
    state.Tb_load = problem.Tb_ext;

    state.Tacc = problem.Tc;
    state.copyC = (problem.Tc != problem.Tc_ext)
               || (!strategy.altCRemainder && (Tc.size() < 4))
               || strategy.forceCopyC
               || (strategy.C.base.getModel() == ModelInvalid);

    state.broadcast = strategy.doubleWA;

    bool cColMajor = isRegisterColMajor(problem.Tc, problem.C, strategy.C);
    state.broadcast |= (Tc == Type::f32 && (cColMajor ? Tb : Ta) == Type::bf16);

    state.Cext_strategy = strategy.C;
    state.Cext_strategy.tileR = state.Cext_strategy.tileC = 0;

    state.lidM = state.inputs.localIDM[0];
    state.lidN = state.inputs.localIDN[0];
    if (strategy.kParallel || strategy.kParallelLocal)
        state.lidK = state.inputs.localIDK[0];

    state.k = state.inputs.k;

    state.lda = state.inputs.lda;
    state.ldb = state.inputs.ldb;

    if (GRF::bytes(hw) == 64) {
        if (!isRegisterColMajor(Ta_ext, problem.A, strategy.A) && strategy.allowDoubleMasking(LoopM)) {
            int ka = strategy.slmA ? strategy.unrollKSLM : strategy.ka_load;
            int effAlign = isBlocklike(strategy.A.accessType) ? problem.A.alignment : 1;
            state.simd32KMasks |= (ka >= 32 * (std::min<int>(4, effAlign) / std::min(4, Ta_ext.paddedSize())));
        }
        if (isRegisterColMajor(Tb_ext, problem.B, strategy.B) && strategy.allowDoubleMasking(LoopN)) {
            int kb = strategy.slmB ? strategy.unrollKSLM : strategy.kb_load;
            int effAlign = isBlocklike(strategy.B.accessType) ? problem.B.alignment : 1;
            state.simd32KMasks |= (kb >= 32 * (std::min<int>(4, effAlign) / std::min(4, Tb_ext.paddedSize())));
        }
    }

    if (state.useTempC) {
        // TODO: consider remainder handling to reduce BW.
        bool cColMajor = isRegisterColMajor(problem.Tc_ext, problem.C, strategy.C);
        state.tempC.layout = cColMajor ? MatrixLayout::Pc : MatrixLayout::Pr;
        state.tempC.crosspack = 1;
        state.tempC.packSize = strategy.unroll[cColMajor ? LoopM : LoopN];
        state.tempC.panelLength = 0;
        state.tempC.setAlignment(64);

        state.tempCStrategy = strategy.C;
        state.tempCStrategy.accessType = AccessType::Block;
        state.tempCStrategy.address2D = false;
        state.tempCStrategy.padded = true;
    }
}

#include "internal/namespace_end.hxx"
