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


#include "compute_utils.hpp"
#include "generator.hpp"
#include "hw_utils.hpp"
#include "layout_utils.hpp"
#include "map.hpp"
#include "ngen_object_helpers.hpp"

using namespace ngen;
using namespace ngen::utils;
using std::vector;

#include "internal/namespace_start.hxx"


// Do one or more outer products (k = 1 slices) of A*B, updating C.
//  ha and hb are the k indices within the A and B chunks, respectively.
//  A_copy, B_copy are the indices of the A, B copies to use.
template <HW hw>
void BLASKernelGenerator<hw>::outerProduct(int h, int ha, int hb, int opCount, bool rem,
                                           const vector<RegisterBlock> &A_layout, const vector<RegisterBlock> &B_layout,
                                           const GRFMultirange &A_regs, const GRFMultirange &B_regs,
                                           const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    if (strategy.dotVL)
        innerProductFMA(h, ha, hb, opCount, A_layout, B_layout, A_regs, B_regs, problem, strategy, state);
    else if (strategy.systolic)
        outerProductSystolic(h, ha, hb, opCount, rem, A_layout, B_layout, A_regs, B_regs, problem, strategy, state);
    else if (hw < HW::Gen12LP && problem.isIGEMM())
        outerProductGen9IGEMM(ha, hb, A_layout, B_layout, A_regs, B_regs, problem, strategy, state);
    else
        outerProductFMA(h, ha, hb, opCount, A_layout, B_layout, A_regs, B_regs, problem, strategy, state);
}

// FMA or dp4a-based outer product implementation.
template <HW hw>
void BLASKernelGenerator<hw>::outerProductFMA(int h, int ha, int hb, int opCount, const vector<RegisterBlock> &A_layout, const vector<RegisterBlock> &B_layout,
                                              const GRFMultirange &A_regs, const GRFMultirange &B_regs,
                                              const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto Ta = problem.Ta, Tb = problem.Tb, Tc = problem.Tc_compute();

    bool mixedMode = ((Tc.real() == Type::f32) && (Ta.real() != Type::f32 || Tb.real() != Type::f32));
    bool useDP4A = (hw >= HW::Gen12LP && problem.isIGEMM());
    bool atomicFMA = strategy.atomicFMA;
    bool noAccSBSet = strategy.extendedAtomicFMA && (hw >= HW::XeHPC);

    int minOPCount = minOuterProductCount(hw, problem, strategy);
    int kChain = opCount / minOPCount;
    int aCP, bCP;
    std::tie(aCP, bCP) = targetKernelCrosspack(hw, problem, strategy);

    int accNum = 0;
    Subregister Clast;
    int nec = elementsPerGRF(hw, Tc);
    bool globalCM = isLayoutColMajor(state.C_layout);
    int fmaSIMD = strategy.fmaSIMD;

    if (!state.Cr_layout.empty()) stub();       /* No repacking support currently */

    bool csplit = false, mixedRC = false;
    int icompCount = 1, ocompCount = 1, ivcompCount = 1, ovcompCount = 1;

    bool bfloat16WA = (Tc.real() == Type::f32) && ((globalCM ? Tb : Ta).real() == Type::bf16);

    // Emit an FMA instruction.
    auto outputFMA = [&](InstructionModifier mod, const Subregister &A, const Subregister &B, const Subregister &C, const RegData &bcastSrc, bool colMajor, int hh, bool ivfirst, bool ivlast) {
        auto Cacc = AccumulatorRegister(accNum).sub(0, Tc.real().ngen());
        auto Csrc = (hh == 0                    && ivfirst) ? C : Cacc;
        auto Cdst = (hh == opCount - minOPCount && ivlast)  ? C : Cacc;
        bool thisNoAccSBSet = (noAccSBSet && Cacc == Csrc);
        if (thisNoAccSBSet) {
            mod |= NoAccSBSet;
            setDefaultAutoSWSB(false);
        }
        if (A.isInvalid() || B.isInvalid()) {
            if (Csrc != Cdst)
                mov(mod, Cdst(1), Csrc(1));
        } else if (useDP4A) {
            auto Ar = A.reinterpret(0, isSigned(A.getType()) ? DataType::d : DataType::ud);
            auto Br = B.reinterpret(0, isSigned(B.getType()) ? DataType::d : DataType::ud);

            colMajor           ? dp4a(mod, Cdst(1), Csrc(1), Ar(1), Br(0))
                               : dp4a(mod, Cdst(1), Csrc(1), Br(1), Ar(0));
        } else if (C.isARF() && hw < HW::XeHP) {
            colMajor           ? mac(mod, C(1), A(1), bcastSrc)
                               : mac(mod, C(1), bcastSrc, B(1));
        } else {
            // On Gen12, always put broadcast in src2 for better bank conflict avoidance.
            colMajor           ? mad(mod, Cdst(1), Csrc(1), A(1), bcastSrc) :
            (hw < HW::Gen12LP) ? mad(mod, Cdst(1), Csrc(1), bcastSrc, B(1)) :
                                 mad(mod, Cdst(1), Csrc(1), B(1), bcastSrc);
        }
        if (thisNoAccSBSet)
            setDefaultAutoSWSB(true);
    };

    ha = align_down(ha, opCount);
    hb = align_down(hb, opCount);

    if (atomicFMA && hw >= HW::XeHPC) {
        // PVC onward can't use {Atomic} on float pipes.
        // Use wrdep to wait on all outstanding A/B dependencies instead.
        atomicFMA = false;
        for (auto &block: A_layout)
            if (block.offsetC >= ha && block.offsetC < ha + opCount)
                wrdep(A_regs[block.offsetReg()]);
        for (auto &block: B_layout)
            if (block.offsetR >= hb && block.offsetR < hb + opCount)
                wrdep(B_regs[block.offsetReg()]);
    }

    // Decide whether to loop in column or row major order.
    //   x = vectorized dimension
    //   y = non-vectorized dimension
    int nx = globalCM ? strategy.unroll[LoopM] : strategy.unroll[LoopN];
    int ny = globalCM ? strategy.unroll[LoopN] : strategy.unroll[LoopM];
    int nx1 = (mixedMode || state.broadcast) ? nx : fmaSIMD;

    // Prepare for chaining FMAs through accumulator registers.
    int necAcc = nec * (csplit ? 2 : 1);
    int accCount = AccumulatorRegister::count(hw, strategy.GRFs, Tc.ngen());
    int accPerFMA = div_up(std::min(nx, fmaSIMD), necAcc);
    int minAccPerFMA = Tc.isFP() ? 1 : 2;
    accPerFMA = std::max(accPerFMA, minAccPerFMA);
    int independentAccs = div_up(accCount, accPerFMA);

    int nx1i = 1, ny1 = 1;
    if (kChain > 1) {
        if (independentAccs < icompCount) hw_unsupported();
        int indepAccComp = div_up(independentAccs, icompCount);

        nx1i = std::min(nx1, indepAccComp * fmaSIMD);
        ny1 = div_up(indepAccComp, div_up(nx1i, fmaSIMD));

        noAccSBSet &= Tc.isFP()
                   && (div_up(nx1i, necAcc) * ny1 * icompCount >= 8)
                   && (nx % nx1i == 0);     /* {NoAccSBSet} requires all 8 accumulator registers to be used */
    } else
        noAccSBSet = false;

    GRFRange broadcastRegs = state.broadcast_regs;
    Subregister lastBcastBase;

    // Last A/B blocks found.
    const RegisterBlock *A_blockLast = nullptr, *B_blockLast = nullptr;

    for (int x0 = 0; x0 < nx; x0 += nx1) {
    for (int ovcomp = 0; ovcomp < ovcompCount; ovcomp++) {
    for (int ocomp = 0; ocomp < ocompCount; ocomp++) {
    for (int y0 = 0; y0 < ny; y0 += ny1) {
    for (int x1 = 0; x1 < nx1 && (x0 + x1) < nx;) {
        int x1New = x1;
        for (int ivcomp = 0; ivcomp < ivcompCount; ivcomp++) {
        for (int hh = 0; hh < opCount; hh += minOPCount) {
            accNum = 0;
            for (int y1 = 0; y1 < ny1 && y0 + y1 < ny; y1++) {
            for (int x1i = x1; (x1i < x1 + nx1i) && (x0 + x1i < nx);) {
                auto x = x0 + x1i;
                auto y = y0 + y1;
                auto i = globalCM ? x : y;
                auto j = globalCM ? y : x;
                auto hha = ha + hh;
                auto hhb = hb + hh;
                auto ia = i, jb = j;

                int fmaCount = 1;

                for (int icomp = 0; icomp < icompCount; icomp++) {
                    // Find the appropriate A and B registers.
                    int na, nb;
                    int vcomp = ivcomp + ovcomp;
                    int ncomp = (vcomp ^ ocomp) + icomp;
                    int compA = globalCM ? vcomp : ncomp;
                    int compB = globalCM ? ncomp : vcomp;

                    const RegisterBlock *A_block, *B_block;
                    Subregister A = findBlockReg(Ta, A_layout, ia, hha, A_regs, na, A_block, compA);
                    Subregister B = findBlockReg(Tb, B_layout, hhb, jb, B_regs, nb, B_block, compB);

                    // Check for expected crosspack.
                    if (globalCM ? (aCP && A_block->crosspack != aCP)
                                 : (bCP && B_block->crosspack != bCP)) stub();

                    // Check if we should specify {Atomic}.
                    bool atomic = (atomicFMA && (A_block == A_blockLast) && (B_block == B_blockLast));
                    A_blockLast = A_block;
                    B_blockLast = B_block;

                    // Find the appropriate C register.
                    int C_buffer = csplit ? 0 : (icomp + ocomp);
                    int compC    = csplit ? ocomp : 0;
                    int nc;
                    const RegisterBlock *C_block;
                    Subregister C = findBlockReg(Tc, state.C_layout, i, j, state.C_regs[C_buffer], nc, C_block, compC);
                    if (C_block->crosspack > 1) stub();

                    // Swap out C register for an accumulator, if necessary.
                    if (strategy.cAccumulators) {
                        auto C_roff = C.getBase() - state.C_regs[0].ranges[0].getBase();
                        if (C_roff < state.C_accCount)
                            C = AccumulatorRegister(C_roff).sub(C.getOffset(), Tc.ngen());
                    }

                    InstructionModifier mod;

                    // Use requested execution size if possible, but limited to available elements.
                    // Decide on kernel type based on register block layouts.
                    bool canColMajor = (A_block->colMajor && globalCM);
                    bool canRowMajor = (!B_block->colMajor && !globalCM);
                    bool colMajor = globalCM;

                    if (!canColMajor && !canRowMajor)
                        fmaCount = 1;
                    else if (canColMajor)
                        fmaCount = rounddown_pow2(std::min({fmaSIMD, na, nc}));
                    else
                        fmaCount = rounddown_pow2(std::min({fmaSIMD, nb, nc}));

                    int simdSize = fmaCount;

                    // Crosspacked kernels: ensure broadcast matrix is contiguous in k.
                    if (minOPCount > 1) {
                        bool nativeDir = (globalCM ? B_block->colMajor : !A_block->colMajor);
                        auto bcastCrosspack = (globalCM ? B_block : A_block)->crosspack;
                        if (nativeDir) {
                            if ((globalCM ? nb : na) < minOPCount) stub();
                            if (bcastCrosspack > 1) stub();
                        } else {
                            if (bcastCrosspack % minOPCount) stub();
                        }
                    }

                    // Add Atomic if appropriate.
                    if (atomic) mod |= Atomic;

                    // Handle broadcast duties.
                    Subregister bcastSrcSub = colMajor ? B : A;
                    RegData bcastSrc = bcastSrcSub;

                    if (state.broadcast) {

                        // Broadcast if necessary: pair of doubles (doubleWA) or single elements.
                        int nbcast = strategy.doubleWA ? 2 : 1;
                        int hs = strategy.doubleWA ? 0 : nbcast;

                        auto bcastType = bcastSrc.getType();
                        Subregister bcastBase = bcastSrcSub;
                        bcastBase.setOffset(bcastBase.getOffset() & ~(nbcast - 1));

                        if (bcastBase != lastBcastBase) {
                            auto bcastRegion = bcastBase(0, nbcast, (nbcast > 1) ? 1 : 0);
                            if (bfloat16WA) {
                                // Upconvert to f32 during broadcast.
                                bcastRegion.setType(DataType::uw);
                                shl(simdSize, broadcastRegs[0].ud(), bcastRegion, 16);
                            } else {
                                moveToIntPipe(simdSize, bcastRegion);
                                mov(simdSize * bcastSrc.getBytes() / bcastRegion.getBytes(),
                                    broadcastRegs[0].retype(bcastRegion.getType()), bcastRegion);
                            }
                        }
                        if (bfloat16WA) bcastType = DataType::f;
                        bcastSrc = broadcastRegs[0].sub(bcastSrc.getOffset() & (nbcast - 1), bcastType)(hs);
                        lastBcastBase = bcastBase;

                    }

                    bool ivfirst = mixedRC || (ivcomp == 0);
                    bool ivlast  = mixedRC || (ivcomp == ivcompCount - 1);

                    // Finally, perform the long-awaited FMA.
                    outputFMA(simdSize | mod, A, B, C, bcastSrc, colMajor, hh, ivfirst, ivlast);
                    Clast = C;

                    if (kChain > 1 && accNum >= accCount) stub();
                    accNum += std::max(minAccPerFMA, div_up(fmaCount, necAcc));
                } /* icomp */

                x1i += fmaCount;
                x1New = x1i;
            } /* x1i */
            } /* y1 */
        } /* hh */
        } /* ivcomp */
        x1 = x1New;
    } /* x1 */
    } /* y0 */
    } /* ocomp */
    } /* ovcomp */
    } /* x0 */
}

// FMA-based inner product (dot) implementation.
template <HW hw>
void BLASKernelGenerator<hw>::innerProductFMA(int h, int ha, int hb, int opCount, const vector<RegisterBlock> &A_layout, const vector<RegisterBlock> &B_layout,
                                              const GRFMultirange &A_regs, const GRFMultirange &B_regs,
                                              const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto Ta = problem.Ta, Tb = problem.Tb, Tc = problem.Tc_compute();
    int vl = strategy.dotVL;
    const auto &C_layout = state.C_layout;
    const auto &C_regs = state.C_regs[0];

    if (!state.Cr_layout.empty()) stub();       /* No repacking support currently */
    if (strategy.kChain > 1) stub();

    bool globalCM = isLayoutColMajor(state.C_layout);
    ha = align_down(ha, opCount);
    hb = align_down(hb, opCount);

    int nx = globalCM ? strategy.unroll[LoopM] : strategy.unroll[LoopN];
    int ny = globalCM ? strategy.unroll[LoopN] : strategy.unroll[LoopM];

    for (int y0 = 0; y0 < ny; y0++) {
    for (int x0 = 0; x0 < nx; x0++) {
    for (int h0 = 0; h0 < vl; ) {
        auto x0c = x0 * vl + h0;
        auto i0 = globalCM ? x0 : y0, i0c = globalCM ? x0c : y0;
        auto j0 = globalCM ? y0 : x0, j0c = globalCM ? y0 : x0c;

        const RegisterBlock *A_block, *B_block, *C_block;
        int na, nb, nc;
        auto A = findBlockReg(Ta, A_layout, i0, h0 + ha, A_regs, na, A_block);
        auto B = findBlockReg(Tb, B_layout, h0 + hb, j0, B_regs, nb, B_block);
        auto C = findBlockReg(Tc, C_layout, i0c, j0c, C_regs, nc, C_block);

        if (A_block->colMajor || !B_block->colMajor) stub();

        int ne = std::min({na, nb, nc, vl, strategy.fmaSIMD});

        mad(ne, C(1), C(1), A(1), B(1));

        h0 += ne;
    } /* h0 */
    } /* x0 */
    } /* y0 */
}

// Gen9 IGEMM outer product implementation.
template <HW hw>
void BLASKernelGenerator<hw>::outerProductGen9IGEMM(int ha, int hb, const vector<RegisterBlock> &A_layout, const vector<RegisterBlock> &B_layout,
                                                    const GRFMultirange &A_regs, const GRFMultirange &B_regs,
                                                    const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto Ta = problem.Ta, Tb = problem.Tb, Tc = problem.Tc_compute();
    DataType tempType = (Ta.isSigned() || Tb.isSigned()) ? DataType::w : DataType::uw;

    struct AddItem {
        int simd;
        RegData dest, src0, src1;
    };
    std::vector<AddItem> adds;

    auto replayAdds = [&]() {
        for (auto &item : adds)
            add(item.simd, item.dest, item.src0, item.src1);
        adds.clear();
    };

    bool globalCM = isLayoutColMajor(state.C_layout);

    if (!state.Cr_layout.empty()) stub();       /* No repacking support currently */

    // Decide whether to loop in column or row major order.
    int nx = globalCM ? strategy.unroll[LoopM] : strategy.unroll[LoopN];
    int ny = globalCM ? strategy.unroll[LoopN] : strategy.unroll[LoopM];

    int tidx = 0;
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx;) {
            auto i = globalCM ? x : y;
            auto j = globalCM ? y : x;

            int fmaCount;

            // Find the appropriate A and B registers.
            int na, nb;
            const RegisterBlock *A_block, *B_block;
            Subregister A = findBlockReg(Ta, A_layout, i, ha, A_regs, na, A_block);
            Subregister B = findBlockReg(Tb, B_layout, hb, j, B_regs, nb, B_block);

            // Find the appropriate C register. Todo: remainders.
            int nc;
            const RegisterBlock *C_block;
            Subregister C = findBlockReg(Tc, state.C_layout, i, j, state.C_regs[0], nc, C_block);

            // No C crosspack support.
            auto cpA = A_block->crosspack, cpB = B_block->crosspack;
            if (C_block->crosspack > 1) stub();

            // Swap out C register for an accumulator, if necessary.
            auto C_roff = C.getBase() - state.C_regs[0].ranges[0].getBase();
            if (C_roff < state.C_accCount)
                C = AccumulatorRegister(C_roff).sub(C.getOffset(), Tc.ngen());

            // Use requested execution size if possible, but limited to available elements.
            // Decide the kernel type based on register block layouts.
            bool canColMajor = (A_block->colMajor && C_block->colMajor);
            bool canRowMajor = (!B_block->colMajor && !C_block->colMajor);
            bool colMajor;

            if (!canColMajor && !canRowMajor) {
                colMajor = true;
                fmaCount = 1;
            } else if (canColMajor) {
                colMajor = true;
                fmaCount = na;
            } else {
                colMajor = false;
                fmaCount = nb;
            }
            fmaCount = rounddown_pow2(std::min({strategy.fmaSIMD, nc, elementsPerGRF<int16_t>(hw)}));

            auto temp = state.tempMul_regs[tidx++];

            if (C.isARF()) {
                if (colMajor)
                    mac(fmaCount, C(1), A(cpA), B(0));
                else
                    mac(fmaCount, C(1), A(0), B(cpB));
            } else {
                if (colMajor)
                    mul(fmaCount, temp[0].sub(0, tempType)(2), A(cpA), B(0));
                else
                    mul(fmaCount, temp[0].sub(0, tempType)(2), A(0), B(cpB));

                adds.push_back({fmaCount, C(1), C(1), temp[0].sub(0, tempType)(2)});
            }

            if (tidx >= int(state.tempMul_regs.size())) {
                tidx = 0;
                replayAdds();
            }

            x += fmaCount;
        }
    }

    replayAdds();

    // A4B4 outer product (4 temporary GRFs per 2 C registers) - 2/3 SP
    //
    // mul (32) temp0.0:w<1> A.0:b<32;16,2> B.0:b<32;16,2>   - EM
    // mul (32) temp2.0:w<1> A.1:b<32;16,2> B.1:b<32;16,2>   - FPU
    // add (16) C.0:d<1> C.0:d<8;8,1> temp0.0:w<16;8,2>      - EM
    // add (16) C.0:d<1> C.0:d<8;8,1> temp0.1:w<16;8,2>      - FPU
    // add (16) C.0:d<1> C.0:d<8;8,1> temp2.0:w<16;8,2>      - EM
    // add (16) C.0:d<1> C.0:d<8;8,1> temp2.1:w<16;8,2>      - FPU

    // Faster A4B4 outer product a la non-VNNI (4 temporary GRFs per 2 C registers) - 4/5 SP
    //
    // mul (32) temp0.0:w<1> A.0:b<32;16,2> B.0:b<32;16,2>   - EM
    // mul (32) temp2.0:w<1> A.1:b<32;16,2> B.1:b<32;16,2>   - FPU
    // add (32) (sat) temp0.0:w<1> temp0.0:w<1> temp2.0:w<1> - EM/FPU
    // add (16) C.0:d<1> C.0:d<8;8,1> temp0.0:w<16;8,2>      - EM
    // add (16) C.0:d<1> C.0:d<8;8,1> temp0.1:w<16;8,2>      - FPU
}

// Accumulate multiple outer products using the systolic array.
template <HW hw>
void BLASKernelGenerator<hw>::outerProductSystolic(int h, int ha, int hb, int opCount, bool rem,
                                                   const vector<RegisterBlock> &A_layout, const vector<RegisterBlock> &B_layout,
                                                   const GRFMultirange &A_regs, const GRFMultirange &B_regs,
                                                   const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto Ta = problem.Ta, Tb = problem.Tb, Tc = problem.Tc_compute();
    bool globalCM = isLayoutColMajor(state.C_layout);
    auto params = systolicParams(hw, problem, strategy);
    auto ksys = params.ksys;
    auto osys = params.osys;
    auto sdepth = params.sdepth;
    auto rcountMax = params.rcountMax;
    int dpaswTile = strategy.dpasw ? (globalCM ? strategy.B.tileC : strategy.A.tileR) : 0;
    bool rsFix = (strategy.readSuppressionWA && hasMasking(globalCM ? A_layout : B_layout));
    bool canAtomicNon8x8 = (hw >= HW::XeHPC);
    bool sum = globalCM ? state.systolicSumA
                        : state.systolicSumB;
    bool snake = strategy.fmaBoustrophedon && !sum;
    bool repackC = !state.Cr_layout.empty();
    bool startRepackC = false, endRepackC = false;
    int Cr_unrollM = 0, Cr_unrollN = 0;
    if (repackC) {
        getLayoutDims(state.Cr_layout, Cr_unrollM, Cr_unrollN);
        auto rphase = align_down(h, opCount) % state.cRepackPeriod;
        startRepackC = rem || (rphase == 0);
        endRepackC = rem || (rphase + opCount >= state.cRepackPeriod);
    }

    RegisterBlock sumBlock;
    sumBlock.colMajor = globalCM;
    sumBlock.crosspack = 1;

    ha = align_down(ha, opCount);
    hb = align_down(hb, opCount);

    // Decide whether to loop in column or row major order, to facilitate macro sequences.
    //  x is the non-accumulating dimension of dpas src1 (V matrix)
    //  y is the non-accumulating dimension of dpas src2 (N matrix)
    int nx = strategy.unroll[globalCM ? LoopM : LoopN];
    int ny = strategy.unroll[globalCM ? LoopN : LoopM];

    int xinc = osys;

    const int compA = 0, compB = 0;
    const int incompCount = 1, oncompCount = 1;

    // Split y loop for k-chaining or boustrophedon ordering.
    bool reverse = false;
    int ny1 = 1;
    if (opCount > ksys || snake)
        ny1 = rcountMax * (strategy.dpasw ? 2 : 1);

    for (int x = 0; x < nx; x += xinc) {
        Subregister A0, B0, C0;
        int rcount = 0, ybase = 0, hhbase = 0;

        auto issueDPAS = [&](bool last) {
            InstructionModifier mod = osys;

            bool useDPASW = strategy.dpasw && ybase < ny;
            auto rc = rcount * (useDPASW ? 2 : 1);
            auto V0 = globalCM ? A0 : B0;
            auto N0 = globalCM ? B0 : A0;
            RegData srcC0 = C0;

            if (rsFix) {
                GRF v0GRF{V0.getBase()};
                mov<uint32_t>(8, v0GRF, v0GRF);
                rsFix = false;
            }

            if (strategy.atomicFMA) {
                if (!last && (rc == 8 || canAtomicNon8x8))
                    mod |= Atomic;
                if (rc != 8 && strategy.extendedAtomicFMA) hw_unsupported();
            }

            if (startRepackC && hhbase == 0)
                srcC0 = null.retype(C0.getType());

            useDPASW ? dpasw(mod, sdepth, rc, C0, srcC0, V0, N0)
                     :  dpas(mod, sdepth, rc, C0, srcC0, V0, N0);
        };

        for (int oncomp = 0; oncomp < oncompCount; oncomp++, reverse = !reverse) {
        for (int y0_ = 0; y0_ < ny + sum; y0_ += ny1) {
            int y0 = (snake && reverse) ? (ny - ny1 - y0_) : y0_;
            for (int hh = 0; hh < opCount; hh += ksys) {
            for (int y1 = 0; y1 < ny1 && y0 + y1 < ny + sum; y1++) {
            for (int incomp = 0; incomp < incompCount; incomp++) {
                int y = y0 + y1;
                int hha = ha + hh, hhb = hb + hh;

                // Find the appropriate A and B registers.
                int na, nb, nc;
                const RegisterBlock *A_block, *B_block, *C_block;
                Subregister A, B, C;

                const int cxCompA = -1, cxCompB = -1, cxCompC = -1, cBuffer = 0;

                if (y < ny) {
                    if (strategy.dpasw && (y % (2 * dpaswTile) >= dpaswTile))
                        continue;

                    int i = globalCM ? x : y;
                    int j = globalCM ? y : x;

                    A = findBlockReg(Ta, A_layout, i, hha, A_regs, na, A_block, cxCompA, compA);
                    B = findBlockReg(Tb, B_layout, hhb, j, B_regs, nb, B_block, cxCompB, compB);
                    if (repackC)
                        C = findBlockReg(Tc, state.Cr_layout, i % Cr_unrollM, j % Cr_unrollN, state.Cr_regs, nc, C_block, cxCompC);
                    else
                        C = findBlockReg(Tc, state.C_layout, i, j, state.C_regs[cBuffer], nc, C_block, cxCompC);
                } else if (state.systolicSumA) {
                    A = findBlockReg(Ta, A_layout, x, hha, A_regs, na, A_block);
                    B = state.sysSumAll1s[0];
                    nb = elementsPerGRF(hw, Tb);
                    B_block = &sumBlock;
                    if (repackC)
                        C = findBlockReg(Tc, state.Asr_layout, x % Cr_unrollM, 0, state.Asr_regs, nc, C_block);
                    else
                        C = findBlockReg(Tc, state.As_layout, x, 0, state.As_regs, nc, C_block);
                } else {
                    A = state.sysSumAll1s[0];
                    na = elementsPerGRF(hw, Ta);
                    A_block = &sumBlock;
                    B = findBlockReg(Tb, B_layout, hhb, x, B_regs, nb, B_block);
                    if (repackC)
                        C = findBlockReg(Tc, state.Bsr_layout, 0, x % Cr_unrollN, state.Bsr_regs, nc, C_block);
                    else
                        C = findBlockReg(Tc, state.Bs_layout, 0, x, state.Bs_regs, nc, C_block);
                }

                int nv = globalCM ? na : nb;
                int nn = globalCM ? nb : na;

                // Verify DPAS requirements.
                if (globalCM) {
                    if (A_block->crosspack * Ta.real() != std::max(4, Ta.real().paddedSize())) stub();
                    if (B_block->crosspack > 1) stub();
                } else {
                    if (B_block->crosspack * Tb.real() != std::max(4, Tb.real().paddedSize())) stub();
                    if (A_block->crosspack > 1) stub();
                }
                if (A_block->colMajor != globalCM || B_block->colMajor != globalCM) stub();
                if (C_block->crosspack > 1) stub();

                if (nv != osys) stub();
                if (nn < ksys) stub();

                // Check if current DPAS can be fused with the previous one.
                bool chain = false;
                if (A0.isValid()) {
                    chain = globalCM ? (elementDiff(hw, B, B0) == (y - ybase) * ksys)
                                     : (elementDiff(hw, A, A0) == (y - ybase) * ksys);
                    chain = chain && (elementDiff(hw, C, C0) == (y - ybase) * osys);
                    chain = chain && (rcount < rcountMax);
                    chain = chain && (hh == hhbase);
                    if (strategy.dpasw)
                        chain = chain && y < ny && (y % (2 * dpaswTile) > 0);
                }

                if (chain)
                    rcount++;
                else {
                    if (strategy.dpasw && y < ny && rcount > 0 && rcount != dpaswTile) stub();
                    if (A0.isValid()) issueDPAS(false);
                    A0 = A; B0 = B; C0 = C; rcount = 1;
                    A0.setType(Ta.ngen());
                    B0.setType(Tb.ngen());
                    C0.setType(Tc.ngen());
                    ybase = y;
                    hhbase = hh;
                }
            } /* incomp loop */
            } /* y1 loop */
            } /* hh loop */
        } /* y loop */
        } /* oncomp loop */

        bool finishChain = !strategy.extendedAtomicFMA || (x + osys >= nx) || (repackC && x >= (Cr_unrollM - 2*xinc));
        issueDPAS(finishChain);

        if (endRepackC) {
            int xr = x - Cr_unrollM + xinc;
            if (xr >= 0)
                outerProductRepackC(xr, xr % Cr_unrollM, xinc, h, problem, strategy, state);
        }
    } /* x loop */

    if (endRepackC) for (int xr = nx - Cr_unrollM + xinc; xr < nx; xr += xinc)
        outerProductRepackC(xr, xr % Cr_unrollM, xinc, h, problem, strategy, state);

}

// Repack (part of) a C tile, converting types and scaling as needed.
//  x0 (resp. xr0) are the offsets into the C tile (resp. repacked C tile) in the vectorized dimension.
template <HW hw>
void BLASKernelGenerator<hw>::outerProductRepackC(int x0, int xr0, int nx, int h,
                                                  const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto Tc = problem.Tc, Tc_compute = problem.Tc_compute();
    const auto &C_layout = state.C_layout, &Cr_layout = state.Cr_layout;
    const auto &C_regs = state.C_regs[0], &Cr_regs = state.Cr_regs;
    int nec = elementsPerGRF(hw, Tc_compute);
    bool globalCM = isLayoutColMajor(C_layout);
    bool scaleA = state.lateScale2DA, scaleB = state.lateScale2DB;

    bool sumA = problem.needsASums();
    bool sumB = problem.needsBSums();
    if (globalCM ? sumB : sumA) stub();

    if (Tc.size() != Tc_compute.size()) stub();
    if (state.C_buffers > 1) stub();

    int ny = strategy.unroll[globalCM ? LoopN : LoopM];
    int nacc = AccumulatorRegister::count(hw, strategy.GRFs, Tc.real().ngen());

    struct WorkItem {
        Subregister C, Cr;
        int simd, iacc;
        std::array<Subregister, 2> scale;
        std::array<int, 2> scaleStride;

        RegData acc(Type T) const {
            return AccumulatorRegister(iacc).retype(T.ngen());
        }
    };

    std::vector<WorkItem> items;
    items.reserve(nacc);

    auto processItems = [&] {
        for (const auto &i: items)
            mov(i.simd, i.acc(Tc), i.Cr(1));
        for (const auto &i : items)
            if (i.scale[1].isValid())
                mul(i.simd, i.acc(Tc), i.acc(Tc), i.scale[1](i.scaleStride[1]));
        for (const auto &i: items) {
            if (i.scale[0].isValid())
                mad(i.simd, i.C(1), i.C(1), i.acc(Tc), i.scale[0](i.scaleStride[0]));
            else
                add(i.simd, i.C(1), i.C(1), i.acc(Tc));
        }
        items.clear();
    };

    int iacc = 0;

    for (int x1 = 0; x1 < nx; x1 += 2 * nec) {
        int x = x0 + x1, xr = xr0 + x1;
        int xchunk = std::min(nx - x1, 2 * nec);
        for (int y = 0; y < ny + sumA + sumB; y++) {
            auto i = globalCM ? x : y;
            auto j = globalCM ? y : x;
            auto ir = globalCM ? xr : y;
            auto jr = globalCM ? y : xr;

            int ne = 0, ner = 0, nes[2] = {0, 0};
            const RegisterBlock *C_block = nullptr, *Cr_block = nullptr;
            const RegisterBlock *sblock = nullptr;
            Subregister C, Cr;

            bool doASum = sumA && y == ny;
            bool doBSum = sumB && y == ny;

            if (y < ny) {
                C  = findBlockReg(Tc, C_layout, i, j, C_regs, ne, C_block);
                Cr = findBlockReg(Tc_compute, Cr_layout, ir, jr, Cr_regs, ner, Cr_block);
            } else if (doASum) {
                C  = findBlockReg(Tc, state.As_layout, x, 0, state.As_regs, ne, C_block);
                Cr = findBlockReg(Tc_compute, state.Asr_layout, xr, 0, state.Asr_regs, ner, Cr_block);
            } else if (doBSum) {
                C  = findBlockReg(Tc, state.Bs_layout, 0, x, state.Bs_regs, ne, C_block);
                Cr = findBlockReg(Tc_compute, state.Bsr_layout, 0, xr, state.Bsr_regs, ner, Cr_block);
            }

            std::array<Subregister, 2> scale;
            std::array<int, 2> scaleStride = {0, 0};
            int nscale = 0;
            if (scaleA && !doBSum) {
                // TODO: handle non-repacked case correctly.
                int js = ((jr + h) / problem.aqGroupK) % state.kaqLate;
                scale[nscale] = findBlockReg(state.Ta_scaleInt, state.Ar_scaleLayout,
                                             i, js, state.Ar_scaleRegs, nes[0], sblock);
                scaleStride[nscale] = globalCM ? 1 : 0;
                nscale++;
            }
            if (scaleB && !doASum) {
                int is = ((ir + h) / problem.bqGroupK) % state.kbqLate;
                scale[nscale] = findBlockReg(state.Tb_scaleInt, state.Br_scaleLayout,
                                             is, j, state.Br_scaleRegs, nes[1], sblock);
                scaleStride[nscale] = globalCM ? 0 : 1;
                nscale++;
            }

            ne = std::min({ne, ner, xchunk});
            if (scaleStride[0] == 1) ne = std::min(ne, nes[0]);
            if (scaleStride[1] == 1) ne = std::min(ne, nes[1]);

            if (ne < xchunk) stub();
            if ((C_block && C_block->crosspack != 1)
                    || (Cr_block && Cr_block->crosspack != 1)) stub();

            WorkItem item = {C, Cr, ne, iacc, scale, scaleStride};
            bool coalesce = false;

            if (!items.empty()) {
                auto &last = items.back();
                coalesce = true;
                coalesce &= (ne == nec && last.simd == nec);
                coalesce &= (C.getBase() == last.C.getBase() + 1);
                coalesce &= (Cr.getBase() == last.Cr.getBase() + 1);
                for (int i = 0; i < 2; i++) if (scale[i].isValid()) {
                    if (scaleStride[i] == 0)
                        coalesce &= (scale[i] == last.scale[i]);
                    else {
                        coalesce &= (scale[i].getBase() == last.scale[i].getBase() + 1)
                                 && (getBytes(scale[i].getType()) == Tc.size());
                    }
                }
            }

            if (coalesce)
                items.back().simd += ne;
            else
                items.push_back(item);

            iacc += (ne > xchunk) ? 2 : 1;
            if (iacc >= nacc) {
                processItems();
                iacc = 0;
            }
        }
    }

    processItems();
}

#include "internal/namespace_end.hxx"
