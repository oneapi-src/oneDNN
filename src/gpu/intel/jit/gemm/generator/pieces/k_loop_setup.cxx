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
#include "generator.hpp"
#include "kernel_queries.hpp"
#include "layout_utils.hpp"
#include "state_utils.hpp"

using namespace ngen;
using namespace ngen::utils;
using std::vector;

#include "internal/namespace_start.hxx"


static void makeAiBiKCloneLayout(HW hw, bool isA, vector<RegisterBlock> &Xi_layout, vector<vector<RegisterBlock>> &Xi_layoutK,
                                 vector<GRFMultirange> &Xi_regsRem, int kx_slm,
                                 const GEMMStrategy &strategy, GEMMState &state);

// Prepare for inner loop. Returns true on success.
template <HW hw>
bool BLASKernelGenerator<hw>::kLoopSetup(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto Ta = problem.Ta, Tb = problem.Tb;
    auto Ta_ext = problem.Ta_ext, Tb_ext = problem.Tb_ext;
    auto Ta_load = state.Ta_load, Tb_load = state.Tb_load;

    auto minOPCount = minOuterProductCount(hw, problem, strategy);
    auto unrollM = strategy.unroll[LoopM];
    auto unrollN = strategy.unroll[LoopN];

    state.barrierReady = false;

    // Get A/B named barrier IDs and prepare barrier message headers.
    // Flag registers have already been allocated.
    auto &barrierHeaderM = state.barrierHeaderM;
    auto &barrierHeaderN = state.barrierHeaderN;
    auto &barrierM = state.barrierM;
    auto &barrierN = state.barrierN;
    bool nbM = strategy.needsNamedBarriersM(problem);
    bool nbN = strategy.needsNamedBarriersN(problem);

    if (nbM) {
        barrierHeaderM = state.ra.alloc();
        if (!is_zero_or_pow2(strategy.wg[LoopM]) || !is_zero_or_pow2(strategy.namedBarriers[LoopM])) stub();
        shr(1, barrierHeaderM.uw(4), state.lidM, ilog2(strategy.wg[LoopM]) - ilog2(strategy.namedBarriers[LoopM]));
    }
    if (nbN) {
        barrierHeaderN = state.ra.alloc();
        if (!is_zero_or_pow2(strategy.wg[LoopN]) || !is_zero_or_pow2(strategy.namedBarriers[LoopN])) stub();
        shr(1, barrierHeaderN.uw(4), state.lidN, ilog2(strategy.wg[LoopN]) - ilog2(strategy.namedBarriers[LoopN]));
    }
    if (nbM) {
        int threadsPerMBar = strategy.wg[LoopM] * strategy.wg[LoopN] / strategy.namedBarriers[LoopM];
        mov(1, barrierHeaderM.uw(5), threadsPerMBar | (threadsPerMBar << 8));
    }
    if (nbN) {
        int threadsPerNBar = strategy.wg[LoopM] * strategy.wg[LoopN] / strategy.namedBarriers[LoopN];
        mov(1, barrierHeaderN.uw(5), threadsPerNBar | (threadsPerNBar << 8));
    }
    if (strategy.kParallelLocal) {
        if (nbM) emad(1, barrierHeaderM.uw(4), barrierHeaderM.uw(4), state.lidK, strategy.namedBarriers[LoopM], strategy, state);
        if (nbN) emad(1, barrierHeaderN.uw(4), barrierHeaderN.uw(4), state.lidK, strategy.namedBarriers[LoopN], strategy, state);
    }
    int offNBM = 0, offNBN = 0;
    if (strategy.needsUnnamedBarrier(problem))
        offNBM++, offNBN++;
    if (nbM && nbN)
        offNBN += strategy.namedBarriers[LoopM] * strategy.wg[LoopK];
    if (nbM && offNBM) add(1, barrierHeaderM.uw(4), barrierHeaderM.uw(4), offNBM);
    if (nbN && offNBN) add(1, barrierHeaderN.uw(4), barrierHeaderN.uw(4), offNBN);
    if (nbM) mov(1, barrierM, barrierHeaderM.uw(4));
    if (nbN) mov(1, barrierN, barrierHeaderN.uw(4));

    // Get tokens for barriers/fences.
    for (int q = 0; q < 2; q++) {
        state.tokenBarrierFence[q] = -1;
        state.modBarrierFence[q] = InstructionModifier{};
    }

    if (hw >= HW::Gen12LP) {
        if (strategy.needsKLoopBarrier() || strategy.xParallel)
            state.tokenBarrierFence[0] = state.tokenAllocator.tryAlloc();
        if (nbM && nbN)
            state.tokenBarrierFence[1] = state.tokenAllocator.tryAlloc();
        for (int q = 0; q < 2; q++)
            if (state.tokenBarrierFence[q] >= 0)
                state.modBarrierFence[q] = SBID(state.tokenBarrierFence[q]);
    }

    // Remainder load preparations.
    auto &ka_loadRem = state.ka_loadRem, &kb_loadRem = state.kb_loadRem;
    ka_loadRem = 1, kb_loadRem = 1;

    // For packed layouts, extend remainder loads to encompass a full logical block.
    int ignore;
    getGranularities(problem.A, ignore, ka_loadRem);
    getGranularities(problem.B, kb_loadRem, ignore);

    ka_loadRem = std::min(ka_loadRem, strategy.ka_load);
    kb_loadRem = std::min(kb_loadRem, strategy.kb_load);

    // With 2D block loads, extend k unroll to at least a full block (array).
    bool a2D = isBlock2D(strategy.A.accessType);
    bool b2D = isBlock2D(strategy.B.accessType);
    bool ai2D = strategy.slmA && isBlock2D(state.Ai_strategy.accessType);
    bool bi2D = strategy.slmB && isBlock2D(state.Bi_strategy.accessType);
    if (a2D || ai2D) {
        ka_loadRem = state.A_layout[0].nc;
        if (!isColMajor(problem.A.layout))
            ka_loadRem *= state.A_layout[0].count;
    }
    if (b2D || bi2D) {
        kb_loadRem = state.B_layout[0].nr;
        if (isColMajor(problem.B.layout))
            kb_loadRem *= state.B_layout[0].count;
    }

    // With regular block loads oriented in the k dimension, do the same, unless it would
    //   involve downgrading to a padded message (scattered byte/D8U32/D16U32).
    auto &A_lateKRem = state.A_lateKRem, &B_lateKRem = state.B_lateKRem;
    A_lateKRem = B_lateKRem = false;

    if (!strategy.slmA && isBlocklike(strategy.A.accessType) && problem.A.layout == MatrixLayout::T && problem.A.alignment >= 4) {
        A_lateKRem = true;
        ka_loadRem = state.A_layout[0].nc;
    }
    if (!strategy.slmB && isBlocklike(strategy.B.accessType) && problem.B.layout == MatrixLayout::N && problem.B.alignment >= 4) {
        B_lateKRem = true;
        kb_loadRem = state.B_layout[0].nr;
    }

    // Try to use descriptor-based remainders if possible.
    auto &A_descRem = state.A_descRem, &B_descRem = state.B_descRem;
    A_descRem = B_descRem = false;

    if (strategy.kDescRem) {
        if (ka_loadRem == 1) {
            int frag = checkDescriptorRemainder(Ta_load, unrollM, strategy.ka_load, true, false, problem.A, strategy.A);
            if (frag > 1) {
                ka_loadRem = frag;
                A_lateKRem = A_descRem = true;
            }
        }
        if (kb_loadRem == 1 && !A_descRem) {
            int frag = checkDescriptorRemainder(Tb_load, strategy.kb_load, unrollN, false, false, problem.B, strategy.B);
            if (frag > 1) {
                kb_loadRem = frag;
                B_lateKRem = B_descRem = true;
            }
        }
    }

    // When A/B are overaligned (e.g. 1b with assumed 4b alignment), and k dimension is contiguous
    //  in memory, can safely expand remainder load k based on the alignment. This avoids
    //  slow memory accesses.
        if (!strategy.slmA && !isColMajor(problem.A.layout)) {
            ka_loadRem = std::max(ka_loadRem, problem.A.alignment / Ta_load);
            ka_loadRem = std::min(ka_loadRem, strategy.ka_load);
        }
        if (!strategy.slmB &&  isColMajor(problem.B.layout)) {
            kb_loadRem = std::max(kb_loadRem, problem.B.alignment / Tb_load);
            kb_loadRem = std::min(kb_loadRem, strategy.kb_load);
        }

    // Fragment the A, B layouts into smaller blocks (usually 1 row/column) for remainder loads.
    if (!getSubblocks(Ta_load, state.A_layoutRem, state.A_addrsRem, state.A_layout, state.A_addrs, true,  0, ka_loadRem, strategy.A.padded, problem.A, strategy.A)) return false;
    if (!getSubblocks(Tb_load, state.B_layoutRem, state.B_addrsRem, state.B_layout, state.B_addrs, false, 0, kb_loadRem, strategy.B.padded, problem.B, strategy.B)) return false;

    // Add k masking now for block 2D loads. Otherwise it is done later, or not at all.
    if (a2D && (ka_loadRem > 1)) addRemainder(Ta_load, state.A_layoutRem, false, true, AvoidFragment, problem.A, strategy.A);
    if (b2D && (kb_loadRem > 1)) addRemainder(Tb_load, state.B_layoutRem, true, false, AvoidFragment, problem.B, strategy.B);

    // Manually set k remainder flags in the overaligned case.
    if (ka_loadRem > 1 && !A_lateKRem) for (auto &block: state.A_layoutRem)
        block.remainderC = true;
    if (kb_loadRem > 1 && !B_lateKRem) for (auto &block: state.B_layoutRem)
        block.remainderR = true;

    // Ai/Bi remainders.
    auto &Ai_layoutRem = state.Ai_layoutRem, &Bi_layoutRem = state.Bi_layoutRem;
    auto &Ai_layoutK = state.Ai_layoutK, &Bi_layoutK = state.Bi_layoutK;
    auto &Ai_addrsRem = state.Ai_addrsRem, &Bi_addrsRem = state.Bi_addrsRem;
    auto &Ai_addrsK = state.Ai_addrsK, &Bi_addrsK = state.Bi_addrsK;
    auto &Ai_regsRem = state.Ai_regsRem, &Bi_regsRem = state.Bi_regsRem;
    auto &Ao_regsRem = state.Ao_regsRem, &Bo_regsRem = state.Bo_regsRem;
    auto &Ai_hasKRem = state.Ai_hasKRem, &Bi_hasKRem = state.Bi_hasKRem;
    auto &Ai_lateKRem = state.Ai_lateKRem, &Bi_lateKRem = state.Bi_lateKRem;
    auto &Ai_remIncrCopy = state.Ai_remIncrCopy, &Bi_remIncrCopy = state.Bi_remIncrCopy;
    auto &Ai_incrementalRem = state.Ai_incrementalRem, &Bi_incrementalRem = state.Bi_incrementalRem;
    auto &aioShareRem = state.aioShareRem, &bioShareRem = state.bioShareRem;
    int ka_slm = state.ka_slm, kb_slm = state.kb_slm;

    Ai_layoutRem = state.Ai_layout;
    Bi_layoutRem = state.Bi_layout;
    Ai_addrsRem = state.Ai_addrs;
    Bi_addrsRem = state.Bi_addrs;
    Ai_regsRem = state.Ai_regs;
    Bi_regsRem = state.Bi_regs;
    Ao_regsRem = state.Ao_regs;
    Bo_regsRem = state.Bo_regs;

    Ai_hasKRem = Ai_lateKRem = false;
    Bi_hasKRem = Bi_lateKRem = false;
    Ai_remIncrCopy = Bi_remIncrCopy = false;

    if (ai2D && (ka_loadRem > 1) && state.Ai_strategy.address2D) {
        Ai_hasKRem = true;
        addRemainder(Ta_ext, state.Ai_layoutRem, false, true, AvoidFragment, state.Ai, state.Ai_strategy);
    }

    if (bi2D && (kb_loadRem > 1) && state.Bi_strategy.address2D) {
        Bi_hasKRem = true;
        addRemainder(Tb_ext, state.Bi_layoutRem, true, false, AvoidFragment, state.Bi, state.Bi_strategy);
    }

    if (strategy.slmA && !Ai_hasKRem)
        Ai_lateKRem |= !isRegisterColMajor(Ta_ext, state.Ai, state.Ai_strategy);
    if (strategy.slmB && !Bi_hasKRem)
        Bi_lateKRem |=  isRegisterColMajor(Tb_ext, state.Bi, state.Bi_strategy);

    Ai_incrementalRem = strategy.slmA && !state.Ai_hasKRem && !state.Ai_lateKRem;
    Bi_incrementalRem = strategy.slmB && !state.Bi_hasKRem && !state.Bi_lateKRem;
    aioShareRem = state.aioShare;
    bioShareRem = state.bioShare;

    if (Ai_incrementalRem) {
        // Prepare to split Ai layout in k dimension. If it's not possible to do in-place, then
        // either redo the layout or copy Ai->Ao incrementally.
        Ai_layoutK.resize(ka_slm);
        Ai_addrsK.resize(ka_slm);
        for (int h = 0; h < ka_slm; h++) {
            bool success = false;

            if (h < int(Ai_addrsK.size())) {
                success = getSubblocks(Ta_ext, Ai_layoutK[h], Ai_addrsK[h], Ai_layoutRem, state.Ai_addrs,
                                       true, h, h + 1, state.Ai_strategy.padded, state.Ai, state.Ai_strategy);
            }

            if (!success && h == 0) stub();

            if (!success) {
                // Maybe the subblock is OK, but we didn't get an address register. Try again without
                //  asking for address registers.
                Ai_addrsK.resize(1);
                success = getSubblocks(Ta_ext, Ai_layoutK[h], Ai_layoutRem,
                                       true, h, h + 1, state.Ai_strategy.padded, state.Ai, state.Ai_strategy, true);
            }

            if (!success) {
                // Can't make a subblock. Will need a new layout or an incremental copy.
                if (strategy.slmUseIncrCopy) {
                    Ai_remIncrCopy = true;
                    Ai_layoutK.resize(1);
                } else
                    makeAiBiKCloneLayout(hw, true, Ai_layoutRem, Ai_layoutK, Ai_regsRem, ka_slm, strategy, state);

                aioShareRem = false;
                if (state.aioShare || state.aoReuseA)
                    Ao_regsRem = state.ra.alloc_range(getRegCount(state.Ao_layout));
                break;
            }
        }
    }

    if (Bi_incrementalRem) {
        Bi_layoutK.resize(kb_slm);
        Bi_addrsK.resize(kb_slm);
        for (int h = 0; h < kb_slm; h++) {
            bool success = false;

            if (h < int(Bi_addrsK.size())) {
                success = getSubblocks(Tb_ext, Bi_layoutK[h], Bi_addrsK[h], Bi_layoutRem, state.Bi_addrs,
                                       false, h, h + 1, state.Bi_strategy.padded, state.Bi, state.Bi_strategy);
            }

            if (!success && h == 0) stub();

            if (!success) {
                Bi_addrsK.resize(1);
                success = getSubblocks(Tb_ext, Bi_layoutK[h], Bi_layoutRem,
                                       false, h, h + 1, state.Bi_strategy.padded, state.Bi, state.Bi_strategy, true);
            }

            if (!success) {
                if (strategy.slmUseIncrCopy) {
                    Bi_remIncrCopy = true;
                    Bi_layoutK.resize(1);
                } else
                    makeAiBiKCloneLayout(hw, false, Bi_layoutRem, Bi_layoutK, Bi_regsRem, kb_slm, strategy, state);

                bioShareRem = false;
                if (state.bioShare || state.boReuseB)
                    Bo_regsRem = state.ra.alloc_range(getRegCount(state.Bo_layout));
                break;
            }
        }
    }

    // Allocate repack registers if we need to assemble multiple loads for
    //  each outer product calculation.
    // TODO: allow allocation to overlap unneeded A/B registers.
    auto &repackA = state.repackA, &repackB = state.repackB;
    auto &repackARem = state.repackARem, &repackBRem = state.repackBRem;
    auto &ka_repackRem = state.ka_repackRem, &kb_repackRem = state.kb_repackRem;

    repackARem = repackA;
    repackBRem = repackB;
    ka_repackRem = repackA ? std::min(ka_loadRem, state.ka_repack) : 0;
    kb_repackRem = repackB ? kb_loadRem : 0;
    if (minOPCount > 1) {
        if (ka_loadRem < minOPCount) {
            ka_repackRem = minOPCount;
            repackARem = true;
        }
        if (kb_loadRem < minOPCount) {
            kb_repackRem = minOPCount;
            repackBRem = true;
        }
    }

    int crosspackA, crosspackB, tileM_A, tileK_A, tileK_B, tileN_B;
    std::tie(crosspackA, crosspackB) = targetKernelCrosspack(hw, problem, strategy);
    std::tie(tileM_A, tileK_A, tileK_B, tileN_B) = targetKernelTiling(hw, problem, strategy);

    if (!repackA && repackARem) {
        makeUnbackedRegLayout(Ta, state.Ar_layout, unrollM, ka_repackRem, isLayoutColMajor(state.A_layout), crosspackA, tileM_A, tileK_A);
        state.Ar_regs = state.ra.alloc_range(getRegCount(state.Ar_layout), getHint(HintType::A0, strategy));
    }

    if (!repackB && repackBRem) {
        makeUnbackedRegLayout(Tb, state.Br_layout, kb_repackRem, unrollN, isLayoutColMajor(state.B_layout), crosspackB, tileK_B, tileN_B);
        state.Br_regs = state.ra.alloc_range(getRegCount(state.Br_layout), getHint(HintType::B0, strategy));
    }

    state.remActiveA = state.remActiveB = state.remActiveSLM = false;
    state.slmRemaskA = state.slmRemaskB = false;
    state.firstKLoopSegment = true;

    return true;
}

// Tear down after a single k loop.
template <HW hw>
void BLASKernelGenerator<hw>::kLoopTeardown(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    if (state.K != state.k)
        state.ra.safeRelease(state.K);
    state.barrierReady = false;
    state.ra.safeRelease(state.barrierHeader);
    state.ra.safeRelease(state.barrierHeaderM);
    state.ra.safeRelease(state.barrierHeaderN);
    safeReleaseMaskAssignments(state.kMasksA, state);
    safeReleaseMaskAssignments(state.kMasksB, state);
    safeReleaseMaskAssignments(state.kMasksAi, state);
    safeReleaseMaskAssignments(state.kMasksBi, state);
    safeReleaseRanges(state.Ao_regsRem, state);
    safeReleaseRanges(state.Bo_regsRem, state);
    state.tokenAllocator.safeRelease(state.tokenBarrierFence[0]);
    state.tokenAllocator.safeRelease(state.tokenBarrierFence[1]);
}


// Prepare for GEMM k loop with m/n masked A/B accesses. Returns true if ka_lda/kb_ldb need recalculating.
template <HW hw>
bool BLASKernelGenerator<hw>::gemmPrepMaskedAB(const GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state)
{
    bool recalc = false;
    bool shrinkUK = false;
    if (!strategy.A.padded && (strategy.remHandling[LoopM] != RemainderHandling::Ignore)) {
        shrinkUK = true;
        if (strategy.ka_load > strategy.ka_load_masked) {
            status << "Downgrading ka_load: " << strategy.ka_load << " -> " << strategy.ka_load_masked << status_stream::endl;
            strategy.ka_load = strategy.ka_load_masked;
            strategy.trimKChain(hw, strategy.ka_load, problem);
            recalc = true;
        }
        // Avoid access patterns that require double masking, unless enabled.
        if (isBlock2D(strategy.A.accessType) || strategy.allowDoubleMasking(LoopM))
            noop();
        else if (!isRegisterColMajor(problem.Ta_ext, problem.A, strategy.A)) {
            transposeAccessType(strategy.A);
            if (strategy.slmA && strategy.coopA == CoopSplit::MN)
                strategy.coopA = CoopSplit::K;
        }
        strategy.slmATrans = false;
        strategy.prefetchA = strategy.prefetchAMasked;
    }
    if (!strategy.B.padded && (strategy.remHandling[LoopN] != RemainderHandling::Ignore)) {
        shrinkUK = true;
        if (strategy.kb_load > strategy.kb_load_masked) {
            status << "Downgrading kb_load: " << strategy.kb_load << " -> " << strategy.kb_load_masked << status_stream::endl;
            strategy.kb_load = strategy.kb_load_masked;
            strategy.trimKChain(hw, strategy.kb_load, problem);
            recalc = true;
        }
        if (isBlock2D(strategy.B.accessType) || strategy.allowDoubleMasking(LoopN))
            noop();
        else if (isRegisterColMajor(problem.Tb_ext, problem.B, strategy.B)) {
            transposeAccessType(strategy.B);
            if (strategy.slmB && strategy.coopB == CoopSplit::MN)
                strategy.coopB = CoopSplit::K;
        }
        strategy.slmBTrans = false;
        strategy.prefetchB = strategy.prefetchBMasked;
    }
    if (shrinkUK && (strategy.unrollK_masked > 0)
                 && (strategy.unroll[LoopK] > strategy.unrollK_masked)) {
        status << "Downgrading k unroll: " << strategy.unroll[LoopK] << " -> " << strategy.unrollK_masked << status_stream::endl;
        strategy.unroll[LoopK] = strategy.unrollK_masked;
    }
    if (shrinkUK && (strategy.unrollKSLMMasked > 0)
                 && (strategy.unrollKSLM > strategy.unrollKSLMMasked)) {
        status << "Downgrading SLM k chunk size: " << strategy.unrollKSLM << " -> " << strategy.unrollKSLMMasked << status_stream::endl;
        strategy.unrollKSLM = strategy.unrollKSLMMasked;
    }
    return recalc;
}


// Calculate kSLMA/kSLMB -- countdown variables for SLM copies.
template <HW hw>
void BLASKernelGenerator<hw>::gemmCalcKSLM(const Subregister &kSLM, const Subregister &lid, int kgran, int kdiv, int krep,
                                           const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, Subregister kBase)
{
    if (kBase.isInvalid()) kBase = state.K;

    if (kdiv == 1)
        mov(1, kSLM, kBase);
    else {
        auto modLID = lid;
        if (krep > 1) {
            if (!is_zero_or_pow2(krep)) stub();
            modLID = state.ra.alloc_sub<uint16_t>();
            shr(1, modLID, lid, ilog2(krep));
        }
        if (!problem.backward())
            emad(1 | sat, kSLM.uw(), kBase.w(), -modLID.w(), kgran, strategy, state);
        else {
            emad(1, kSLM, strategy.unrollKSLM - kgran, -modLID, kgran, strategy, state);
            add(1, kSLM, kBase, state.kSLMCountUp ? +kSLM : -kSLM);
        }
        if (krep > 1) state.ra.safeRelease(modLID);
    }
}

template <HW hw>
void BLASKernelGenerator<hw>::gemmCalcKSLMA(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, Subregister kBase)
{
    int kgran, kdiv, krep;
    switch (state.effCoopA) {
        case CoopSplit::MN:
            kgran = strategy.unrollKSLM;
            kdiv = 1;
            krep = strategy.wg[LoopN];
            break;
        case CoopSplit::Linear:
            kgran = std::max<int>(state.Ai.crosspack, state.Ai.tileC);
            kdiv = strategy.unrollKSLM / kgran;
            krep = strategy.wg[LoopN] / kdiv;
            if (krep > 0)
                break;
            /* fall through: only split in k dimension */
        case CoopSplit::K:
        case CoopSplit::FullK:
            kgran = state.ka_slm;
            kdiv = strategy.wg[LoopN];
            krep = 1;
            break;
        default: stub();
    }
    gemmCalcKSLM(state.kSLMA, state.lidN, kgran, kdiv, krep, problem, strategy, state, kBase);
}

template <HW hw>
void BLASKernelGenerator<hw>::gemmCalcKSLMB(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, Subregister kBase)
{
    int kgran, kdiv, krep;
    switch (state.effCoopB) {
        case CoopSplit::MN:
            kgran = strategy.unrollKSLM;
            kdiv = 1;
            krep = strategy.wg[LoopM];
            break;
        case CoopSplit::Linear:
            kgran = std::max<int>(state.Bi.crosspack, state.Bi.tileR);
            kdiv = strategy.unrollKSLM / kgran;
            krep = strategy.wg[LoopM] / kdiv;
            if (krep > 0)
                break;
            /* fall through: only split in k dimension */
        case CoopSplit::K:
        case CoopSplit::FullK:
            kgran = state.kb_slm;
            kdiv = strategy.wg[LoopM];
            krep = 1;
            break;
        default: stub();
    }
    gemmCalcKSLM(state.kSLMB, state.lidM, kgran, kdiv, krep, problem, strategy, state, kBase);
}

// Calculate barrier count for a k loop.
template <HW hw>
void BLASKernelGenerator<hw>::gemmCalcKLoopBarrierCount(Subregister &count, const Subregister &k, int cooldown,
                                                        const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    int barrierFreq = strategy.barrierFreq;
    int unrollK     = strategy.unroll[LoopK];
    int unrollKSLM  = strategy.unrollKSLM;

    if (count.isInvalid())
        count = state.ra.alloc_sub<uint32_t>();

    if (barrierFreq > 0) {
        bool maySkipSplitBarrier = strategy.splitBarrier && (cooldown > 0) && !state.splitBarrierAlways;
        if (maySkipSplitBarrier)
            cmp(1 | ge | state.flagAP, k, cooldown);
        add(1 | sat, count, k, barrierFreq - cooldown - unrollK);
        divDown(count, count, barrierFreq, strategy, state);
        if (strategy.splitBarrier) {
            maySkipSplitBarrier ? add(1 | state.flagAP, count, count, 1)
                                : add(1,                count, count, 1);
        }
    } else if (strategy.slmBuffers > 0) {
        if (!is_zero_or_pow2(unrollKSLM)) stub();

        if (strategy.slmBuffers == 1) {
            add(1 | sat, count, k, unrollKSLM - 1);
            if (unrollKSLM == 2)
                and_(1, count, count, ~uint32_t(1));
            else {
                shr(1, count, count, uint16_t(ilog2(unrollKSLM)));
                shl(1, count, count, 1);
            }
        } else {
            add(1 | sat, count, k, unrollKSLM - 1);
            shr(1, count, count, uint16_t(ilog2(unrollKSLM)));
        }
    } else
        mov(1, count, 0);
}

// Make a remainder layout by duplicating the k = 0 slice, allocating extra registers as needed.
static void makeAiBiKCloneLayout(HW hw, bool isA, vector<RegisterBlock> &Xi_layout, vector<vector<RegisterBlock>> &Xi_layoutK,
                                 vector<GRFMultirange> &Xi_regsRem, int kx_slm,
                                 const GEMMStrategy &strategy, GEMMState &state)
{
    auto regCountK = getRegCount(Xi_layoutK[0]);
    auto regCount = regCountK * kx_slm;
    auto offsetK = isA ? &RegisterBlock::offsetC : &RegisterBlock::offsetR;

    Xi_layout = Xi_layoutK[0];

    for (int h1 = 1; h1 < kx_slm; h1++) {
        Xi_layoutK[h1] = Xi_layoutK[h1 - 1];
        for (auto &block: Xi_layoutK[h1]) {
            block.offsetBytes += regCountK * GRF::bytes(hw);

            auto oblock = block;
            oblock.*offsetK += h1;
            Xi_layout.push_back(std::move(oblock));
        }
    }

    int extraRegs = regCount - Xi_regsRem[0].getLen();
    if (extraRegs > 0) {
        for (int q = 0; q < strategy.slmCopies; q++)
            Xi_regsRem[q].append(state.ra.alloc_range(extraRegs));
    }
}

#include "internal/namespace_end.hxx"
