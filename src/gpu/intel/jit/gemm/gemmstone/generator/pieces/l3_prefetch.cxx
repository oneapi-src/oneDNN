
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


#include "generator.hpp"
#include "cooperative_split.hpp"
#include "loop_sequencer.hpp"

using namespace ngen;

#include "internal/namespace_start.hxx"


template <HW hw>
void BLASKernelGenerator<hw>::gemmInitL3Prefetch(bool nextWave, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    if (!strategy.prefetchABL3) return;

    status << (nextWave ? "Start L3 prefetch on next tile"
                        : "Prepare for L3 prefetch") << status_stream::endl;

    auto Ta_ext = problem.Ta_ext, Tb_ext = problem.Tb_ext;
    bool doA = strategy.l3PrefetchA;
    bool doB = strategy.l3PrefetchB;

    auto &gidMN = state.groupIDMN;
    auto gcMN = state.inputs.groupCountMN;
    Subregister nextGroupID;

    if (nextWave) {
        nextGroupID = state.ra.alloc_sub<uint32_t>();
        add(1, nextGroupID, gidMN, state.inputs.groupStride);
    } else
        nextGroupID = gidMN;

    if (gcMN.isInvalid()) {
        gcMN = state.ra.alloc_sub<uint32_t>();
        emul(1, gcMN, state.inputs.groupCountM, state.inputs.groupCountN, strategy, state);
    }

    if (state.nextGroupIDM.isInvalid()) state.nextGroupIDM = state.ra.alloc_sub<uint32_t>();
    if (state.nextGroupIDN.isInvalid()) state.nextGroupIDN = state.ra.alloc_sub<uint32_t>();

    auto &nextFlagL3PFA = state.nextFlagL3PFA;
    auto &nextFlagL3PFB = state.nextFlagL3PFB;
    if (nextFlagL3PFA.isInvalid() && nextFlagL3PFB.isInvalid()) {
        auto storage = state.ra.alloc_sub<uint32_t>();
        if (doA) nextFlagL3PFA = storage.uw(0);
        if (doB) nextFlagL3PFB = storage.uw(1);
    }

    gemmLinearOrder(nextGroupID, state.nextGroupIDM, state.nextGroupIDN,
                    nextFlagL3PFA, nextFlagL3PFB,
                    problem, strategy, state);

#if 1
    if (gpu_utils::dev_getenv("ALL_PF",0)) {
        if (doA) mov(1, nextFlagL3PFA, 0xFFFF);
        if (doB) mov(1, nextFlagL3PFB, 0xFFFF);
    }
#endif

    if (nextWave) {
        cmp(1 | ge | state.flagAP, nextGroupID, gcMN);
        if (doA) mov(1 | state.flagAP, nextFlagL3PFA, 0);
        if (doB) mov(1 | state.flagAP, nextFlagL3PFB, 0);
    }

    auto nextI0 = state.ra.alloc_sub<uint32_t>();
    auto nextJ0 = state.ra.alloc_sub<uint32_t>();

    if (doA) mulConstant(1, nextI0, state.nextGroupIDM, strategy.wgTile(LoopM));
    if (doB) mulConstant(1, nextJ0, state.nextGroupIDN, strategy.wgTile(LoopN));

    auto coopSplitA = naturalSplitA(problem.A.layout);
    auto coopSplitB = naturalSplitB(problem.B.layout);

    auto effApL3 = state.inputs.A;
    auto effBpL3 = state.inputs.B;
    Address2DParams Apl3_params, Bpl3_params;

    Apl3_params.rows = state.inputs.m;
    Apl3_params.cols = state.fullK;
    Apl3_params.offR = nextI0;
    Bpl3_params.rows = state.fullK;
    Bpl3_params.cols = state.inputs.n;
    Bpl3_params.offC = nextJ0;

    int ma_prefetchL3, ka_prefetchL3;
    int kb_prefetchL3, nb_prefetchL3;

    if (doA) coopSplit(true,  ma_prefetchL3, ka_prefetchL3, strategy.unroll[LoopM], strategy.ka_prefetchL3, strategy.wgTile(LoopM), coopSplitA, problem.A, strategy);
    if (doB) coopSplit(false, kb_prefetchL3, nb_prefetchL3, strategy.kb_prefetchL3, strategy.unroll[LoopN], strategy.wgTile(LoopN), coopSplitB, problem.B, strategy);

    std::swap(state.effCoopA, coopSplitA);  /* Temporarily override A/B splitting */
    std::swap(state.effCoopB, coopSplitB);

    if (doA) gemmApplyWorkshareOffset(true,  effApL3, state.inputs.A, Apl3_params, problem.A, strategy.AB_prefetchL3, ma_prefetchL3, ka_prefetchL3, problem, strategy, state);
    if (doB) gemmApplyWorkshareOffset(false, effBpL3, state.inputs.B, Bpl3_params, problem.B, strategy.AB_prefetchL3, kb_prefetchL3, nb_prefetchL3, problem, strategy, state);

    std::swap(state.effCoopA, coopSplitA);
    std::swap(state.effCoopB, coopSplitB); /* ... and restore */

    if (!strategy.AB_prefetchL3.address2D) {
        if (doA) gemmOffsetAm(nextI0, effApL3, problem.A, problem, strategy, state);
        if (doB) gemmOffsetBn(nextJ0, effBpL3, problem.B, problem, strategy, state);
    }

    if (strategy.kParallelLocal) stub();

    if (doA && state.Apl3_layout.empty()) {
        state.flagL3PFA = state.raVFlag.alloc();
        if (!getRegLayout(Ta_ext, state.Apl3_layout, ma_prefetchL3, ka_prefetchL3, false, false, false, AvoidFragment, 0, 0, problem.A, strategy.AB_prefetchL3)) stub();
        for (auto &block: state.Apl3_layout)
            block.flag[0] = state.flagL3PFA;
        allocAddrRegs(state.Apl3_addrs, state.Apl3_layout, problem.A, strategy.AB_prefetchL3, state);
    }

    if (doB && state.Bpl3_layout.empty()) {
        state.flagL3PFB = state.raVFlag.alloc();
        if (!getRegLayout(Tb_ext, state.Bpl3_layout, kb_prefetchL3, nb_prefetchL3, false, false, false, AvoidFragment, 0, 0, problem.B, strategy.AB_prefetchL3)) stub();
        for (auto &block: state.Bpl3_layout)
            block.flag[0] = state.flagL3PFB;
        allocAddrRegs(state.Bpl3_addrs, state.Bpl3_layout, problem.B, strategy.AB_prefetchL3, state);
    }

    if (doA) setupAddr(Ta_ext, state.Apl3_addrs, effApL3, state.Apl3_layout, state.inputs.lda, problem.A, strategy.AB_prefetchL3, strategy, state, Apl3_params);
    if (doB) setupAddr(Tb_ext, state.Bpl3_addrs, effBpL3, state.Bpl3_layout, state.inputs.ldb, problem.B, strategy.AB_prefetchL3, strategy, state, Bpl3_params);

    if (doA) mov(1, state.flagL3PFA, nextFlagL3PFA);
    if (doB) mov(1, state.flagL3PFB, nextFlagL3PFB);

    state.ra.safeRelease(nextI0);
    state.ra.safeRelease(nextJ0);
    if (nextWave)
        state.ra.safeRelease(nextGroupID);
    state.ra.safeRelease(gcMN);
    state.ra.claim(state.inputs.groupCountMN);
    if (effApL3 != state.inputs.A) state.ra.safeRelease(effApL3);
    if (effBpL3 != state.inputs.B) state.ra.safeRelease(effBpL3);
    for (auto off: {&Address2DParams::offR, &Address2DParams::offC}) {
        if (Apl3_params.*off != state.A_params.*off) state.ra.safeRelease(Apl3_params.*off);
        if (Bpl3_params.*off != state.B_params.*off) state.ra.safeRelease(Bpl3_params.*off);
    }
}

template <HW hw>
void BLASKernelGenerator<hw>::gemmWarmupL3Prefetch(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    if (!strategy.prefetchABL3) return;

    status << "L3 prefetch warmup" << status_stream::endl;

    using namespace loop_sequencer;
    LoopSequencer ls;

    gemmScheduleL3Prefetches(&ls, problem, strategy, state);
    gemmScheduleL3PrefetchIncs(&ls, problem, strategy, state, false);

    std::vector<Label> labels;
    int lastThresh = -1;

    using CT = LoopSequencer::CallbackType;

    ls.setCallback(CT::JumpIfLT, [&](int thresh, int label) {
        if (size_t(label) >= labels.size())
            labels.resize(label + 1);
        if (thresh != lastThresh)
            cmp(1 | lt | state.flagAP, state.k, thresh);
        jmpi(1 | state.flagAP, labels[label]);
        lastThresh = thresh;
    });
    ls.setCallback(CT::JumpTarget, [&](int label, int) {
        mark(labels[label]);
    });
    ls.setCallback(CT::Jump, [&](int label, int) {
        if (size_t(label) >= labels.size())
            labels.resize(label + 1);
        jmpi(1, labels[label]);
    });

    ls.analyze();
    ls.materialize(0, strategy.prefetchABL3);
}

template <HW hw>
void BLASKernelGenerator<hw>::gemmScheduleL3Prefetches(void *lsPtr, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    using namespace loop_sequencer;
    auto &ls = *reinterpret_cast<loop_sequencer::LoopSequencer*>(lsPtr);

    auto reqL3PFA = every(strategy.ka_prefetchL3);
    auto reqL3PFB = every(strategy.kb_prefetchL3);

    if (strategy.l3PrefetchA) ls.schedule(reqL3PFA, [&](Iteration h) {
        gemmALoad(GRFMultirange(), state.Apl3_layout, state.Apl3_addrs,
                  problem.A, strategy.AB_prefetchL3, problem, strategy, state);
    });

    if (strategy.l3PrefetchB) ls.schedule(reqL3PFB, [&](Iteration h) {
        gemmBLoad(GRFMultirange(), state.Bpl3_layout, state.Bpl3_addrs,
                  problem.B, strategy.AB_prefetchL3, problem, strategy, state);
    });
}

template <HW hw>
void BLASKernelGenerator<hw>::gemmScheduleL3PrefetchIncs(void *lsPtr, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, bool allowDelay)
{
    using namespace loop_sequencer;
    auto &ls = *reinterpret_cast<loop_sequencer::LoopSequencer*>(lsPtr);

    auto reqL3PFA = every(strategy.ka_prefetchL3);
    auto reqL3PFB = every(strategy.kb_prefetchL3);

    allowDelay &= strategy.delayABInc;
    auto delayA = allowDelay ? (strategy.ka_prefetchL3 >> 1) : 0;
    auto delayB = allowDelay ? (strategy.kb_prefetchL3 >> 1) : 0;

    if (strategy.l3PrefetchA) ls.schedule(reqL3PFA.delay(delayA), [&](Iteration h) {
        gemmAIncrement(problem.Ta_ext, state.Apl3_layout, state.Apl3_addrs,
                       problem.A, strategy.AB_prefetchL3, strategy.ka_prefetchL3,
                       problem, strategy, state);
    });

    if (strategy.l3PrefetchB) ls.schedule(reqL3PFB.delay(delayB), [&](Iteration h) {
        gemmBIncrement(problem.Tb_ext, state.Bpl3_layout, state.Bpl3_addrs,
                       problem.B, strategy.AB_prefetchL3, strategy.kb_prefetchL3,
                       problem, strategy, state);
    });
}

template <HW hw>
void BLASKernelGenerator<hw>::gemmTeardownL3Prefetch(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    // Not much teardown to do. Free flags (we will restore them next loop),
    //   but leave address registers in place.
    state.raVFlag.safeRelease(state.flagL3PFA);
    state.raVFlag.safeRelease(state.flagL3PFB);
}

#include "internal/namespace_end.hxx"
