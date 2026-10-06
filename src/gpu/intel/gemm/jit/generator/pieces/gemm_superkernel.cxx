/*******************************************************************************
* Copyright 2019 Intel Corporation
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


#include "gemmstone/generator.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;


template <HW hw>
void Generator<hw>::gemmSuperkernelInitState(GEMMSuperkernelProblem &problem, GEMMSuperkernelStrategy &strategy, GEMMSuperkernelState &state)
{
    if (strategy.persistent)
        interface.requireGlobalAtomics();

    gemmInitState(problem, strategy.substrategies[0], state, true);

    state.isNested |= strategy.persistent;

    state.inputsSK.surfacePlan = interface.getArgumentSurface("plan");
    state.inputsSK.planCount = interface.getArgument("plan_count");
    state.inputsSK.localID = interface.getLocalID(0);
    state.inputsSK.localSize = interface.getLocalSize(0);

    state.ra.claim(state.inputsSK.localID);
    state.ra.claim(state.inputsSK.localSize);
    state.ra.claim(state.inputsSK.planCount);
}

// Create a GEMM superkernel.
template <HW hw>
void Generator<hw>::gemmSuperkernel(GEMMSuperkernelProblem problem, GEMMSuperkernelStrategy strategy, const InterfaceHandler &interface_)
{
    auto &strategy0 = strategy.substrategies[0];
    bool persistent = strategy.persistent;

    GEMMSuperkernelState state(hw, strategy);

    // Set up.
    setDefaultNoMask();
    setDefaultAutoSWSB();
    interface = interface_;
    gemmSuperkernelInitState(problem, strategy, state);
#ifdef STANDALONE
    gemmSuperkernelNotes(problem, strategy);
#endif
    state.ra.safeRelease(state.inputs.localIDN);
    state.ra.safeRelease(state.inputs.localSizeN);
#ifdef ASM_OUTPUT
    interface.dumpAssignments(std::cout);
#endif

    for (auto &ss: strategy.substrategies) {
        if (!ss.A.base.isStateless()) ss.A.base.setIndex(state.inputs.surfaceA);
        if (!ss.B.base.isStateless()) ss.B.base.setIndex(state.inputs.surfaceB);
        if (!ss.C.base.isStateless()) ss.C.base.setIndex(state.inputs.surfaceC[0]);
    }

    // Prevent unhelpful layouts.
    if (problem.A.layout == MatrixLayout::PackedRows)    stub();
    if (problem.B.layout == MatrixLayout::PackedColumns) stub();

    Label loopSK, loopSKEnd;

    // Prologue.
    prologue(strategy0, state);

    // Grab fused ID if needed.
    getFusedID(1, problem, strategy0, state);

    // Get my plan ID and convert to offset in plan.
    auto idX = r0.ud(1);
    auto header = state.ra.alloc();
    auto poff = header.ud(2);
    constexpr uint16_t eltSz = 8;

    auto temp = state.ra.alloc_sub<uint32_t>();

    mulConstant(1, temp, state.inputsSK.planCount, strategy.subgroupSize());
    mul(1, poff, idX, state.inputsSK.localSize);
    add(1, poff, poff, state.inputsSK.localID.uw(0));
    cmp<uint32_t>(1 | ge | f0[0], poff, temp);
    if (eltSz < strategy.subgroupSize())
        shr(1, poff, poff, ilog2(strategy.subgroupSize() / eltSz));
    else if (eltSz > strategy.subgroupSize())
        mulConstant(1, poff, poff, eltSz / strategy.subgroupSize());

    state.ra.safeRelease(temp);
    state.ra.safeRelease(state.inputsSK.localID);
    state.ra.safeRelease(state.inputsSK.localSize);

    if (persistent)
        add(1, poff, poff, eltSz);

    // Move r0 to acc0 if configured.
    moveR0(strategy0, state);

    // Quick exit for extra threads (uniform WG).
    jmpi(1 | f0[0], loopSKEnd);

    // Retrieve plan element.
    auto pdata = state.ra.alloc(getHint(HintType::TempComp0, strategy0));
    load(8, pdata, aligned_block_oword(1), Surface(state.inputsSK.surfacePlan), header);
    state.ra.safeRelease(header);

    gemmScaleInputs(problem, strategy0, state); // Scale inputs while waiting.

    state.i0 = pdata.d(0);
    state.j0 = pdata.d(1);

    state.ra.safeRelease(pdata);
    state.ra.claim(state.i0);
    state.ra.claim(state.j0);

    auto flagKID0 = f1[0];
    auto flagKID1 = f1[1];

    if (strategy.multiM) cmp(1 | lt | flagKID0, null.d(), state.i0, 0);
    if (strategy.multiN) cmp(1 | lt | flagKID1, null.d(), state.j0, 0);
    and_(2, state.i0.ud()(1), state.i0.ud()(1), uint32_t(0x7FFFFFFF));

    // Initial offset of A/B/C.
    gemmOffsetABC(true, state.i0, state.j0, Subregister(), state.i0, state.j0, problem, strategy0, state);
    gemmSetupABC(problem, strategy0, state);

    // Save i0, j0 for later.
    state.last_i0 = state.ra.alloc_sub<int32_t>(getHint(HintType::LongTerm, strategy0));
    state.last_j0 = state.ra.alloc_sub<int32_t>(getHint(HintType::LongTerm, strategy0));
    mov(1, state.last_i0, state.i0);
    mov(1, state.last_j0, state.j0);

    // Top of superkernel loop.
    status << "Begin superkernel loop" << status_stream::endl;
    mark(loopSK); {
        // Dispatch appropriate kernel, supporting up to 4 subkernels.
        int kidx = 0;
        Label labelM1, labelM0N1, labelM1N1, labelKernelDone;
        if (strategy.multiM) jmpi(1 | flagKID0, labelM1);
        if (strategy.multiN) jmpi(1 | flagKID1, labelM0N1);

        gemmSubkernel(problem, strategy.substrategies[kidx++], state);

        if (strategy.multiN) {
            jmpi(1, labelKernelDone);
            mark(labelM0N1);
            gemmSubkernel(problem, strategy.substrategies[kidx++], state);
        }

        if (strategy.multiM) {
            jmpi(1, labelKernelDone);

            mark(labelM1);
            if (strategy.multiN) jmpi(1 | flagKID1, labelM1N1);

            gemmSubkernel(problem, strategy.substrategies[kidx++], state);

            if (strategy.multiN) {
                jmpi(1, labelKernelDone);
                mark(labelM1N1);
                gemmSubkernel(problem, strategy.substrategies[kidx++], state);
            }
        }

        mark(labelKernelDone);

        if (persistent) {
            // Get next plan element via atomic increment of plan ID counter.
            auto header = state.ra.alloc();
            auto nextID = state.ra.alloc(getHint(HintType::TempComp1, strategy0));
            auto pdata = state.ra.alloc(getHint(HintType::TempComp0, strategy0));

            mov<uint32_t>(8, header, uint16_t(0));
            atomic(AtomicOp::inc, 1, nextID, scattered_dword(), Surface(state.inputsSK.surfacePlan), header);

            // Load next plan element, or exit if no more work.
            mulConstant<uint32_t>(1, header[2], nextID[0], eltSz);      cmp<uint32_t>(1 | ge | f0[0], null, nextID[0], state.inputsSK.planCount);
            add<uint32_t>(1, header[2], header[2], eltSz);

            jmpi(1 | f0[0], loopSKEnd);

            load(8, pdata, aligned_block_oword(1), Surface(state.inputsSK.surfacePlan), header);
            state.ra.safeRelease(header);
            state.ra.safeRelease(nextID);

            // Load next (i0, j0) and kernel IDs.
            auto in_i0 = pdata.d(0);
            auto in_j0 = pdata.d(1);

            if (strategy.multiM) cmp(1 | lt | flagKID0, null.d(), in_i0, 0);
            if (strategy.multiN) cmp(1 | lt | flagKID1, null.d(), in_j0, 0);
            and_(1, state.i0.ud(), in_i0.ud(), uint32_t(0x7FFFFFFF));
            and_(1, state.j0.ud(), in_j0.ud(), uint32_t(0x7FFFFFFF));

            // Get difference in i0 and j0...
            add(1, in_i0, state.i0, -state.last_i0);
            add(1, in_j0, state.j0, -state.last_j0);

            // ... save current (i0, j0) for later...
            mov(1, state.last_i0, state.i0);
            mov(1, state.last_j0, state.j0);

            // ...and offset A, B, C appropriately.
            gemmOffsetABC(false, in_i0, in_j0, Subregister(), state.i0, state.j0, problem, strategy0, state);

            state.ra.safeRelease(pdata);

            state.ra.safeRelease(state.i0);
            state.ra.safeRelease(state.j0);

            // Ready for the next kernel.
            jmpi(1, loopSK);
        }
    }
    mark(loopSKEnd);

    epilogue(strategy.substrategies[0], state);
    padding();
}

GEMMSTONE_NAMESPACE_END
