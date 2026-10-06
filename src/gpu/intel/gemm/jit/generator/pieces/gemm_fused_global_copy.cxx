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
#include "kernel_queries.hpp"
#include "ngen_object_helpers.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;
using namespace ngen::utils;


template <typename T>
static inline Subregister getTaskField(HW hw, const RegisterRange &regs, T fused::FusedTask::Task::* field)
{
    fused::FusedTask::Task dummy;
    auto offset = (char *)(&(dummy.*field)) - (char *)(&dummy);
    return regs[int(offset / GRF::bytes(hw))].sub<T>((offset % GRF::bytes(hw)) / sizeof(T));
}

// GEMM fused kernel generation interface.
template <HW hw>
void Generator<hw>::gemmFused(FusedGEMMProblem problem, FusedGEMMStrategy strategy, const InterfaceHandler &interface_)
{
    using namespace fused;
    using Task = FusedTask::Task;

    auto Tc = problem.Tc();
    bool simtCF = strategy.fused;
    bool a64 = strategy.statelessPlan;

    auto jump = [&](Label &target) {
        simtCF ? goto12(16, target) : jmpi(1, target);
    };
    auto jumpIf = [&](const FlagRegister &flag, Label &target) {
        simtCF ? goto12(16 | flag | any16h, target) : jmpi(1 | flag, target);
    };
    auto label = [&](Label &here) {
        mark(here);
        if (simtCF) join(16);
    };

    Label lPersistentLoopTop;
    Label lGEMM, lCopy, lOut[2], lClaimNextTask[2];
    Label lWait, lWaitLoopTop, lWaitCopy, lNoWork;
    Label lFollowers, lGEMMExit, lDispatchCopy, lExit;

    FusedGEMMState state(hw);

    // Set up.
    interface = interface_;
    gemmFusedInitState(problem, strategy, state);
#ifdef STANDALONE
    gemmFusedKernelNotes(problem, strategy);
#endif
#ifdef ASM_OUTPUT
    interface.dumpAssignments(std::cout);
#endif
    setDefaultNoMask();
    setDefaultAutoSWSB();

    // Prologue.
    prologue(strategy);

    // Divide out subgroup size from local ID.
    removeSG(problem, strategy, state);

    if (strategy.persistent)
        mark(lPersistentLoopTop);

    // Variables for following code.
    bool atomicLoad = (hw >= HW::XeHP);

    auto planBase = a64 ? A64 : Surface(state.inputs.surfacePlan);
    auto headerGRFs = GRF::bytesToGRFs(hw, 64);
    auto alloc64 = [&](int n = 1){ return state.ra.alloc_range(headerGRFs * n); };

    GRFRange headers = alloc64(2);
    GRFRange header[2] = {headers[0]-headers[headerGRFs-1],
                          headers[headerGRFs]-headers[2*headerGRFs-1]};
    GRFRange addrHeader[2] = {alloc64(a64 ? 2 : 1), alloc64(a64 ? 2 : 1)};
    GRF addrHeaderBlock[2];
    if (a64) {
        addrHeaderBlock[0] = addrHeader[0][0];
        addrHeaderBlock[1] = addrHeader[1][0];
    } else {
        addrHeaderBlock[0] = state.ra.alloc();
        addrHeaderBlock[1] = state.ra.alloc();
    }
    GRFRange addrSLMTask;
    addrSLMTask = state.ra.alloc_range(1);
    auto addrTask = state.ra.alloc().ud();
    auto incTP = alloc64();
    GRFRange task[2] = {alloc64(), alloc64()};
    auto temp = state.ra.alloc().ud();
    auto tries = state.ra.alloc_sub<uint32_t>();
    auto moreGEMM = f0[1];
    auto moreCopy = f1[1];
    auto leader = f1[0];
    auto readMask = temp.uw(6);

    // Split followers from leader thread.
    cmp(16 | eq | leader, state.inputs.localID.uw(0), 0);

    mov(1, addrSLMTask[0].ud(2), 0);

    simtCF ? goto12(16 | ~leader, lFollowers)
           :   jmpi(1  | ~leader, lFollowers);

    // Leader finds work to do.
    status << "Initialize and check thread ID for copy/gemm preference" << status_stream::endl;

    if (a64) {
        if (simtCF) { mov(1, f0[0], 0); }                       mov(8, addrHeader[0][1].uw(), Immediate::uv(0,1,2,3,4,5,6,7));
        and_(1, temp, sr0[0], 0x7);                             shl(8, addrHeader[0][0].ud(0)(2), addrHeader[0][1].uw(), 2);
        bfi1(1, readMask, state.inputs.slotCount1, 0);
        mov<uint32_t>(16, incTP, 0);                            eadd<uint64_t>(8, addrHeader[0][0], addrHeader[0][0].ud(0)(2), state.inputs.plan, strategy, state);
        mov(1, tries, 2);                                       eadd<uint64_t>(8, addrHeader[0][headerGRFs], addrHeader[0][0], 8 * 4, strategy, state);
        cmp(1 | ge | f0[0], temp, state.inputs.tidThresh);      eadd<uint64_t>(8, addrHeader[1][0], addrHeader[0][0], 16 * 4, strategy, state);
        mov<uint32_t>(1, incTP[0], 1);                          eadd<uint64_t>(8, addrHeader[1][headerGRFs], addrHeader[0][0], 24 * 4, strategy, state);
    } else {
        if (simtCF) mov(1, f0[0], 0);
        and_(1, temp, sr0[0], 0x7);                             mov(8, addrHeader[0][0].uw(), Immediate::uv(0,1,2,3,4,5,6,7));
        bfi1(1, readMask, state.inputs.slotCount1, 0);
        mov<uint32_t>(16, incTP, 0);                            shl<uint32_t>(8, addrHeader[0][0], addrHeader[0][0].uw(), 2);
        mov(1, tries, 2);
        cmp(1 | ge | f0[0], temp, state.inputs.tidThresh);      add<uint32_t>(8, addrHeader[0][1], addrHeader[0][0], 8 * 4);
        mov<uint32_t>(1, incTP[0], 1);                          add<uint32_t>(16, addrHeader[1][0], addrHeader[0][0], 16 * 4);
    }

    jumpIf(f0[0], lCopy);

    // Main work-finding logic.
#define TASK_FIELD(f) getTaskField(hw, task[copy], &f)
    auto gotTask = [&](bool copy, bool noWork = false) {
        auto wgIDSave = TASK_FIELD(Task::triggerA);
        auto wgID = header[copy][0].uw(0);

        if (noWork)
            mov(1, wgIDSave, 0xFFFFFFFF);
        else
            mov(1, wgIDSave, wgID);

        store(16, block_oword(4), SLM, addrSLMTask, task[copy]);

        if (copy)
            mov<uint32_t>(16, task[0], task[1]);
        jump(lOut[noWork || copy]);
    };

    auto getTask = [&](bool copy, Label &lTop) {
        Label lNotFirstGEMM, lUnsuccessful, lNextTask;
        auto &more = copy ? moreCopy : moreGEMM;

        mov(1, f1[0], readMask);
        if (a64 && GRF::bytes(hw) == 64)
            atomic(AtomicOp::add, 16 | f1[0], header[copy], D32, planBase, addrHeader[copy], incTP);
        else
        if (a64) {
            atomic(AtomicOp::add, 8 | f1[0] | M0, header[copy][0], scattered_dword(), planBase, addrHeader[copy][0], incTP[0]);
            atomic(AtomicOp::add, 8 | f1[0] | M8, header[copy][1], scattered_dword(), planBase, addrHeader[copy][2], incTP[1]);
        } else
            atomic(AtomicOp::add, 16 | f1[0], header[copy], scattered_dword(), planBase, addrHeader[copy], incTP);

        // We have claimed the following work.
        auto taskID = header[copy][0].uw(1);
        auto wgID =   header[copy][0].uw(0);

        if (!copy) cmp(1 | gt | f0[0], taskID, 0);

        // Get task address. Copy tasks come first, then GEMM.
        copy ? add(1, addrTask[2], taskID, 2)
             : eadd3(1, addrTask[2], taskID, state.inputs.copyTasks, 2);
        mulConstant(1, addrTask[2], addrTask[2], a64 ? 64 : (64 / 16));
        if (a64)
            eadd(1, addrTask.uq(0), addrTask.ud(2), state.inputs.plan, strategy, state);

        // As an optimization, check whether first GEMM is ready by looking at input triggers.
        if (!copy) {
            jumpIf(f0[0], lNotFirstGEMM);
            status << "GEMM 0 special handling" << status_stream::endl;
            cmp(1 | eq | f0[0], header[copy][0].ud(1), state.inputs.cc0Trigger);
            cmp(1 | f0[0] | eq | f0[0], header[copy][0].ud(2), state.inputs.cc1Trigger);
            jumpIf(f0[0], lNextTask);
            mov(1, moreGEMM, 0xFFFF);
            mov<uint32_t>(1, task[0], 0x01000000);   // op 0, slotA = 0, slotB = 1
            mov<uint32_t>(1, TASK_FIELD(Task::triggerA), state.inputs.cc0Trigger);
            mov<uint32_t>(1, TASK_FIELD(Task::triggerB), state.inputs.cc1Trigger);
            jump(lUnsuccessful);
            label(lNotFirstGEMM);
        }

        // Load task.
        load(16, task[copy], block_oword(4), planBase, addrTask);

        cmp(1 | ge | f0[0], wgID, TASK_FIELD(Task::wgCount));
        jumpIf(f0[0], lNextTask);

        gotTask(copy);

        label(lNextTask);
        status << "Task complete; check next task" << status_stream::endl;
        add(1, taskID, taskID, 1);
        mov(1, more, 0);
        a64 ? eadd<uint64_t>(1, addrTask.uq(0), addrTask.uq(0), 64, strategy, state)
            : add<uint32_t>(1, addrTask[2], addrTask[2], 64 / 16);
        cmp(1 | ge | f0[0], taskID, copy ? state.inputs.copyTasks : state.inputs.gemmTasks);
        jumpIf(f0[0], lUnsuccessful);

        mov(1, more, 0xFFFF);
        load(16, task[copy], block_oword(4), planBase, addrTask);

        auto grfOffset = header[copy].getBase() * GRF::bytes(hw);
        shl(2, a0, TASK_FIELD(Task::slotA)(1), 2);
        if (grfOffset >= 16 * GRF::bytes(hw)) {
            add(2, a0, a0, grfOffset);
            grfOffset = 0;
        }

        cmp(1 | eq | f0[0], indirect[a0[0]].ud(grfOffset + 4), TASK_FIELD(Task::triggerA));
        if (!copy) {
            doReadSuppressionWA(strategy, state);
            cmp(1 | f0[0] | eq | f0[0], indirect[a0[1]].ud(grfOffset + 4), TASK_FIELD(Task::triggerB));
        }
        jumpIf(~f0[0], lUnsuccessful);

        label(lClaimNextTask[copy]);
        status << "Attempt to claim next task" << status_stream::endl;
        mov(1, wgID, 1);
        atomic(AtomicOp::umax, 1, temp, scattered_dword(), planBase, addrHeader[copy], header[copy]);
        mov(1, wgID, 0);
        cmp(1 | ge | f0[0], temp.uw(1), taskID);
        jumpIf(f0[0], lTop);

        gotTask(copy);

        label(lUnsuccessful);
    };
#undef TASK_FIELD

    label(lGEMM);
    status << "Try to find available GEMM work" << status_stream::endl;
    getTask(false, lGEMM);
    add(1 | le | f0[0], tries, tries, -1);
    jumpIf(f0[0], lWait);
    label(lCopy);
    status << "Try to find available copy work" << status_stream::endl;
    getTask(true, lCopy);
    add(1 | le | f0[0], tries, tries, -1);
    jumpIf(~f0[0], lGEMM);

    label(lWait);
    status << "No tasks ready to run" << status_stream::endl;
    if (hw >= HW::XeHPC) {
        or_(1 | ze | f0[0], null.uw(), moreGEMM, moreCopy);
        jmpi(1 | f0[0], lNoWork);
    } else {
        simtCF ? goto12(16 | ~moreGEMM | anyv, lNoWork)
               :   jmpi(1  | ~moreGEMM | anyv, lNoWork);
    }

    mov(1, tries, 1);
    if (!a64) {
        mov(1, addrHeaderBlock[0].ud(2), 0);
        mov(1, addrHeaderBlock[1].ud(2), 64 / 16);
    }
    add(2, temp.uw(0)(1), getTaskField(hw, task[0], &Task::slotA)(1), 1);
    add(2, temp.uw(2)(1), getTaskField(hw, task[1], &Task::slotA)(1), 1);

    shl(4, a0, temp.uw(), 2);
    add(2, a0, a0, header[0].getBase() * GRF::bytes(hw));
    add(1, a0[2], a0[2], header[1].getBase() * GRF::bytes(hw));

    Subregister waitMasks[2] = {temp.uw(12), temp.uw(14)};
    shl(4, temp.uw(), tries.uw(), temp.uw());
    or_(1, temp.uw(0), temp.uw(0), temp.uw(1));
    or_(2, waitMasks[0](2), temp.uw(0)(2), 1);

    RegisterRange zeros;
    if (atomicLoad) {
        zeros = alloc64();
        mov<uint32_t>(16, zeros, 0);
    }

    auto doWait = [&](bool copy) {
        auto flag = f1[0];
        pause(strategy);

        mov(1, flag, waitMasks[copy]);
        mov<uint32_t>(1, temp, header[copy]);

        doReadSuppressionWA(strategy, state);
        if (!atomicLoad)
            load(16 | flag, header[copy], block_oword(4), planBase, addrHeaderBlock[copy]);
        else if (a64 && GRF::bytes(hw) == 64)
            atomic(AtomicOp::or_, 16 | flag, header[copy], D32, planBase, addrHeader[copy], zeros);
        else if (!a64)
            atomic(AtomicOp::or_, 16 | flag, header[copy], scattered_dword(), planBase, addrHeader[copy], zeros);
        else {
            atomic(AtomicOp::or_, 8 | flag | M0, header[copy][0], scattered_dword(), planBase, addrHeader[copy][0], zeros);
            atomic(AtomicOp::or_, 8 | flag | M8, header[copy][2], scattered_dword(), planBase, addrHeader[copy][1], zeros);
        }

        cmp(1 | gt | f0[0], header[copy][0].uw(1), temp.uw(1));
        jumpIf(f0[0], copy ? lCopy : lGEMM);

        cmp(1 | eq | f0[0], indirect[a0[copy ? 2 : 0]].ud(), getTaskField(hw, task[copy], &Task::triggerA));
        if (!copy) {
            doReadSuppressionWA(strategy, state);
            cmp(1 | f0[0] | eq | f0[0], indirect[a0[1]].ud(), getTaskField(hw, task[copy], &Task::triggerB));
        }
        jumpIf(f0[0], lClaimNextTask[copy]);
    };

    auto waitCount = temp.ud(4);
    if (strategy.waitTimeout > 0)
        mov(1, waitCount, strategy.waitTimeout);

    label(lWaitLoopTop);
    if (strategy.waitTimeout > 0) {
        add(1 | le | f0[0], waitCount, waitCount, -1);
        jmpi(1 | f0[0], lNoWork);
    }
    status << "Wait for GEMM work" << status_stream::endl;
    jumpIf(~moreGEMM, lWaitCopy);
    doWait(false);
    label(lWaitCopy);
    status << "Wait for copy work" << status_stream::endl;
    jumpIf(~moreCopy, lWaitLoopTop);
    doWait(true);
    jump(lWaitLoopTop);

    state.ra.safeRelease(zeros);

    label(lNoWork);
    status << "No more work available" << status_stream::endl;
    gotTask(false, true);

#define TASK_FIELD(f) getTaskField(hw, task[0], &f)

    label(lOut[0]);

    status << "Reset counters after final GEMM" << status_stream::endl;

    add(2, temp, header[0][0].uw(), 1);
    cmp<uint16_t>(1 | eq | f0[0], temp[0], TASK_FIELD(Task::wgCount));
    cmp<uint16_t>(1 | f0[0] | eq | f0[0], temp[1], state.inputs.gemmTasks);
    jumpIf(~f0[0], lOut[1]);

    memfence(temp);
    fencewait();
    if (!a64) mov(1, addrHeaderBlock[0].ud(2), 0);
    zeroMatrix(headers, strategy);
    store(16, block_oword(8), planBase, addrHeaderBlock[0], headers);

    label(lOut[1]);

    status << "SLM synchronization" << status_stream::endl;
    slmfence(temp);
    fencewait();
    mov(1, leader, 0xFFFF);     // May have been trashed earlier.

    // Followers enter here.
    mark(lFollowers);
    status << "Follower threads enter here" << status_stream::endl;

    if (simtCF) join(16);

    // Followers read task from SLM.
    barrier(temp);
    doReadSuppressionWA(strategy, state);
    load(16 | ~leader, task[0], block_oword(4), SLM, addrSLMTask);

    // Barrier to finish reading SLM if we use SLM later.
    if (std::max(strategy.gemm.slmBuffersA, strategy.gemm.slmBuffersB) > 0)
        barrier(temp);

    // Release temporary registers.
    state.ra.safeRelease(headers);
    header[0] = invalid;
    header[1] = invalid;
    state.ra.safeRelease(temp);
    state.ra.safeRelease(addrHeader[0]);
    state.ra.safeRelease(addrHeader[1]);
    if (!a64) {
        state.ra.safeRelease(addrHeaderBlock[0]);
        state.ra.safeRelease(addrHeaderBlock[1]);
    }
    state.ra.safeRelease(addrSLMTask);
    state.ra.safeRelease(addrTask);
    state.ra.safeRelease(incTP);
    state.ra.safeRelease(task[1]);
    state.ra.safeRelease(tries);

    // Free up scheduling-related inputs.
    if (!strategy.persistent) {
        if (!a64)
            state.ra.safeRelease(state.inputs.plan);
        state.ra.safeRelease(state.inputs.copyTasks);
        state.ra.safeRelease(state.inputs.gemmTasks);
        state.ra.safeRelease(state.inputs.tidThresh);
        state.ra.safeRelease(state.inputs.slotCount1);
        state.ra.safeRelease(state.inputs.cc0Trigger);
        state.ra.safeRelease(state.inputs.cc1Trigger);
    }

    // task[0] contains current task.
    // task[0].triggerA/B repurposed to hold WG ID and flat local ID, respectively.
    auto groupIDFlat = TASK_FIELD(Task::triggerA);
    auto localIDFlat = TASK_FIELD(Task::triggerB).uw();
    auto op = TASK_FIELD(Task::op);
    auto slotW = state.ra.alloc_sub<uint32_t>();

    // EOT if no work for us.
    cmp(1 | eq | f0[0], groupIDFlat, 0xFFFFFFFF);
    jmpi(1 | f0[0], lExit);

    // Get effective global ID as seen by the GEMM/copy kernel.
    status << "Calculate logical global ID for current task" << status_stream::endl;
    state.groupIDX = state.ra.alloc_sub<uint32_t>();
    state.groupIDY = state.ra.alloc_sub<uint32_t>();
    divDown(state.groupIDY, groupIDFlat, TASK_FIELD(Task::size1), TASK_FIELD(Task::recip1), f1[0], strategy, state);
    mul(1, state.groupIDX, state.groupIDY, TASK_FIELD(Task::size1));
    mov(1, localIDFlat, state.inputs.localID);
    add(1, state.groupIDX, -state.groupIDX, groupIDFlat);

    and_(1 | ne | f0[0], null.ud(), op, 0x8000);
    moveR0(strategy, state);
    mov(2, slotW.uw()(1), TASK_FIELD(Task::slotA)(1));
    jmpi(1 | f0[0], lDispatchCopy);

    // GEMM.
    auto globalInnerLoop = strategy.gemm.loopOrder[0];
    auto localInnerLoop = strategy.fused ? strategy.gemm.fusedLoop : globalInnerLoop;
    int nx = strategy.gemm.wg[localInnerLoop];
    if (!nx || !is_zero_or_pow2(nx)) stub();

    status << "Prepare GEMM kernel inputs" << status_stream::endl;
    state.localIDX = state.ra.alloc().uw();
    state.localIDY = state.ra.alloc().uw();
    and_(1, state.localIDX, state.inputs.localID, nx - 1);
    shr(1, state.localIDY, state.inputs.localID, ilog2(nx));

    if (problem.gemm.gemmt()) {
        cmp(1 | ne | f0[0], TASK_FIELD(Task::offsetCO_diag).d(), 0);
        cmp(1 | ne | f1[0], TASK_FIELD(Task::m), TASK_FIELD(Task::n));
    }

    if (state.inputs.beta_real.isValid()) {
        and_(1 | ne | f0[1], null.ud(), op, FlagNoninitialKBlock);
        mov(1 | f0[1], state.inputs.beta_real, cast(problem.gemm.Ts.real(), 1));
        if (state.inputs.beta_imag.isValid())
            mov(1 | f0[1], state.inputs.beta_imag, cast(problem.gemm.Ts.real(), 0));
    }

    auto offsetA = TASK_FIELD(Task::offsetA).ud();
    auto offsetB = TASK_FIELD(Task::offsetB).ud();

    GEMMState stateGEMM(hw, strategy.gemm);
    static_cast<CommonState&>(stateGEMM) = static_cast<CommonState&>(state);

    if (!strategy.persistent) {
        stateGEMM.ra.release(state.inputs.A);
        stateGEMM.ra.release(state.inputs.B);
        if (!strategy.gemm.A.base.isStateless() && !strategy.gemm.B.base.isStateless())
            stateGEMM.ra.release(state.inputs.X);
        stateGEMM.ra.release(state.inputs.lda);
        stateGEMM.ra.release(state.inputs.ldb);
        stateGEMM.ra.release(state.inputs.slotSize);
        stateGEMM.ra.release(state.inputs.localID);
        if (problem.alphaScaledCopy()) {
            stateGEMM.ra.release(state.inputs.alpha_real);
            stateGEMM.ra.release(state.inputs.alpha_imag);
        }
        stateGEMM.ra.release(slotW);
    }

    stateGEMM.inputs.A = state.inputs.X;
    stateGEMM.inputs.B = state.inputs.X;
    stateGEMM.inputs.C[0] = state.inputs.C;
    stateGEMM.inputs.coPtr = state.inputs.CO;
    stateGEMM.inputs.surfaceA = state.inputs.surfaceX;
    stateGEMM.inputs.surfaceB = state.inputs.surfaceX;
    stateGEMM.inputs.surfaceC[0] = state.inputs.surfaceC;
    stateGEMM.inputs.surfaceCO = state.inputs.surfaceCO;
    stateGEMM.inputs.offsetA = offsetA;
    stateGEMM.inputs.offsetB = offsetB;
    stateGEMM.inputs.offsetC[0] = TASK_FIELD(Task::offsetC);
    stateGEMM.inputs.offsetCO = TASK_FIELD(Task::offsetCO_diag).ud();
    stateGEMM.inputs.lda = TASK_FIELD(Task::ldx);
    stateGEMM.inputs.ldb = TASK_FIELD(Task::ldx);
    stateGEMM.inputs.ldc[0] = state.inputs.ldc;
    stateGEMM.inputs.m = TASK_FIELD(Task::m).d();
    stateGEMM.inputs.n = TASK_FIELD(Task::n).d();
    stateGEMM.inputs.k = TASK_FIELD(Task::k_z0).d();
    if (!problem.alphaScaledCopy()) {
        stateGEMM.inputs.alpha_real = state.inputs.alpha_real;
        stateGEMM.inputs.alpha_imag = state.inputs.alpha_imag;
    }
    stateGEMM.inputs.beta_real = state.inputs.beta_real;
    stateGEMM.inputs.beta_imag = state.inputs.beta_imag;
    stateGEMM.inputs.ao = state.inputs.ao;
    stateGEMM.inputs.bo = state.inputs.bo;
    if (problem.gemm.updateType != UpdateType::Full)
        stateGEMM.inputs.diagC = TASK_FIELD(Task::offsetCO_diag).d();
    stateGEMM.inputs.flags = op;
    stateGEMM.inputs.groupIDM = (globalInnerLoop == LoopM) ? state.groupIDX : state.groupIDY;
    stateGEMM.inputs.groupIDN = (globalInnerLoop == LoopN) ? state.groupIDX : state.groupIDY;
    stateGEMM.inputs.localIDM = (localInnerLoop  == LoopM) ? state.localIDX : state.localIDY;
    stateGEMM.inputs.localIDN = (localInnerLoop  == LoopN) ? state.localIDX : state.localIDY;
    stateGEMM.fusedGEMM.active = true;
    stateGEMM.fusedGEMM.surfacePlan = state.inputs.surfacePlan;
    stateGEMM.fusedGEMM.plan = state.inputs.plan;
    stateGEMM.fusedGEMM.slotA = TASK_FIELD(Task::slotA);
    stateGEMM.fusedGEMM.slotB = TASK_FIELD(Task::slotB);
    stateGEMM.fusedGEMM.localIDFlat = localIDFlat;
    stateGEMM.fusedGEMM.needLateGEMMDone = f1[0];
    stateGEMM.isNested = true;
    stateGEMM.lid0 = localIDFlat;

    if (!problem.mock.gemm) {
        status << "GEMM kernel" << status_stream::endl;
        gemm(problem.gemm, strategy.gemm, stateGEMM);
    } else {
        stateGEMM.flagAP = stateGEMM.fusedGEMM.needLateGEMMDone = f0[0];
        and_(16 | nz | f0[0], null.uw(), stateGEMM.inputs.flags.uw(), FlagEarlyFusedGEMMDone);
    }

    if (strategy.kBlockFence) {
        Label skipFence;

        and_(1 | nz | f0[1], null.uw(), ~stateGEMM.inputs.flags.uw(), FlagNonfinalKBlock | FlagLateFusedGEMMDone);
        jmpi(1 | f0[1], skipFence);
        auto temp = state.ra.alloc();
        useR0(state, [&](const GRF &r0_info) {
            globalMemBarrier(temp, r0_info, strategy);
        });
        state.ra.safeRelease(temp);
        mark(skipFence);
    }

    gemmFusedFinishRead(problem.gemm, strategy.gemm, stateGEMM, true);

    mark(lGEMMExit);
    if (simtCF) join(16);
    strategy.persistent ? jmpi(1, lPersistentLoopTop)
                        : epilogue(strategy, state);

    state.ra.safeRelease(state.localIDX);
    state.ra.safeRelease(state.localIDY);

    mark(lDispatchCopy);

    Label lEnterCopy[2][2][2];

    status << "Copy kernel dispatch" << status_stream::endl;

    bool isComplex = Tc.isComplex();
    auto offsetD = TASK_FIELD(Task::offsetC).ud();
    auto jumpOffset = state.ra.alloc_sub<uint32_t>();

    shl(1, jumpOffset, op.ub(0), 4);
    if (isGen12)
        add(1, jumpOffset, jumpOffset, 16);
    jmpi(1, jumpOffset);
    for (bool conj : {false, true})
        for (bool trans : {false, true})
            for (bool copyB : {false, true})
                if (!conj || isComplex)
                    jmpi(1, lEnterCopy[conj][copyB][trans]);

    for (bool conj : {false, true}) {
        if (conj && !isComplex) break;

        for (bool copyB : {false, true}) {
            for (bool trans : {false, true}) {
                mark(lEnterCopy[conj][copyB][trans]);
                auto &problemCopy = problem.copy[copyB][trans];
                auto &strategyCopy = strategy.copy[copyB][trans];

                problemCopy.conjugate = conj;

                CopyState stateCopy(hw);
                static_cast<CommonState&>(stateCopy) = static_cast<CommonState&>(state);

                if (!strategy.persistent) {
                    if (!problem.alphaScaledCopy() || copyB) {
                        stateCopy.ra.release(state.inputs.alpha_real);
                        stateCopy.ra.release(state.inputs.alpha_imag);
                    }
                    stateCopy.ra.release(state.inputs.beta_real);
                    stateCopy.ra.release(state.inputs.beta_imag);
                    if (!strategyCopy.S.base.isStateless())
                        stateCopy.ra.release(!copyB ? state.inputs.A : state.inputs.B);
                    stateCopy.ra.release(!copyB ? state.inputs.B : state.inputs.A);
                    stateCopy.ra.release(state.inputs.C);
                    stateCopy.ra.release(state.inputs.CO);
                    stateCopy.ra.release(state.inputs.ldc);
                    stateCopy.ra.release(state.inputs.slotSize);
                    stateCopy.ra.release(state.inputs.beta_real);
                    stateCopy.ra.release(state.inputs.beta_imag);
                    stateCopy.ra.release(state.inputs.ao);
                    stateCopy.ra.release(state.inputs.bo);
                    stateCopy.ra.release(slotW);
                }

                stateCopy.inputs.S = !copyB ? state.inputs.A : state.inputs.B;
                stateCopy.inputs.D = state.inputs.X;
                stateCopy.inputs.surfaceS = !copyB ? state.inputs.surfaceA : state.inputs.surfaceB;
                stateCopy.inputs.surfaceD = state.inputs.surfaceX;
                stateCopy.inputs.offsetS = TASK_FIELD(Task::offsetA);
                stateCopy.inputs.offsetD = offsetD;
                stateCopy.inputs.lds = !copyB ? state.inputs.lda : state.inputs.ldb;
                stateCopy.inputs.ldd = TASK_FIELD(Task::ldx);
                stateCopy.inputs.m = TASK_FIELD(Task::m).d();
                stateCopy.inputs.n = TASK_FIELD(Task::n).d();
                if (problem.alphaScaledCopy() && !copyB) {
                    stateCopy.inputs.alpha_real = state.inputs.alpha_real;
                    stateCopy.inputs.alpha_imag = state.inputs.alpha_imag;
                }
                stateCopy.inputs.localIDW = state.inputs.localID;
                stateCopy.inputs.groupIDW = state.groupIDX;
                if (strategyCopy.zParallel) {
                    stateCopy.inputs.blockZ = TASK_FIELD(Task::k_z0);
                    stateCopy.inputs.groupIDZ = state.groupIDY;
                }
                stateCopy.fusedGEMM.active = true;
                stateCopy.isNested = true;
                stateCopy.lid0 = localIDFlat;

                if (!problem.mock.copies) {
                    status << "AB"[copyB] << "NT"[trans] << " copy kernel" << status_stream::endl;
                    copy(problemCopy, strategyCopy, stateCopy);
                }

                status << "Increment copy counter" << status_stream::endl;

                auto incHeader = state.ra.alloc();

                useR0(state, [&](const GRF &r0_info) {
                    globalMemBarrier(incHeader, r0_info, strategy);
                });

                shl<uint32_t>(1, incHeader[0], TASK_FIELD(Task::slotA), 2);
                cmp(1 | eq | f1[0], localIDFlat, 0);
                add<uint32_t>(1, incHeader[0], incHeader[0], 4);
                if (a64) eadd(1, incHeader.uq(0), incHeader.ud(0), state.inputs.plan, strategy, state);
                atomic(AtomicOp::inc, 1 | f1[0], scattered_dword(), planBase, incHeader);
                state.ra.safeRelease(incHeader);

                strategy.persistent ? jmpi(1, lPersistentLoopTop)
                                    : epilogue(strategy, state);
            }
        }
    }

    state.ra.safeRelease(task[0]);
#undef TASK_FIELD

    // Early exit (no more work). r0 has not been moved yet.
    mark(lExit);
    state.r0_info = r0;
    epilogue(strategy, state);
    padding();
}

// Initialize state for gemmFused.
template <HW hw>
void Generator<hw>::gemmFusedInitState(FusedGEMMProblem &problem, const FusedGEMMStrategy &strategy, FusedGEMMState &state)
{
    using namespace fused;
    initState(problem, strategy, state);

    interface.requireSLM(std::max(sizeof(FusedTask), gemmSLMSize(hw, problem.gemm, strategy.gemm)));
    interface.requireBarrier();
    if (strategy.gemm.systolic)
        interface.requireDPAS();
    interface.requireGlobalAtomics();
    interface.requireWorkgroup(strategy.wg() * strategy.subgroupSize, 1, 1);
    interface.finalize();

#if defined(ASM_OUTPUT)
    interface.generateDummyCL(std::cerr);
#endif

    // Get input arguments.
    state.inputs.groupID = r0.ud(1);
    state.inputs.localID = interface.getLocalID(0);

    state.inputs.A = interface.getArgumentIfExists("A");
    state.inputs.B = interface.getArgumentIfExists("B");
    state.inputs.C = interface.getArgumentIfExists("C");
    state.inputs.X = interface.getArgumentIfExists("X");
    state.inputs.plan = interface.getArgumentIfExists("plan");
    state.inputs.surfaceA = interface.getArgumentSurfaceIfExists("A");
    state.inputs.surfaceB = interface.getArgumentSurfaceIfExists("B");
    state.inputs.surfaceC = interface.getArgumentSurfaceIfExists("C");
    state.inputs.surfaceX = interface.getArgumentSurfaceIfExists("X");
    state.inputs.surfacePlan = interface.getArgumentSurfaceIfExists("plan");

    state.inputs.lda = interface.getArgumentIfExists("lda");
    state.inputs.ldb = interface.getArgumentIfExists("ldb");
    state.inputs.ldc = interface.getArgumentIfExists("ldc");
    state.inputs.alpha_real = interface.getArgumentIfExists("alpha_real");
    state.inputs.alpha_imag = interface.getArgumentIfExists("alpha_imag");
    state.inputs.beta_real = interface.getArgumentIfExists("beta_real");
    state.inputs.beta_imag = interface.getArgumentIfExists("beta_imag");

    if (problem.gemm.aOffset != ABOffset::None) {
        auto abo = interface.getArgumentIfExists("abo");
        if (abo.isValid()) {
            // A/B offset are two words packed into a single dword argument.
            state.inputs.ao = abo.w(0);
            state.inputs.bo = abo.w(1);
        } else {
            state.inputs.ao = interface.getArgument("ao");
            state.inputs.bo = interface.getArgument("bo");
        }
    }

    if (problem.gemm.usesCOPtr()) {
        state.inputs.CO = interface.getArgumentIfExists("co_ptr");
        state.inputs.surfaceCO = interface.getArgumentSurfaceIfExists("co_ptr");
    }

    state.inputs.copyTasks = interface.getArgument("copyTasks");
    state.inputs.gemmTasks = interface.getArgument("gemmTasks");
    state.inputs.slotSize = interface.getArgument("slotSize");
    state.inputs.cc0Trigger = interface.getArgument("cc0Trigger");
    state.inputs.cc1Trigger = interface.getArgument("cc1Trigger");

    auto ttsc = interface.getArgument("tidThreshSlotCount1");
    state.inputs.tidThresh = ttsc.uw(0);
    state.inputs.slotCount1 = ttsc.uw(1);

    // For now, reinterpret ld variables to 32-bit if they are 64-bit.
    state.inputs.lda = state.inputs.lda.ud();
    state.inputs.ldb = state.inputs.ldb.ud();
    state.inputs.ldc = state.inputs.ldc.ud();

    // Claim input registers.
    state.ra.claim(r0);

    bool needABase = (strategy.copy[0][0].S.base.isStateless() || strategy.copy[0][1].S.base.isStateless());
    bool needBBase = (strategy.copy[1][0].S.base.isStateless() || strategy.copy[1][1].S.base.isStateless());
    bool needCBase = strategy.gemm.C.base.isStateless();
    bool needCOBase = strategy.gemm.CO.base.isStateless();
    bool needXBase = strategy.gemm.A.base.isStateless() || strategy.gemm.B.base.isStateless();
    for (bool copyB : {false, true})
        for (bool trans: {false, true})
            needXBase |= strategy.copy[copyB][trans].D.base.isStateless();
    bool needPBase = strategy.statelessPlan;

    if (needABase) state.ra.claim(state.inputs.A);
    if (needBBase) state.ra.claim(state.inputs.B);
    if (needCBase) state.ra.claim(state.inputs.C);
    if (needCOBase) state.ra.claim(state.inputs.CO);
    if (needXBase) state.ra.claim(state.inputs.X);
    if (needPBase) state.ra.claim(state.inputs.plan);
    if (!needABase) state.inputs.A = invalid;
    if (!needBBase) state.inputs.B = invalid;
    if (!needCBase) state.inputs.C = invalid;
    if (!needCOBase) state.inputs.CO = invalid;
    if (!needXBase) state.inputs.X = invalid;
    if (!needPBase) state.inputs.plan = invalid;

    state.ra.claim(state.inputs.lda);
    state.ra.claim(state.inputs.ldb);
    state.ra.claim(state.inputs.ldc);

    state.ra.claim(state.inputs.alpha_real);
    state.ra.claim(state.inputs.beta_real);

    if (problem.gemm.aOffset != ABOffset::None) {
        state.ra.claim(state.inputs.ao);
        state.ra.claim(state.inputs.bo);
    }

    state.ra.claim(state.inputs.copyTasks);
    state.ra.claim(state.inputs.gemmTasks);
    state.ra.claim(state.inputs.tidThresh);
    state.ra.claim(state.inputs.slotSize);
    state.ra.claim(state.inputs.cc0Trigger);
    state.ra.claim(state.inputs.cc1Trigger);
    state.ra.claim(state.inputs.localID);

    state.allocEmulate64Temp(strategy.emulate);
}

// Callback from gemmSubkernel when it is ready for next copies to start.
// Called twice: (1) late = false: just after k loop, when done reading A/B
//               (2) late = true:  at end of GEMM kernel (used for early exit + k blocking)
template <HW hw>
void Generator<hw>::gemmFusedFinishRead(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, bool late)
{
    using namespace fused;
    if (!state.fusedGEMM.active)
        return;

    status << "Increment GEMM counters" << status_stream::endl;

    Label lSkip;
    auto incHeader = state.ra.alloc();
    bool a64 = state.fusedGEMM.plan.isValid();

    auto flag = late ? state.fusedGEMM.needLateGEMMDone
                     : state.flagAP;
    auto mod  = late ? (1 | ~flag | nz | flag)
                     : (1 | nz | flag);
    and_<uint32_t>(mod, null.ud(), state.inputs.flags, late ? FlagLateFusedGEMMDone : FlagEarlyFusedGEMMDone);
    jmpi(1 | ~flag, lSkip);

    useR0(state, [&](const GRF &r0_info) {
        barrier(incHeader, r0_info);
    });
    cmp(2 | eq | state.flagAP, state.fusedGEMM.localIDFlat, 0);
    shl<uint32_t>(2, incHeader, state.fusedGEMM.slotA(1), 2);
    if (a64) {
        add<uint32_t>(2, incHeader[0](2), incHeader, sizeof(FusedTask) + 4);
        eadd(2, incHeader.uq(0)(1), incHeader.ud(0)(2), state.fusedGEMM.plan, strategy, state);
    } else
        add<uint32_t>(2, incHeader, incHeader, sizeof(FusedTask) + 4);
    atomic(AtomicOp::inc, 2 | state.flagAP, scattered_dword(), a64 ? A64 : Surface(state.fusedGEMM.surfacePlan), incHeader);

    mark(lSkip);
    state.ra.safeRelease(incHeader);
}

GEMMSTONE_NAMESPACE_END
