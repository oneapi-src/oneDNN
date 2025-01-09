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

using namespace ngen;

#include "internal/namespace_start.hxx"

// goto instruction with Gen12 semantics.
template <HW hw>
void BLASKernelGenerator<hw>::goto12(const InstructionModifier &mod, Label &jip, Label &uip, bool branchCtrl)
{
    InstructionModifier mmod = mod;
    if (hw == HW::Gen9 && !branchCtrl) {
        if (mmod.getPredCtrl() == PredCtrl::None) stub();
        mmod.setPredInv(!mmod.isPredInv());
    }
    goto_(mmod, jip, uip, branchCtrl);
}

// Compare to zero.
template <HW hw>
void BLASKernelGenerator<hw>::cmp0(const InstructionModifier &mod, RegData src0)
{
    mov(mod, null.retype(src0.getType()), abs(src0));
}

// Synchronize on all pipes and OOO operations.
template <HW hw>
void BLASKernelGenerator<hw>::syncall()
{
    if (hw == HW::Gen12LP)
        sync.allwr(SWSB(1));
    else if (hw >= HW::XeHP)
        sync.allwr(SWSB<AllPipes>(1));
}

// Simple do-while loop macro for the backward conditional branch at end of loop.
template <HW hw>
void BLASKernelGenerator<hw>::simtDoWhileLoop(const InstructionModifier &mod, Label &dest)
{
    Label next;

    goto12(mod, next, dest, true);
    mark(next);
    join(mod.getExecSize());
}

// Barrier for all threads in the workgroup, except padding threads.
template <HW hw>
void BLASKernelGenerator<hw>::activeThreadBarrier(const GRF &temp, const GRF &r0_info, const CommonStrategy &strategy)
{
    if (hw >= HW::XeHPG && strategy.activeThreads > 0)
        barrier(temp, strategy.activeThreads, r0_info);
    else
        barrier(temp, r0_info);
}

template <HW hw>
void BLASKernelGenerator<hw>::activeThreadBarrierSignal(const GRF &temp, const GRF &r0_info, const CommonStrategy &strategy)
{
    if (hw >= HW::XeHPG && strategy.activeThreads > 0)
        barriersignal(temp, strategy.activeThreads, r0_info);
    else
        barriersignal(temp, r0_info);
}

// Barrier with SLM fence.
template <HW hw>
void BLASKernelGenerator<hw>::slmBarrier(const GRF &temp, const GRF &r0_info, const CommonStrategy &strategy)
{
    if (hw >= HW::Gen11) {
        slmfence(temp, r0_info);
        fencewait();
    }
    activeThreadBarrier(temp, r0_info, strategy);
}

// Global memory fence.
template <HW hw>
void BLASKernelGenerator<hw>::globalMemFence(const GRF &temp, const GRF &r0_info, const CommonStrategy &strategy)
{
    if (hw >= HW::XeHPG && !strategy.multitile)
        memfence(FenceScopeLSC::Tile, FlushTypeLSC::None, temp, r0_info);
    else
        memfence(temp, r0_info);
}

// Barrier with global memory fence.
template <HW hw>
void BLASKernelGenerator<hw>::globalMemBarrier(const GRF &temp, const GRF &r0_info, const CommonStrategy &strategy)
{
    globalMemFence(temp, r0_info, strategy);
    fencewait();
    activeThreadBarrier(temp, r0_info, strategy);
}

// Pause for a short period of time.
template <HW hw>
void BLASKernelGenerator<hw>::pause(const CommonStrategy &strategy)
{
    if (hw == HW::Gen9) for (int i = 0; i < 8; i++)
        mov<uint32_t>(8 | Switch, null, acc0);
    else if (hw >= HW::Gen11 && hw != HW::XeHPC)
        mov(1 | Switch, tm0[4], strategy.pauseCycles);
    else for (int i = 0; i < (strategy.pauseCycles / 8); i++)
        mov<uint32_t>(1 | SWSB(1), null, acc0);
}

// Clear read suppresion data on ALU pipes.
template <HW hw>
void BLASKernelGenerator<hw>::doReadSuppressionWA(const CommonStrategy &strategy, CommonState &state)
{
    GRF temp;
    bool freeTemp = false;

    if (!strategy.readSuppressionWA)
        return;

    temp = state.ra.try_alloc();
    if (temp.isValid())
        freeTemp = true;
    else
        temp = GRF(strategy.GRFs - 1);

    auto rI = temp.uw(0)(1);
    auto rF = temp.f(4)(1);

    csel(4, rI, rI, rI, rI);    // Clear read suppression data in int pipe.
    csel(4, rF, rF, rF, rF);    // Clear read suppression data in float pipe.

    if (freeTemp)
        state.ra.safeRelease(temp);
}

#include "internal/namespace_end.hxx"
