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


#include "state_utils.hpp"
#include "generator.hpp"
#include "hw_utils.hpp"

using namespace ngen;
using std::vector;

#include "internal/namespace_start.hxx"


// Release fused remainder-related state variables.
void releaseFusedRemainders(GEMMState &state)
{
    state.ra.safeRelease(state.remFusedStorage);
    state.remaindersFused[LoopM] = Subregister{};
    state.remaindersFused[LoopN] = Subregister{};
}

void releaseCoopRemainders(GEMMState &state)
{
    for (LoopType loop: {LoopM, LoopN, LoopK})
        if (state.remaindersCoop[loop] != state.remainders[loop])
            state.ra.safeRelease(state.remaindersCoop[loop]);
}

template <HW hw>
void BLASKernelGenerator<hw>::saveMNLocalIDs(const GEMMStrategy &strategy, GEMMState &state)
{
    state.lidStorage = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
    state.lidM = state.lidStorage.uw(0);
    state.lidN = state.lidStorage.uw(1);
    mov(1, state.lidM, state.inputs.localIDM);
    mov(1, state.lidN, state.inputs.localIDN);
}

template <HW hw>
void BLASKernelGenerator<hw>::saveKLocalIDSize(const GEMMStrategy &strategy, GEMMState &state)
{
    state.lidszKStorage = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
    state.lidK = state.lidszKStorage.uw(0);
    state.lszK = state.lidszKStorage.uw(1);
    mov(1, state.lidK, state.inputs.localIDK);
    mov(1, state.lszK, state.inputs.localSizeK);
}

template <HW hw>
void BLASKernelGenerator<hw>::releaseSavedMNLocalIDs(GEMMState &state)
{
    state.ra.safeRelease(state.lidStorage);
    state.lidStorage = invalid;
    state.lidM = invalid;
    state.lidN = invalid;
}

// Allocate temporary registers for emulating atomic addition.
void allocEAtomicAddRegs(HW hw, Type T, const vector<RegisterBlock> &layout,
                         const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy, CommonState &state,
                         const FlagRegister &flag)
{
    if (hasNativeAtomicAdd(hw, T.real(), atype, astrategy)) return;

    int maxNReg = 0;
    for (const auto &block : layout)
        maxNReg = std::max(maxNReg, block.nregs());

    if (maxNReg == 0) return;

    state.eatomicAddRegs[0] = state.ra.alloc_range(maxNReg * 2);
    state.eatomicAddRegs[1] = state.ra.alloc_range(maxNReg);
    state.vflagEAtomicAdd = flag.isValid() ? flag
                                           : state.allocVFlag(hw);
}

void freeEAtomicAddRegs(CommonState &state, const FlagRegister &flag)
{
    state.ra.safeRelease(state.eatomicAddRegs[0]);
    state.ra.safeRelease(state.eatomicAddRegs[1]);
    if (flag.isInvalid())
        state.raVFlag.release(state.vflagEAtomicAdd);
}

void releaseMaskAssignments(vector<MaskAssignment> &assignments, CommonState &state, int start)
{
    for (size_t an = start; an < assignments.size(); an++)
        state.raVFlag.release(assignments[an].flag);

    state.wipeActiveVFlags();
}

void reclaimMaskAssignments(vector<MaskAssignment> &assignments, CommonState &state, int start)
{
    for (size_t an = start; an < assignments.size(); an++)
        state.raVFlag.claim(assignments[an].flag);
}

void safeReleaseMaskAssignments(vector<MaskAssignment> &assignments, CommonState &state, int start)
{
    releaseMaskAssignments(assignments, state, start);
    assignments.resize(start);
}

RegData getMaskFlag(HW hw, VirtualFlag vflag, CommonState &state)
{
    if (state.vflagsEnabled()) {
        return state.vflagStorage.sub(hw, vflag.idx, DataType::uw)
                                 .reinterpret(0, vflag.n == 2 ? DataType::ud : DataType::uw);
    } else if (!state.raVFlag.isVirtual(vflag)) {
        auto pflag = vflag.toPhysical();
        state.usePhysicalFlag(pflag);
        return pflag;
    } else
        stub("Need virtual flag registers");
}


#include "internal/namespace_end.hxx"
