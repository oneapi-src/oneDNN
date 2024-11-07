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

#include "internal/namespace_end.hxx"
