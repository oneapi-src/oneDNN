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


#ifndef GEMMSTONE_GUARD_TOKEN_ALLOC_UTILS_HPP
#define GEMMSTONE_GUARD_TOKEN_ALLOC_UTILS_HPP

#include "internal/ngen_includes.hpp"
#include "strategy.hpp"
#include "state.hpp"

#include "internal/namespace_start.hxx"

// Allocate tokens for a layout keyed to the given source/destination registers.
bool allocateTokens(const std::vector<RegisterBlock> &layout, const GRFMultirange &regs, CommonState &state,
                    const std::vector<ngen::GRFRange> &addrs = std::vector<ngen::GRFRange>());

// Clear token allocations that are mapped to specific registers.
void clearMappedTokenAllocations(ngen::HW hw, CommonState &state);

// Clear all token allocations.
void clearTokenAllocations(ngen::HW hw, CommonState &state);

#include "internal/namespace_end.hxx"

#endif /* header guard */
