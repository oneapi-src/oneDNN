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

#ifndef MICROKERNEL_PROVIDER_HPP
#define MICROKERNEL_PROVIDER_HPP

#include "config.hpp"
#include "gpu/intel/microkernels/package.hpp"
#include "kernel_selector.hpp"
#include "kernel_evaluator.hpp"

#include "internal/namespace_start.hxx"

/* Hardware information for microkernel provider */
struct HWInformation {
    uint32_t gmdid;
    int euCount;
    bool systolicAvailable;
};

/* Main entrypoint for microkernel auto-selection */
micro::Package selectGEMMMicrokernel(micro::GEMMProtocol protocol, HWInformation hwInfo, SizeParams sizes, const GEMMProblem &problem,
                                     const std::vector<StrategyRequirement> &reqs = std::vector<StrategyRequirement>(),
                                     void (*strategyAdjuster)(GEMMStrategy &strategy) = nullptr);

/* Helpers */
static inline int alignmentForLD(int ld)
{
    for (int x = 1; x <= 64; x <<= 1)
        if (ld & x) return x;
    return 128;
};

#include "internal/namespace_end.hxx"

#endif /* header guard */
