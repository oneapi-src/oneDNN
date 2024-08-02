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


#ifndef GEMMSTONE_GUARD_COMPUTE_UTILS_HPP
#define GEMMSTONE_GUARD_COMPUTE_UTILS_HPP

#include <tuple>

#include "layout_utils.hpp"
#include "problem.hpp"
#include "strategy.hpp"

#include "internal/namespace_start.hxx"


// The systolic array performs a series of GEMVs with a single fixed-size matrix.
// The size of the matrix is osys x ksys with vectors of size ksys x 1.
// The number of GEMVs (with same matrix) is given by the (variable) repeat count.
struct SystolicParams {
    int opsPerChan;     // # of FMAs/stage
    int sdepth;         // Number of stages (systolic depth)
    int rcountMax;      // Maximum repeat count (# of RHS)
    int ksys;           // Total number of FMAs
    int osys;           // Output vector length
};

static inline SystolicParams systolicParams(ngen::HW hw, const GEMMProblem &problem, const GEMMStrategy &strategy)
{
    SystolicParams params;
    params.opsPerChan = std::max(1, std::min(4 / problem.Ta.real(), 4 / problem.Tb.real()));
    params.sdepth = 8;
    params.ksys = params.sdepth * params.opsPerChan;
    params.osys = ngen::GRF::bytes(hw) / std::max(problem.Tc_compute().real().size(), 4);
    params.rcountMax = 8;

    return params;
}

// Return # of outer products performed at once.
static inline int minOuterProductCount(ngen::HW hw, const GEMMProblem &problem, const GEMMStrategy &strategy)
{
    if (strategy.systolic) {
        auto params = systolicParams(hw, problem, strategy);
        return params.ksys;
    }
    if (strategy.dotVL)
        return strategy.dotVL;
    if (hw >= ngen::HW::Gen12LP && problem.isIGEMM())
        return 4;
    return 1;
}

// Return # of outer products performed at once.
static inline int outerProductCount(ngen::HW hw, const GEMMProblem &problem, const GEMMStrategy &strategy)
{
    return minOuterProductCount(hw, problem, strategy) * strategy.kChain;
}

// Get the A and B crosspacks needed by the kernel. 0 indicates any crosspack is OK.
static inline std::tuple<int,int> targetKernelCrosspack(ngen::HW hw, const GEMMProblem &problem, const GEMMStrategy &strategy)
{
    int opBatch = minOuterProductCount(hw, problem, strategy);
    bool aColMajor = isRegisterColMajor(problem.Ta, problem.A, strategy.A);
    bool bColMajor = isRegisterColMajor(problem.Tb, problem.B, strategy.B);
    bool cColMajor = isRegisterColMajor(problem.Tc, problem.C, strategy.C);

    if (strategy.systolic) {
        return cColMajor ? std::make_tuple(std::max(1, 4 / problem.Ta.real()), 1)
                         : std::make_tuple(1, std::max(1, 4 / problem.Tb.real()));
    }
    if (strategy.dotVL)
        return std::make_tuple(1, 1);
    if (opBatch == 1) {
        return cColMajor ? std::make_tuple(1, 0)
                         : std::make_tuple(0, 1);
    } else {
        bool bcastOK = cColMajor ? bColMajor : !aColMajor;

        return cColMajor ? std::make_tuple(opBatch, bcastOK ? 1 : opBatch)
                         : std::make_tuple(bcastOK ? 1 : opBatch, opBatch);
    }
}

// Get the A and B crosspacks to use for SLM data.
static inline std::tuple<int,int> targetSLMCrosspack(ngen::HW hw, const GEMMProblem &problem, const GEMMStrategy &strategy)
{
    int opBatch = minOuterProductCount(hw, problem, strategy);

    if (strategy.systolic) {
        bool cColMajor = isRegisterColMajor(problem.Tc, problem.C, strategy.C);
        return cColMajor ? std::make_tuple(std::max(1, 4 / problem.Ta.real()), opBatch)
                         : std::make_tuple(opBatch, std::max(1, 4 / problem.Tb.real()));
    }
    if (strategy.dotVL)
        return std::make_tuple(1, 1);

    return std::make_tuple(opBatch, opBatch);
}

// Get the A and B tiling needed by the kernel.
// Return value is in the format {A_tileR, A_tileC, B_tileR, B_tileC}.
static inline std::tuple<int,int,int,int> targetKernelTiling(ngen::HW hw, const GEMMProblem &problem, const GEMMStrategy &strategy)
{
    if (strategy.systolic) {
        auto params = systolicParams(hw, problem, strategy);
        bool cColMajor = isRegisterColMajor(problem.Tc, problem.C, strategy.C);
        auto tileO_V = params.osys;
        auto tileI_N = params.ksys;
        if (strategy.unroll[cColMajor ? LoopN : LoopM] == 1)
            tileI_N = 0;
        return cColMajor ? std::make_tuple(tileO_V, 0, tileI_N, 0)
                         : std::make_tuple(0, tileI_N, 0, tileO_V);
    }
    return std::make_tuple(0,0,0,0);
}

#include "internal/namespace_end.hxx"

#endif
