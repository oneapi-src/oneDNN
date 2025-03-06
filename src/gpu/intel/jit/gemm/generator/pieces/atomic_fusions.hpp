/*******************************************************************************
* INTEL CONFIDENTIAL
* Copyright 2025 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/


#ifndef GEMMSTONE_GUARD_ATOMIC_FUSIONS_HPP
#define GEMMSTONE_GUARD_ATOMIC_FUSIONS_HPP

#include "problem.hpp"
#include "strategy.hpp"

#include "internal/namespace_start.hxx"

// Calculate per-thread stride within temporary C memory.
inline int tempCThreadStride(const GEMMProblem &problem, const GEMMStrategy &strategy)
{
    int stride = strategy.unroll[LoopM] * strategy.unroll[LoopN];
    if (problem.sumA) stride += strategy.unroll[LoopM];
    if (problem.sumB) stride += strategy.unroll[LoopN];
    stride *= problem.Tc;
    stride = align_up(stride, 64);
    return stride;
}


// Calculate per-workgroup stride within temporary C memory.
inline int tempCWGStride(const GEMMProblem &problem, const GEMMStrategy &strategy) {
    return tempCThreadStride(problem, strategy) * strategy.wg[LoopM] * strategy.wg[LoopN];
}

#include "internal/namespace_end.hxx"

#endif /* header guard */
