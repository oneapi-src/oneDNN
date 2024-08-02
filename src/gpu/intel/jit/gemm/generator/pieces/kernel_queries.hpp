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


#ifndef GEMMSTONE_GUARD_KERNEL_QUERIES_HPP
#define GEMMSTONE_GUARD_KERNEL_QUERIES_HPP

#include "internal/ngen_includes.hpp"
#include "problem.hpp"
#include "strategy.hpp"
#include "state.hpp"

#include "internal/namespace_start.hxx"


// Return amount of SLM needed by a GEMM kernel.
size_t gemmSLMSize(ngen::HW hw, const GEMMProblem &problem, const GEMMStrategy &strategy, bool computeMax = false);

// Return amount of per-k SLM needed by a GEMM kernel.
size_t gemmPerKSLMSize(ngen::HW hw, const GEMMProblem &problem, const GEMMStrategy &strategy);

// Decide whether C layout needs m/n remainder handling.
void getCRemainders(ngen::HW hw, const GEMMProblem &problem, const GEMMStrategy &strategy, bool &remM_C, bool &remN_C);

// Check if i0/j0 need to be saved across the k loop.
bool keepIJ0(const GEMMProblem &problem, const GEMMStrategy &strategy);

// Check if h0 needs to be saved across the k loop.
bool keepH0(const GEMMProblem &problem, const GEMMStrategy &strategy);

#include "internal/namespace_end.hxx"

#endif /* header guard */
