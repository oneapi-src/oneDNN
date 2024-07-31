/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
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

#ifndef STRATEGY_PARSER_HPP
#define STRATEGY_PARSER_HPP

#include "gen_gemm_kernel_generator.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

void parseStrategy(const char *str, ngen::HW hw, const GEMMProblem &problem,
        GEMMStrategy &strategy);

void adjustStrategy(ngen::HW hw, const GEMMProblem &problem,
        GEMMStrategy &strategy, const char *tags = nullptr);

const char *parsePrecision(const char *s, Type &precision);
const char *parsePrecisions(const char *s, Type &precision1, Type &precision2);

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif /* header guard */
