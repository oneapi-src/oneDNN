/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef COMPILER_WORKAROUNDS_HPP
#define COMPILER_WORKAROUNDS_HPP

// Workaround 01: clang.
//
// Clang has an issue [1] with `#pragma omp simd` that might lead to segfault.
// The essential conditions are:
//  1. Optimization level is O1 or O2. Surprisingly, O3 is fine.
//  2. Conditional check inside the vectorization loop.
// Since there is no reliable way to determine the first condition, we disable
// vectorization for clang altogether for now.
//
// [1] https://bugs.llvm.org/show_bug.cgi?id=48104
#if (defined __clang_major__) && (__clang_major__ >= 6)
#define CLANG_WA_01_SAFE_TO_USE_OMP_SIMD 0
#else
#define CLANG_WA_01_SAFE_TO_USE_OMP_SIMD 1
#endif

#endif // COMPILER_WORKAROUNDS_HPP
