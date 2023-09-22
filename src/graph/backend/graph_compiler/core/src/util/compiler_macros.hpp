/*******************************************************************************
 * Copyright 2022-2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_COMPILER_MACROS_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_COMPILER_MACROS_HPP

#if defined(__GNUC__)
#define SC_GNUC_VERSION_GE(x) (__GNUC__ >= (x))
#else
#define SC_GNUC_VERSION_GE(x) 0
#endif

#if defined(__GNUC__)
#define SC_GNUC_VERSION_LT(x) (__GNUC__ < (x))
#else
#define SC_GNUC_VERSION_LT(x) 0
#endif

// a macro which does nothing. It is used as placeholder which workarounds a
// bug in parsing of "omp parallel for" in icx compiler
#define SC_NO_OP()

#if defined(__INTEL_LLVM_COMPILER)
// the dpcpp spec says that we also need to find existence of
// SYCL_LANGUAGE_VERSION, which is not true
#define SC_IS_DPCPP() 1
#else
#define SC_IS_DPCPP() 0
#endif

#if defined(__clang__)
#define SC_IS_CLANG() 1
#else
#define SC_IS_CLANG() 0
#endif

#ifdef _MSC_VER
// returns if the compiler is really MSVC (not dpcpp simulation)
#define SC_IS_MSVC() (!SC_IS_DPCPP() && !SC_IS_CLANG())
#else
#define SC_IS_MSVC() 0
#endif

#endif
