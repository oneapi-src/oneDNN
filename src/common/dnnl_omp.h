/*******************************************************************************
 * Copyright 2020 NEC Labs America, Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License"); you may not
* use this file except in compliance with the License.  You may obtain a copy
* of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
* License for the specific language governing permissions and limitations under
* the License.
*******************************************************************************/
#ifndef DNNL_OMP_H
#define DNNL_OMP_H
/** \file
 * More standalone than dnnl_threading, \b only defining macros.
 *
 * - PRAGMA_OMP(...)
 * - PRAGMA_OMP_SIMD(...)
 *
 * Both will expand to nothing if you pre-defining ENABLE_OMP_MACROS 0.
 * Include dnnl_thread.hpp instead if you want to use the balancing
 * and parallel_nd tools.
 *
 * Typically included via \ref dnnl_thread.hpp, which adds \c parallel_nd etc.
 *
 * \sa dnnl_optimize.h for compiler-specific optimizations unrelated to OpenMP
 */

#include "z_magic.hpp"

#if !defined(ENABLE_OMP_MACROS)
#define ENABLE_OMP_MACROS 1
#endif

#if !ENABLE_OMP_MACROS
#define PRAGMA_OMP(...)
#define PRAGMA_OMP_SIMD(...)

#else
#define PRAGMA_OMP(...) PRAGMA_MACRO(CHAIN2(omp, __VA_ARGS__))

#if defined(_MSC_VER) && !defined(__clang__) && !defined(__INTEL_COMPILER)
#define collapse(x)
#endif

#ifndef PRAGMA_OMP_SIMD
// MSVC still supports omp 2.0 only
#if defined(_MSC_VER) && !defined(__clang__) && !defined(__INTEL_COMPILER)
#define PRAGMA_OMP_SIMD(...)
#elif defined(__ve)
#define PRAGMA_OMP_SIMD(...)
#else
#define PRAGMA_OMP_SIMD(...) PRAGMA_MACRO(CHAIN2(omp, simd __VA_ARGS__))
#endif // defined(_MSC_VER) && !defined(__clang__) && !defined(__INTEL_COMPILER)
#endif // ndef PRAGMA_OMP_SIMD

// process simdlen; it is supported for Clang >= 3.9; ICC >= 17.0; GCC >= 6.1
// No support on Windows.
#if (defined(__clang_major__) \
        && (__clang_major__ < 3 \
                || (__clang_major__ == 3 && __clang_minor__ < 9))) \
        || (defined(__INTEL_COMPILER) && __INTEL_COMPILER < 1700) \
        || (!defined(__INTEL_COMPILER) && !defined(__clang__) \
                && (defined(_MSC_VER) || __GNUC__ < 6 \
                        || (__GNUC__ == 6 && __GNUC_MINOR__ < 1)))
#define simdlen(x)
#endif // long simdlen if

// deprecated ancient macros with dnnl equivalents
#define OMPSIMD(...) ("OMPSIMD deprecated" ^ "use PRAGMA_OMP_SIMD instead")
#define OMP(...) ("OMP deprecated" ^ "use PRAGMA_OMP instead")

#endif // ENABLE_OMP_MACROS
// vim: et ts=4 sw=4 cindent cino=+2s,^=l0,\:0,N-s
#endif // DNNL_OMP_H
