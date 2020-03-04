/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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

#ifndef OS_BLAS_HPP
#define OS_BLAS_HPP

/** \file
 * DNNL provides gemm functionality on its own using jit generated
 * kernels. This is the only official supported option for the
 * default build target.
 *
 * However, for non-jit builds and debugging (ex. CPU dispatch to non-jit code)
 * we keep the ability to use cblas functions from the other libraries. The
 * following macros affect the behavior:
 *
 * - DNNL_USE_CBLAS allow using sgemm and other regular BLAS functionality
 * - DNNL_USE_MKL (implies DNNL_USE_CBLAS) same as above + allow using igemm
 *   and packed gemm from Intel MKL library.
 *
 *  New:  USE_MKL and USE_CBLAS compile flags were replaced by above cmake flags
 *  that are set like
 *    `cmake -DDNNL_CPU_EXTERNAL_GEMM=NONE|MKL|CBLAS`.
 *  USE_MKL and USE_CBLAS are now propagated via dnnl_config.h & cpu_target.h
 */
#include "cpu_target.h" // we want USE_MKL and USE_CBLAS shortcuts
#include "dnnl_config.h" // provides DNNL_USE_MKL and DNNL_USE_CBLAS

#if defined(USE_MKL)

#include "mkl_version.h"

#define USE_MKL_PACKED_GEMM (INTEL_MKL_VERSION >= 20170000)
#define USE_MKL_IGEMM \
    (INTEL_MKL_VERSION >= 20180000 && __INTEL_MKL_BUILD_DATE >= 20170628)

#include "mkl_cblas.h"
#if !defined(USE_CBLAS)
// XXX deprecated configuration? no longer available via cmake
#error "Is it safe to deprecate this case?"
#define cblas_sgemm(...) assert(!"CBLAS is unavailable")
#endif

#else /* defined(USE_MKL) */

#define USE_MKL_PACKED_GEMM 0
#define USE_MKL_IGEMM 0

#if defined(USE_CBLAS)

#if defined(_SX)
extern "C" {
#endif

#include "cblas.h" /* Maybe a system/cmake cblas works for you? */

#if defined(_SX)
}
#endif

#else /* defined(USE_CBLAS) */

/* put the stubs to make a code compilable but not workable */
#define cblas_sgemm(...) assert(!"CBLAS is unavailable")

#endif /* defined(USE_CBLAS) */
#endif /* defined(USE_MKL) */

namespace mkldnn {
namespace impl {
namespace cpu {

#if defined(USE_MKL) && defined(USE_CBLAS)
typedef MKL_INT cblas_int;

#elif defined(USE_CBLAS)
typedef int cblas_int;

#if defined(_SX) /* this cblas.h is peculiar... */
typedef CBLAS_ORDER CBLAS_LAYOUT;
#endif
#endif

} // namespace cpu
} // namespace impl
} // namespace mkldnn

#endif /* OS_BLAS_HPP */

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
