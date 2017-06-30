/*******************************************************************************
* Copyright 2016-2017 Intel Corporation, NEC Laboratories America LLC
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
 * Common stuff respecting USE_MKL and USE_CBLAS compile flags
 * 
 *  USE_MKL  USE_CBLAS effect
 *  -------  --------- ------
 *  yes      yes       normal compile: jit *may* be preferred over MKL cblas_sgemm
 *  yes      no        jit calls OK; assert if cblas_sgemm is ever called
 *  no       yes       system-dependent (non-MKL) cblas
 *  no       no        gemm convolution (or other blas) N/A; create stubs
 */ 
#ifdef USE_MKL
#include "mkl_cblas.h"
#ifndef USE_CBLAS
#define cblas_sgemm(...) assert(!"CBLAS is unavailable")
#endif

#else // non-MKL: platform-dependent non-jit gemm might be available...
#if defined(_SX)
extern "C" {
#include "cblas.h" // CHECK: does SX also have a fortran API sgemm?
}
// #elif <additional plateforms HERE...>
#else
#include "cblas.h" // Maybe a system/cmake cblas works for you?
#endif
#endif
#endif // OS_BLAS_HPP

namespace mkldnn {
namespace impl {

#if defined(USE_MKL) && defined(USE_CBLAS)
typedef MKL_INT cblas_int;

#elif defined(USE_CBLAS) // modify for your particular cblas.h ...
typedef int cblas_int;

#if defined(_SX) // this cblas.h is peculiar...
typedef CBLAS_ORDER CBLAS_LAYOUT;
#endif //SX
#endif

}//impl::
}//mkldnn::

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
