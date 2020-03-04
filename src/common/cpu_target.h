/*******************************************************************************
* Copyright 2020 NEC Labs America
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
#ifndef CPU_TARGET_H
#define CPU_TARGET_H
/** \file
 * extra build-time cpu/isa macros, based dnnl_config.h ONLY.
 * \sa dnnl_config.h
 */

#include "dnnl_config.h"

#if defined(JITFUNCS) || defined(TARGET_VANILLA)
#error "JITFUNCS and TARGET_VANILLA compiler options are removed -- see dnnl_config.h"
#endif

/// @defgroup cpu_target DNNL cpu and JIT macros
/**
 * 
 * can run on any CPU.
 * It removes all xbyak code, and runs... slowly, but faster now that a
 * reference gemm implementation is available.
 *
 * \sa dnnl_config.h
 *
 */
//@{
//@}

// XXX reason for DNNL_X86_64 existing?
//     How is it different from TARGET_X86?  Is this for AARCH64 or such?
#if defined(__x86_64__) || defined(_M_X64)
#define DNNL_X86_64
#endif

// easier to read versions of some dnnl_config.h 0/1 values
#define TARGET_X86 DNNL_TARGET_X86
#define TARGET_X86_JIT DNNL_TARGET_X86_JIT
#define TARGET_VE DNNL_TARGET_VE
#define TARGET_VEDNN DNNL_TARGET_VEDNN
#define TARGET_VEJIT DNNL_TARGET_VEJIT

// src code uses shorter versions of these dnnl_config.h build options
#if defined(DNNL_USE_MKL) && !defined(USE_MKL)
#define USE_MKL DNNL_USE_MKL
#endif
#if defined(DNNL_USE_CBLAS) && !defined(USE_CBLAS)
#define USE_CBLAS DNNL_USE_CBLAS
#endif

// sanity checks...

// exactly one cpu target must be active
#if !TARGET_X86 && !TARGET_VE
#error "cpu target no true TARGET_foo !"
#endif
#if TARGET_X86 + TARGET_VE != 1
#error "exactly one of TARGET_VE and TARGET_X86 should be set"
#endif

// vim: et ts=4 sw=4 cindent cino=+2s,^=l0,\:0,N-s
#endif // CPU_TARGET_H
