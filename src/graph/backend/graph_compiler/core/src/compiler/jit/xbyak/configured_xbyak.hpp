/*******************************************************************************
 * Copyright 2021-2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_CONFIGURED_XBYAK_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_CONFIGURED_XBYAK_HPP

#ifdef XBYAK_XBYAK_H_
#error "Don't #include xbyak.h directly! #include this file instead."
#endif

/**
 * WARNING!!!
 * xbyak.h will define certain classes, etc. differently depending on if/how
 * certain preprocessor variables are defined prior to #include<xbyak.h>.
 *
 * In order to avoid mayhem that would stem from defining these things
 * differently in MKLDNN vs. Graphcompiler, #include *this* header file instead
 * of directly #include<xbyak.h>. This header file does what's needed to match
 * MKLDNN.
 */

/// NOTE: The following #define statements, etc. are copied from
/// 3rdparty/mkl-dnn/src/cpu/X86/cpu_isa_traits.hpp to ensure
/// consistency.

/// TODO: If/when such consistency is no longer required between
/// Graphcompiler's and mkldnn's Xbyak usage, we should reexamine the design
/// choices implied by these #defines.

#define XBYAK64
#define XBYAK_NO_OP_NAMES

/* in order to make selinux happy memory that would be marked with X-bit should
 * be obtained with mmap */
#define XBYAK_USE_MMAP_ALLOCATOR

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
/* turn off `size_t to other-type implicit casting` warning
 * currently we have a lot of jit-generated instructions that
 * take uint32_t, but we pass size_t (e.g. due to using sizeof).
 * FIXME: replace size_t parameters with the appropriate ones */
#pragma warning(disable : 4267)
#endif

#include <common/compiler_workarounds.hpp>

#if defined(__GNUG__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

#if defined(__GNUG__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#endif
