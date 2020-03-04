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

#ifndef CPU_BARRIER_HPP
#define CPU_BARRIER_HPP

#include "cpu_isa_traits.hpp"
// dnnl DEFAULT always provides a 'barrier' implementation, even for DNNL_RUNTIME_SEQ
//     but DNNL_RUNTIME_SEQ probably should avoid calling barrier()...
//
// so BARRIER_SHOULD_THROW for debug build with DNNL_RUNTIME_SEQ
//     might want to catch illegal attempts to invoke barrier()
#ifdef NDEBUG
#define BARRIER_SHOULD_THROW \
    0 /*dnnl release behavior, always provide barrier()*/
#else
#define BARRIER_SHOULD_THROW (DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_SEQ)
#endif
#if BARRIER_SHOULD_THROW
#include <stdexcept>
#endif

#include <assert.h>

#if TARGET_X86_JIT
#include "jit_generator.hpp"
#endif // TARGET_X86_JIT
#include "utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

namespace simple_barrier {

#ifdef _WIN32
#define CTX_ALIGNMENT 64
#elif TARGET_VE
#define CTX_ALIGNMENT 128 /*VE LLC cache line bytes*/
#else
#define CTX_ALIGNMENT 4096
#endif

STRUCT_ALIGN(
        CTX_ALIGNMENT, struct ctx_t {
            enum { CACHE_LINE_SIZE = 64 };
            volatile size_t ctr;
            char pad1[CACHE_LINE_SIZE - 1 * sizeof(size_t)];
            volatile size_t sense;
            char pad2[CACHE_LINE_SIZE - 1 * sizeof(size_t)];
        });

/* TODO: remove ctx_64_t once batch normalization switches to barrier-less
 * implementation.
 * Different alignments of context structure affect performance differently for
 * convolution and batch normalization. Convolution performance becomes more
 * stable with page alignment compared to cache line size alignment.
 * Batch normalization (that creates C / simd_w barriers) degrades with page
 * alignment due to significant overhead of ctx_init in case of mb=1. */
STRUCT_ALIGN(
        64, struct ctx_64_t {
            enum { CACHE_LINE_SIZE = 64 };
            volatile size_t ctr;
            char pad1[CACHE_LINE_SIZE - 1 * sizeof(size_t)];
            volatile size_t sense;
            char pad2[CACHE_LINE_SIZE - 1 * sizeof(size_t)];
        });

#if BARRIER_SHOULD_THROW
struct ctx_t {};

// In C++14 can be constexpr, not C++11
inline void ctx_init(ctx_t *ctx_t) {}
inline void barrier(ctx_t *ctx, int nthr) {
    if (nthr > 1)
        throw std::runtime_error(
                "nthr must be <= 1 (compiled without _MULTI_THREAD)");
}

#else // assume multiple thread support

template <typename ctx_t>
inline void ctx_init(ctx_t *ctx) {
    *ctx = utils::zero<ctx_t>();
}
void barrier(ctx_t *ctx, int nthr);

#endif

#if TARGET_X86_JIT
/** injects actual barrier implementation into another jitted code
 * @param code      jit_generator object where the barrier is to be injected
 * @param reg_ctx   read-only register with pointer to the barrier context
 * @param reg_nnthr read-only register with the # of synchronizing threads
 */
void generate(jit_generator &code, Xbyak::Reg64 reg_ctx, Xbyak::Reg64 reg_nthr);
#endif // TARGET_X86_JIT

} // namespace simple_barrier

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino=+2s,^=l0,\:4,N-s
#endif // CPU_BARRIER_HPP
