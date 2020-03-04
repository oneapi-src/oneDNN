/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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

#ifndef CPU_ISA_TRAITS_HPP
#define CPU_ISA_TRAITS_HPP
/** \file
 * This file has been branched off of jit_generator.hpp to provide those "jit"
 * utilities/macros that are also useful to non-jit programs.
 */

#include "dnnl.h"
#include "dnnl_config.h"
#include "dnnl_types.h"

#include "dnnl_thread.hpp"
#include "utils.hpp"

#if defined(_WIN32) && !defined(__GNUC__)
#define STRUCT_ALIGN(al, ...) __declspec(align(al)) __VA_ARGS__
#elif defined(__ve)
#define STRUCT_ALIGN(al, ...) \
    __VA_ARGS__ __attribute__((__aligned__((al) > 16 ? 16 : (al))))
#else
#define STRUCT_ALIGN(al, ...) __VA_ARGS__ __attribute__((__aligned__(al)))
#endif

// Any restrictions on alignas(expression)?
#ifdef __ve
#define alignas(x) alignas((x) > 16 ? 16 : (x))
#endif

// How is the restrict keyword handled? (disallow it as you encounter errors, please)
// Actually dnnl sources seem to use __restrict throughout, avoiding C++ restrict keyword

#if defined( \
        __ve) // cross-compilers may or may not be happy with restrict or __restrict
#elif defined(WIN32)
#elif defined(__INTEL_COMPILER)
//#   define restrict __restrict__ /* historical? */
#elif defined(__GNUC__) /* may catch many compilers */
//#   define restrict __restrict__
#else
#endif // restrict keyword handling

#if TARGET_X86_JIT
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
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"
#endif

namespace dnnl {
namespace impl {
namespace cpu {

// generic, from jit_generator.hpp
typedef enum {
    PAGE_4K = 4096,
    PAGE_2M = 2097152,
} cpu_page_size_t;

#if defined(__ve)
enum { CACHE_LINE_SIZE = 128 }; // is ADB cache line size the relevant quantity?
#else
enum { CACHE_LINE_SIZE = 64 };
#endif

enum cpu_isa_bit_t : unsigned {
    /** impls using only C/C++ code. Typicall ref impls, but may be built
     * to use external gemm routines (cblas/MKL). inline assembler only when
     * absolutely necessary (by checking compile flags). */
    vanilla_bit = 1U << 0,
    /// @defgroup x86_jit_flags x86 JIT levels
    //@{
    x86_common_bit = 1U << 1, ///< [for completeness] x86 [+ jit allowed]
    sse41_bit = 1u << 2,
    avx_bit = 1u << 3,
    avx2_bit = 1u << 4,
    avx512_common_bit = 1u << 5,
    avx512_mic_bit = 1u << 6,
    avx512_mic_4ops_bit = 1u << 7,
    avx512_core_bit = 1u << 8,
    avx512_core_vnni_bit = 1u << 9,
    avx512_core_bf16_bit = 1u << 10,
    x86_11_bit = 1u << 11, ///< future use
    x86_12_bit = 1u << 12, ///< future use
    x86_13_bit = 1u << 13, ///< future use
    x86_14_bit = 1u << 14, ///< future use
    x86_15_bit = 1u << 15, ///< future use
    x86_bits = 0xffffu, ///< set of x86 flags
    //x86_jit_bits        = x86_bits ^ vanilla_bit,
    ///@}
    /// @defgroup VE_extensions VE extended capabilities
    ///@{
    /// perhaps remove ve_common* (force it identical to vanilla)?
    ve_common_bit = 1u << 16, ///< without libvednn support
    ve_vednn_bit = 1u << 17, ///< DNNL_CPU_VE with libvednn support
    ve_vejit_bit = 1u << 18, ///< VE with jit support, also via libvednn
    ve_bits = vanilla_bit | 0x80000u, ///< set of VE flags
    ///@}
};

/** These are bitmasks for CPU isa capabilities.
 * vanilla, common and full are cross-platform, supported for all
 * DNNL_CPU build targets.  common and full could be identical to
 * vanilla.
 *
 * - "vanilla" means portable C/C++ reference implementations.
 * - "common" should run on any version of the CPU, but without the
 *   restrictions of "vanilla".
 *   (i.e. cpu-specific intrinsics, libraries, jit might be included)
 * - "full" always means "full set of features"
 *
 * TODO: full back to all? any change to common (any vs all too confusing).
 */
// false if a cpu_isa_t doesn't make sense.
enum cpu_isa_t : unsigned {
    /** replaces old "isa_any" value of zero with it characteristic
     * behavior that `mayiuse(isa_unkown)==false`. */
    isa_unknown = 0,

    /** \enum vanilla
     * cpu-agnostic reference implementations.
     * mayiuse(vanilla) will always be true.
     * \note In rare cases, a bare minimum of inline assembler might be
     *       used, as in cpu_barrier.
     */
    /** \enum isa_any
     * For multiple cpus, isa_any represents the "most basic" ISA, and has
     * mayiuse(isa_any)==true.  It need not be cross-platform, like vanilla. */
    vanilla = vanilla_bit,
    /// @defgroup x86_jit_masks x86-specific jit masks
    //@{
    x86_common = x86_common_bit | vanilla, // like old "all"
    sse41 = sse41_bit | x86_common,
    avx = avx_bit | sse41,
    avx2 = avx2_bit | avx,
    //
    avx512_common = avx512_common_bit | vanilla,
    avx512_mic = avx512_mic_bit | avx512_common,
    avx512_mic_4ops = avx512_mic_4ops_bit | avx512_mic,
    //
    avx512_core = avx512_core_bit | avx512_common,
    avx512_core_vnni = avx512_core_vnni_bit | avx512_core,
    avx512_core_bf16 = avx512_core_bf16_bit | avx512_core_vnni,
    x86_full = x86_bits,
    //@}
    /// @defgroup ve_jit_masks VE-specific implementation masks
    //@{
    ve_common
    = ve_common_bit | vanilla, // ref impls allowed to expand on vanilla
    vednn = ve_vednn_bit | ve_common,
    vejit = ve_vejit_bit | vednn,
    ve_full = ve_bits,
//@}
/// @defgroup cpu_agnostic_masks VANILLA and FULL DNNL_ISA settings
//@{
// Now (mainly) just have VANILLA and FULL that get remapped somehow
// Well, let's keep VANILLA as is (could remap to x86_common or ve_common?)
#if DNNL_CPU == DNNL_CPU_X86
    isa_any = x86_common,
    isa_full = x86_full, // all varieties of x86 jit
#elif DNNL_CPU == DNNL_CPU_VE
    isa_any = ve_common,
    isa_full = ve_full, // all types of VE optimizations
#else
#error "isa_any and isa_full for this cpu MUST be defined"
#endif
    //@}
};

/** map environment DNNL_MAX_CPU strings to public \c dnnl_cpu_isa_t values.
 * Only the ones we want to use for CPU dispatch or \c mayiuse need definition.
 * May also supply other lookup data for particular ISA targets. */
template <cpu_isa_t>
struct cpu_isa_traits {}; /* ::vlen -> 32 (for avx2) */

template <>
struct cpu_isa_traits<vanilla> { // MUST work for any cpu (X86, VE, ...)
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_vanilla;
    static constexpr const char *user_option_env = "VANILLA";
};

template <>
struct cpu_isa_traits<x86_common> {
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_any;
    static constexpr const char *user_option_env = "ANY";
};

template <>
struct cpu_isa_traits<ve_common> {
    // no Vmm typedef for nc++ (though available in clang cross-compiler)
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_any;
    static constexpr const char *user_option_env = "ANY";
    // 2 kBit vector regs (256 double or 512 packed float)
    static constexpr int vlen_shift = 8; // CHECK! also try nc++ -mpacked-float
    static constexpr int vlen = (1 << (vlen_shift - 1)) * 8;
    static constexpr int n_vregs = 64;
};

/// x86-specific
//@{
template <>
struct cpu_isa_traits<sse41> {
#if TARGET_X86_JIT
    typedef Xbyak::Xmm Vmm;
#endif // TARGET_X86_JIT
    static constexpr int vlen_shift = 4;
    static constexpr int vlen = 16;
    static constexpr int n_vregs = 16;
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_sse41;
    static constexpr const char *user_option_env = "SSE41";
};

template <>
struct cpu_isa_traits<avx> {
#if TARGET_X86_JIT
    typedef Xbyak::Ymm Vmm;
#endif // TARGET_X86_JIT
    static constexpr int vlen_shift = 5;
    static constexpr int vlen = 32; // 256-bit regs --> 32 *bytes*
    static constexpr int n_vregs = 16;
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_avx;
    static constexpr const char *user_option_env = "AVX";
};

template <>
struct cpu_isa_traits<avx2> : public cpu_isa_traits<avx> {
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_avx2;
    static constexpr const char *user_option_env = "AVX2";
};

template <>
struct cpu_isa_traits<avx512_common> { // for convenience
#if TARGET_X86_JIT
    typedef Xbyak::Zmm Vmm;
#endif // TARGET_X86_JIT
    static constexpr int vlen_shift = 6;
    static constexpr int vlen = 64;
    static constexpr int n_vregs = 32;
};

template <>
struct cpu_isa_traits<avx512_core> : public cpu_isa_traits<avx512_common> {
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_avx512_core;
    static constexpr const char *user_option_env = "AVX512_CORE";
};

template <>
struct cpu_isa_traits<avx512_mic> : public cpu_isa_traits<avx512_common> {
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_avx512_mic;
    static constexpr const char *user_option_env = "AVX512_MIC";
};

template <>
struct cpu_isa_traits<avx512_mic_4ops> : public cpu_isa_traits<avx512_mic> {
    static constexpr dnnl_cpu_isa_t user_option_val
            = dnnl_cpu_isa_avx512_mic_4ops;
    static constexpr const char *user_option_env = "AVX512_MIC_4OPS";
};

template <>
struct cpu_isa_traits<avx512_core_vnni> : public cpu_isa_traits<avx512_core> {
    static constexpr dnnl_cpu_isa_t user_option_val
            = dnnl_cpu_isa_avx512_core_vnni;
    static constexpr const char *user_option_env = "AVX512_CORE_VNNI";
};

template <>
struct cpu_isa_traits<avx512_core_bf16> : public cpu_isa_traits<avx512_core> {
    static constexpr dnnl_cpu_isa_t user_option_val
            = dnnl_cpu_isa_avx512_core_bf16;
    static constexpr const char *user_option_env = "AVX512_CORE_BF16";
};
//@}

template <>
struct cpu_isa_traits<x86_full> : public cpu_isa_traits<avx512_core> {
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_full;
    static constexpr const char *user_option_env = "ALL";
};
template <>
struct cpu_isa_traits<ve_full> : public cpu_isa_traits<ve_common> {
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_full;
    static constexpr const char *user_option_env = "ALL";
};

/// only for cross-compilation builds
//@{
template <>
struct cpu_isa_traits<vednn> : public cpu_isa_traits<ve_common> {
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_vednn;
    static constexpr const char *user_option_env = "VEDNN";
};

template <>
struct cpu_isa_traits<vejit> : public cpu_isa_traits<ve_common> {
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_vejit;
    static constexpr const char *user_option_env = "VEJIT";
};
//@}

// END cpu_isa_traits (vector register defaults)

cpu_isa_t DNNL_API get_max_cpu_isa(bool soft = false);

// Note: also in include/dnnl.h and include/dnnl.hpp ?
dnnl::impl::status_t DNNL_API set_max_cpu_isa(dnnl_cpu_isa_t isa, bool force);

namespace {

/** Convert from public dnnl_cpu_isa value to internal bitflags value.
 * Since dnnl.h dnnl_cpu_isa values and cpu_isa values are now decoupled,
 * portable tests should *not* rely on equality between values of the
 * two enum types.
 * \return the appropriate cpu_isa enum value for the build target,
 *         or \c unknown.
 * \todo \c from_dnnl sig could use C++ enum as "cpu_isa_t from_dnnl(cpu_isa isa)"
 * \todo perhaps remove this function \sa tests/gtests/test_isa_iface.cpp */
static inline cpu_isa_t from_dnnl(dnnl_cpu_isa_t isa) {
    static int const verbose = 0;
    using namespace dnnl::impl;
    using namespace dnnl::impl::cpu;

    // return \c unknown if unrecognized
    cpu_isa_t isa_to_set = isa_unknown;
#define HANDLE_CASE(CPU_ISA_T) \
    case cpu_isa_traits<CPU_ISA_T>::user_option_val: \
        isa_to_set = CPU_ISA_T; \
        break;

    // convert a dnnl.h \c isa value to internal \c cpu_isa_t
    switch (isa) {
        // Note cases here should match init_max_cpu_isa()
        // All target cpus support "VANILLA", "ANY" ... "ALL"
        HANDLE_CASE(vanilla);
        HANDLE_CASE(isa_any); // x86_common/ve_common (almost same as vanilla)
        HANDLE_CASE(isa_full); // covers x86_full and ve_full
#if TARGET_X86
        //HANDLE_CASE(x86_common);// "ALL" (mostly vanilla, but jit technically allowed)
        //HANDLE_CASE(x86_full);  // "FULL"
        HANDLE_CASE(sse41); // x86 jit with vec ops
        HANDLE_CASE(avx);
        HANDLE_CASE(avx2);
        HANDLE_CASE(avx512_mic);
        HANDLE_CASE(avx512_mic_4ops);
        HANDLE_CASE(avx512_core);
        HANDLE_CASE(avx512_core_vnni);
        HANDLE_CASE(avx512_core_bf16);
#elif TARGET_VE
        //HANDLE_CASE(ve_common); // "ALL" (identical to vanilla, for now)
        //HANDLE_CASE(ve_full);   // "FULL" vednn + libvednn jit
        HANDLE_CASE(vednn); // vanilla + libvednn "C" api
        HANDLE_CASE(vejit); // vednn + libvednn jit
#endif
        default: /*unknown*/;
    }
#undef HANDLE_CASE
    if (verbose)
        printf(" from_dnnl(0x%lx) --> cpu_isa_t(0x%lx)\n", (long)isa,
                (long)isa_to_set);
    return isa_to_set;
}

#if TARGET_X86_JIT
static Xbyak::util::Cpu cpu;
/** mayiuse is a runtime check that an ISA is available on the target CPU,
 * and we respect a \c soft runtime limit.  Build time x86 limits are mostly
 * enforced by \c cpu_engine (by pruning the impl lists at compile time).
 *
 * \note It is \b not required that \c mayiuse invoke \c get_max_cpu_isa.
 * Ex.VANILLA build can only use reference impls, an easy test.
 * Tests for set-once behavior should \e explicitly call \c get_max_cpu_isa.
 */
static inline bool mayiuse(const cpu_isa_t cpuIsaT, const bool soft = false) {
    // reminder:
    static_assert(
            isa_any == (int)x86_common, "changed VE alias for cpuIsaT 'any'?");
    static_assert(
            isa_full == (int)x86_full, "changed VE alias for cpuIsaT 'full'?");
    using namespace Xbyak::util;

    // Say no quickly if cpuIsaT fails runtime CPU dispatch check
    //                or cpuIsaT is for a different cpu .
    unsigned cpu_isa_mask = get_max_cpu_isa(soft);
    if ((cpu_isa_mask & cpuIsaT) != cpuIsaT) return false;
    assert(cpuIsaT != isa_unknown /*0*/); // old 'isa_any' enum

    switch (cpuIsaT) {
        case vanilla: return true;
        case isa_any: return true;
        case sse41: return cpu.has(Cpu::tSSE41);
        case avx: return cpu.has(Cpu::tAVX);
        case avx2: return cpu.has(Cpu::tAVX2);
        case avx512_common: return cpu.has(Cpu::tAVX512F);
        case avx512_core:
            return cpu.has(Cpu::tAVX512F) && cpu.has(Cpu::tAVX512BW)
                    && cpu.has(Cpu::tAVX512VL) && cpu.has(Cpu::tAVX512DQ);
        case avx512_core_vnni:
            return cpu.has(Cpu::tAVX512F) && cpu.has(Cpu::tAVX512BW)
                    && cpu.has(Cpu::tAVX512VL) && cpu.has(Cpu::tAVX512DQ)
                    && cpu.has(Cpu::tAVX512_VNNI);
        case avx512_mic:
            return cpu.has(Cpu::tAVX512F) && cpu.has(Cpu::tAVX512CD)
                    && cpu.has(Cpu::tAVX512ER) && cpu.has(Cpu::tAVX512PF);
        case avx512_mic_4ops:
            return mayiuse(avx512_mic, soft) && cpu.has(Cpu::tAVX512_4FMAPS)
                    && cpu.has(Cpu::tAVX512_4VNNIW);
        case avx512_core_bf16:
            return mayiuse(avx512_core_vnni, soft)
                    && cpu.has(Cpu::tAVX512_BF16);
        default: assert(!"unhandled x86 mayiuse"); return false;
    }
    return false;
}

#elif TARGET_X86 || (TARGET_VE && DNNL_ISA <= DNNL_ISA_VE)
// cmake -DDNNL_ISA=VANILLA build for x86 has only the VANILLA impls
// cmake -DDNNL_ISA=ANY build for VE also only has the VANILLA impls (so far)
static inline constexpr bool mayiuse(
        cpu_isa_t const cpuIsaT, bool const soft = false) {
    // Exceptional case:
    //   - non-jit x86 build: only valid cpu_isa_t is vanilla
    //   - mayiuse does NOT imply a call to get_max_cpu_isa.
    // tests assuming that get_max_cpu_isa was called must do so explicitly.
    return cpuIsaT == vanilla;
}

#elif TARGET_VE // Example non-x86 target cpu

// On Intel master, this is simplified because only the ANY==VANILLA build
// is available.  On fork, change mayiuse as you add support for non-VANILLA
// implementations. VEDNN and VEJIT cmake targets on Intel master are examples.
static inline bool mayiuse(cpu_isa_t const cpuIsaT, bool const soft = false) {

#ifdef DNNL_ENABLE_MAX_CPU_ISA
    // Say no quickly if cpuIsaT fails runtime CPU dispatch check
    //                or cpuIsaT is for a different cpu .

    //unsigned cpu_isa_mask = get_max_cpu_isa(soft);
    // Simpler... non-VANILLA VE impls are not in Intel master.
    //            forks begin with only VANILLA, so runtime dispatch irrelevant
    unsigned cpu_isa_mask = ve_full; // enough to reject mayiuse(wrong-cpu)
    if ((cpu_isa_mask & cpuIsaT) != cpuIsaT) return false;
    assert(cpuIsaT != unknown /*0*/);

    // Do check any compile-time DNNL_ISA cap.
    static_assert(
            isa_full == (int)ve_full, "changed VE alias for cpuIsaT 'full'?");
    static_assert(
            isa_any == (int)ve_bits, "changed VE alias for cpuIsaT 'all'?");
    // we check for cpuIsaT at or under the compile-time DNNL_ISA cap,
    // but have not checked for runtime get_max_cpu_isa(soft) CPU dispatch limit
    return (cpuIsaT == vanilla || cpuIsaT == ve_common)
            ? true
            //: cpuIsaT==isa_any? true /*aliased to ve_common*/
            : cpuIsaT == vednn ? (DNNL_ISA >= DNNL_ISA_VEDNN)
                               : cpuIsaT == vejit
                            ? (DNNL_ISA >= DNNL_ISA_VEJIT)
                            // isa_full could be different, in principle?
                            : cpuIsaT == isa_full
                                    ? (DNNL_ISA >= DNNL_ISA_VE_FULL)
                                    : false;
#else
    // -DDNNL_ISA=VEDNN or VEJIT or FULL are not available in Intel master
    return ((void)soft, (void)cpuIsaT,
            cpuIsaT == vanilla || cpuIsaT == isa_any);
#endif // DNNL_ENABLE_MAX_CPU_ISA
}

#else
#error "unhandled DNNL_CPU target cpu -- please write a 'mayiuse' function"
#endif // TARGET_* mayiuse variations
} // namespace

namespace {
inline unsigned int get_cache_size(int level, bool per_core = true) {
    unsigned int l = level - 1;
#if TARGET_X86_JIT
    unsigned const cpuDataCacheLevels = cpu.getDataCacheLevels();
#else
    unsigned const cpuDataCacheLevels = 0; // use default settings
#endif
    // Currently, if XByak is not able to fetch the cache topology
    // we default to 32KB of L1, 512KB of L2 and 1MB of L3 per core.
    if (cpuDataCacheLevels == 0) {
#if TARGET_VE
        const int L1_cache_per_core = 32000; // each, for data and instruction
        const int L2_cache_per_core = 256000;
        const int L3_cache_per_core
                = 16 * 1024 * 1024 / 8; // 16G per chip of 8 processors (today)
#else
        const int L1_cache_per_core = 32000;
        const int L2_cache_per_core = 512000;
        const int L3_cache_per_core = 1024000;
#if !TARGET_X86
#warning "Guessed cache sizes for this CPU!"
#endif
#endif
        int num_cores = per_core ? 1 : dnnl_get_max_threads();
        switch (l) {
            case (0): return L1_cache_per_core * num_cores;
            case (1): return L2_cache_per_core * num_cores;
            case (2): return L3_cache_per_core * num_cores;
            default: return 0;
        }
    }
#if TARGET_X86_JIT
    if (l < cpuDataCacheLevels) {
        return cpu.getDataCacheSize(l)
                / (per_core ? cpu.getCoresSharingDataCache(l) : 1);
    }
#endif
    return 0;
}

inline bool isa_has_bf16(cpu_isa_t isa) {
    return TARGET_X86 && isa == avx512_core_bf16;
}

} // namespace

/* whatever is required to generate string literals... */
#include "z_magic.hpp"
/* clang-format off */
#if TARGET_X86_JIT
#define JIT_IMPL_NAME_HELPER(prefix, isa, suffix_if_any) \
    ( (isa) == isa_any            ? (prefix STRINGIFY(any)) \
    : (isa) == sse41              ? (prefix STRINGIFY(sse41)) \
    : (isa) == avx                ? (prefix STRINGIFY(avx)) \
    : (isa) == avx2               ? (prefix STRINGIFY(avx2)) \
    : (isa) == avx512_common      ? (prefix STRINGIFY(avx512_common)) \
    : (isa) == avx512_core        ? (prefix STRINGIFY(avx512_core)) \
    : (isa) == avx512_core_vnni   ? (prefix STRINGIFY(avx512_core_vnni)) \
    : (isa) == avx512_mic         ? (prefix STRINGIFY(avx512_mic)) \
    : (isa) == avx512_mic_4ops    ? (prefix STRINGIFY(avx512_mic_4ops)) \
    : (isa) == avx512_core_bf16   ? (prefix STRINGIFY(avx512_core_bf16)) \
    : prefix suffix_if_any)

#else
// non-jit has no use for this macro. Can VANILLA automatically add "ref"?
#define JIT_IMPL_NAME_HELPER(prefix, isa, suffix_if_any) \
    prefix "OOPS_no_x86_jit" suffix_if_any
#endif // TARGET_X86_JIT
/* clang-format on */

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino=+2s,^=l0,\:0,N-s
#endif
