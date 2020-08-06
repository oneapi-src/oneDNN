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

#ifndef CPU_AARCH64_CPU_ISA_TRAITS_HPP
#define CPU_AARCH64_CPU_ISA_TRAITS_HPP

#include <type_traits>

#include "dnnl_types.h"

#include "common/dnnl_thread.hpp"
#include "common/utils.hpp"

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

#include "cpu/aarch64/xbyak_translator_aarch64/xbyak/xbyak.h"
#include "cpu/aarch64/xbyak_translator_aarch64/xbyak/xbyak_util.h"


namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

enum cpu_isa_bit_t : unsigned {
    simd_bit = 1u << 0,
    sve_bit  = 1u << 1,
};

enum cpu_isa_t : unsigned {
    isa_any = 0u,
    simd    = simd_bit,
    sve     = sve_bit | simd,
    isa_all = ~0u,
};

const char *get_isa_info();

cpu_isa_t DNNL_API get_max_cpu_isa_mask(bool soft = false);
status_t set_max_cpu_isa(dnnl_cpu_isa_t isa);
dnnl_cpu_isa_t get_effective_cpu_isa();

template <cpu_isa_t>
struct cpu_isa_traits {}; /* ::vlen -> 32 (for avx2) */

template <>
struct cpu_isa_traits<isa_all> {
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_all;
    static constexpr const char *user_option_env = "ALL";
};

template <>
struct cpu_isa_traits<simd> {
    typedef Xbyak::Xbyak_aarch64::VReg4S Vmm;
    static constexpr int vlen_shift = 4;
    static constexpr int vlen = 16;
    static constexpr int n_vregs = 32;
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_simd;
    static constexpr const char *user_option_env = "SIMD";
};

template <>
struct cpu_isa_traits<sve> {
    typedef Xbyak::Xbyak_aarch64::ZRegS Vmm;
    static constexpr int vlen_shift = 6;
    static constexpr int vlen = 64;
    static constexpr int n_vregs = 32;
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_sve;
    static constexpr const char *user_option_env = "SVE";
};

namespace {

static Xbyak::util::Cpu cpu;
static inline bool mayiuse(const cpu_isa_t cpu_isa, bool soft = false) {
    using namespace Xbyak::util;

    unsigned cpu_isa_mask = aarch64::get_max_cpu_isa_mask(soft);
    if ((cpu_isa_mask & cpu_isa) != cpu_isa) return false;

    switch (cpu_isa) {
        case simd:
            return true && cpu.has(Cpu::tSIMD);
        case sve:
            return true && cpu.has(Cpu::tSVE);
        case isa_any: return true;
        case isa_all: return false;
    }
    return false;
}

inline bool isa_has_bf16(cpu_isa_t isa) {
    return false;
}

} // namespace

/* whatever is required to generate string literals... */
#include "common/z_magic.hpp"
/* clang-format off */
#define JIT_IMPL_NAME_HELPER(prefix, isa, suffix_if_any) \
    ((isa) == isa_any ? prefix STRINGIFY(any) : \
    ((isa) == simd ? prefix STRINGIFY(simd) : \
    ((isa) == sve ? prefix STRINGIFY(sve) : \
    prefix suffix_if_any)))
/* clang-format on */

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
