/*******************************************************************************
* Copyright 2018-2023 Intel Corporation
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

#ifndef CPU_X64_CPU_ISA_TRAITS_HPP
#define CPU_X64_CPU_ISA_TRAITS_HPP

#include <functional>
#include <type_traits>
#include <unordered_map>

#include "oneapi/dnnl/dnnl_types.h"

#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#define XBYAK64
#define XBYAK_NO_OP_NAMES
/* in order to make selinux happy memory that would be marked with X-bit should
 * be obtained with mmap */
#define XBYAK_USE_MMAP_ALLOCATOR

#define XBYAK_NO_EXCEPTION
#ifndef NDEBUG
#undef XBYAK_NO_EXCEPTION
#endif

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
/* turn off `size_t to other-type implicit casting` warning
 * currently we have a lot of jit-generated instructions that
 * take uint32_t, but we pass size_t (e.g. due to using sizeof).
 * FIXME: replace size_t parameters with the appropriate ones */
#pragma warning(disable : 4267)
#endif
#include "common/compiler_workarounds.hpp"
#include "cpu/x64/xbyak/xbyak.h"
#include "cpu/x64/xbyak/xbyak_util.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

// Maximum number of features + hints that can be specified via bits
static constexpr int cpu_isa_total_bits = sizeof(unsigned) * 8;

enum cpu_isa_bit_t : unsigned {
    // Fill in features from least significant bit to most significant bit
    sse41_bit = 1u << 0,
    avx_bit = 1u << 1,
    avx2_bit = 1u << 2,
    avx_vnni_bit = 1u << 3,
    avx_vnni_2_bit = 1u << 4,
    avx512_core_bit = 1u << 5,
    avx512_core_vnni_bit = 1u << 6,
    avx512_core_bf16_bit = 1u << 7,
    avx512_core_fp16_bit = 1u << 8,
    amx_tile_bit = 1u << 9,
    amx_int8_bit = 1u << 10,
    amx_bf16_bit = 1u << 11,
    amx_fp16_bit = 1u << 12,
    // Fill in hints from most significant bit to least significant bit
    prefer_ymm_bit = 1u << (cpu_isa_total_bits - 1),
};

dnnl_cpu_isa_hints_t DNNL_API get_cpu_isa_hints(bool soft = false);
status_t set_cpu_isa_hints(dnnl_cpu_isa_hints_t isa_hints);

namespace cpu_isa_hints_utils {
/* hints_1 | hints_2 | ... | hints_n where hints_i are hint specific
   bits declared inside the cpu_isa_bit_t */
static constexpr unsigned hints_mask = prefer_ymm_bit;

static unsigned cvt2mask(dnnl_cpu_isa_hints_t hints) {
    static const std::unordered_map<dnnl_cpu_isa_hints_t, unsigned,
            std::hash<int>>
            hints_map = {{dnnl_cpu_isa_no_hints, 0},
                    {dnnl_cpu_isa_prefer_ymm, prefer_ymm_bit}};

    auto iter = hints_map.find(hints);
    if (iter != hints_map.end())
        return iter->second;
    else {
        assert(!"unexpected CPU ISA hint");
        return 0;
    }
}

static bool is_hints_bit_set(cpu_isa_bit_t hint_bit, bool soft) {
    const dnnl_cpu_isa_hints_t hints = get_cpu_isa_hints(soft);
    const unsigned cur_hints_mask = cpu_isa_hints_utils::cvt2mask(hints);
    return (cur_hints_mask & hint_bit) == hint_bit;
}
} // namespace cpu_isa_hints_utils

enum cpu_isa_t : unsigned {
    isa_undef = 0u,
    sse41 = sse41_bit,
    avx = avx_bit | sse41,
    avx2 = avx2_bit | avx,
    avx2_vnni = avx_vnni_bit | avx2,
    avx2_vnni_2 = avx2_vnni | avx_vnni_2_bit,
    avx512_core = avx512_core_bit | avx2,
    avx512_core_vnni = avx512_core_vnni_bit | avx512_core,
    avx512_core_bf16 = avx512_core_bf16_bit | avx512_core_vnni,
    avx512_core_bf16_ymm = prefer_ymm_bit | avx512_core_bf16,
    amx_tile = amx_tile_bit,
    amx_int8 = amx_int8_bit | amx_tile,
    amx_bf16 = amx_bf16_bit | amx_tile,
    amx_fp16 = amx_fp16_bit | amx_tile,
    avx512_core_fp16 = avx512_core_fp16_bit | avx512_core_bf16 | avx_vnni_bit,
    avx512_core_amx = avx512_core_fp16 | amx_int8 | amx_bf16,
    avx512_core_amx_fp16 = avx512_core_amx | amx_fp16,
    // NOTES: 1. isa_all by default has no isa specific hints
    //        2. avx2_vnni_2 is under preview support and turned off by default
    //        3. avx512_core_amx_fp16 is under preview support and turned off
    //          by default
    isa_all
    = ~0u & ~avx_vnni_2_bit & ~amx_fp16_bit & ~cpu_isa_hints_utils::hints_mask,
};

enum class cpu_isa_cmp_t {
    // List of infix comparison relations between two cpu_isa_t
    // where we take isa_1 and isa_2 to be two cpu_isa_t instances.

    // isa_1 SUBSET isa_2 if all feature flags supported by isa_1
    // are supported by isa_2 as well (equality allowed)
    SUBSET,

    // isa_1 SUPERSET isa_2 if all feature flags supported by isa_2
    // are supported by isa_1 as well (equality allowed)
    SUPERSET,

    // Few more options that (depending upon need) can be enabled in future

    // 1. PROPER_SUBSET: isa_1 SUBSET isa_2 and isa_1 != isa_2
    // 2. PROPER_SUPERSET: isa_1 SUPERSET isa_2 and isa_1 != isa_2
};

const char *get_isa_info();

cpu_isa_t get_max_cpu_isa();
cpu_isa_t DNNL_API get_max_cpu_isa_mask(bool soft = false);
status_t set_max_cpu_isa(dnnl_cpu_isa_t isa);
dnnl_cpu_isa_t get_effective_cpu_isa();

static inline bool compare_isa(
        cpu_isa_t isa_1, cpu_isa_cmp_t cmp, cpu_isa_t isa_2) {
    assert(isa_1 != isa_all);
    assert(isa_2 != isa_all);
    // Comparison with `isa_all` is illegal.
    if (utils::one_of(isa_all, isa_1, isa_2)) return false;

    // By default, comparison between ISA ignores ISA specific hints
    unsigned mask_1
            = static_cast<unsigned>(isa_1) & ~cpu_isa_hints_utils::hints_mask;
    unsigned mask_2
            = static_cast<unsigned>(isa_2) & ~cpu_isa_hints_utils::hints_mask;
    unsigned mask_common = mask_1 & mask_2;

    switch (cmp) {
        case cpu_isa_cmp_t::SUBSET: return mask_1 == mask_common;
        case cpu_isa_cmp_t::SUPERSET: return mask_2 == mask_common;
        default: assert(!"unsupported comparison of isa"); return false;
    }
}

static inline bool is_subset(cpu_isa_t isa_1, cpu_isa_t isa_2) {
    return compare_isa(isa_1, cpu_isa_cmp_t::SUBSET, isa_2);
}

static inline bool is_superset(cpu_isa_t isa_1, cpu_isa_t isa_2) {
    return compare_isa(isa_1, cpu_isa_cmp_t::SUPERSET, isa_2);
}

template <typename Vmm>
struct vreg_traits {};

template <>
struct vreg_traits<Xbyak::Zmm> {
    typedef Xbyak::Ymm Vmm_lower_t;
    static constexpr size_t vlen = 64;
};

template <>
struct vreg_traits<Xbyak::Ymm> {
    typedef Xbyak::Xmm Vmm_lower_t;
    static constexpr size_t vlen = 32;
};

template <>
struct vreg_traits<Xbyak::Xmm> {
    typedef Xbyak::Xmm Vmm_lower_t;
    static constexpr size_t vlen = 16;
};

template <cpu_isa_t>
struct cpu_isa_traits {}; /* ::vlen -> 32 (for avx2) */

// pack struct so it can fit into a single 64-byte cache line
#pragma pack(push, 1)
struct palette_config_t {
    uint8_t palette_id;
    uint8_t startRow;
    uint8_t reserved[14];
    uint16_t cols[16];
    uint8_t rows[16];
};
#pragma pack(pop)

template <>
struct cpu_isa_traits<isa_all> {
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_default;
    static constexpr const char *user_option_env = "default";
};

template <>
struct cpu_isa_traits<sse41> {
    typedef Xbyak::Xmm Vmm;
    static constexpr int vlen_shift = 4;
    static constexpr int vlen = vreg_traits<Vmm>::vlen;
    static constexpr int n_vregs = 16;
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_sse41;
    static constexpr const char *user_option_env = "sse41";
};

template <>
struct cpu_isa_traits<avx> {
    typedef Xbyak::Ymm Vmm;
    static constexpr int vlen_shift = 5;
    static constexpr int vlen = vreg_traits<Vmm>::vlen;
    static constexpr int n_vregs = 16;
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_avx;
    static constexpr const char *user_option_env = "avx";
};

template <>
struct cpu_isa_traits<avx2> : public cpu_isa_traits<avx> {
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_avx2;
    static constexpr const char *user_option_env = "avx2";
};

template <>
struct cpu_isa_traits<avx2_vnni> : public cpu_isa_traits<avx2> {
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_avx2_vnni;
    static constexpr const char *user_option_env = "avx2_vnni";
};

template <>
struct cpu_isa_traits<avx2_vnni_2> : public cpu_isa_traits<avx2> {
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_avx2_vnni_2;
    static constexpr const char *user_option_env = "avx2_vnni_2";
};

template <>
struct cpu_isa_traits<avx512_core> {
    typedef Xbyak::Zmm Vmm;
    static constexpr int vlen_shift = 6;
    static constexpr int vlen = vreg_traits<Vmm>::vlen;
    static constexpr int n_vregs = 32;
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_avx512_core;
    static constexpr const char *user_option_env = "avx512_core";
};

template <>
struct cpu_isa_traits<avx512_core_vnni> : public cpu_isa_traits<avx512_core> {
    static constexpr dnnl_cpu_isa_t user_option_val
            = dnnl_cpu_isa_avx512_core_vnni;
    static constexpr const char *user_option_env = "avx512_core_vnni";
};

template <>
struct cpu_isa_traits<avx512_core_bf16> : public cpu_isa_traits<avx512_core> {
    static constexpr dnnl_cpu_isa_t user_option_val
            = dnnl_cpu_isa_avx512_core_bf16;
    static constexpr const char *user_option_env = "avx512_core_bf16";
};

template <>
struct cpu_isa_traits<avx512_core_amx> {
    typedef Xbyak::Zmm Vmm;
    static constexpr dnnl_cpu_isa_t user_option_val
            = dnnl_cpu_isa_avx512_core_amx;
    static constexpr const char *user_option_env = "avx512_core_amx";
};

template <>
struct cpu_isa_traits<avx512_core_fp16> : public cpu_isa_traits<avx512_core> {
    static constexpr dnnl_cpu_isa_t user_option_val
            = dnnl_cpu_isa_avx512_core_fp16;
    static constexpr const char *user_option_env = "avx512_core_fp16";
};

template <>
struct cpu_isa_traits<avx512_core_amx_fp16> {
    typedef Xbyak::Zmm Vmm;
    static constexpr dnnl_cpu_isa_t user_option_val
            = dnnl_cpu_isa_avx512_core_amx_fp16;
    static constexpr const char *user_option_env = "avx512_core_amx_fp16";
};

inline const Xbyak::util::Cpu &cpu() {
    const static Xbyak::util::Cpu cpu_;
    return cpu_;
}

namespace amx {

// Return the target palette for AMX instructions. Currently this is `0` if AMX
// instructions are not supported, and `1` if they are.
int get_target_palette();

int get_max_tiles(int palette);
int get_max_column_bytes(int palette);
int get_max_rows(int palette);
bool DNNL_API is_available();

} // namespace amx

namespace {

static inline bool mayiuse(const cpu_isa_t cpu_isa, bool soft = false) {
    using namespace Xbyak::util;

    unsigned cpu_isa_mask = x64::get_max_cpu_isa_mask(soft);
    unsigned cpu_isa_no_hints = cpu_isa & ~cpu_isa_hints_utils::hints_mask;

    if ((cpu_isa_mask & cpu_isa_no_hints) != cpu_isa_no_hints) return false;

    switch (cpu_isa) {
        case sse41: return cpu().has(Cpu::tSSE41);
        case avx: return cpu().has(Cpu::tAVX);
        case avx2: return cpu().has(Cpu::tAVX2);
        case avx2_vnni: return mayiuse(avx2, soft) && cpu().has(Cpu::tAVX_VNNI);
        case avx2_vnni_2:
            return mayiuse(avx2_vnni, soft) && cpu().has(Cpu::tAVX_VNNI_INT8)
                    && cpu().has(Cpu::tAVX_NE_CONVERT);
        case avx512_core:
            return cpu().has(Cpu::tAVX512F) && cpu().has(Cpu::tAVX512BW)
                    && cpu().has(Cpu::tAVX512VL) && cpu().has(Cpu::tAVX512DQ);
        case avx512_core_vnni:
            return cpu().has(Cpu::tAVX512F) && cpu().has(Cpu::tAVX512BW)
                    && cpu().has(Cpu::tAVX512VL) && cpu().has(Cpu::tAVX512DQ)
                    && cpu().has(Cpu::tAVX512_VNNI);
        case avx512_core_bf16:
            return mayiuse(avx512_core_vnni, soft)
                    && cpu().has(Cpu::tAVX512_BF16);
        case avx512_core_bf16_ymm:
            return mayiuse(avx512_core_bf16, soft)
                    && cpu_isa_hints_utils::is_hints_bit_set(
                            prefer_ymm_bit, soft);
        case avx512_core_fp16:
            return cpu().has(Cpu::tAVX512_FP16)
                    && mayiuse(avx512_core_bf16, soft)
                    && mayiuse(avx2_vnni, soft);
        case amx_tile:
            return cpu().has(Cpu::tAMX_TILE) && x64::amx::is_available();
        case amx_int8:
            return mayiuse(amx_tile, soft) && cpu().has(Cpu::tAMX_INT8);
        case amx_bf16:
            return mayiuse(amx_tile, soft) && cpu().has(Cpu::tAMX_BF16);
        case amx_fp16:
            return mayiuse(amx_tile, soft) && cpu().has(Cpu::tAMX_FP16);
        case avx512_core_amx:
            return mayiuse(amx_int8, soft) && mayiuse(amx_bf16, soft)
                    && mayiuse(avx512_core_fp16, soft);
        case avx512_core_amx_fp16:
            return mayiuse(avx512_core_amx, soft) && mayiuse(amx_fp16, soft);
        case isa_undef: return true;
        case isa_all: return false;
    }
    return false;
}

static inline bool isa_has_int8_vnni(cpu_isa_t isa) {
    return is_superset(isa, avx512_core_vnni) || is_superset(isa, avx2_vnni);
}

static inline bool isa_has_s8s8(cpu_isa_t isa) {
    return is_superset(isa, amx_int8) || is_superset(isa, avx2_vnni_2);
}

static inline bool isa_has_bf16(cpu_isa_t isa) {
    return is_superset(isa, avx512_core_bf16);
}

static inline bool isa_has_masks(cpu_isa_t isa) {
    return is_superset(isa, avx512_core);
}

static inline int isa_max_vlen(cpu_isa_t isa) {
    if (is_superset(isa, avx512_core))
        return cpu_isa_traits<avx512_core>::vlen;
    else if (is_superset(isa, avx2))
        return cpu_isa_traits<avx2>::vlen;
    else if (is_superset(isa, sse41))
        return cpu_isa_traits<sse41>::vlen;
    assert(!"ISA Error");
    return 0;
}

static inline int isa_num_vregs(cpu_isa_t isa) {
    if (is_superset(isa, avx512_core))
        return cpu_isa_traits<avx512_core>::n_vregs;
    else if (is_superset(isa, avx2))
        return cpu_isa_traits<avx2>::n_vregs;
    else if (is_superset(isa, sse41))
        return cpu_isa_traits<sse41>::n_vregs;
    assert(!"ISA Error");
    return 0;
}

} // namespace

/* whatever is required to generate string literals... */
#include "common/z_magic.hpp"
/* clang-format off */
#define JIT_IMPL_NAME_HELPER(prefix, isa, suffix_if_any) \
    ((isa) == isa_undef ? prefix STRINGIFY(undef) : \
    (isa) == sse41 ? prefix STRINGIFY(sse41) : \
    (isa) == avx ? prefix STRINGIFY(avx) : \
    (isa) == avx2 ? prefix STRINGIFY(avx2) : \
    (isa) == avx2_vnni ? prefix STRINGIFY(avx2_vnni) : \
    (isa) == avx2_vnni_2 ? prefix STRINGIFY(avx2_vnni_2) : \
    (isa) == avx512_core ? prefix STRINGIFY(avx512_core) : \
    (isa) == avx512_core_vnni ? prefix STRINGIFY(avx512_core_vnni) : \
    (isa) == avx512_core_bf16 ? prefix STRINGIFY(avx512_core_bf16) : \
    (isa) == avx512_core_fp16 ? prefix STRINGIFY(avx512_core_fp16) : \
    (isa) == avx512_core_amx ? prefix STRINGIFY(avx512_core_amx) : \
    (isa) == avx512_core_amx_fp16 ? prefix STRINGIFY(avx512_core_amx_fp16) : \
    prefix suffix_if_any)
/* clang-format on */

inline size_t data_type_vnni_granularity(data_type_t data_type) {
    using namespace data_type;
    switch (data_type) {
        case f32:
        case s32: return size_t(1);
        case f16:
        case bf16: return size_t(2);
        case s8:
        case u8: return size_t(4);
        case data_type::undef:
        default: assert(!"unknown data_type");
    }
    return size_t(0); /* should not be reachable */
}

template <cpu_isa_t isa>
inline size_t data_type_vnni_simd_elems(data_type_t data_type) {
    const size_t dt_size = types::data_type_size(data_type);
    assert(dt_size > 0);
    return cpu_isa_traits<isa>::vlen / dt_size;
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
