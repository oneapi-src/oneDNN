/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#include <cstring>
#include <mutex>
#include <string>

#include "common/utils.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace {
#ifdef DNNL_ENABLE_MAX_CPU_ISA
cpu_isa_t init_max_cpu_isa() {
    cpu_isa_t max_cpu_isa_val = isa_all;
    static std::string isa_val = getenv_string_user("MAX_CPU_ISA");
    if (!isa_val.empty()) {

#define IF_HANDLE_CASE(cpu_isa) \
    if (isa_val.compare(cpu_isa_traits<cpu_isa>::user_option_env) == 0) \
    max_cpu_isa_val = cpu_isa
#define ELSEIF_HANDLE_CASE(cpu_isa) else IF_HANDLE_CASE(cpu_isa)

        IF_HANDLE_CASE(isa_all);
        ELSEIF_HANDLE_CASE(sse41);
        ELSEIF_HANDLE_CASE(avx);
        ELSEIF_HANDLE_CASE(avx2);
        ELSEIF_HANDLE_CASE(avx2_vnni);
        ELSEIF_HANDLE_CASE(avx2_vnni_2);
        ELSEIF_HANDLE_CASE(avx512_core);
        ELSEIF_HANDLE_CASE(avx512_core_vnni);
        ELSEIF_HANDLE_CASE(avx512_core_bf16);
        ELSEIF_HANDLE_CASE(avx512_core_fp16);
        ELSEIF_HANDLE_CASE(avx512_core_amx);
        ELSEIF_HANDLE_CASE(avx512_core_amx_fp16);

#undef IF_HANDLE_CASE
#undef ELSEIF_HANDLE_CASE
    }
    return max_cpu_isa_val;
}

set_once_before_first_get_setting_t<cpu_isa_t> &max_cpu_isa() {
    static set_once_before_first_get_setting_t<cpu_isa_t> max_cpu_isa_setting(
            init_max_cpu_isa());
    return max_cpu_isa_setting;
}
#endif

#ifdef DNNL_ENABLE_CPU_ISA_HINTS
dnnl_cpu_isa_hints_t init_cpu_isa_hints() {
    dnnl_cpu_isa_hints_t cpu_isa_hints_val = dnnl_cpu_isa_no_hints;
    static std::string hints_val = getenv_string_user("CPU_ISA_HINTS");
    if (!hints_val.empty()) {
        if (hints_val.compare("prefer_ymm") == 0)
            cpu_isa_hints_val = dnnl_cpu_isa_prefer_ymm;
    }
    return cpu_isa_hints_val;
}

set_once_before_first_get_setting_t<dnnl_cpu_isa_hints_t> &cpu_isa_hints() {
    static set_once_before_first_get_setting_t<dnnl_cpu_isa_hints_t>
            cpu_isa_hints_setting(init_cpu_isa_hints());
    return cpu_isa_hints_setting;
}
#endif
} // namespace

struct isa_info_t {
    isa_info_t(cpu_isa_t aisa) : isa(aisa) {};

    // this converter is needed as code base defines certain ISAs
    // that the library does not expose (e.g. avx512_core_bf16_ymm),
    // so the internal and external enum types do not coincide.
    dnnl_cpu_isa_t convert_to_public_enum(void) const {
        switch (isa) {
            case avx512_core_amx_fp16: return dnnl_cpu_isa_avx512_core_amx_fp16;
            case avx512_core_amx: return dnnl_cpu_isa_avx512_core_amx;
            case avx512_core_fp16: return dnnl_cpu_isa_avx512_core_fp16;
            case avx512_core_bf16_ymm: // fallback to avx512_core_bf16
            case avx512_core_bf16: return dnnl_cpu_isa_avx512_core_bf16;
            case avx512_core_vnni: return dnnl_cpu_isa_avx512_core_vnni;
            case avx512_core: return dnnl_cpu_isa_avx512_core;
            case avx2_vnni_2: return dnnl_cpu_isa_avx2_vnni_2;
            case avx2_vnni: return dnnl_cpu_isa_avx2_vnni;
            case avx2: return dnnl_cpu_isa_avx2;
            case avx: return dnnl_cpu_isa_avx;
            case sse41: return dnnl_cpu_isa_sse41;
            default: return dnnl_cpu_isa_default;
        }
    }

    const char *get_name() const {
        switch (isa) {
            case avx512_core_amx_fp16:
                return "Intel AVX-512 with float16, Intel DL Boost and "
                       "bfloat16 support and Intel AMX with bfloat16, float16 "
                       "and 8-bit integer support (preview support)";
            case avx512_core_amx:
                return "Intel AVX-512 with float16, Intel DL Boost and "
                       "bfloat16 support and Intel AMX with bfloat16 and 8-bit "
                       "integer support";
            case avx512_core_fp16:
                return "Intel AVX-512 with float16, Intel DL Boost and "
                       "bfloat16 support ";
            case avx512_core_bf16_ymm:
                return "Intel AVX-512 with Intel DL Boost and bfloat16 support "
                       "on Ymm/Zmm";
            case avx512_core_bf16:
                return "Intel AVX-512 with Intel DL Boost and bfloat16 support";
            case avx512_core_vnni: return "Intel AVX-512 with Intel DL Boost";
            case avx512_core:
                return "Intel AVX-512 with AVX512BW, AVX512VL, and AVX512DQ "
                       "extensions";
            case avx2_vnni_2:
                return "Intel AVX2 with Intel DL Boost, float16 and bfloat16 "
                       "support (preview support)";
            case avx2_vnni: return "Intel AVX2 with Intel DL Boost";
            case avx2: return "Intel AVX2";
            case avx: return "Intel AVX";
            case sse41: return "Intel SSE4.1";
            default: return "Intel 64";
        }
    }

    cpu_isa_t isa;
};

static isa_info_t get_isa_info_t(void) {
#ifdef DNNL_ENABLE_MAX_CPU_ISA
    // descending order due to mayiuse check
#define HANDLE_CASE(cpu_isa) \
    if (mayiuse(cpu_isa)) return isa_info_t(cpu_isa);
    HANDLE_CASE(avx512_core_amx_fp16);
    HANDLE_CASE(avx512_core_amx);
    HANDLE_CASE(avx512_core_fp16);
    HANDLE_CASE(avx512_core_bf16_ymm);
    HANDLE_CASE(avx512_core_bf16);
    HANDLE_CASE(avx512_core_vnni);
    HANDLE_CASE(avx512_core);
    HANDLE_CASE(avx2_vnni_2);
    HANDLE_CASE(avx2_vnni);
    HANDLE_CASE(avx2);
    HANDLE_CASE(avx);
    HANDLE_CASE(sse41);
#undef HANDLE_CASE
#endif
    return isa_info_t(isa_undef);
}

const char *get_isa_info() {
    return get_isa_info_t().get_name();
}

cpu_isa_t get_max_cpu_isa() {
    return get_isa_info_t().isa;
}

cpu_isa_t get_max_cpu_isa_mask(bool soft) {
    MAYBE_UNUSED(soft);
#ifdef DNNL_ENABLE_MAX_CPU_ISA
    return max_cpu_isa().get(soft);
#else
    return isa_all;
#endif
}

dnnl_cpu_isa_hints_t get_cpu_isa_hints(bool soft) {
    MAYBE_UNUSED(soft);
#ifdef DNNL_ENABLE_CPU_ISA_HINTS
    return cpu_isa_hints().get(soft);
#else
    return dnnl_cpu_isa_no_hints;
#endif
}

status_t set_max_cpu_isa(dnnl_cpu_isa_t isa) {
    using namespace dnnl::impl::status;
#ifdef DNNL_ENABLE_MAX_CPU_ISA
    using namespace dnnl::impl;
    using namespace dnnl::impl::cpu;

    cpu_isa_t isa_to_set = isa_undef;
#define HANDLE_CASE(cpu_isa) \
    case cpu_isa_traits<cpu_isa>::user_option_val: isa_to_set = cpu_isa; break;
    switch (isa) {
        HANDLE_CASE(isa_all);
        HANDLE_CASE(sse41);
        HANDLE_CASE(avx);
        HANDLE_CASE(avx2);
        HANDLE_CASE(avx2_vnni);
        HANDLE_CASE(avx2_vnni_2);
        HANDLE_CASE(avx512_core);
        HANDLE_CASE(avx512_core_vnni);
        HANDLE_CASE(avx512_core_bf16);
        HANDLE_CASE(avx512_core_amx);
        HANDLE_CASE(avx512_core_fp16);
        HANDLE_CASE(avx512_core_amx_fp16);
        default: return invalid_arguments;
    }
    assert(isa_to_set != isa_undef);
#undef HANDLE_CASE

    if (max_cpu_isa().set(isa_to_set))
        return success;
    else
        return invalid_arguments;
#else
    return unimplemented;
#endif
}

dnnl_cpu_isa_t get_effective_cpu_isa() {
    return get_isa_info_t().convert_to_public_enum();
}

status_t set_cpu_isa_hints(dnnl_cpu_isa_hints_t isa_hints) {
    using namespace dnnl::impl::status;
#ifdef DNNL_ENABLE_CPU_ISA_HINTS
    using namespace dnnl::impl;
    using namespace dnnl::impl::cpu;

    if (cpu_isa_hints().set(isa_hints))
        return success;
    else
        return runtime_error;
#else
    return unimplemented;
#endif
}

namespace amx {

int get_max_palette() {
    if (mayiuse(amx_tile)) {
        static const unsigned int EAX = []() {
            unsigned int data[4] = {};
            Xbyak::util::Cpu::getCpuidEx(0x1D, 0, data);
            return data[0];
        }();
        return EAX;
    } else {
        return 0;
    }
}
int get_target_palette() {
    constexpr int max_supported_palette = 1;
    return nstl::min(max_supported_palette, get_max_palette());
}

namespace {
enum class info_kind_t { max_tiles, max_column_bytes, max_rows };

std::vector<int> get_palettes_info(info_kind_t info_kind) {
    std::vector<int> palettes_info;
    for (int p = 1; p <= get_max_palette(); p++) {
        unsigned int data[4] = {};
        const unsigned int &EBX = data[1];
        const unsigned int &ECX = data[2];
        Xbyak::util::Cpu::getCpuidEx(0x1D, p, data);

        switch (info_kind) {
            case info_kind_t::max_tiles:
                palettes_info.push_back(EBX >> 16);
                break;
            case info_kind_t::max_column_bytes:
                palettes_info.push_back((EBX << 16) >> 16);
                break;
            case info_kind_t::max_rows:
                palettes_info.push_back((ECX << 16) >> 16);
                break;
            default: assert(!"unknown info_kind"); break;
        }
    }
    assert((int)palettes_info.size() == get_max_palette());
    return palettes_info;
}

} // namespace

int get_max_tiles(int palette) {
    if (mayiuse(amx_tile)) {
        if (palette > get_max_palette() || palette <= 0) return -1;
        static const std::vector<int> palettes
                = get_palettes_info(info_kind_t::max_tiles);
        return palettes.at(palette - 1);
    } else {
        return 0;
    }
}

int get_max_column_bytes(int palette) {
    if (mayiuse(amx_tile)) {
        if (palette > get_max_palette() || palette <= 0) return -1;
        static const std::vector<int> palettes
                = get_palettes_info(info_kind_t::max_column_bytes);
        return palettes.at(palette - 1);
    } else {
        return 0;
    }
}

int get_max_rows(int palette) {
    if (mayiuse(amx_tile)) {
        if (palette > get_max_palette() || palette <= 0) return -1;
        static const std::vector<int> palettes
                = get_palettes_info(info_kind_t::max_rows);
        return palettes.at(palette - 1);
    } else {
        return 0;
    }
}

namespace {
#ifdef __linux__
#include <sys/syscall.h>

#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18
#define XFEATURE_MASK_XTILECFG (1 << XFEATURE_XTILECFG)
#define XFEATURE_MASK_XTILEDATA (1 << XFEATURE_XTILEDATA)
#define XFEATURE_MASK_XTILE (XFEATURE_MASK_XTILECFG | XFEATURE_MASK_XTILEDATA)
#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023

bool init() {
    unsigned long bitmask = 0;
    long status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
    if (0 != status) return false;
    if (bitmask & XFEATURE_MASK_XTILEDATA) return true;

    status = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    if (0 != status)
        return false; // XFEATURE_XTILEDATA setup is failed, TMUL usage is not allowed
    status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);

    // XFEATURE_XTILEDATA setup is failed, can't use TMUL
    if (0 != status || !(bitmask & XFEATURE_MASK_XTILEDATA)) return false;

    // XFEATURE_XTILEDATA set successfully, TMUL usage is allowed
    return true;
}
#elif defined(_WIN32)
bool init() {
    // XSAVE feature must be supported in order to check AMX state.
    const bool xsave_supported = cpu().has(Xbyak::util::Cpu::tOSXSAVE);
    if (!xsave_supported) return false;

    // AMX state is controlled by TILECFG and TILEDATA features defined in
    // XCR0[18:17].
    uint64_t xcr0_features = Xbyak::util::Cpu::getXfeature();
    return ((xcr0_features >> 17) & 3) == 3;
}
#else
bool init() {
    // Disable AMX by default to avoid potential crashes.
    return false;
}
#endif

set_once_before_first_get_setting_t<bool> &amx_setting() {
    static set_once_before_first_get_setting_t<bool> setting(init());
    return setting;
}
} // namespace

bool is_available() {
    return amx_setting().get();
}
} // namespace amx

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
