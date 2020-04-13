/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include <atomic>
#include <cstring>
#include <mutex>

#include "common/utils.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

namespace {
#ifdef DNNL_ENABLE_MAX_CPU_ISA

// A setting (basically a value) that can be set() multiple times until the
// time first time the get() method is called. The set() method is expected to
// be as expensive as a busy-waiting spinlock. The get() method is expected to
// be asymptotically as expensive as a single lock-prefixed memory read. The
// get() method also has a 'soft' mode when the setting is not locked for
// re-setting. This is used for testing purposes.
template <typename T>
struct set_before_first_get_setting_t {
private:
    T value_;
    bool initialized_;
    std::atomic<unsigned> state_;
    enum : unsigned { idle = 0, busy_setting = 1, locked_after_a_get = 2 };

public:
    set_before_first_get_setting_t(T init = T(0))
        : value_ {init}, initialized_ {false}, state_ {0} {}

    bool set(T new_value) {
        if (state_.load() == locked_after_a_get) return false;

        while (true) {
            unsigned expected = idle;
            if (state_.compare_exchange_weak(expected, busy_setting)) break;
            if (expected == locked_after_a_get) return false;
        }

        value_ = new_value;
        initialized_ = true;
        state_.store(idle);
        return true;
    }

    bool initialized() { return initialized_; }

    T get(bool soft = false) {
        if (!soft && state_.load() != locked_after_a_get) {
            while (true) {
                unsigned expected = idle;
                if (state_.compare_exchange_weak(expected, locked_after_a_get))
                    break;
                if (expected == locked_after_a_get) break;
            }
        }
        return value_;
    }
};

set_before_first_get_setting_t<cpu_isa_t> &max_cpu_isa() {
    static set_before_first_get_setting_t<cpu_isa_t> max_cpu_isa_setting;
    return max_cpu_isa_setting;
}

bool init_max_cpu_isa() {
    if (max_cpu_isa().initialized()) return false;

    cpu_isa_t max_cpu_isa_val = isa_all;
    char buf[64];
    if (getenv("DNNL_MAX_CPU_ISA", buf, sizeof(buf)) > 0) {

#define IF_HANDLE_CASE(cpu_isa) \
    if (std::strcmp(buf, cpu_isa_traits<cpu_isa>::user_option_env) == 0) \
    max_cpu_isa_val = cpu_isa
#define ELSEIF_HANDLE_CASE(cpu_isa) else IF_HANDLE_CASE(cpu_isa)

        IF_HANDLE_CASE(isa_all);
        ELSEIF_HANDLE_CASE(sse41);
        ELSEIF_HANDLE_CASE(avx);
        ELSEIF_HANDLE_CASE(avx2);
        ELSEIF_HANDLE_CASE(avx512_mic);
        ELSEIF_HANDLE_CASE(avx512_mic_4ops);
        ELSEIF_HANDLE_CASE(avx512_core);
        ELSEIF_HANDLE_CASE(avx512_core_vnni);
        ELSEIF_HANDLE_CASE(avx512_core_bf16);

#undef IF_HANDLE_CASE
#undef ELSEIF_HANDLE_CASE
    }

    return max_cpu_isa().set(max_cpu_isa_val);
}
#endif
} // namespace

namespace x64 {

const char *get_isa_info() {
    /* oneDNN CPU ISA info */
    const char *ISA_ANY = "Intel 64";
    const char *SSE41 = "Intel SSE4.1";
    const char *AVX = "Intel AVX";
    const char *AVX2 = "Intel AVX2";
    const char *AVX512_COMMON = "Intel AVX-512";
    const char *AVX512_CORE
            = "Intel AVX-512 with AVX512BW, AVX512VL, and AVX512DQ extensions";
    const char *AVX512_CORE_VNNI = "Intel AVX-512 with Intel DL Boost";
    const char *AVX512_MIC
            = "Intel AVX-512 with AVX512CD, AVX512ER, and AVX512PF extensions";
    const char *AVX512_MIC_4OPS
            = "Intel AVX-512 with AVX512_4FMAPS and AVX512_4VNNIW extensions";
    const char *AVX512_CORE_BF16
            = "Intel AVX-512 with Intel DL Boost and bfloat16 support";

    if (mayiuse(avx512_core_bf16)) return AVX512_CORE_BF16;
    if (mayiuse(avx512_mic_4ops)) return AVX512_MIC_4OPS;
    if (mayiuse(avx512_mic)) return AVX512_MIC;
    if (mayiuse(avx512_core_vnni)) return AVX512_CORE_VNNI;
    if (mayiuse(avx512_core)) return AVX512_CORE;
    if (mayiuse(avx512_common)) return AVX512_COMMON;
    if (mayiuse(avx2)) return AVX2;
    if (mayiuse(avx)) return AVX;
    if (mayiuse(sse41)) return SSE41;

    return ISA_ANY;
}

cpu::cpu_isa_t get_max_cpu_isa(bool soft) {
    MAYBE_UNUSED(soft);
#ifdef DNNL_ENABLE_MAX_CPU_ISA
    init_max_cpu_isa();
    return max_cpu_isa().get(soft);
#else
    return isa_all;
#endif
}

status_t set_max_cpu_isa(dnnl_cpu_isa_t isa) {
    using namespace dnnl::impl::status;
#ifdef DNNL_ENABLE_MAX_CPU_ISA
    using namespace dnnl::impl;
    using namespace dnnl::impl::cpu;

    cpu_isa_t isa_to_set = isa_any;
#define HANDLE_CASE(cpu_isa) \
    case cpu_isa_traits<cpu_isa>::user_option_val: isa_to_set = cpu_isa; break;
    switch (isa) {
        HANDLE_CASE(isa_all);
        HANDLE_CASE(sse41);
        HANDLE_CASE(avx);
        HANDLE_CASE(avx2);
        HANDLE_CASE(avx512_mic);
        HANDLE_CASE(avx512_mic_4ops);
        HANDLE_CASE(avx512_core);
        HANDLE_CASE(avx512_core_vnni);
        HANDLE_CASE(avx512_core_bf16);
        default: return invalid_arguments;
    }
    assert(isa_to_set != isa_any);
#undef HANDLE_CASE

    if (max_cpu_isa().set(isa_to_set))
        return success;
    else
        return invalid_arguments;
#else
    return unimplemented;
#endif
}
} // namespace x64

} // namespace cpu
} // namespace impl
} // namespace dnnl
