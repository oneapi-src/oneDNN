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

#include "cpu_isa_traits.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

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

// Attempt a "set before 1st get" of the max_cpu_isa() flag
bool init_max_cpu_isa() {
    if (max_cpu_isa().initialized()) { return false; }

    cpu_isa_t max_cpu_isa_val = isa_full; // x86:x86_full, VE:ve_full, ...

    char buf[64];
    if (getenv("DNNL_MAX_CPU_ISA", buf, sizeof(buf)) > 0) {
        // string value --> cpu_isa_t max_cpu_isa_val;
#define IF_HANDLE_CASE(CPU_ISA_T) \
    if (std::strcmp(buf, cpu_isa_traits<CPU_ISA_T>::user_option_env) == 0) \
    max_cpu_isa_val = CPU_ISA_T
#define ELSEIF_HANDLE_CASE(CPU_ISA_T) else IF_HANDLE_CASE(CPU_ISA_T)
        // allow case-insensitive compare
        for (size_t i = 0u; i < sizeof(buf) && buf[i]; ++i)
            buf[i] = toupper(buf[i]);
        //printf(" getenv DNNL_MAX_CPU_ISA --> %s\n", &buf[0]);

        IF_HANDLE_CASE(vanilla); // "VANILLA" --> (CPU-agnostic ref impls)
        ELSEIF_HANDLE_CASE(isa_any); // "ANY" --> x86_common or ve_common or ...
        ELSEIF_HANDLE_CASE(isa_full); // "ALL" --> x86_full or ve_full or ...
#if TARGET_X86
        ELSEIF_HANDLE_CASE(sse41);
        ELSEIF_HANDLE_CASE(avx);
        ELSEIF_HANDLE_CASE(avx2);
        ELSEIF_HANDLE_CASE(avx512_mic);
        ELSEIF_HANDLE_CASE(avx512_mic_4ops);
        ELSEIF_HANDLE_CASE(avx512_core);
        ELSEIF_HANDLE_CASE(avx512_core_vnni);
        ELSEIF_HANDLE_CASE(avx512_core_bf16);
        //else printf("Bad DNNL_MAX_CPU_ISA=%s environment for x86", buf);
#elif TARGET_VE
        ELSEIF_HANDLE_CASE(vednn);
        ELSEIF_HANDLE_CASE(vejit);
        //else printf("Bad DNNL_MAX_CPU_ISA=%s environment value for VE", buf);
#endif

#undef IF_HANDLE_CASE
#undef ELSEIF_HANDLE_CASE
    }

    //printf("init_max_cpu_isa->0x%x\n", max_cpu_isa_val);
    return max_cpu_isa().set(max_cpu_isa_val);
}
#endif // DNNL_ENABLE_MAX_CPU_ISA

#if defined(DNNL_ENABLE_MAX_CPU_ISA)
cpu_isa_t get_max_cpu_isa(bool soft) {
    MAYBE_UNUSED(soft);
    init_max_cpu_isa();
    return max_cpu_isa().get(soft);
}

#else
cpu_isa_t get_max_cpu_isa(bool soft) {
    MAYBE_UNUSED(soft);
    return isa_full;
}
#endif // defined(DNNL_ENABLE_MAX_CPU_ISA)

} // namespace cpu
} // namespace impl
} // namespace dnnl

/** set max_cpu_isa() to the \c cpu_isa_t mask corresponding to a
 * \c dnnl_cpu_isa_t.
 * When a \c cpu_isa trait field matches the dnnl \c isa value,
 * remember \c cpu_isa value.
 *
 * This decouples dnnl cpu_isa values [public] from cpu_isa [internal] ones.
 *
 * \c max_cpu_isa() ensures we set a value at most once (subsequent calls
 * should fail).
 */
dnnl_status_t dnnl_set_max_cpu_isa(dnnl_cpu_isa_t isa) {
    using namespace dnnl::impl::status;
#ifdef DNNL_ENABLE_MAX_CPU_ISA
    using namespace dnnl::impl;
    using namespace dnnl::impl::cpu;

    // XXX see if we can get rid of from_dnnl, at least in the header (see also tests/gtests/test_isa_*)
    cpu_isa_t isa_to_set = ::dnnl::impl::cpu::from_dnnl(isa);
    if (isa_to_set == isa_unknown) return invalid_arguments;

    if (::dnnl::impl::cpu::max_cpu_isa().set(isa_to_set))
        return success;
    else
        return invalid_arguments;
#else
    return unimplemented;
#endif // DNNL_ENABLE_MAX_CPU_ISA
}
// vim: et ts=4 sw=4 cindent cino=+2s,^=l0,\:0,N-s
