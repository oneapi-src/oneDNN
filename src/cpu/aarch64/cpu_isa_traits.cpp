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

#include "cpu/aarch64/cpu_isa_traits.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

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
        ELSEIF_HANDLE_CASE(simd);
        ELSEIF_HANDLE_CASE(sve);

#undef IF_HANDLE_CASE
#undef ELSEIF_HANDLE_CASE
    }

    return max_cpu_isa().set(max_cpu_isa_val);
}
#endif
} // namespace

struct isa_info_t {
    isa_info_t(cpu_isa_t aisa) : isa(aisa) {};

    // this converter is needed as code base defines certain ISAs
    // that the library does not expose (eg SVE),
    // so the internal and external enum types do not coincide.
    dnnl_cpu_isa_t convert_to_public_enum(void) const {
        switch (isa) {
            case sve: return dnnl_cpu_isa_sve;
            case simd: return dnnl_cpu_isa_simd;
            default: return dnnl_cpu_isa_all;
        }
    }

    const char *get_name() const {
        switch (isa) {
            case sve: return "AARCH64 SVE";
            case simd: return "AARCH64 SIMD";
            default: return "AARCH64";
        }
    }

    cpu_isa_t isa;
};

static const isa_info_t get_isa_info_t(void) {
    // descending order due to mayiuse check
#define HANDLE_CASE(cpu_isa) \
    if (mayiuse(cpu_isa)) return isa_info_t(cpu_isa);
    HANDLE_CASE(sve);
    HANDLE_CASE(simd);
#undef HANDLE_CASE
    return isa_info_t(isa_any);
}

const char *get_isa_info() {
    return get_isa_info_t().get_name();
}

cpu_isa_t get_max_cpu_isa_mask(bool soft) {
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
        HANDLE_CASE(simd);
        HANDLE_CASE(sve);
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

dnnl_cpu_isa_t get_effective_cpu_isa() {
    return get_isa_info_t().convert_to_public_enum();
}
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
