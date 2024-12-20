/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef GPU_INTEL_JIT_V2_CONV_DEBUG_SETTINGS_HPP
#define GPU_INTEL_JIT_V2_CONV_DEBUG_SETTINGS_HPP

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {
namespace conv {

struct debug_settings_t;
const debug_settings_t &debug_settings();
#ifdef DNNL_DEV_MODE
void set_debug_settings(const debug_settings_t &new_settings);
#endif

struct debug_settings_t {
    bool skip_zero_out = false;
};

struct debug_settings_guard_t {
    debug_settings_guard_t(const debug_settings_t &new_settings) {
        active = true;
        old_settings_ = debug_settings();
        set_debug_settings(new_settings);
    }

    debug_settings_guard_t(debug_settings_guard_t &&other) {
        active = true;
        old_settings_ = other.old_settings_;
        other.active = false;
    }

    debug_settings_guard_t(const debug_settings_guard_t &) = delete;
    debug_settings_guard_t &operator=(const debug_settings_guard_t &) = delete;

    ~debug_settings_guard_t() {
        if (active) set_debug_settings(old_settings_);
    }

private:
    debug_settings_t old_settings_;
    bool active = false;
};

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
