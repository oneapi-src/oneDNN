/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#ifndef GPU_INTEL_JIT_V2_CONV_PLAN_PRESET_HPP
#define GPU_INTEL_JIT_V2_CONV_PLAN_PRESET_HPP

#include "gpu/intel/jit/v2/conv/kernel_desc.hpp"
#include "gpu/intel/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {
namespace conv {

class plan_preset_t {
public:
    bool is_set() const {
        if (!env_desc_.is_empty()) return true;
        if (!tls_desc_.is_empty()) return true;
        return false;
    }

    const kernel_desc_t &get() const {
        if (!env_desc_.is_empty()) return env_desc_;
        if (!tls_desc_.is_empty()) return tls_desc_;
        ir_error_not_expected();
        return env_desc_;
    }

    struct guard_t {
        guard_t(const kernel_desc_t &desc, kernel_desc_t *desc_ptr)
            : desc_ptr_(desc_ptr) {
            *desc_ptr_ = desc;
        }

        guard_t(guard_t &&other) {
            desc_ptr_ = other.desc_ptr_;
            other.desc_ptr_ = nullptr;
        }

        guard_t(const guard_t &) = delete;
        guard_t &operator=(const guard_t &) = delete;

        ~guard_t() {
            if (desc_ptr_) *desc_ptr_ = kernel_desc_t();
        }

    private:
        kernel_desc_t *desc_ptr_ = nullptr;
    };

    guard_t make_guard(const kernel_desc_t &desc) {
        return guard_t(desc, &tls_desc_);
    }

    static plan_preset_t &instance() {
        static plan_preset_t _instance;
        return _instance;
    }

private:
    plan_preset_t() {
        auto s_desc = gpu_utils::dev_getenv("desc", std::string());
        if (!s_desc.empty()) env_desc_.set(s_desc);
    }

    static kernel_desc_t env_desc_;
    static thread_local kernel_desc_t tls_desc_;
};

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
