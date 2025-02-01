/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#ifndef GPU_INTEL_JIT_V2_CONV_DEBUG_HPP
#define GPU_INTEL_JIT_V2_CONV_DEBUG_HPP

#include "gpu/intel/jit/v2/conv/kernel_desc.hpp"
#include "gpu/intel/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {
namespace conv {

class debug_t {
public:
    static bool init_kernel_desc(kernel_desc_t &desc) {
        if (!env_desc().is_empty()) {
            desc = env_desc();
            return true;
        }
        if (!tls_desc_.is_empty()) {
            desc = tls_desc_;
            return true;
        }
        return false;
    }

    struct kernel_desc_setter_t {
        kernel_desc_setter_t(const kernel_desc_t &desc, kernel_desc_t *desc_ptr)
            : desc_ptr_(desc_ptr) {
            *desc_ptr_ = desc;
        }

        kernel_desc_setter_t(kernel_desc_setter_t &&other) {
            desc_ptr_ = other.desc_ptr_;
            other.desc_ptr_ = nullptr;
        }

        kernel_desc_setter_t(const kernel_desc_setter_t &) = delete;
        kernel_desc_setter_t &operator=(const kernel_desc_setter_t &) = delete;

        ~kernel_desc_setter_t() {
            if (desc_ptr_) *desc_ptr_ = kernel_desc_t();
        }

    private:
        kernel_desc_t *desc_ptr_ = nullptr;
    };

    static kernel_desc_setter_t make_kernel_desc_setter(
            const kernel_desc_t &desc) {
        return kernel_desc_setter_t(desc, &tls_desc_);
    }

    static debug_t &instance() {
        static debug_t _instance;
        return _instance;
    }

private:
    static kernel_desc_t &env_desc() {
        static kernel_desc_t _env_desc = []() {
            kernel_desc_t d;
            auto s_desc = gpu_utils::dev_getenv("desc", std::string());
            if (!s_desc.empty()) d.set(s_desc);
            return d;
        }();
        return _env_desc;
    }

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
