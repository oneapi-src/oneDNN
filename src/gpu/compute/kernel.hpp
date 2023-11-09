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

#ifndef GPU_COMPUTE_KERNEL_HPP
#define GPU_COMPUTE_KERNEL_HPP

#include <memory>
#include <utility>

#include "common/stream.hpp"
#include "gpu/compute/context.hpp"
#include "gpu/compute/kernel_arg_list.hpp"
#include "gpu/compute/utils.hpp"
#include "gpu/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace compute {

class compute_engine_t;
class kernel_impl_t;

class kernel_t {
public:
    kernel_t(std::nullptr_t) : impl_(nullptr) {}
    kernel_t(std::shared_ptr<kernel_impl_t> &impl) : impl_(impl) {}
    kernel_t(std::shared_ptr<kernel_impl_t> &&impl) : impl_(std::move(impl)) {}

    kernel_t() = default;
    kernel_t(kernel_t &&other) = default;
    kernel_t(const kernel_t &other) = default;
    kernel_t &operator=(const kernel_t &other) = default;
    kernel_t &operator=(kernel_t &&other) = default;

    virtual ~kernel_t() = default;

    operator bool() const { return bool(impl_); }

    kernel_impl_t *impl() const { return impl_.get(); }

    status_t parallel_for(stream_t &stream, const nd_range_t &range,
            const kernel_arg_list_t &arg_list, const event_t &deps,
            event_t &out_dep) const;

    status_t parallel_for(
            stream_t &stream, const std::function<void(void *)> &cgf) const;

    status_t get_binary_size(const engine_t *engine, size_t *binary_size) const;
    status_t get_binary(
            const engine_t *engine, compute::binary_t &binary) const;

    const std::vector<scalar_type_t> &arg_types() const;

    void save_output_events();

    bool is_on(const compute_engine_t &) const;

    status_t dump() const;

    std::string name() const;

private:
    std::shared_ptr<kernel_impl_t> impl_;
};

class kernel_impl_t {
public:
    kernel_impl_t() = default;

    kernel_impl_t(const kernel_impl_t &) = delete;
    kernel_impl_t &operator=(const kernel_impl_t &) = delete;
    virtual ~kernel_impl_t() = default;

    virtual status_t parallel_for(stream_t &stream, const nd_range_t &range,
            const kernel_arg_list_t &arg_list, const event_t &deps,
            event_t &out_dep) {
        gpu_assert(false) << "unimplemented function parallel_for() called";
        return status::runtime_error;
    }

    virtual status_t parallel_for(
            stream_t &stream, const std::function<void(void *)> &cgf) {
        gpu_assert(false) << "unimplemented function parallel_for() called";
        return status::runtime_error;
    }

    virtual status_t get_binary_size(
            const engine_t *engine, size_t *binary_size) const {
        gpu_assert(false) << "unimplemented function get_binary_size() called";
        return status::runtime_error;
    }
    virtual status_t get_binary(
            const engine_t *engine, compute::binary_t &binary) const {
        gpu_assert(false) << "unimplemented function get_binary() called";
        return status::runtime_error;
    }

    virtual const std::vector<scalar_type_t> &arg_types() const {
        static const std::vector<scalar_type_t> dummy;
        return dummy;
    }

    virtual void save_output_events() {}

    virtual bool is_on(const compute_engine_t &) const {
        gpu_assert(false) << "unimplemented function is_on() called";
        return false;
    }

    virtual status_t dump() const {
        gpu_assert(false) << "unimplemented function dump() called";
        return status::runtime_error;
    }

    virtual std::string name() const {
        gpu_assert(false) << "unimplemented function name() called";
        return "unknown";
    }
};

inline status_t kernel_t::parallel_for(stream_t &stream,
        const nd_range_t &range, const kernel_arg_list_t &arg_list,
        const event_t &deps, event_t &out_dep) const {
    return impl_->parallel_for(stream, range, arg_list, deps, out_dep);
}
inline status_t kernel_t::parallel_for(
        stream_t &stream, const std::function<void(void *)> &cgf) const {
    return impl_->parallel_for(stream, cgf);
}

inline status_t kernel_t::get_binary_size(
        const engine_t *engine, size_t *binary_size) const {
    return impl_->get_binary_size(engine, binary_size);
}

inline status_t kernel_t::get_binary(
        const engine_t *engine, compute::binary_t &binary) const {
    return impl_->get_binary(engine, binary);
}

inline const std::vector<scalar_type_t> &kernel_t::arg_types() const {
    return impl_->arg_types();
}

inline void kernel_t::save_output_events() {
    return impl_->save_output_events();
}

inline bool kernel_t::is_on(const compute_engine_t &engine) const {
    return impl_->is_on(engine);
}

inline status_t kernel_t::dump() const {
    if (!gpu_utils::is_jit_dump_enabled()) return status::success;
    return impl_->dump();
}

inline std::string kernel_t::name() const {
    return impl_->name();
}

class kernel_bundle_t {
public:
    kernel_bundle_t() = default;
    kernel_bundle_t(std::vector<kernel_t> &&kernels,
            const std::vector<const char *> &kernel_names) {
        for (size_t i = 0; i < kernels.size(); i++) {
            bundle[kernel_names[i]] = std::move(kernels[i]);
        }
    }
    // Copies may be expensive, require explicit clone
    kernel_bundle_t(const kernel_bundle_t &other) = delete;
    kernel_bundle_t &operator=(const kernel_bundle_t &other) = delete;
    kernel_bundle_t(kernel_bundle_t &&other) = default;
    kernel_bundle_t &operator=(kernel_bundle_t &&other) = default;

    status_t get_kernels(std::vector<kernel_t> &kernels,
            const std::vector<const char *> &kernel_names) const {
        kernels = std::vector<kernel_t>(kernel_names.size());
        for (size_t i = 0; i < kernel_names.size(); i++) {
            if (!kernel_names[i]) continue;
            auto kernel_entry = bundle.find(kernel_names[i]);
            if (kernel_entry == bundle.end()) return status::runtime_error;
            kernels[i] = kernel_entry->second;
        }
        return status::success;
    }

    status_t get_kernels(const engine_t *engine, std::vector<kernel_t> &kernels,
            const std::vector<const char *> &kernel_names) const {
        auto &compute_engine
                = *utils::downcast<const compute_engine_t *>(engine);
        if (!is_on(compute_engine)) return status::runtime_error;
        return get_kernels(kernels, kernel_names);
    }

    bool is_on(const compute_engine_t &engine) const {
        // All kernels are required to be located in the same context.
        return !bundle.empty() && bundle.begin()->second.is_on(engine);
    }

    std::unordered_map<std::string, kernel_t> bundle;
};

} // namespace compute
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_COMPUTE_KERNEL_HPP
