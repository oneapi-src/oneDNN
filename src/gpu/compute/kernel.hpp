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

namespace dnnl {
namespace impl {
namespace gpu {
namespace compute {

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
        assert(!"unexpected");
        return status::runtime_error;
    }

    virtual status_t parallel_for(
            stream_t &stream, const std::function<void(void *)> &cgf) {
        assert(!"unexpected");
        return status::runtime_error;
    }

    virtual status_t get_binary_size(
            const engine_t *engine, size_t *binary_size) const {
        assert(!"unexpected");
        return status::runtime_error;
    }
    virtual status_t get_binary(
            const engine_t *engine, compute::binary_t &binary) const {
        assert(!"unexpected");
        return status::runtime_error;
    }

    virtual const std::vector<scalar_type_t> &arg_types() const {
        static const std::vector<scalar_type_t> dummy;
        return dummy;
    }

    virtual void save_output_events() {}
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

} // namespace compute
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_COMPUTE_KERNEL_HPP
