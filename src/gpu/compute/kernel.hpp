/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
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
#include "gpu/compute/kernel_arg_list.hpp"
#include "gpu/compute/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace compute {

class program_list_t;
class kernel_impl_t;

class kernel_t {
public:
    using id_t = intptr_t;
    kernel_t(kernel_impl_t *impl) : impl_(impl) {}

    kernel_t() = default;
    kernel_t(kernel_t &&other) = default;
    kernel_t(const kernel_t &other) = default;
    kernel_t &operator=(const kernel_t &other) = default;
    virtual ~kernel_t() = default;

    operator bool() const { return bool(impl_); }
    id_t id() const;

    kernel_impl_t *impl() const { return impl_.get(); }

    status_t parallel_for(stream_t &stream, const nd_range_t &range,
            const kernel_arg_list_t &arg_list) const;

    status_t parallel_for(
            stream_t &stream, const std::function<void(void *)> &cgf) const;

    status_t realize(kernel_t *kernel, const engine_t *engine,
            program_list_t *programs) const;

    void clear();
    status_t binary_size(size_t *binary_size) const;

    status_t binary(engine_t *engine, compute::binary_t &binary) const;
    const std::shared_ptr<compute::binary_t> &binary() const;

    const std::vector<scalar_type_t> &arg_types() const;

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
            const kernel_arg_list_t &arg_list) {
        assert(!"unexpected");
        return status::runtime_error;
    }

    virtual status_t parallel_for(
            stream_t &stream, const std::function<void(void *)> &cgf) const {
        assert(!"unexpected");
        return status::runtime_error;
    }

    virtual status_t realize(kernel_t *kernel, const engine_t *engine,
            program_list_t *programs) const = 0;

    virtual void clear() = 0;
    virtual status_t binary_size(size_t *binary_size) const {
        assert(!"unexpected");
        return status::runtime_error;
    }
    virtual status_t binary(engine_t *engine, compute::binary_t &binary) const {
        assert(!"unexpected");
        return status::runtime_error;
    }

    virtual const std::vector<scalar_type_t> &arg_types() const = 0;
    virtual const std::shared_ptr<compute::binary_t> &binary() const = 0;
};

inline kernel_t::id_t kernel_t::id() const {
    return reinterpret_cast<id_t>(impl_.get());
}
inline status_t kernel_t::parallel_for(stream_t &stream,
        const nd_range_t &range, const kernel_arg_list_t &arg_list) const {
    return impl_->parallel_for(stream, range, arg_list);
}
inline status_t kernel_t::parallel_for(
        stream_t &stream, const std::function<void(void *)> &cgf) const {
    return impl_->parallel_for(stream, cgf);
}

inline status_t kernel_t::realize(kernel_t *kernel, const engine_t *engine,
        program_list_t *programs) const {
    return impl_->realize(kernel, engine, programs);
}

inline void kernel_t::clear() {
    impl_->clear();
}

inline status_t kernel_t::binary_size(size_t *binary_size) const {
    return impl_->binary_size(binary_size);
}

inline status_t kernel_t::binary(
        engine_t *engine, compute::binary_t &binary) const {
    return impl_->binary(engine, binary);
}

inline const std::shared_ptr<compute::binary_t> &kernel_t::binary() const {
    return impl_->binary();
}

inline const std::vector<scalar_type_t> &kernel_t::arg_types() const {
    return impl_->arg_types();
}

} // namespace compute
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_COMPUTE_KERNEL_HPP
