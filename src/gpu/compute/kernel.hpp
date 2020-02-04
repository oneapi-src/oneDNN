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

class kernel_impl_t;

class kernel_t {
public:
    kernel_t(kernel_impl_t *impl) : impl_(impl) {}

    kernel_t() = default;
    kernel_t(kernel_t &&other) = default;
    kernel_t(const kernel_t &other) = default;
    kernel_t &operator=(const kernel_t &other) = default;
    virtual ~kernel_t() = default;

    operator bool() const { return bool(impl_); }

    status_t parallel_for(stream_t &stream, const nd_range_t &range,
            const kernel_arg_list_t &arg_list) const;

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
            const kernel_arg_list_t &arg_list) const = 0;
};

inline status_t kernel_t::parallel_for(stream_t &stream,
        const nd_range_t &range, const kernel_arg_list_t &arg_list) const {
    return impl_->parallel_for(stream, range, arg_list);
}

class binary_impl_t;

class binary_t {
public:
    using id_t = intptr_t;
    binary_t(binary_impl_t *impl) : impl_(impl) {}

    binary_t() = default;
    binary_t(binary_t &&other) = default;
    binary_t(const binary_t &other) = default;
    binary_t &operator=(const binary_t &other) = default;
    virtual ~binary_t() = default;

    operator bool() const { return bool(impl_); }
    size_t size() const;
    const char *name() const;
    const unsigned char *data() const;
    id_t get_id() const;

private:
    std::shared_ptr<binary_impl_t> impl_;
};

class binary_impl_t {
public:
    binary_impl_t() = default;

    binary_impl_t(const binary_impl_t &) = delete;
    binary_impl_t &operator=(const binary_impl_t &) = delete;
    virtual ~binary_impl_t() = default;

    virtual size_t size() const = 0;
    virtual const char *name() const = 0;
    virtual const unsigned char *data() const = 0;
};

inline binary_t::id_t binary_t::get_id() const {
    return reinterpret_cast<id_t>(impl_.get());
}
inline size_t binary_t::size() const {
    return impl_->size();
}
inline const char *binary_t::name() const {
    return impl_->name();
}
inline const unsigned char *binary_t::data() const {
    return impl_->data();
}

} // namespace compute
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_COMPUTE_KERNEL_HPP
