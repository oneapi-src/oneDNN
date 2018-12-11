/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef SYCL_MEMORY_STORAGE_HPP
#define SYCL_MEMORY_STORAGE_HPP

#include "c_types_map.hpp"
#include "memory_storage.hpp"
#include "utils.hpp"

#include "mkldnn_support.hpp"

#include <CL/sycl.hpp>
#include <memory>

namespace mkldnn {
namespace impl {
namespace sycl {

class sycl_memory_storage_t : public memory_storage_t
{
public:
    sycl_memory_storage_t(engine_t *engine,
            unsigned flags, size_t size, void *handle);
    sycl_memory_storage_t(sycl_memory_storage_t &&other)
        : memory_storage_t(other.engine()), buffer_(std::move(other.buffer_)) {}

    sycl_memory_storage_t(const sycl_memory_storage_t &) = delete;
    sycl_memory_storage_t &operator=(const sycl_memory_storage_t &) = delete;

    virtual status_t get_data_handle(void **handle) const override {
        *handle = static_cast<void *>(buffer_.get());
        return status::success;
    }

    virtual status_t set_data_handle(void *handle) override {
        auto *untyped_buf_ptr = static_cast<untyped_sycl_buffer_t *>(handle);
        buffer_.reset(new untyped_sycl_buffer_t(std::move(*untyped_buf_ptr)));
        return status::success;
    }

    virtual status_t map_data(void **mapped_ptr) const override;
    virtual status_t unmap_data(void *mapped_ptr) const override;

    untyped_sycl_buffer_t &buffer() const { return *buffer_; }

#if 0
    template <typename T, int ndims = 1>
    cl::sycl::buffer<T, ndims> buffer() const {
        return buffer->to_sycl_buffer<T, ndims>();
    }
#endif

private:
    std::unique_ptr<untyped_sycl_buffer_t> buffer_;
};

} // namespace sycl
} // namespace impl
} // namespace mkldnn

#endif
