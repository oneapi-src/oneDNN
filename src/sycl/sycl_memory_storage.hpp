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

#include "common/c_types_map.hpp"
#include "common/memory_storage.hpp"
#include "common/utils.hpp"
#include "sycl/sycl_utils.hpp"

#include "mkldnn.hpp"

#include <CL/sycl.hpp>
#include <functional>
#include <memory>

namespace mkldnn {
namespace impl {
namespace sycl {

class sycl_memory_storage_t : public memory_storage_impl_t
{
public:
    sycl_memory_storage_t(engine_t *engine, unsigned flags, size_t size,
            size_t alignment, void *handle);

    virtual status_t get_data_handle(void **handle) const override {
#if MKLDNN_SYCL_MEMORY_API == MKLDNN_SYCL_MEMORY_API_VPTR
        *handle = vptr_.get();
#else
        *handle = static_cast<void *>(buffer_.get());
#endif
        return status::success;
    }

    virtual status_t set_data_handle(void *handle) override {
#if MKLDNN_SYCL_MEMORY_API == MKLDNN_SYCL_MEMORY_API_VPTR
        assert(mkldnn::is_sycl_vptr(handle));
        vptr_.reset(handle, [](void *){});
#else
        auto *buf_u8_ptr = static_cast<buffer_u8_t *>(handle);
        buffer_.reset(new buffer_u8_t(*buf_u8_ptr));
#endif
        return status::success;
    }

    virtual status_t map_data(void **mapped_ptr) const override;
    virtual status_t unmap_data(void *mapped_ptr) const override;

#if MKLDNN_SYCL_MEMORY_API == MKLDNN_SYCL_MEMORY_API_VPTR
    void *vptr() const { return vptr_.get(); }
#else
    buffer_u8_t &buffer() const { return *buffer_; }
#endif

    virtual uintptr_t base_offset() const override { return 0; }

private:
#if MKLDNN_SYCL_MEMORY_API == MKLDNN_SYCL_MEMORY_API_VPTR
    std::unique_ptr<void, std::function<void()>> vptr_;
#else
    std::shared_ptr<buffer_u8_t> buffer_;
#endif
};

} // namespace sycl
} // namespace impl
} // namespace mkldnn

#endif
