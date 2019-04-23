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

#include "sycl_memory_storage.hpp"

#include "common/guard_manager.hpp"
#include "common/utils.hpp"
#include "sycl/sycl_engine.hpp"

namespace mkldnn {
namespace impl {
namespace sycl {

sycl_memory_storage_t::sycl_memory_storage_t(
        engine_t *engine, unsigned flags, size_t size, void *handle)
    : memory_storage_t(engine) {
    // Do not allocate memory if one of these is true:
    // 1) size is 0
    // 2) handle is nullptr and flags have use_backend_ptr
    if ((size == 0) || (!handle && (flags & memory_flags_t::alloc) == 0))
        return;

    //auto *sycl_engine = utils::downcast<sycl_engine_base_t *>(engine);
    if (flags & memory_flags_t::alloc) {
        buffer_.reset(new untyped_sycl_buffer_t(data_type::u8, size));
    } else if (flags & memory_flags_t::use_backend_ptr) {
        auto *untyped_buf = static_cast<untyped_sycl_buffer_t *>(handle);
        auto &buf = untyped_buf->sycl_buffer<uint8_t>();
        buffer_.reset(new untyped_sycl_buffer_t(buf));
    } else if (flags & memory_flags_t::use_host_ptr) {
        buffer_.reset(new untyped_sycl_buffer_t(handle, data_type::u8, size));
    }
}

status_t sycl_memory_storage_t::map_data(void **mapped_ptr) const {
    if (!buffer_) {
        *mapped_ptr = nullptr;
        return status::success;
    }

    auto &guard_manager = guard_manager_t::instance();

    std::function<void()> unmap_callback;
    *mapped_ptr = buffer_->map_data<cl::sycl::access::mode::read_write>(
            [&](std::function<void()> f) { unmap_callback = f; });

    return guard_manager.enter(this, unmap_callback);
}

status_t sycl_memory_storage_t::unmap_data(void *mapped_ptr) const {
    if (!mapped_ptr)
        return status::success;

    auto &guard_manager = guard_manager_t::instance();
    return guard_manager.exit(this);
}

} // namespace sycl
} // namespace impl
} // namespace mkldnn
