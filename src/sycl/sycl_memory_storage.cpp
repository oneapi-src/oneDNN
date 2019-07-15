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

#include <algorithm>

#include "sycl_memory_storage.hpp"

#include "common/guard_manager.hpp"
#include "common/utils.hpp"
#include "sycl/sycl_engine.hpp"

namespace mkldnn {
namespace impl {
namespace sycl {

struct map_tag;
struct use_host_ptr_tag;

sycl_memory_storage_t::sycl_memory_storage_t(
        engine_t *engine, unsigned flags, size_t size, void *handle)
    : memory_storage_impl_t(engine, size) {
    // Do not allocate memory if one of these is true:
    // 1) size is 0
    // 2) handle is nullptr and flags have use_backend_ptr
    if ((size == 0) || (!handle && (flags & memory_flags_t::alloc) == 0))
        return;

    if (flags & memory_flags_t::alloc) {
#if MKLDNN_SYCL_MEMORY_API == MKLDNN_SYCL_MEMORY_API_VPTR
        void *vptr_alloc = mkldnn::sycl_malloc(size);
        vptr_.reset(vptr_alloc, [](void *ptr) { mkldnn::sycl_free(ptr); });
#else
        buffer_.reset(new buffer_u8_t(cl::sycl::range<1>(size)));
#endif
    } else if (flags & memory_flags_t::use_backend_ptr) {
#if MKLDNN_SYCL_MEMORY_API == MKLDNN_SYCL_MEMORY_API_VPTR
        assert(mkldnn::is_sycl_vptr(handle));
        vptr_.reset(handle, [](void *) {});
#else
        auto &buf_u8 = *static_cast<buffer_u8_t *>(handle);
        buffer_.reset(new buffer_u8_t(buf_u8));
#endif
    } else if (flags & memory_flags_t::use_host_ptr) {
#if MKLDNN_SYCL_MEMORY_API == MKLDNN_SYCL_MEMORY_API_VPTR
        void *vptr_alloc = mkldnn::sycl_malloc(size);

        auto buf = mkldnn::get_sycl_buffer(vptr_alloc);
        {
            auto acc = buf.get_access<cl::sycl::access::mode::write>();
            uint8_t *handle_u8 = static_cast<uint8_t *>(handle);
            for (size_t i = 0; i < size; i++)
                acc[i] = handle_u8[i];
        }

        vptr_.reset(vptr_alloc, [](void *ptr) {
            auto buf = mkldnn::get_sycl_buffer(ptr);
            auto acc = buf.get_access<cl::sycl::access::mode::read>();
            uint8_t *handle_u8 = static_cast<uint8_t *>(handle);
            for (size_t i = 0; i < size; i++)
                handle_u8[i] = acc[i];
            mkldnn::sycl_free(ptr);
        });

#else
        buffer_.reset(new buffer_u8_t(
                static_cast<uint8_t *>(handle), cl::sycl::range<1>(size)));
#endif
    }
}

#if MKLDNN_SYCL_MEMORY_API == MKLDNN_SYCL_MEMORY_API_VPTR
sycl_memory_storage_t::~sycl_memory_storage_t() {
}
#endif

status_t sycl_memory_storage_t::map_data(void **mapped_ptr) const {
#if MKLDNN_SYCL_MEMORY_API == MKLDNN_SYCL_MEMORY_API_VPTR
    if (!vptr_) {
        *mapped_ptr = nullptr;
        return status::success;
    }
#else
    if (!buffer_) {
        *mapped_ptr = nullptr;
        return status::success;
    }
#endif

    auto &guard_manager = guard_manager_t<map_tag>::instance();

#if MKLDNN_SYCL_MEMORY_API == MKLDNN_SYCL_MEMORY_API_VPTR
    auto buf = mkldnn::get_sycl_buffer(vptr_);

    auto acc = buf.get_access<cl::sycl::access::mode::read_write>();
    auto *acc_ptr = new decltype(acc)(acc);
    *mapped_ptr = static_cast<void *>(acc_ptr->get_pointer());
    auto unmap_callback = [=]() { delete acc_ptr; };
#else
    auto acc = buffer_->get_access<cl::sycl::access::mode::read_write>();
    auto *acc_ptr = new decltype(acc)(acc);
    *mapped_ptr = static_cast<void *>(acc_ptr->get_pointer());
    auto unmap_callback = [=]() { delete acc_ptr; };

#endif
    return guard_manager.enter(this, unmap_callback);
}

status_t sycl_memory_storage_t::unmap_data(void *mapped_ptr) const {
    if (!mapped_ptr)
        return status::success;

    auto &guard_manager = guard_manager_t<map_tag>::instance();
    return guard_manager.exit(this);
}

} // namespace sycl
} // namespace impl
} // namespace mkldnn
