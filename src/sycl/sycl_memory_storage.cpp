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

sycl_memory_storage_t::sycl_memory_storage_t(engine_t *engine, unsigned flags,
        size_t size, size_t alignment, void *handle)
    : memory_storage_impl_t(engine, size) {

    UNUSED(alignment);

    // Do not allocate memory if one of these is true:
    // 1) size is 0
    // 2) handle is nullptr and flags have use_backend_ptr
    if ((size == 0) || (!handle && (flags & memory_flags_t::alloc) == 0))
        return;

    if (flags & memory_flags_t::alloc) {
#if MKLDNN_SYCL_MEMORY_API == MKLDNN_SYCL_MEMORY_API_BUFFER
        buffer_.reset(new buffer_u8_t(cl::sycl::range<1>(size)));
#elif MKLDNN_SYCL_MEMORY_API == MKLDNN_SYCL_MEMORY_API_USM
        auto *sycl_engine = utils::downcast<sycl_engine_base_t *>(engine);
        auto &sycl_dev = sycl_engine->device();
        auto &sycl_ctx = sycl_engine->context();

        void *usm_ptr_alloc = cl::sycl::malloc_shared(size, sycl_dev, sycl_ctx);
        usm_ptr_ = decltype(usm_ptr_)(usm_ptr_alloc,
                [&](void *ptr) { cl::sycl::free(ptr, sycl_ctx); });
#elif MKLDNN_SYCL_MEMORY_API == MKLDNN_SYCL_MEMORY_API_VPTR
        void *vptr_alloc = mkldnn::sycl_malloc(size);
        vptr_ptr_ = decltype(vptr_ptr_)(
                vptr_alloc, [](void *ptr) { mkldnn::sycl_free(ptr); });
#endif
    } else if (flags & memory_flags_t::use_backend_ptr) {
#if MKLDNN_SYCL_MEMORY_API == MKLDNN_SYCL_MEMORY_API_BUFFER
        auto &buf_u8 = *static_cast<buffer_u8_t *>(handle);
        buffer_.reset(new buffer_u8_t(buf_u8));
#elif MKLDNN_SYCL_MEMORY_API == MKLDNN_SYCL_MEMORY_API_VPTR
        assert(mkldnn::is_sycl_vptr(handle));
        vptr_ = decltype(vptr_)((handle, [](void *) {});
#elif MKLDNN_SYCL_MEMORY_API == MKLDNN_SYCL_MEMORY_API_USM
        usm_ptr_ = decltype(usm_ptr_)(handle, [](void *) {});
#endif
    }
}

#if MKLDNN_SYCL_MEMORY_API == MKLDNN_SYCL_MEMORY_API_VPTR
// TODO: why???
sycl_memory_storage_t::~sycl_memory_storage_t() {
}
#endif

status_t sycl_memory_storage_t::map_data(void **mapped_ptr) const {
#if MKLDNN_SYCL_MEMORY_API == MKLDNN_SYCL_MEMORY_API_USM
    *mapped_ptr = usm_ptr_.get();
    return status::success;
#endif

#if MKLDNN_SYCL_MEMORY_API == MKLDNN_SYCL_MEMORY_API_BUFFER
    if (!buffer_) {
        *mapped_ptr = nullptr;
        return status::success;
    }
#elif MKLDNN_SYCL_MEMORY_API == MKLDNN_SYCL_MEMORY_API_VPTR
    if (!vptr_) {
        *mapped_ptr = nullptr;
        return status::success;
    }
#endif

    auto &guard_manager = guard_manager_t<map_tag>::instance();

#if MKLDNN_SYCL_MEMORY_API == MKLDNN_SYCL_MEMORY_API_BUFFER
    auto acc = buffer_->get_access<cl::sycl::access::mode::read_write>();
    auto *acc_ptr = new decltype(acc)(acc);
    *mapped_ptr = static_cast<void *>(acc_ptr->get_pointer());
    auto unmap_callback = [=]() { delete acc_ptr; };
    return guard_manager.enter(this, unmap_callback);
#elif MKLDNN_SYCL_MEMORY_API == MKLDNN_SYCL_MEMORY_API_VPTR
    auto buf = mkldnn::get_sycl_buffer(vptr_);

    auto acc = buf.get_access<cl::sycl::access::mode::read_write>();
    auto *acc_ptr = new decltype(acc)(acc);
    *mapped_ptr = static_cast<void *>(acc_ptr->get_pointer());
    auto unmap_callback = [=]() { delete acc_ptr; };
    return guard_manager.enter(this, unmap_callback);
#endif
}

status_t sycl_memory_storage_t::unmap_data(void *mapped_ptr) const {
#if MKLDNN_SYCL_MEMORY_API == MKLDNN_SYCL_MEMORY_API_USM
    return status::success;
#endif

    if (!mapped_ptr)
        return status::success;

    auto &guard_manager = guard_manager_t<map_tag>::instance();
    return guard_manager.exit(this);
}

} // namespace sycl
} // namespace impl
} // namespace mkldnn
