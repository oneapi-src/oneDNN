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

#include "sycl/sycl_buffer_memory_storage.hpp"
#include "sycl/sycl_engine_base.hpp"

#include "common/memory.hpp"
#include "common/memory_map_manager.hpp"
#include "common/utils.hpp"
#include "sycl/sycl_stream.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

namespace {
template <::sycl::access_mode mode>
gpu::sycl::sycl_memory_arg_t<mode> get_memory_arg(
        const sycl_buffer_memory_storage_t *storage, stream_t *stream,
        ::sycl::handler &cgh) {
    void *handle = nullptr;
    storage->get_data_handle(&handle);
    if (!handle) {
        auto *sycl_stream = utils::downcast<sycl_stream_t *>(stream);
        return {sycl_stream->get_dummy_accessor<mode>(cgh)};
    }
    return {storage->buffer().get_access<mode>(cgh)};
}

} // namespace

struct map_buffer_tag;

sycl_buffer_memory_storage_t::sycl_buffer_memory_storage_t(engine_t *engine)
    : sycl_memory_storage_base_t(engine) {}

sycl_buffer_memory_storage_t::sycl_buffer_memory_storage_t(
        engine_t *engine, const memory_storage_t *parent_storage)
    : sycl_memory_storage_base_t(engine, parent_storage) {}

status_t sycl_buffer_memory_storage_t::map_data(
        void **mapped_ptr, stream_t *stream, size_t) const {
    if (!buffer_) {
        *mapped_ptr = nullptr;
        return status::success;
    }

    auto &map_manager = memory_map_manager_t<map_buffer_tag>::instance();

    auto acc = buffer_->get_host_access();
    auto *acc_ptr = new decltype(acc)(acc);
    *mapped_ptr = static_cast<void *>(acc_ptr->get_pointer());
    auto unmap_callback = [acc_ptr](stream_t *, void *) {
        delete acc_ptr;
        return status::success;
    };
    return map_manager.map(this, stream, *mapped_ptr, unmap_callback);
}

status_t sycl_buffer_memory_storage_t::unmap_data(
        void *mapped_ptr, stream_t *stream) const {
    if (!mapped_ptr) return status::success;

    auto &map_manager = memory_map_manager_t<map_buffer_tag>::instance();
    return map_manager.unmap(this, stream, mapped_ptr);
}

std::unique_ptr<memory_storage_t> sycl_buffer_memory_storage_t::get_sub_storage(
        size_t offset, size_t size) const {
    auto storage = utils::make_unique<sycl_buffer_memory_storage_t>(
            engine(), parent_storage());
    if (!storage) return nullptr;

    status_t status
            = storage->init(memory_flags_t::use_runtime_ptr, 0, nullptr);
    if (status != status::success) return nullptr;

    if (engine()->kind() == engine_kind::cpu) {
        storage->buffer_ = buffer_;
    } else {
        buffer_u8_t *sub_buffer = buffer_
                ? new buffer_u8_t(parent_buffer(), base_offset_ + offset, size)
                : nullptr;
        storage->buffer_.reset(sub_buffer);
        storage->base_offset_ = base_offset_ + offset;
    }

    return storage;
}

std::unique_ptr<memory_storage_t> sycl_buffer_memory_storage_t::clone() const {
    auto storage = utils::make_unique<sycl_buffer_memory_storage_t>(engine());
    if (!storage) return nullptr;

    status_t status
            = storage->init(memory_flags_t::use_runtime_ptr, 0, nullptr);
    if (status != status::success) return nullptr;

    storage->buffer_ = buffer_;
    storage->base_offset_ = base_offset_;
    return storage;
}

status_t sycl_buffer_memory_storage_t::init_allocate(size_t size) {
    const auto &device
            = utils::downcast<sycl_engine_base_t *>(engine())->device();
    if (size > device.get_info<::sycl::info::device::max_mem_alloc_size>()) {
        return status::out_of_memory;
    }

    buffer_ = std::make_shared<buffer_u8_t>(::sycl::range<1>(size));
    if (!buffer_) return status::out_of_memory;
    return status::success;
}

buffer_u8_t &sycl_buffer_memory_storage_t::parent_buffer() const {
    return utils::downcast<const sycl_buffer_memory_storage_t *>(
            parent_storage())
            ->buffer();
}

gpu::sycl::sycl_in_memory_arg_t sycl_buffer_memory_storage_t::get_in_memory_arg(
        stream_t *stream, ::sycl::handler &cgh) const {
    return get_memory_arg<::sycl::access::mode::read>(this, stream, cgh);
}

gpu::sycl::sycl_out_memory_arg_t
sycl_buffer_memory_storage_t::get_out_memory_arg(
        stream_t *stream, ::sycl::handler &cgh) const {
    return get_memory_arg<::sycl::access::mode::write>(this, stream, cgh);
}

gpu::sycl::sycl_inout_memory_arg_t
sycl_buffer_memory_storage_t::get_inout_memory_arg(
        stream_t *stream, ::sycl::handler &cgh) const {
    return get_memory_arg<::sycl::access::mode::read_write>(this, stream, cgh);
}

} // namespace sycl
} // namespace impl
} // namespace dnnl
