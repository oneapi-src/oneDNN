/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#include "xpu/sycl/buffer_memory_storage.hpp"

#include "common/memory.hpp"
#include "common/memory_map_manager.hpp"
#include "common/stream.hpp"
#include "common/utils.hpp"

#include "xpu/sycl/engine_impl.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace sycl {

namespace {
template <::sycl::access_mode mode>
memory_arg_t<mode> get_memory_arg(const buffer_memory_storage_t *storage,
        stream_t *stream, ::sycl::handler &cgh) {
    void *handle = nullptr;
    storage->get_data_handle(&handle);
    if (!handle) {
        auto *sycl_stream_impl
                = utils::downcast<xpu::sycl::stream_impl_t *>(stream->impl());
        return {sycl_stream_impl->get_dummy_accessor<mode>(cgh)};
    }
    ::sycl::id<1> offset(storage->offset());
    ::sycl::range<1> range(storage->buffer().size() - storage->offset());

    return {storage->buffer().get_access<mode>(cgh, range, offset)};
}

} // namespace

struct map_buffer_tag;

buffer_memory_storage_t::buffer_memory_storage_t(engine_t *engine)
    : memory_storage_base_t(engine) {}

buffer_memory_storage_t::buffer_memory_storage_t(
        engine_t *engine, const memory_storage_t *root_storage)
    : memory_storage_base_t(engine, root_storage) {}

status_t buffer_memory_storage_t::map_data(
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

status_t buffer_memory_storage_t::unmap_data(
        void *mapped_ptr, stream_t *stream) const {
    if (!mapped_ptr) return status::success;

    auto &map_manager = memory_map_manager_t<map_buffer_tag>::instance();
    return map_manager.unmap(this, stream, mapped_ptr);
}

std::unique_ptr<memory_storage_t> buffer_memory_storage_t::get_sub_storage(
        size_t offset, size_t size) const {
    auto storage = utils::make_unique<buffer_memory_storage_t>(
            engine(), root_storage());
    if (!storage) return nullptr;

    status_t status
            = storage->init(memory_flags_t::use_runtime_ptr, 0, nullptr);
    if (status != status::success) return nullptr;

    if (engine()->kind() == engine_kind::cpu) {
        storage->buffer_ = buffer_;
        storage->base_offset_ = base_offset_ + offset;
    } else {
        const auto *sycl_engine_impl
                = utils::downcast<const xpu::sycl::engine_impl_t *>(
                        engine()->impl());
        MAYBE_UNUSED(sycl_engine_impl);
        // TODO: Generalize gpu_assert to make it available for use in the xpu
        // space.
        assert(IMPLICATION(
                xpu::sycl::is_intel_device(sycl_engine_impl->device()),
                offset % sycl_engine_impl->get_buffer_alignment() == 0));
        xpu::sycl::buffer_u8_t *sub_buffer = buffer_
                ? new xpu::sycl::buffer_u8_t(
                        parent_buffer(), base_offset_ + offset, size)
                : nullptr;
        storage->buffer_.reset(sub_buffer);
        storage->base_offset_ = base_offset_ + offset;
    }

    return storage;
}

std::unique_ptr<memory_storage_t> buffer_memory_storage_t::clone() const {
    auto storage = utils::make_unique<buffer_memory_storage_t>(engine());
    if (!storage) return nullptr;

    status_t status
            = storage->init(memory_flags_t::use_runtime_ptr, 0, nullptr);
    if (status != status::success) return nullptr;

    storage->buffer_ = buffer_;
    storage->base_offset_ = base_offset_;
    storage->set_offset(offset());

    return storage;
}

std::unique_ptr<memory_storage_t> buffer_memory_storage_t::clone_ptr_off(
        size_t offset) const {
    auto storage = clone();
    storage->set_offset(offset + this->offset());

    return storage;
}

status_t buffer_memory_storage_t::init_allocate(size_t size) {
    const auto &device = utils::downcast<const xpu::sycl::engine_impl_t *>(
            engine()->impl())
                                 ->device();
    if (size > device.get_info<::sycl::info::device::max_mem_alloc_size>()) {
        return status::out_of_memory;
    }

    buffer_ = std::make_shared<xpu::sycl::buffer_u8_t>(::sycl::range<1>(size));
    if (!buffer_) return status::out_of_memory;
    return status::success;
}

xpu::sycl::buffer_u8_t &buffer_memory_storage_t::parent_buffer() const {
    return utils::downcast<const buffer_memory_storage_t *>(root_storage())
            ->buffer();
}

in_memory_arg_t buffer_memory_storage_t::get_in_memory_arg(
        stream_t *stream, ::sycl::handler &cgh) const {
    return get_memory_arg<::sycl::access::mode::read>(this, stream, cgh);
}

out_memory_arg_t buffer_memory_storage_t::get_out_memory_arg(
        stream_t *stream, ::sycl::handler &cgh) const {
    return get_memory_arg<::sycl::access::mode::write>(this, stream, cgh);
}

inout_memory_arg_t buffer_memory_storage_t::get_inout_memory_arg(
        stream_t *stream, ::sycl::handler &cgh) const {
    return get_memory_arg<::sycl::access::mode::read_write>(this, stream, cgh);
}
} // namespace sycl
} // namespace xpu
} // namespace impl
} // namespace dnnl
