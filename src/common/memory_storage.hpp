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

#ifndef MEMORY_STORAGE_HPP
#define MEMORY_STORAGE_HPP

#include "common/c_types_map.hpp"
#include "common/utils.hpp"

#include <assert.h>
#include <memory>

namespace mkldnn {
namespace impl {

// Memory storage is an abstraction providing interfaces to:
// - set/get the underlying data handle (in form of void pointer)
// - map/unmap the data to the host
//
// Memory storage is engine-specific and has different implementations for
// different engines.
struct memory_storage_impl_t : public c_compatible {
    memory_storage_impl_t(engine_t *engine, size_t size)
        : engine_(engine), size_(size) {}
    virtual ~memory_storage_impl_t() = default;

    engine_t *engine() const { return engine_; }

    size_t size() const { return size_; }

    virtual status_t get_data_handle(void **handle) const = 0;
    virtual status_t set_data_handle(void *handle) = 0;

    virtual status_t map_data(void **mapped_ptr) const {
        return get_data_handle(mapped_ptr);
    }

    virtual status_t unmap_data(void *mapped_ptr) const {
        UNUSED(mapped_ptr);
        return status::success;
    }

    // Offset in bytes from the "perfectly" aligned address
    virtual uintptr_t base_offset() const = 0;

private:
    engine_t *engine_;
    size_t size_;

    MKLDNN_DISALLOW_COPY_AND_ASSIGN(memory_storage_impl_t);
};

struct memory_storage_t : public c_compatible {
public:
    static memory_storage_t &empty_storage() {
        static memory_storage_t instance;
        return instance;
    }

    memory_storage_t(memory_storage_impl_t *impl = nullptr, size_t offset = 0)
        : impl_(impl), offset_(offset) {}
    memory_storage_t(const memory_storage_t &other, size_t offset)
        : impl_(other.impl_), offset_(other.offset() + offset) {
        assert(offset <= other.size());
    }

    engine_t *engine() const { return impl_ ? impl_->engine() : nullptr; }

    memory_storage_impl_t *impl() const {
        return impl_ ? impl_.get() : nullptr;
    }

    size_t offset() const { return offset_; }

    size_t size() const { return impl_ ? (impl_->size() - offset()) : 0; }

    void set_offset(size_t offset) {
        assert(impl_);
        offset_ = offset;
    }

    // Returns the associated data handle, offset is ignored
    void *data_handle() const {
        if (!impl_) return nullptr;

        void *handle;
        status_t status = impl_->get_data_handle(&handle);
        assert(status == status::success);
        MAYBE_UNUSED(status);
        return handle;
    }

    // Returns the associated data handle, offset is ignored
    status_t get_data_handle(void **handle) const {
        if (!impl_) return status::invalid_arguments;

        return impl_->get_data_handle(handle);
    }

    // Sets the associated data handle, offset is ignored
    status_t set_data_handle(void *handle) {
        if (!impl_) {
            assert(!"not expected");
            return status::invalid_arguments;
        }

        return impl_->set_data_handle(handle);
    }

    status_t map_data(void **mapped_ptr) const {
        if (!impl_) {
            *mapped_ptr = nullptr;
            return status::success;
        }

        void *ptr;
        CHECK(impl_->map_data(&ptr));

        auto *ptr_u8 = static_cast<uint8_t *>(ptr);
        *mapped_ptr = ptr_u8 + offset();
        return status::success;
    }

    status_t unmap_data(void *mapped_ptr) const {
        if (!impl_) return status::success;

        auto *ptr_u8 = static_cast<uint8_t *>(mapped_ptr);
        return impl_->unmap_data(ptr_u8 - offset());
    }

    // Returns true if the pointer associated with the storage is NULL
    bool is_null() const {
        if (!impl_) return true;

        return !data_handle();
    }

    explicit operator bool() const { return !is_null(); }

    uintptr_t base_offset() const { return impl_ ? impl_->base_offset() : 0; }

private:
    std::shared_ptr<memory_storage_impl_t> impl_;
    size_t offset_;
};

} // namespace impl
} // namespace mkldnn

#endif
