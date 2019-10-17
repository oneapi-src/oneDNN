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

#ifndef CPU_MEMORY_STORAGE_HPP
#define CPU_MEMORY_STORAGE_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "common/memory.hpp"
#include "common/memory_storage.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

class cpu_memory_storage_t : public memory_storage_t {
public:
    cpu_memory_storage_t(engine_t *engine)
        : memory_storage_t(engine), data_(nullptr, release) {}

    status_t init(unsigned flags, size_t size, void *handle) {
        // Do not allocate memory if one of these is true:
        // 1) size is 0
        // 2) handle is nullptr and 'alloc' flag is not set
        if (size == 0 || (!handle && !(flags & memory_flags_t::alloc))) {
            data_ = decltype(data_)(handle, release);
            return status::success;
        }
        if (flags & memory_flags_t::alloc) {
            void *data_ptr = malloc(size, 64);
            if (data_ptr == nullptr) return status::out_of_memory;
            data_ = decltype(data_)(data_ptr, destroy);
        } else if (flags & memory_flags_t::use_runtime_ptr) {
            data_ = decltype(data_)(handle, release);
        }
        return status::success;
    }

    virtual status_t get_data_handle(void **handle) const override {
        *handle = data_.get();
        return status::success;
    }

    virtual status_t set_data_handle(void *handle) override {
        data_ = decltype(data_)(handle, release);
        return status::success;
    }

    virtual std::unique_ptr<memory_storage_t> get_sub_storage(
            size_t offset, size_t size) const override {
        void *sub_ptr = reinterpret_cast<uint8_t *>(data_.get()) + offset;
        auto sub_storage = new cpu_memory_storage_t(this->engine());
        sub_storage->init(memory_flags_t::use_runtime_ptr, size, sub_ptr);
        return std::unique_ptr<memory_storage_t>(sub_storage);
    }

private:
    std::unique_ptr<void, void (*)(void *)> data_;

    DNNL_DISALLOW_COPY_AND_ASSIGN(cpu_memory_storage_t);

    static void release(void *ptr) {}
    static void destroy(void *ptr) { free(ptr); }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
