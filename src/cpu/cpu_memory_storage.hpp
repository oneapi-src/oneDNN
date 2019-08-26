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

#include <functional>
#include <memory>

#include "common/c_types_map.hpp"
#include "common/memory.hpp"
#include "common/memory_storage.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

class cpu_memory_storage_t : public memory_storage_impl_t {
public:
    cpu_memory_storage_t(engine_t *engine, unsigned flags, size_t size,
            size_t alignment, void *handle)
        : memory_storage_impl_t(engine, size) {
        if (alignment == 0) alignment = 64;

        if (size == 0 || (!handle && (flags & memory_flags_t::alloc) == 0)) {
            return;
        }
        if (flags & memory_flags_t::alloc) {
            void *data_ptr = malloc(size, (int)alignment);
            data_ = decltype(data_)(data_ptr, [](void *ptr) { free(ptr); });
        } else if (flags & memory_flags_t::use_backend_ptr) {
            data_ = decltype(data_)(handle, [](void *) {});
        }
    }

    virtual status_t get_data_handle(void **handle) const override {
        *handle = data_.get();
        return status::success;
    }

    virtual status_t set_data_handle(void *handle) override {
        data_ = decltype(data_)(handle, [](void *) {});
        return status::success;
    }

    virtual uintptr_t base_offset() const override {
        return reinterpret_cast<uintptr_t>(data_.get());
    }

private:
    std::unique_ptr<void, std::function<void(void *)>> data_;

    DNNL_DISALLOW_COPY_AND_ASSIGN(cpu_memory_storage_t);
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
