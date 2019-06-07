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

#include "common/c_types_map.hpp"
#include "common/memory_storage.hpp"
#include "common/utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

class cpu_memory_storage_t : public memory_storage_t
{
public:
    cpu_memory_storage_t(
            engine_t *engine, unsigned flags, size_t size, void *handle)
        : memory_storage_t(engine) {
        if (size == 0 || (!handle && (flags & memory_flags_t::alloc) == 0)) {
            data_ = nullptr;
            is_owned_ = false;
            return;
        }
        if (flags & memory_flags_t::alloc) {
            data_ = malloc(size, 64);
            is_owned_ = true;
        } else if (flags & memory_flags_t::use_backend_ptr) {
            data_ = handle;
            is_owned_ = false;
        }
    }

    virtual ~cpu_memory_storage_t() override {
        if (is_owned_) {
            free(data_);
        }
    }

    virtual status_t get_data_handle(void **handle) const override {
        *handle = data_;
        return status::success;
    }

    virtual status_t set_data_handle(void *handle) override {
        if (is_owned_) {
            free(data_);
        }
        data_ = handle;
        is_owned_ = false;
        return status::success;
    }

private:
    void *data_ = nullptr;
    bool is_owned_ = false;

    MKLDNN_DISALLOW_COPY_AND_ASSIGN(cpu_memory_storage_t);
};

} // namespace cpu
} // namespace impl
} // namespace mkldnn

#endif
