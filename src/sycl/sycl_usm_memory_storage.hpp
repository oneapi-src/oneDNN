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

#ifndef SYCL_USM_MEMORY_STORAGE_HPP
#define SYCL_USM_MEMORY_STORAGE_HPP

#include "dnnl_config.h"

#ifdef DNNL_SYCL_INTEL

#include "common/memory_storage.hpp"
#include "common/utils.hpp"
#include "sycl/sycl_engine_base.hpp"
#include "sycl/sycl_memory_storage_base.hpp"

#include <functional>
#include <memory>
#include <CL/sycl.hpp>

namespace dnnl {
namespace impl {
namespace sycl {

class sycl_usm_memory_storage_t : public sycl_memory_storage_base_t {
public:
    sycl_usm_memory_storage_t(engine_t *engine, unsigned flags, size_t size,
            size_t alignment, void *handle)
        : sycl_memory_storage_base_t(engine, size) {
        UNUSED(alignment);

        // Do not allocate memory if one of these is true:
        // 1) size is 0
        // 2) handle is nullptr and flags have use_runtime_ptr
        if ((size == 0) || (!handle && (flags & memory_flags_t::alloc) == 0))
            return;

        if (flags & memory_flags_t::alloc) {
            auto *sycl_engine = utils::downcast<sycl_engine_base_t *>(engine);
            auto &sycl_dev = sycl_engine->device();
            auto &sycl_ctx = sycl_engine->context();

            void *usm_ptr_alloc
                    = cl::sycl::malloc_shared(size, sycl_dev, sycl_ctx);
            usm_ptr_ = decltype(usm_ptr_)(usm_ptr_alloc,
                    [&](void *ptr) { cl::sycl::free(ptr, sycl_ctx); });
        } else if (flags & memory_flags_t::use_runtime_ptr) {
            usm_ptr_ = decltype(usm_ptr_)(handle, [](void *) {});
        }
    }

    void *usm_ptr() const { return usm_ptr_.get(); }

    memory_api_kind_t memory_api_kind() const override {
        return memory_api_kind_t::usm;
    }

    virtual status_t get_data_handle(void **handle) const override {
        *handle = usm_ptr_.get();
        return status::success;
    }

    virtual status_t set_data_handle(void *handle) override {
        usm_ptr_ = decltype(usm_ptr_)(handle, [](void *) {});
        return status::success;
    }

    virtual uintptr_t base_offset() const override {
        return reinterpret_cast<uintptr_t>(usm_ptr_.get());
    }

    virtual bool is_host_accessible() const override { return true; }

private:
    std::unique_ptr<void, std::function<void(void *)>> usm_ptr_;
};

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif

#endif // SYCL_USM_MEMORY_STORAGE_HPP
