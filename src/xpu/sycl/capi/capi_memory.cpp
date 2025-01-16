/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#include "oneapi/dnnl/dnnl_sycl.h"

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/memory.hpp"
#include "common/utils.hpp"

#include "xpu/sycl/c_types_map.hpp"
#include "xpu/sycl/engine_factory.hpp"
#include "xpu/sycl/memory_storage.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::xpu::sycl;

using dnnl::impl::engine_t;
using dnnl::impl::memory_desc_t;
using dnnl::impl::memory_t;
using dnnl::impl::status_t;
using ::sycl::context;
using ::sycl::get_pointer_type;

status_t dnnl_sycl_interop_memory_create_v2(memory_t **memory,
        const memory_desc_t *md, engine_t *engine, memory_kind_t memory_kind,
        int nhandles, void **handles) {

    bool ok = !utils::any_null(memory, md, engine, handles) && nhandles > 0
            && engine->runtime_kind() == runtime_kind::sycl;
    if (!ok) return status::invalid_arguments;

    const auto mdw = memory_desc_wrapper(md);
    if (mdw.format_any() || mdw.has_runtime_dims_or_strides())
        return status::invalid_arguments;

    std::vector<unsigned> flags_vec(nhandles);
    std::vector<void *> handles_vec(nhandles);
    for (int i = 0; i < nhandles; i++) {
        unsigned f = (handles[i] == DNNL_MEMORY_ALLOCATE)
                ? memory_flags_t::alloc
                : memory_flags_t::use_runtime_ptr;
        void *h = (handles[i] == DNNL_MEMORY_ALLOCATE) ? nullptr : handles[i];
        flags_vec[i] = f;
        handles_vec[i] = h;
    }

    bool is_usm = memory_kind == memory_kind::usm;
    std::vector<std::unique_ptr<memory_storage_t>> mem_storages(nhandles);

    if (is_usm) {
        for (int i = 0; i < nhandles; i++) {
            if (handles[i] != DNNL_MEMORY_NONE
                    && handles[i] != DNNL_MEMORY_ALLOCATE) {
                const auto *sycl_engine_impl
                        = utils::downcast<const xpu::sycl::engine_impl_t *>(
                                engine->impl());
                auto &sycl_ctx = sycl_engine_impl->context();
                ::sycl::usm::alloc ptr_type
                        = get_pointer_type(handles[i], sycl_ctx);
                if (ptr_type == ::sycl::usm::alloc::unknown
                        && !engine->mayiuse_system_memory_allocators())
                    return status::invalid_arguments;
            }
            size_t sz = dnnl_memory_desc_get_size_v2(md, i);
            mem_storages[i].reset(new xpu::sycl::usm_memory_storage_t(engine));
            if (!mem_storages[i]) return status::out_of_memory;
            CHECK(mem_storages[i]->init(flags_vec[i], sz, handles_vec[i]));
        }
    } else {
        for (int i = 0; i < nhandles; i++) {
            size_t sz = dnnl_memory_desc_get_size_v2(md, i);
            mem_storages[i].reset(
                    new xpu::sycl::buffer_memory_storage_t(engine));
            if (!mem_storages[i]) return status::out_of_memory;
            CHECK(mem_storages[i]->init(flags_vec[i], sz, handles_vec[i]));
        }
    }

    return safe_ptr_assign(
            *memory, new memory_t(engine, md, std::move(mem_storages)));
}

status_t dnnl_sycl_interop_memory_create(memory_t **memory,
        const memory_desc_t *md, engine_t *engine, memory_kind_t memory_kind,
        void *handle) {
    using namespace dnnl::impl;

    bool ok = !utils::any_null(memory, md, engine)
            && engine->runtime_kind() == runtime_kind::sycl;
    if (!ok) return status::invalid_arguments;

    const auto mdw = memory_desc_wrapper(md);
    if (mdw.format_any() || mdw.has_runtime_dims_or_strides())
        return status::invalid_arguments;

    size_t size = mdw.size();
    unsigned flags = (handle == DNNL_MEMORY_ALLOCATE)
            ? memory_flags_t::alloc
            : memory_flags_t::use_runtime_ptr;
    void *handle_ptr = (handle == DNNL_MEMORY_ALLOCATE) ? nullptr : handle;

    bool is_usm = memory_kind == memory_kind::usm;

    std::unique_ptr<memory_storage_t> mem_storage;
    if (is_usm) {
        if (handle != DNNL_MEMORY_NONE && handle != DNNL_MEMORY_ALLOCATE) {
            const auto *sycl_engine_impl
                    = utils::downcast<const xpu::sycl::engine_impl_t *>(
                            engine->impl());
            auto &sycl_ctx = sycl_engine_impl->context();
            ::sycl::usm::alloc ptr_type = get_pointer_type(handle, sycl_ctx);
            if (ptr_type == ::sycl::usm::alloc::unknown
                    && !engine->mayiuse_system_memory_allocators())
                return status::invalid_arguments;
        }

        mem_storage.reset(new usm_memory_storage_t(engine));
    } else
        mem_storage.reset(new buffer_memory_storage_t(engine));
    if (!mem_storage) return status::out_of_memory;

    CHECK(mem_storage->init(flags, size, handle_ptr));

    return safe_ptr_assign(
            *memory, new memory_t(engine, md, std::move(mem_storage)));
}

status_t dnnl_sycl_interop_memory_set_buffer(memory_t *memory, void *buffer) {
    using namespace dnnl::impl;

    bool ok = !utils::any_null(memory, buffer)
            && memory->engine()->runtime_kind() == runtime_kind::sycl;
    if (!ok) return status::invalid_arguments;

    std::unique_ptr<memory_storage_t> mem_storage(
            new buffer_memory_storage_t(memory->engine()));
    if (!mem_storage) return status::out_of_memory;

    size_t size = memory_desc_wrapper(memory->md()).size();
    CHECK(mem_storage->init(memory_flags_t::use_runtime_ptr, size, buffer));
    CHECK(memory->reset_memory_storage(std::move(mem_storage)));

    return status::success;
}

status_t dnnl_sycl_interop_memory_get_memory_kind(
        const memory_t *memory, memory_kind_t *memory_kind) {
    using namespace dnnl::impl;

    bool ok = !utils::any_null(memory, memory_kind)
            && memory->engine()->runtime_kind() == runtime_kind::sycl;
    if (!ok) return status::invalid_arguments;

    *memory_kind = utils::downcast<const memory_storage_base_t *>(
            memory->memory_storage())
                           ->memory_kind();
    return status::success;
}
