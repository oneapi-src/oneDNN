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

#include "sycl/sycl_engine.hpp"
#include "xpu/sycl/c_types_map.hpp"
#include "xpu/sycl/memory_storage.hpp"

using namespace dnnl::impl::sycl;
using namespace dnnl::impl::xpu::sycl;

using dnnl::impl::engine_t;
using dnnl::impl::memory_desc_t;
using dnnl::impl::memory_t;
using dnnl::impl::status_t;
using ::sycl::context;
using ::sycl::get_pointer_type;

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
            auto *sycl_engine = utils::downcast<sycl_engine_base_t *>(engine);
            auto &sycl_ctx = sycl_engine->context();
            ::sycl::usm::alloc ptr_type = get_pointer_type(handle, sycl_ctx);
            if (ptr_type == ::sycl::usm::alloc::unknown
                    && !sycl_engine->mayiuse_system_memory_allocators())
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
