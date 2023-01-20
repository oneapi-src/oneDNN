/*******************************************************************************
* Copyright 2016-2023 Intel Corporation
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

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "oneapi/dnnl/dnnl.h"
#include "oneapi/dnnl/dnnl.hpp"

#ifdef DNNL_WITH_SYCL
#include "oneapi/dnnl/dnnl_sycl.h"
#endif

#include "c_types_map.hpp"
#include "engine.hpp"
#include "memory.hpp"
#include "memory_desc_wrapper.hpp"
#include "stream.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::status;
using namespace dnnl::impl::data_type;

namespace dnnl {
namespace impl {
memory_desc_t glob_zero_md = memory_desc_t();
}
} // namespace dnnl

namespace {
// Returns the size required for memory descriptor mapping.
// Caveats:
// 1. If memory descriptor with run-time parameters, the mapping cannot be done;
//    hence return DNNL_RUNTIME_SIZE_VAL
// 2. Otherwise, the size returned includes `offset0` and holes (for the case
//    of non-trivial strides). Strictly speaking, the mapping should happen only
//    for elements accessible with `md.off_l(0 .. md.nelems())`. However, for
//    the sake of simple implementation let's have such limitation hoping that
//    no one will do concurrent mapping for overlapping memory objects.
//
// XXX: remove limitation mentioned in 2nd bullet.
size_t memory_desc_map_size(const memory_desc_t *md, int index = 0) {
    auto mdw = memory_desc_wrapper(md);

    if (mdw.has_runtime_dims_or_strides()) return DNNL_RUNTIME_SIZE_VAL;
    if (mdw.offset0() == 0) return mdw.size(index);

    memory_desc_t md_no_offset0 = *md;
    md_no_offset0.offset0 = 0;
    return memory_desc_wrapper(md_no_offset0).size(index)
            + md->offset0 * mdw.data_type_size();
}
} // namespace

dnnl_memory::dnnl_memory(dnnl::impl::engine_t *engine,
        const dnnl::impl::memory_desc_t *md, const std::vector<unsigned> &flags,
        const std::vector<void *> &handles)
    : engine_(engine), md_(*md) {

    const size_t nhandles = handles.size();
    std::vector<std::unique_ptr<dnnl::impl::memory_storage_t>> mem_storages(
            nhandles);
    for (size_t i = 0; i < nhandles; i++) {
        const size_t size = memory_desc_wrapper(md_).size((int)i);
        memory_storage_t *memory_storage_ptr;
        status_t status = engine->create_memory_storage(
                &memory_storage_ptr, flags[i], size, handles[i]);
        if (status != success) return;
        mem_storages[i].reset(memory_storage_ptr);
    }

    memory_storages_ = std::move(mem_storages);
}

dnnl_memory::dnnl_memory(dnnl::impl::engine_t *engine,
        const dnnl::impl::memory_desc_t *md,
        std::unique_ptr<dnnl::impl::memory_storage_t> &&memory_storage)
    : engine_(engine), md_(*md) {
    this->reset_memory_storage(std::move(memory_storage));
}

status_t dnnl_memory::set_data_handle(void *handle, int index) const {
    using namespace dnnl::impl;
    void *old_handle;
    CHECK(memory_storage(index)->get_data_handle(&old_handle));
    if (handle != old_handle) {
        CHECK(memory_storage(index)->set_data_handle(handle));
    }
    return status::success;
}

status_t dnnl_memory::reset_memory_storage(
        std::unique_ptr<dnnl::impl::memory_storage_t> &&memory_storage) {
    if (memory_storage) {
        if (memory_storages_.empty())
            memory_storages_.emplace_back(std::move(memory_storage));
        else
            memory_storages_[0] = std::move(memory_storage);
    } else {
        memory_storage_t *memory_storage_ptr;
        status_t status = engine_->create_memory_storage(
                &memory_storage_ptr, use_runtime_ptr, 0, nullptr);
        if (status != status::success) return status;

        if (memory_storages_.empty())
            memory_storages_.emplace_back(memory_storage_ptr);
        else
            memory_storages_[0].reset(memory_storage_ptr);
    }

    return status::success;
}

status_t dnnl_memory_create(memory_t **memory, const memory_desc_t *md,
        engine_t *engine, void *handle) {
#ifdef DNNL_WITH_SYCL
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
    if (engine->kind() == engine_kind::gpu)
#endif
        return dnnl_sycl_interop_memory_create(
                memory, md, engine, dnnl_sycl_interop_usm, handle);
#endif
    if (any_null(memory, engine)) return invalid_arguments;

    memory_desc_t z_md = types::zero_md();
    if (md == nullptr) md = &z_md;

    const auto mdw = memory_desc_wrapper(md);
    if (mdw.format_any() || mdw.has_runtime_dims_or_strides())
        return invalid_arguments;

    unsigned flags = (handle == DNNL_MEMORY_ALLOCATE)
            ? memory_flags_t::alloc
            : memory_flags_t::use_runtime_ptr;
    void *handle_ptr = (handle == DNNL_MEMORY_ALLOCATE) ? nullptr : handle;
    auto _memory = new memory_t(engine, md, flags, handle_ptr);
    if (_memory == nullptr) return out_of_memory;
    if (_memory->memory_storage() == nullptr) {
        delete _memory;
        return out_of_memory;
    }
    *memory = _memory;
    return success;
}

status_t dnnl_memory_create_v2(memory_t **memory, const memory_desc_t *md,
        engine_t *engine, int nhandles, void **handles) {
    const bool args_ok = !any_null(memory, engine, handles) && nhandles > 0;
    if (!args_ok) return invalid_arguments;
#ifdef DNNL_WITH_SYCL
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
    if (engine->kind() == engine_kind::gpu)
#endif
        return dnnl_sycl_interop_memory_create(
                memory, md, engine, dnnl_sycl_interop_usm, handles[0]);
#endif
    memory_desc_t z_md = types::zero_md();
    if (md == nullptr) md = &z_md;

    const auto mdw = memory_desc_wrapper(md);
    if (mdw.format_any() || mdw.has_runtime_dims_or_strides())
        return invalid_arguments;

    std::vector<unsigned> flags_vec(nhandles);
    std::vector<void *> handles_vec(nhandles);
    for (size_t i = 0; i < handles_vec.size(); i++) {
        unsigned f = (handles[i] == DNNL_MEMORY_ALLOCATE)
                ? memory_flags_t::alloc
                : memory_flags_t::use_runtime_ptr;
        void *h = (handles[i] == DNNL_MEMORY_ALLOCATE) ? nullptr : handles[i];
        flags_vec[i] = f;
        handles_vec[i] = h;
    }

    auto _memory = new memory_t(engine, md, flags_vec, handles_vec);
    if (_memory == nullptr) return out_of_memory;
    for (size_t i = 0; i < handles_vec.size(); i++) {
        if (_memory->memory_storage((int)i) == nullptr) {
            delete _memory;
            return out_of_memory;
        }
    }
    *memory = _memory;
    return success;
}

status_t dnnl_memory_get_memory_desc(
        const memory_t *memory, const memory_desc_t **md) {
    if (any_null(memory, md)) return invalid_arguments;
    *md = memory->md();
    return success;
}

status_t dnnl_memory_get_engine(const memory_t *memory, engine_t **engine) {
    if (any_null(memory, engine)) return invalid_arguments;
    *engine = memory->engine();
    return success;
}

status_t dnnl_memory_get_data_handle(const memory_t *memory, void **handle) {
    if (any_null(handle)) return invalid_arguments;
    if (memory == nullptr) {
        *handle = nullptr;
        return success;
    }
    return memory->get_data_handle(handle);
}

status_t dnnl_memory_set_data_handle(memory_t *memory, void *handle) {
    if (any_null(memory)) return invalid_arguments;
    CHECK(memory->set_data_handle(handle));
    return status::success;
}

status_t dnnl_memory_get_data_handle_v2(
        const memory_t *memory, void **handle, int index) {
    if (any_null(handle)) return invalid_arguments;
    if (memory == nullptr) {
        *handle = nullptr;
        return success;
    }
    return memory->get_data_handle(handle, index);
}

status_t dnnl_memory_set_data_handle_v2(
        memory_t *memory, void *handle, int index) {
    if (any_null(memory)) return invalid_arguments;
    CHECK(memory->set_data_handle(handle, index));
    return status::success;
}

status_t dnnl_memory_map_data_v2(
        const memory_t *memory, void **mapped_ptr, int index) {
    const bool args_ok = !any_null(memory, mapped_ptr)
            && (index >= 0 && index < (int)memory->get_num_handles());
    if (!args_ok) return invalid_arguments;

    const memory_desc_t *md = memory->md();
    // See caveats in the comment to `memory_desc_map_size()` function.
    const size_t map_size = memory_desc_map_size(md, index);

    if (map_size == 0) {
        *mapped_ptr = nullptr;
        return success;
    } else if (map_size == DNNL_RUNTIME_SIZE_VAL) {
        return invalid_arguments;
    }

    return memory->memory_storage(index)->map_data(
            mapped_ptr, nullptr, map_size);
}

status_t dnnl_memory_unmap_data_v2(
        const memory_t *memory, void *mapped_ptr, int index) {
    const bool args_ok = !any_null(memory)
            && (index >= 0 && index < (int)memory->get_num_handles());
    if (!args_ok) return invalid_arguments;
    return memory->memory_storage(index)->unmap_data(mapped_ptr, nullptr);
}

status_t dnnl_memory_map_data(const memory_t *memory, void **mapped_ptr) {
    return dnnl_memory_map_data_v2(memory, mapped_ptr, 0);
}

status_t dnnl_memory_unmap_data(const memory_t *memory, void *mapped_ptr) {
    return dnnl_memory_unmap_data_v2(memory, mapped_ptr, 0);
}

status_t dnnl_memory_destroy(memory_t *memory) {
    delete memory;
    return success;
}

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
