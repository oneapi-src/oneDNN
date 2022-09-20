/*******************************************************************************
* Copyright 2016-2022 Intel Corporation
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
size_t memory_desc_map_size(const memory_desc_t *md) {
    auto mdw = memory_desc_wrapper(md);

    if (mdw.has_runtime_dims_or_strides()) return DNNL_RUNTIME_SIZE_VAL;
    if (mdw.offset0() == 0) return mdw.size();

    memory_desc_t md_no_offset0 = *md;
    md_no_offset0.offset0 = 0;
    return memory_desc_wrapper(md_no_offset0).size()
            + md->offset0 * mdw.data_type_size();
}
} // namespace

dnnl_memory::dnnl_memory(dnnl::impl::engine_t *engine,
        const dnnl::impl::memory_desc_t *md, unsigned flags, void *handle)
    : engine_(engine), md_(*md) {
    const size_t size = memory_desc_wrapper(md_).size();

    memory_storage_t *memory_storage_ptr;
    status_t status = engine->create_memory_storage(
            &memory_storage_ptr, flags, size, handle);
    if (status != success) return;

    memory_storage_.reset(memory_storage_ptr);
}

dnnl_memory::dnnl_memory(dnnl::impl::engine_t *engine,
        const dnnl::impl::memory_desc_t *md,
        std::unique_ptr<dnnl::impl::memory_storage_t> &&memory_storage)
    : engine_(engine), md_(*md) {
    this->reset_memory_storage(std::move(memory_storage));
}

status_t dnnl_memory::set_data_handle(void *handle) {
    using namespace dnnl::impl;

    void *old_handle;
    CHECK(memory_storage()->get_data_handle(&old_handle));

    if (handle != old_handle) {
        CHECK(memory_storage_->set_data_handle(handle));
    }
    return status::success;
}

status_t dnnl_memory::reset_memory_storage(
        std::unique_ptr<dnnl::impl::memory_storage_t> &&memory_storage) {
    if (memory_storage) {
        memory_storage_ = std::move(memory_storage);
    } else {
        memory_storage_t *memory_storage_ptr;
        status_t status = engine_->create_memory_storage(
                &memory_storage_ptr, use_runtime_ptr, 0, nullptr);
        if (status != status::success) return status;

        memory_storage_.reset(memory_storage_ptr);
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

status_t dnnl_memory_map_data(const memory_t *memory, void **mapped_ptr) {
    bool args_ok = !any_null(memory, mapped_ptr);
    if (!args_ok) return invalid_arguments;

    const memory_desc_t *md = memory->md();
    // See caveats in the comment to `memory_desc_map_size()` function.
    const size_t map_size = memory_desc_map_size(md);

    if (map_size == 0) {
        *mapped_ptr = nullptr;
        return success;
    } else if (map_size == DNNL_RUNTIME_SIZE_VAL) {
        return invalid_arguments;
    }

    return memory->memory_storage()->map_data(mapped_ptr, nullptr, map_size);
}

status_t dnnl_memory_unmap_data(const memory_t *memory, void *mapped_ptr) {
    bool args_ok = !any_null(memory);
    if (!args_ok) return invalid_arguments;

    return memory->memory_storage()->unmap_data(mapped_ptr, nullptr);
}

status_t dnnl_memory_destroy(memory_t *memory) {
    delete memory;
    return success;
}

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
