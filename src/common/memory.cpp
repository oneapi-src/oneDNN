/*******************************************************************************
* Copyright 2016-2019 Intel Corporation
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

#include "dnnl.h"

#include "c_types_map.hpp"
#include "engine.hpp"
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
bool memory_desc_sanity_check(int ndims, const dims_t dims,
        data_type_t data_type, format_kind_t format_kind) {
    if (ndims == 0) return true;

    bool ok = dims != nullptr && 0 < ndims && ndims <= DNNL_MAX_NDIMS
            && one_of(data_type, f16, bf16, f32, s32, s8, u8);
    if (!ok) return false;

    bool has_runtime_dims = false;
    for (int d = 0; d < ndims; ++d) {
        if (dims[d] != DNNL_RUNTIME_DIM_VAL && dims[d] < 0) return false;
        if (dims[d] == DNNL_RUNTIME_DIM_VAL) has_runtime_dims = true;
    }

    if (has_runtime_dims) {
        // format `any` is currently not supported for run-time dims
        if (format_kind == format_kind::any) return false;
    }

    return true;
}

bool memory_desc_sanity_check(const memory_desc_t *md) {
    if (md == nullptr) return false;
    return memory_desc_sanity_check(
            md->ndims, md->dims, md->data_type, format_kind::undef);
}
} // namespace

dnnl_memory::dnnl_memory(dnnl::impl::engine_t *engine,
        const dnnl::impl::memory_desc_t *md, unsigned flags, void *handle)
    : engine_(engine), md_(*md) {
    const size_t size = memory_desc_wrapper(md_).size();

    memory_storage_t *memory_storage_ptr;
    status_t status = engine->create_memory_storage(
            &memory_storage_ptr, flags, size, handle);
    assert(status == status::success);
    if (status != status::success) return;

    memory_storage_.reset(memory_storage_ptr);
    if (!(flags & omit_zero_pad)) zero_pad();
}

status_t dnnl_memory_desc_init_by_tag(memory_desc_t *memory_desc, int ndims,
        const dims_t dims, data_type_t data_type, format_tag_t tag) {
    if (any_null(memory_desc)) return invalid_arguments;
    if (ndims == 0 || tag == format_tag::undef) {
        *memory_desc = types::zero_md();
        return success;
    }

    format_kind_t format_kind = types::format_tag_to_kind(tag);

    /* memory_desc != 0 */
    bool args_ok = !any_null(memory_desc)
            && memory_desc_sanity_check(ndims, dims, data_type, format_kind);
    if (!args_ok) return invalid_arguments;

    auto md = memory_desc_t();
    md.ndims = ndims;
    array_copy(md.dims, dims, ndims);
    md.data_type = data_type;
    array_copy(md.padded_dims, dims, ndims);
    md.format_kind = format_kind;

    status_t status = success;
    if (tag == format_tag::undef) {
        status = invalid_arguments;
    } else if (tag == format_tag::any) {
        // nop
    } else if (format_kind == format_kind::blocked) {
        status = memory_desc_wrapper::compute_blocking(md, tag);
    } else {
        assert(!"unreachable");
        status = invalid_arguments;
    }

    if (status == success) *memory_desc = md;

    return status;
}

status_t dnnl_memory_desc_init_by_strides(memory_desc_t *memory_desc, int ndims,
        const dims_t dims, data_type_t data_type, const dims_t strides) {
    if (any_null(memory_desc)) return invalid_arguments;
    if (ndims == 0) {
        *memory_desc = types::zero_md();
        return success;
    }

    /* memory_desc != 0 */
    bool args_ok = !any_null(memory_desc)
            && memory_desc_sanity_check(
                    ndims, dims, data_type, format_kind::undef);
    if (!args_ok) return invalid_arguments;

    auto md = memory_desc_t();
    md.ndims = ndims;
    array_copy(md.dims, dims, ndims);
    md.data_type = data_type;
    array_copy(md.padded_dims, dims, ndims);
    md.format_kind = format_kind::blocked;

    dims_t default_strides = {0};
    if (strides == nullptr) {
        bool has_runtime_strides = false;
        default_strides[md.ndims - 1] = 1;
        for (int d = md.ndims - 2; d >= 0; --d) {
            if (md.padded_dims[d] == DNNL_RUNTIME_DIM_VAL)
                has_runtime_strides = true;
            default_strides[d] = has_runtime_strides
                    ? DNNL_RUNTIME_DIM_VAL
                    : default_strides[d + 1] * md.padded_dims[d + 1];
        }
        strides = default_strides;
    } else {
        /* TODO: add sanity check for the provided strides */
    }

    array_copy(md.format_desc.blocking.strides, strides, md.ndims);

    *memory_desc = md;

    return status::success;
}

status_t dnnl_memory_desc_init_submemory(memory_desc_t *md,
        const memory_desc_t *parent_md, const dims_t dims,
        const dims_t offsets) {
    if (any_null(md, parent_md) || !memory_desc_sanity_check(parent_md))
        return invalid_arguments;

    const memory_desc_wrapper src_d(parent_md);
    if (src_d.has_runtime_dims_or_strides()) return unimplemented;

    for (int d = 0; d < src_d.ndims(); ++d) {
        if (utils::one_of(DNNL_RUNTIME_DIM_VAL, dims[d], offsets[d]))
            return unimplemented;

        if (dims[d] < 0 || offsets[d] < 0
                || (offsets[d] + dims[d] > src_d.dims()[d]))
            return invalid_arguments;
    }

    if (src_d.format_kind() != format_kind::blocked) return unimplemented;

    dims_t blocks;
    src_d.compute_blocks(blocks);

    memory_desc_t dst_d = *parent_md;
    auto &dst_d_blk = dst_d.format_desc.blocking;

    /* TODO: put this into memory_desc_wrapper */
    for (int d = 0; d < src_d.ndims(); ++d) {
        /* very limited functionality for now */
        const bool ok = true && offsets[d] % blocks[d] == 0 /* [r1] */
                && src_d.padded_offsets()[d] == 0
                && (false || dims[d] % blocks[d] == 0 || dims[d] < blocks[d]);
        if (!ok) return unimplemented;

        const bool is_right_border = offsets[d] + dims[d] == src_d.dims()[d];

        dst_d.dims[d] = dims[d];
        dst_d.padded_dims[d] = is_right_border
                ? src_d.padded_dims()[d] - offsets[d]
                : dst_d.dims[d];
        dst_d.padded_offsets[d] = src_d.padded_offsets()[d];
        dst_d.offset0 += /* [r1] */
                offsets[d] / blocks[d] * dst_d_blk.strides[d];
    }

    *md = dst_d;

    return success;
}

int dnnl_memory_desc_equal(const memory_desc_t *lhs, const memory_desc_t *rhs) {
    if (lhs == rhs) return 1;
    if (any_null(lhs, rhs)) return 0;
    return memory_desc_wrapper(*lhs) == memory_desc_wrapper(*rhs);
}

status_t dnnl_memory_desc_reshape(memory_desc_t *out_md,
        const memory_desc_t *in_md, int ndims, const dims_t dims) {
    if (any_null(out_md, in_md) || !memory_desc_sanity_check(in_md)
            || !memory_desc_sanity_check(
                    ndims, dims, in_md->data_type, in_md->format_kind)
            || !(in_md->format_kind == format_kind::blocked)
            || types::is_zero_md(in_md))
        return invalid_arguments;

    // TODO: right now only appending is supported.
    if (ndims < in_md->ndims) return invalid_arguments;

    // TODO: right now the functionality is limited to append ones to the end.
    for (int d = 0; d < in_md->ndims; ++d)
        if (dims[d] != in_md->dims[d]) return invalid_arguments;
    for (int d = in_md->ndims; d < ndims; ++d)
        if (dims[d] != 1) return invalid_arguments;

    auto md = memory_desc_t();

    // copy values from in_md to md
    md.ndims = ndims;
    array_copy(md.dims, in_md->dims, ndims);
    md.data_type = in_md->data_type;
    array_copy(md.padded_dims, in_md->padded_dims, ndims);
    md.format_kind = in_md->format_kind;
    array_copy(md.padded_offsets, in_md->padded_offsets, ndims);
    md.offset0 = in_md->offset0;

    // copy blocking_desc values from in_md to md
    const auto &bds = in_md->format_desc.blocking;
    auto &bd = md.format_desc.blocking;
    array_copy(bd.strides, bds.strides, ndims);
    bd.inner_nblks = bds.inner_nblks;
    array_copy(bd.inner_blks, bds.inner_blks, ndims);
    array_copy(bd.inner_idxs, bds.inner_idxs, ndims);

    // assign new values
    for (int d = in_md->ndims; d < ndims; ++d) {
        md.dims[d] = 1;
        md.padded_dims[d] = 1;
        bd.strides[d] = bd.strides[d - 1];
    }

    *out_md = md;
    return success;
}

size_t dnnl_memory_desc_get_size(const memory_desc_t *md) {
    if (md == nullptr) return 0;
    return memory_desc_wrapper(*md).size();
}

status_t dnnl_memory_create(memory_t **memory, const memory_desc_t *md,
        engine_t *engine, void *handle) {
    if (any_null(memory, engine)) return invalid_arguments;

    memory_desc_t z_md = types::zero_md();
    if (md == nullptr) md = &z_md;

    const auto mdw = memory_desc_wrapper(md);
    if (mdw.format_any() || mdw.has_runtime_dims_or_strides())
        return invalid_arguments;

    unsigned flags = (handle == DNNL_MEMORY_ALLOCATE)
            ? memory_flags_t::alloc
            : memory_flags_t::use_runtime_ptr;
    auto _memory = new memory_t(engine, md, flags, handle);
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
    return memory->set_data_handle(handle);
}

status_t dnnl_memory_map_data(const memory_t *memory, void **mapped_ptr) {
    bool args_ok = !any_null(memory, mapped_ptr);
    if (!args_ok) return invalid_arguments;

    return memory->memory_storage()->map_data(mapped_ptr);
}

status_t dnnl_memory_unmap_data(const memory_t *memory, void *mapped_ptr) {
    bool args_ok = !any_null(memory);
    if (!args_ok) return invalid_arguments;

    return memory->memory_storage()->unmap_data(mapped_ptr);
}

status_t dnnl_memory_destroy(memory_t *memory) {
    delete memory;
    return success;
}

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
