/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#include <CL/cl.h>

#include "common/engine.hpp"
#include "common/stream.hpp"

#include "xpu/ocl/buffer_memory_storage.hpp"
#include "xpu/ocl/engine_impl.hpp"
#include "xpu/ocl/stream_impl.hpp"
#include "xpu/ocl/usm_utils.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace ocl {

status_t buffer_memory_storage_t::init_allocate(size_t size) {
    auto context
            = utils::downcast<const xpu::ocl::engine_impl_t *>(engine()->impl())
                      ->context();
    cl_int err;
    mem_object_ = clCreateBuffer_wrapper(
            context, CL_MEM_READ_WRITE, size, nullptr, &err);
    OCL_CHECK(err);
    return status::success;
}

namespace {
status_t get_map_queue(cl_command_queue &queue, impl::engine_t *engine,
        impl::stream_t *stream) {
    if (stream == nullptr) {
        status_t status = engine->get_service_stream(stream);
        if (status != status::success) { return status::runtime_error; }
    }
    queue = utils::downcast<xpu::ocl::stream_impl_t *>(stream->impl())->queue();
    return status::success;
}
} // namespace

status_t buffer_memory_storage_t::map_data(
        void **mapped_ptr, impl::stream_t *stream, size_t) const {
    if (!mem_object()) {
        *mapped_ptr = nullptr;
        return status::success;
    }

    cl_mem_flags mem_flags;
    OCL_CHECK(clGetMemObjectInfo(mem_object(), CL_MEM_FLAGS, sizeof(mem_flags),
            &mem_flags, nullptr));

    size_t mem_bytes;
    OCL_CHECK(clGetMemObjectInfo(
            mem_object(), CL_MEM_SIZE, sizeof(mem_bytes), &mem_bytes, nullptr));

    cl_map_flags map_flags = 0;
    if (mem_flags & CL_MEM_READ_WRITE) {
        map_flags |= CL_MAP_READ;
        map_flags |= CL_MAP_WRITE;
    } else if (mem_flags & CL_MEM_READ_ONLY) {
        map_flags |= CL_MAP_READ;
    } else if (mem_flags & CL_MEM_WRITE_ONLY) {
        map_flags |= CL_MAP_WRITE;
    }

    cl_command_queue queue;
    CHECK(get_map_queue(queue, engine(), stream));

    // Use blocking operation to simplify the implementation and API
    cl_int err;
    *mapped_ptr = clEnqueueMapBuffer(queue, mem_object(), CL_TRUE, map_flags, 0,
            mem_bytes, 0, nullptr, nullptr, &err);
    return xpu::ocl::convert_to_dnnl(err);
}

status_t buffer_memory_storage_t::unmap_data(
        void *mapped_ptr, impl::stream_t *stream) const {
    if (!mapped_ptr) return status::success;
    cl_command_queue queue;
    CHECK(get_map_queue(queue, engine(), stream));
    OCL_CHECK(clEnqueueUnmapMemObject(queue, mem_object_,
            const_cast<void *>(mapped_ptr), 0, nullptr, nullptr));
    OCL_CHECK(clFinish(queue));
    return status::success;
}

std::unique_ptr<memory_storage_t> buffer_memory_storage_t::get_sub_storage(
        size_t offset, size_t size) const {
    // Fast return on size = 0.
    // It also seems clCreateSubBuffer() does not work properly for such case.
    // Assumption: returned sub-storage won't be used for extracting cl_mem.
    if (size == 0) return nullptr;

    cl_mem_flags mem_flags;
    cl_int err;
    err = clGetMemObjectInfo(
            mem_object(), CL_MEM_FLAGS, sizeof(mem_flags), &mem_flags, nullptr);

    // TODO: Generalize gpu_assert to make it available for use in the xpu
    // space.
    assert(err == CL_SUCCESS);
    if (err != CL_SUCCESS) return nullptr;

    const auto *ocl_engine_impl
            = utils::downcast<const xpu::ocl::engine_impl_t *>(
                    engine()->impl());
    MAYBE_UNUSED(ocl_engine_impl);

    assert(size != 0);
    assert(offset % ocl_engine_impl->get_buffer_alignment() == 0);

    cl_buffer_region buffer_region = {base_offset_ + offset, size};
    xpu::ocl::wrapper_t<cl_mem> sub_buffer
            = clCreateSubBuffer(parent_mem_object(), mem_flags,
                    CL_BUFFER_CREATE_TYPE_REGION, &buffer_region, &err);
    assert(err == CL_SUCCESS);
    if (err != CL_SUCCESS) return nullptr;

    auto sub_storage
            = new buffer_memory_storage_t(this->engine(), root_storage());
    if (sub_storage) {
        sub_storage->init(memory_flags_t::use_runtime_ptr, size, sub_buffer);
        sub_storage->base_offset_ = base_offset_ + offset;
    }
    return std::unique_ptr<memory_storage_t>(sub_storage);
}

std::unique_ptr<memory_storage_t> buffer_memory_storage_t::clone() const {
    auto storage = new buffer_memory_storage_t(engine());
    if (storage) storage->init(memory_flags_t::use_runtime_ptr, 0, mem_object_);
    return std::unique_ptr<memory_storage_t>(storage);
}

cl_mem buffer_memory_storage_t::parent_mem_object() const {
    return utils::downcast<const buffer_memory_storage_t *>(root_storage())
            ->mem_object();
}

} // namespace ocl
} // namespace xpu
} // namespace impl
} // namespace dnnl
