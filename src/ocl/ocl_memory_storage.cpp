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

#include <CL/cl.h>

#include "ocl/ocl_engine.hpp"
#include "ocl/ocl_stream.hpp"

#include "ocl/ocl_memory_storage.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

ocl_memory_storage_t::ocl_memory_storage_t(
        engine_t *engine, unsigned flags, size_t size, void *handle)
    : memory_storage_t(engine) {
    // Do not allocate memory if one of these is true:
    // 1) size is 0
    // 2) handle is nullptr and 'alloc' flag is not set
    if ((size == 0) || (!handle && (flags & memory_flags_t::alloc) == 0)) {
        mem_object_ = nullptr;
        return;
    }
    auto *ocl_engine = utils::downcast<ocl_engine_t *>(engine);
    cl_int err;
    if (flags & memory_flags_t::alloc) {
        mem_object_ = clCreateBuffer(
                ocl_engine->context(), CL_MEM_READ_WRITE, size, nullptr, &err);
        OCL_CHECK_V(err);
    } else if (flags & memory_flags_t::use_backend_ptr) {
        mem_object_ = static_cast<cl_mem>(handle);
        OCL_CHECK_V(clRetainMemObject(mem_object_));
    }
}

status_t ocl_memory_storage_t::map_data(void **mapped_ptr) const {
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

    auto *ocl_engine = utils::downcast<ocl_engine_t *>(engine());
    auto *service_stream
            = utils::downcast<ocl_stream_t *>(ocl_engine->service_stream());

    // Use blocking operation to simplify the implementation and API
    cl_int err;
    *mapped_ptr = clEnqueueMapBuffer(service_stream->queue(), mem_object(),
            CL_TRUE, map_flags, 0, mem_bytes, 0, nullptr, nullptr, &err);
    return ocl_utils::convert_to_mkldnn(err);
}

status_t ocl_memory_storage_t::unmap_data(void *mapped_ptr) const {
    if (!mapped_ptr)
        return status::success;

    auto *ocl_engine = utils::downcast<ocl_engine_t *>(engine());
    auto *service_stream
            = utils::downcast<ocl_stream_t *>(ocl_engine->service_stream());
    auto service_queue = service_stream->queue();

    OCL_CHECK(clEnqueueUnmapMemObject(service_queue, mem_object_,
            const_cast<void *>(mapped_ptr), 0, nullptr, nullptr));
    return status::success;
}

} // namespace ocl
} // namespace impl
} // namespace mkldnn
