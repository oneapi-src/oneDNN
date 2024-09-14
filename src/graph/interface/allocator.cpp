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

#include "oneapi/dnnl/dnnl_graph.h"
#include "oneapi/dnnl/dnnl_graph_sycl.h"

#include "common/engine.hpp"
#include "common/rw_mutex.hpp"

#include "graph/interface/allocator.hpp"
#include "graph/interface/c_types_map.hpp"

#include "graph/utils/utils.hpp"

#ifdef DNNL_WITH_SYCL
#include "oneapi/dnnl/dnnl_sycl.h"
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "oneapi/dnnl/dnnl_graph_ocl.h"
#include "oneapi/dnnl/dnnl_ocl.h"

#endif

using namespace dnnl::impl::graph;

status_t DNNL_API dnnl_graph_allocator_create(allocator_t **allocator,
        host_allocate_f host_malloc, host_deallocate_f host_free) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
    UNUSED(allocator);
    UNUSED(host_malloc);
    UNUSED(host_free);
    return status::invalid_arguments;
#else
    if (utils::any_null(host_malloc, host_free)) {
        *allocator = new dnnl_graph_allocator();
    } else {
        *allocator = new dnnl_graph_allocator(host_malloc, host_free);
    }
    return status::success;
#endif
}

status_t DNNL_API dnnl_graph_sycl_interop_allocator_create(
        allocator_t **allocator, sycl_allocate_f sycl_malloc,
        sycl_deallocate_f sycl_free) {
#ifdef DNNL_WITH_SYCL
    if (utils::any_null(sycl_malloc, sycl_free)) {
        *allocator = new dnnl_graph_allocator();
    } else {
        *allocator = new dnnl_graph_allocator(sycl_malloc, sycl_free);
    }
    return status::success;
#else
    UNUSED(allocator);
    UNUSED(sycl_malloc);
    UNUSED(sycl_free);
    return status::unimplemented;
#endif
}

status_t DNNL_API dnnl_graph_allocator_destroy(allocator_t *allocator) {
    if (allocator == nullptr) return status::invalid_arguments;
    delete allocator;
    return status::success;
}

status_t DNNL_API dnnl_graph_make_engine_with_allocator(engine_t **engine,
        engine_kind_t kind, size_t index, const allocator_t *alloc) {
    auto ret = dnnl_engine_create(engine, kind, index);
    if (ret != status::success) return ret;

    (*engine)->set_allocator(const_cast<allocator_t *>(alloc));
    return status::success;
}

status_t DNNL_API dnnl_graph_sycl_interop_make_engine_with_allocator(
        engine_t **engine, const void *device, const void *context,
        const allocator_t *alloc) {
#ifdef DNNL_WITH_SYCL
    auto ret = dnnl_sycl_interop_engine_create(engine, device, context);
    if (ret != status::success) return ret;

    (*engine)->set_allocator(const_cast<allocator_t *>(alloc));
    return status::success;
#else
    UNUSED(engine);
    UNUSED(device);
    UNUSED(context);
    UNUSED(alloc);
    return status::unimplemented;
#endif
}

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
status_t DNNL_API dnnl_graph_ocl_interop_allocator_create(
        allocator_t **allocator, ocl_allocate_f ocl_malloc,
        ocl_deallocate_f ocl_free) {
    if (utils::any_null(ocl_malloc, ocl_free)) {
        *allocator = new dnnl_graph_allocator();
    } else {
        *allocator = new dnnl_graph_allocator(ocl_malloc, ocl_free);
    }
    return status::success;
}

status_t DNNL_API dnnl_graph_ocl_interop_make_engine_with_allocator(
        engine_t **engine, cl_device_id device, cl_context context,
        const allocator_t *alloc) {
    auto ret = dnnl_ocl_interop_engine_create(engine, device, context);
    if (ret != status::success) return ret;

    (*engine)->set_allocator(const_cast<allocator_t *>(alloc));
    return status::success;
}

status_t DNNL_API
dnnl_graph_ocl_interop_make_engine_from_cache_blob_with_allocator(
        engine_t **engine, cl_device_id device, cl_context context,
        const allocator_t *alloc, size_t size, const uint8_t *cache_blob) {
    auto ret = dnnl_ocl_interop_engine_create_from_cache_blob(
            engine, device, context, size, cache_blob);
    if (ret != status::success) return ret;

    (*engine)->set_allocator(const_cast<allocator_t *>(alloc));
    return status::success;
}
#endif
