/*******************************************************************************
* Copyright 2020-2025 Intel Corporation
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

#include "common/engine.hpp"

#include "graph/interface/allocator.hpp"
#include "graph/interface/tensor.hpp"

#include "graph/utils/utils.hpp"

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
static const size_t DNNL_CPU_MEMALIGNMENT = 64;
#endif

#ifdef DNNL_WITH_SYCL
#include "oneapi/dnnl/dnnl_sycl.hpp"
static const size_t DNNL_SYCL_MEMALIGNMENT = 64;
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "xpu/ocl/engine_factory.hpp"
static const size_t DNNL_OCL_MEMALIGNMENT = 0;
using namespace dnnl::impl::gpu::intel;
#endif

using namespace dnnl::impl::graph;

static void *tensor_malloc(
        size_t size, const engine_t *eng, allocator_t::mem_type_t type) {
    const auto *alc = static_cast<dnnl::impl::graph::allocator_t *>(
            eng->get_allocator());
#ifdef DNNL_WITH_SYCL
    void *dev_ptr {nullptr};
    dnnl_sycl_interop_engine_get_device(const_cast<engine_t *>(eng), &dev_ptr);
    auto *dev = static_cast<sycl::device *>(dev_ptr);
    void *ctx_ptr {nullptr};
    dnnl_sycl_interop_engine_get_context(const_cast<engine_t *>(eng), &ctx_ptr);
    auto *ctx = static_cast<sycl::context *>(ctx_ptr);
#endif
    if (eng->kind() == engine_kind::cpu) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
        return alc->allocate(size, *dev, *ctx, {type, DNNL_SYCL_MEMALIGNMENT});
#else
        return alc->allocate(size, {type, DNNL_CPU_MEMALIGNMENT});
#endif
    } else if (eng->kind() == engine_kind::gpu) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
        return alc->allocate(size, *dev, *ctx, {type, DNNL_SYCL_MEMALIGNMENT});
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        auto *ocl_engine = utils::downcast<const ocl::engine_t *>(eng);
        const cl_device_id &ocl_dev = ocl_engine->device();
        const cl_context &ocl_ctx = ocl_engine->context();
        return alc->allocate(
                size, ocl_dev, ocl_ctx, {type, DNNL_OCL_MEMALIGNMENT});
#else
        assertm(false, "Unsupported gpu runtime");
        return nullptr;
#endif
    } else {
        assertm(false, "Unsupported engine kind");
        return nullptr;
    }
}

static void tensor_free(void *p, const engine_t *eng) {
    const auto *alc = static_cast<dnnl::impl::graph::allocator_t *>(
            eng->get_allocator());
#ifdef DNNL_WITH_SYCL
    void *dev_ptr {nullptr};
    dnnl_sycl_interop_engine_get_device(const_cast<engine_t *>(eng), &dev_ptr);
    auto *dev = static_cast<sycl::device *>(dev_ptr);
    void *ctx_ptr {nullptr};
    dnnl_sycl_interop_engine_get_context(const_cast<engine_t *>(eng), &ctx_ptr);
    auto *ctx = static_cast<sycl::context *>(ctx_ptr);
#endif
    if (eng->kind() == engine_kind::cpu) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
        alc->deallocate(p, *dev, *ctx, {});
#else
        alc->deallocate(p);
#endif
    } else if (eng->kind() == engine_kind::gpu) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
        alc->deallocate(p, *dev, *ctx, {});
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        auto *ocl_engine = utils::downcast<const ocl::engine_t *>(eng);
        const cl_device_id &ocl_dev = ocl_engine->device();
        const cl_context &ocl_ctx = ocl_engine->context();
        return alc->deallocate(p, ocl_dev, ocl_ctx, {});
#else
        assertm(false, "Unsupported gpu runtime");
#endif
    } else {
        assertm(false, "Unsupported engine kind");
    }
}

dnnl_graph_tensor::dnnl_graph_tensor(
        const logical_tensor_t &lt, const engine_t *eng, void *handle)
    : lt_(lt), eng_(eng) {
    if (handle == DNNL_MEMORY_ALLOCATE) {
        size_t num_bytes = logical_tensor_wrapper_t(lt).size();

        void *data
                = tensor_malloc(num_bytes, eng, allocator_t::mem_type_t::temp);
        assertm(data, "Can't allocate memory for a tensor!");
        handle_.reset(data, [eng](void *p) { tensor_free(p, eng); });
    } else {
        handle_.reset(handle, dummy_destructor);
    }
}

status_t DNNL_API dnnl_graph_tensor_create(tensor_t **tensor,
        const logical_tensor_t *logical_tensor, engine_t *eng, void *handle) {
    if (utils::any_null(tensor, logical_tensor, eng))
        return status::invalid_arguments;

    *tensor = new tensor_t {*logical_tensor, eng, handle};
    if (*tensor == nullptr) return status::out_of_memory;
    if (handle == DNNL_MEMORY_ALLOCATE
            && (*tensor)->get_data_handle() == nullptr) {
        delete *tensor;
        *tensor = nullptr;
        return status::out_of_memory;
    }
    return status::success;
}

status_t DNNL_API dnnl_graph_tensor_destroy(tensor_t *tensor) {
    delete tensor;
    return status::success;
}

status_t DNNL_API dnnl_graph_tensor_get_data_handle(
        const tensor_t *tensor, void **handle) {
    if (utils::any_null(tensor, handle)) return status::invalid_arguments;

    *handle = tensor->get_data_handle();
    return status::success;
}

status_t DNNL_API dnnl_graph_tensor_set_data_handle(
        tensor_t *tensor, void *handle) {
    if (tensor == nullptr) return status::invalid_arguments;

    tensor->set_data_handle(handle);
    return status::success;
}

status_t DNNL_API dnnl_graph_tensor_get_engine(
        const tensor_t *tensor, engine_t **engine) {
    if (utils::any_null(tensor, engine)) return status::invalid_arguments;

    *engine = const_cast<engine_t *>(tensor->get_engine());

    return status::success;
}

dnnl_status_t DNNL_API dnnl_graph_tensor_get_logical_tensor(
        const tensor_t *tensor, logical_tensor_t *logical_tensor) {
    if (utils::any_null(tensor, logical_tensor))
        return status::invalid_arguments;

    *logical_tensor = tensor->get_logical_tensor();
    return status::success;
}
