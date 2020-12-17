/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "c_types_map.hpp"
#include "engine.hpp"
#include "utils.hpp"

#if DNNL_GRAPH_WITH_SYCL
#include <CL/sycl.hpp>
#endif

using namespace llga::impl;

status_t DNNL_GRAPH_API dnnl_graph_engine_create(
        engine_t **created_engine, engine_kind_t kind, int32_t device_id) {
    *created_engine = new engine_t {kind, device_id};
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_sycl_interop_engine_create(
        engine_t **created_engine, const void *dev, const void *ctx) {
#if DNNL_GRAPH_WITH_SYCL
    if (utils::any_null(created_engine, dev, ctx)) {
        return status::invalid_argument;
    }

    auto &sycl_dev = *static_cast<const cl::sycl::device *>(dev);
    auto &sycl_ctx = *static_cast<const cl::sycl::context *>(ctx);

    engine_kind_t kind;
    if (sycl_dev.is_gpu())
        kind = engine_kind::gpu;
    else if (sycl_dev.is_cpu() || sycl_dev.is_host())
        kind = engine_kind::cpu;
    else
        return status::invalid_argument;

    *created_engine = new engine_t {kind, sycl_dev, sycl_ctx};

    return status::success;
#else
    UNUSED(created_engine);
    UNUSED(dev);
    UNUSED(ctx);
    return status::unsupported;
#endif
}

status_t DNNL_GRAPH_API dnnl_graph_engine_destroy(engine_t *engine) {
    delete engine;
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_engine_set_allocator(
        engine_t *engine, allocator_t *allocator) {
    if (utils::any_null(engine, allocator)) { return status::invalid_argument; }

    engine->set_allocator(allocator);
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_engine_get_device_handle(
        const engine_t *engine, void **handle) {
    *handle = engine->get_device_handle();
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_engine_get_device_id(
        const engine_t *engine, int32_t *device_id) {
    *device_id = engine->device_id();
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_engine_get_kind(
        const engine_t *engine, engine_kind_t *kind) {
    *kind = engine->kind();
    return status::success;
}
