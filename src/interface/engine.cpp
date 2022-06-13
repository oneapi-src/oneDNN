/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#include "interface/c_types_map.hpp"
#include "interface/engine.hpp"

#include "utils/utils.hpp"

#ifdef DNNL_GRAPH_WITH_SYCL
#include <CL/sycl.hpp>
#endif

using namespace dnnl::graph::impl;

status_t DNNL_GRAPH_API dnnl_graph_engine_create(
        engine_t **engine, engine_kind_t kind, size_t index) {
    if (kind == engine_kind::gpu) { return status::invalid_arguments; }
#ifdef DNNL_GRAPH_CPU_SYCL
    UNUSED(engine);
    UNUSED(kind);
    UNUSED(index);
    return status::invalid_arguments;
#else
    *engine = new engine_t {kind, index};
    return status::success;
#endif
}

status_t DNNL_GRAPH_API dnnl_graph_engine_create_with_allocator(
        engine_t **engine, engine_kind_t kind, size_t index,
        const allocator_t *alloc) {
#ifdef DNNL_GRAPH_CPU_SYCL
    UNUSED(engine);
    UNUSED(kind);
    UNUSED(index);
    UNUSED(alloc);
    return status::invalid_arguments;
#else
    *engine = new engine_t {kind, index, alloc};
    return status::success;
#endif
}

status_t DNNL_GRAPH_API dnnl_graph_sycl_interop_engine_create(
        engine_t **engine, const void *dev, const void *ctx) {
#ifdef DNNL_GRAPH_WITH_SYCL
    if (utils::any_null(engine, dev, ctx)) { return status::invalid_arguments; }

    auto &sycl_dev = *static_cast<const ::sycl::device *>(dev);
    auto &sycl_ctx = *static_cast<const ::sycl::context *>(ctx);

    engine_kind_t kind;
    if (sycl_dev.is_gpu()) {
#ifdef DNNL_GRAPH_GPU_SYCL
        kind = engine_kind::gpu;
#else
        return status::invalid_arguments;
#endif
    } else if (sycl_dev.is_cpu() || sycl_dev.is_host()) {
#ifdef DNNL_GRAPH_CPU_SYCL
        kind = engine_kind::cpu;
#else
        return status::invalid_arguments;
#endif
    } else {
        return status::invalid_arguments;
    }

    *engine = new engine_t {kind, sycl_dev, sycl_ctx};

    return status::success;
#else
    UNUSED(engine);
    UNUSED(dev);
    UNUSED(ctx);
    return status::unimplemented;
#endif
}

status_t DNNL_GRAPH_API dnnl_graph_sycl_interop_engine_create_with_allocator(
        engine_t **engine, const void *dev, const void *ctx,
        const allocator_t *alloc) {
#ifdef DNNL_GRAPH_WITH_SYCL
    if (utils::any_null(engine, dev, ctx)) { return status::invalid_arguments; }

    auto &sycl_dev = *static_cast<const ::sycl::device *>(dev);
    auto &sycl_ctx = *static_cast<const ::sycl::context *>(ctx);

    engine_kind_t kind;
    if (sycl_dev.is_gpu()) {
#ifdef DNNL_GRAPH_GPU_SYCL
        kind = engine_kind::gpu;
#else
        return status::invalid_arguments;
#endif
    } else if (sycl_dev.is_cpu() || sycl_dev.is_host()) {
#ifdef DNNL_GRAPH_CPU_SYCL
        kind = engine_kind::cpu;
#else
        return status::invalid_arguments;
#endif
    } else {
        return status::invalid_arguments;
    }

    *engine = new engine_t {kind, sycl_dev, sycl_ctx, alloc};

    return status::success;
#else
    UNUSED(engine);
    UNUSED(dev);
    UNUSED(ctx);
    return status::unimplemented;
#endif
}

status_t DNNL_GRAPH_API dnnl_graph_engine_destroy(engine_t *engine) {
    delete engine;
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_engine_get_kind(
        const engine_t *engine, engine_kind_t *kind) {
    *kind = engine->kind();
    return status::success;
}
