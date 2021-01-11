/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include <vector>

#include "oneapi/dnnl/dnnl_graph.h"
#include "oneapi/dnnl/dnnl_graph_sycl.h"

#include "c_types_map.hpp"
#include "partition.hpp"
#include "stream.hpp"
#include "utils.hpp"

#if DNNL_GRAPH_WITH_SYCL
#include <CL/sycl.hpp>
#endif

using namespace dnnl::graph::impl;

///
/// \brief dnnl_graph_thread_pool_t
///
status_t DNNL_GRAPH_API dnnl_graph_thread_pool_create(
        thread_pool_t **created_thread_pool, int32_t num_threads) {
    *created_thread_pool = new thread_pool_t {num_threads};
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_thread_pool_destroy(
        thread_pool_t *thread_pool) {
    delete thread_pool;
    return status::success;
}

///
/// \brief dnnl_graph_stream_attr_t
///
status_t DNNL_GRAPH_API dnnl_graph_stream_attr_create(
        stream_attr_t **created_stream_attr, thread_pool_t *thread_pool) {
    *created_stream_attr = new stream_attr_t {thread_pool};
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_stream_attr_destroy(
        stream_attr_t *stream_attr) {
    delete stream_attr;
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_stream_create(
        stream_t **created_stream, const engine_t *engine) {
    *created_stream = new stream_t {engine};
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_sycl_interop_stream_create(
        stream_t **created_stream, const engine_t *engine, const void *queue) {
#if DNNL_GRAPH_WITH_SYCL
    if (utils::any_null(created_stream, engine, queue)) {
        return status::invalid_argument;
    }
    auto &sycl_queue = *static_cast<const cl::sycl::queue *>(queue);
    *created_stream = new stream_t {engine, sycl_queue};
    return status::success;
#else
    UNUSED(created_stream);
    UNUSED(engine);
    UNUSED(queue);
    return status::unsupported;
#endif
}

status_t DNNL_GRAPH_API dnnl_graph_stream_create_with_attr(
        stream_t **created_stream, const engine_t *engine,
        const stream_attr_t *attr) {
    *created_stream = new stream_t {engine, attr};
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_stream_create_sycl_with_attr(
        stream_t **created_stream, const engine_t *engine, const void *queue,
        const stream_attr_t *attr) {
#if DNNL_GRAPH_WITH_SYCL
    if (utils::any_null(created_stream, engine, queue, attr)) {
        return status::invalid_argument;
    }
    auto &sycl_queue = *static_cast<const cl::sycl::queue *>(queue);
    *created_stream = new stream_t {engine, sycl_queue, attr};
    return status::success;
#else
    UNUSED(created_stream);
    UNUSED(engine);
    UNUSED(queue);
    UNUSED(attr);
    return status::unsupported;
#endif
}

status_t DNNL_GRAPH_API dnnl_graph_stream_destroy(stream_t *stream) {
    delete stream;
    return status::success;
}
