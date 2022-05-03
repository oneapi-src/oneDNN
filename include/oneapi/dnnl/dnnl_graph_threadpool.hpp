/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#ifndef ONEAPI_DNNL_DNNL_GRAPH_THREADPOOL_HPP
#define ONEAPI_DNNL_DNNL_GRAPH_THREADPOOL_HPP

#include "oneapi/dnnl/dnnl_graph.hpp"
#include "oneapi/dnnl/dnnl_graph_threadpool.h"
#include "oneapi/dnnl/dnnl_graph_threadpool_iface.hpp"

/// @addtogroup dnnl_graph_api
/// @{

namespace dnnl {
namespace graph {

/// @addtogroup dnnl_graph_api_interop
/// @{

/// @addtogroup dnnl_graph_api_threadpool_interop Threadpool interop API
/// API extensions to interact with the underlying Threadpool run-time.
/// @{

/// Threadpool interoperability namespace
namespace threadpool_interop {

/// Constructs an execution stream for the specified engine and threadpool.
///
/// @param aengine Engine to create the stream on.
/// @param threadpool Pointer to an instance of a C++ class that implements
///     dnnl::graph::threadpool_iface interface.
/// @returns An execution stream.
inline dnnl::graph::stream make_stream(
        const dnnl::graph::engine &aengine, threadpool_iface *threadpool) {
    dnnl_graph_stream_t c_stream = nullptr;
    error::check_succeed(dnnl_graph_threadpool_interop_stream_create(
                                 &c_stream, aengine.get(), threadpool),
            "could not create stream");
    return stream(c_stream);
}

/// Returns the pointer to a threadpool that is used by an execution stream.
///
/// @param astream An execution stream.
/// @returns Output pointer to an instance of a C++ class that implements
///     dnnl::graph::threadpool_iface interface or NULL if the stream was
///     created without threadpool.
inline threadpool_iface *get_threadpool(const dnnl::graph::stream &astream) {
    void *tp;
    error::check_succeed(dnnl_graph_threadpool_interop_stream_get_threadpool(
                                 astream.get(), &tp),
            "could not get stream threadpool");
    return static_cast<threadpool_iface *>(tp);
}

} // namespace threadpool_interop

/// @} dnnl_graph_api_threadpool_interop

/// @} dnnl_graph_api_interop

} // namespace graph
} // namespace dnnl

/// @} dnnl_graph_api

#endif
