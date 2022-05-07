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

#ifndef ONEAPI_DNNL_DNNL_GRAPH_THREADPOOL_H
#define ONEAPI_DNNL_DNNL_GRAPH_THREADPOOL_H

#include "oneapi/dnnl/dnnl_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @addtogroup dnnl_graph_api
/// @{

/// @addtogroup dnnl_graph_api_interop
/// @{

/// @addtogroup dnnl_graph_api_threadpool_interop
/// @{

/// Creates an execution stream with specified threadpool.
///
/// @param stream Output execution stream.
/// @param engine Engine to create the execution stream on.
/// @param threadpool Pointer to an instance of a C++ class that implements
///     dnnl::graph::threadpool_iface interface.
/// @returns #dnnl_graph_success on success and a status describing the
///     error otherwise.
dnnl_graph_status_t DNNL_GRAPH_API dnnl_graph_threadpool_interop_stream_create(
        dnnl_graph_stream_t *stream, const_dnnl_graph_engine_t engine,
        void *threadpool);

/// Returns a threadpool to be used by the execution stream.
///
/// @param astream Execution stream.
/// @param threadpool Output pointer to an instance of a C++ class that
///     implements dnnl::graph::threadpool_iface interface. Set to NULL if the
///     stream was created without threadpool.
/// @returns #dnnl_graph_success on success and a status describing the
///     error otherwise.
dnnl_graph_status_t DNNL_GRAPH_API
dnnl_graph_threadpool_interop_stream_get_threadpool(
        dnnl_graph_stream_t astream, void **threadpool);

/// @} dnnl_graph_api_threadpool_interop

/// @} dnnl_graph_api_interop

/// @} dnnl_graph_api

#ifdef __cplusplus
}
#endif

#endif
