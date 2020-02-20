/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#ifndef CAPI_HPP
#define CAPI_HPP

#include "dnnl.h"

extern "C" {

/** Creates an @p engine of particuler @p kind associated with given SYCL
 * device and context objects. */
dnnl_status_t dnnl_engine_create_sycl(dnnl_engine_t *engine,
        dnnl_engine_kind_t kind, const void *device, const void *context);

/** Returns a SYCL context associated with @p engine. */
dnnl_status_t dnnl_engine_get_sycl_context(dnnl_engine_t engine, void **ctx);

/** Returns a SYCL device associated with @p engine. */
dnnl_status_t dnnl_engine_get_sycl_device(dnnl_engine_t engine, void **dev);

/** Creates an execution @p stream for given @p engine associated with SYCL
 * queue object @p queue.  */
dnnl_status_t dnnl_stream_create_sycl(
        dnnl_stream_t *stream, dnnl_engine_t engine, void *queue);

/** Returns a SYCL command queue associated with @p stream. */
dnnl_status_t dnnl_stream_get_sycl_queue(dnnl_stream_t stream, void **queue);
}

#endif
