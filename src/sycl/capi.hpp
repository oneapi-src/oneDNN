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

#ifndef CAPI_HPP
#define CAPI_HPP

#include "mkldnn.h"

extern "C" {

/** Creates an @p engine of particuler @p kind associated with given SYCL
 * device and context objects. */
mkldnn_status_t mkldnn_engine_create_sycl(mkldnn_engine_t *engine,
        mkldnn_engine_kind_t kind, const void *device, const void *context);

/** Returns a SYCL context associated with @p engine. */
mkldnn_status_t mkldnn_engine_get_sycl_context(
        mkldnn_engine_t engine, void **ctx);

/** Returns a SYCL device associated with @p engine. */
mkldnn_status_t mkldnn_engine_get_sycl_device(
        mkldnn_engine_t engine, void **dev);

/** Creates an execution @p stream for given @p engine associated with SYCL
 * queue object @p queue.  */
mkldnn_status_t mkldnn_stream_create_sycl(
        mkldnn_stream_t *stream, mkldnn_engine_t engine, void *queue);

/** Returns a SYCL command queue associated with @p stream. */
mkldnn_status_t mkldnn_stream_get_sycl_queue(
        mkldnn_stream_t stream, void **queue);

}

#endif
