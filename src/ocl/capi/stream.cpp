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

#include "mkldnn.h"

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/stream.hpp"
#include "common/utils.hpp"
#include "ocl/ocl_engine.hpp"
#include "ocl/ocl_stream.hpp"

using namespace mkldnn::impl;
using namespace mkldnn::impl::ocl;

status_t mkldnn_stream_create_ocl(
        stream_t **stream, engine_t *engine, cl_command_queue queue) {
    bool args_ok = true
            && !utils::any_null(stream, engine, queue)
            && engine->backend_kind() == backend_kind::ocl;

    if (!args_ok)
        return status::invalid_arguments;

    auto *ocl_engine = utils::downcast<ocl_engine_t *>(engine);
    return ocl_engine->create_stream(stream, queue);
}

status_t mkldnn_stream_get_ocl_command_queue(
        stream_t *stream, cl_command_queue *queue) {
    bool args_ok = true
            && !utils::any_null(queue, stream)
            && stream->engine()->backend_kind() == backend_kind::ocl;

    if (!args_ok)
        return status::invalid_arguments;

    auto *ocl_stream = utils::downcast<ocl_stream_t *>(stream);
    *queue = ocl_stream->queue();
    return status::success;
}
