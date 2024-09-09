/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#include <memory>

#include <CL/cl.h>

#include "oneapi/dnnl/dnnl_ocl.h"

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/stream.hpp"
#include "common/utils.hpp"

#include "xpu/ocl/stream_impl.hpp"

using namespace dnnl::impl;

status_t dnnl_ocl_interop_stream_create(
        stream_t **stream, engine_t *engine, cl_command_queue queue) {
    bool args_ok = !utils::any_null(stream, engine, queue)
            && engine->runtime_kind() == runtime_kind::ocl;

    if (!args_ok) return status::invalid_arguments;

    unsigned flags;
    CHECK(dnnl::impl::xpu::ocl::stream_impl_t::init_flags(&flags, queue));

    std::unique_ptr<dnnl::impl::stream_impl_t> stream_impl(
            new dnnl::impl::xpu::ocl::stream_impl_t(queue, flags));
    if (!stream_impl) return status::out_of_memory;

    CHECK(engine->create_stream(stream, stream_impl.get()));
    // stream (`s`) takes ownership of `stream_impl` if `create_stream` call
    // is successful.
    stream_impl.release();
    return status::success;
}

status_t dnnl_ocl_interop_stream_get_command_queue(
        stream_t *stream, cl_command_queue *queue) {
    bool args_ok = !utils::any_null(queue, stream)
            && stream->engine()->runtime_kind() == runtime_kind::ocl;

    if (!args_ok) return status::invalid_arguments;

    const auto *ocl_stream_impl
            = utils::downcast<const xpu::ocl::stream_impl_t *>(stream->impl());
    *queue = const_cast<xpu::ocl::stream_impl_t *>(ocl_stream_impl)->queue();
    return status::success;
}
