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

#include "oneapi/dnnl/dnnl_sycl.h"

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/stream.hpp"
#include "common/utils.hpp"

#include "xpu/sycl/stream_impl.hpp"

using dnnl::impl::engine_t;
using dnnl::impl::status_t;
using dnnl::impl::stream_t;

status_t dnnl_sycl_interop_stream_create(
        stream_t **stream, engine_t *engine, void *queue) {
    using namespace dnnl::impl;
    bool args_ok = true && !utils::any_null(stream, engine, queue)
            && engine->runtime_kind() == runtime_kind::sycl;
    if (!args_ok) return status::invalid_arguments;

    auto &sycl_queue = *static_cast<::sycl::queue *>(queue);

    unsigned flags;
    CHECK(dnnl::impl::xpu::sycl::stream_impl_t::init_flags(&flags, sycl_queue));

    std::unique_ptr<dnnl::impl::stream_impl_t> stream_impl(
            new dnnl::impl::xpu::sycl::stream_impl_t(sycl_queue, flags));
    if (!stream_impl) return status::out_of_memory;

    CHECK(engine->create_stream(stream, stream_impl.get()));
    // `create_stream` captures `stream_impl_ptr` internally. To avoid double
    // free of the same object, `stream_impl` releases it and delegates freeing
    // part to the stream.
    stream_impl.release();
    return status::success;
}

status_t dnnl_sycl_interop_stream_get_queue(stream_t *stream, void **queue) {
    using namespace dnnl::impl;
    bool args_ok = true && !utils::any_null(queue, stream)
            && stream->engine()->runtime_kind() == runtime_kind::sycl;

    if (!args_ok) return status::invalid_arguments;

    auto *sycl_stream_impl
            = utils::downcast<dnnl::impl::xpu::sycl::stream_impl_t *>(
                    stream->impl());
    auto &sycl_queue = *sycl_stream_impl->queue();
    *queue = static_cast<void *>(&sycl_queue);
    return status::success;
}
