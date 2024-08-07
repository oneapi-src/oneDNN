/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#include "oneapi/dnnl/dnnl_ocl.h"

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/memory.hpp"
#include "common/primitive_desc_iface.hpp"
#include "common/primitive_iface.hpp"
#include "common/utils.hpp"

#include "xpu/ocl/c_types_map.hpp"
#include "xpu/ocl/memory_storage.hpp"
#include "xpu/ocl/stream_impl.hpp"
#include "xpu/ocl/utils.hpp"

using namespace dnnl::impl;

status_t dnnl_ocl_interop_primitive_execute(
        const primitive_iface_t *primitive_iface, stream_t *stream, int nargs,
        const dnnl_exec_arg_t *args, const cl_event *deps, int ndeps,
        cl_event *return_event) {
    const bool ok = !utils::any_null(primitive_iface, stream)
            && primitive_iface->engine() == stream->engine()
            && primitive_iface->engine()->runtime_kind() == runtime_kind::ocl
            && IMPLICATION(nargs > 0, args != nullptr)
            && IMPLICATION(ndeps > 0, deps != nullptr);
    if (!ok) return status::invalid_arguments;

    auto *ocl_stream_impl
            = utils::downcast<xpu::ocl::stream_impl_t *>(stream->impl());

    stream->before_exec_hook();

    if (deps != nullptr) {
        std::vector<xpu::ocl::wrapper_t<cl_event>> events(ndeps);
        for (int i = 0; i < ndeps; i++) {
            events[i] = xpu::ocl::wrapper_t<cl_event>(deps[i], true);
        }
        ocl_stream_impl->ocl_ctx().set_deps(events);
    }

    // run primitive
    exec_args_t exec_args;
    CHECK(cvt_primitive_args(
            primitive_iface->pd()->impl().get(), nargs, args, exec_args));

    exec_ctx_t ctx(stream, std::move(exec_args));
    CHECK(primitive_execute(primitive_iface, ctx));

    // return output event
    if (return_event != nullptr) {
        if (ocl_stream_impl->flags() & stream_flags::in_order) {
            *return_event = nullptr;
        } else {
            auto last_event = ocl_stream_impl->get_output_event();
            *return_event = last_event.release();
        }
    }

    stream->after_exec_hook();

    return status::success;
}
