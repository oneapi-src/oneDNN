/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#include "oneapi/dnnl/dnnl_sycl.h"

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/primitive_desc_iface.hpp"
#include "common/primitive_iface.hpp"
#include "common/utils.hpp"

#include "xpu/sycl/engine_factory.hpp"

using dnnl::impl::status_t;
using dnnl::impl::stream_t;

status_t dnnl_sycl_interop_primitive_execute(
        const primitive_iface_t *primitive_iface, stream_t *stream, int nargs,
        const dnnl_exec_arg_t *args, const void *deps_, void *return_event_) {
    using namespace dnnl::impl;
    bool ok = !utils::any_null(primitive_iface, stream)
            && primitive_iface->engine() == stream->engine()
            && primitive_iface->engine()->runtime_kind() == runtime_kind::sycl
            && IMPLICATION(nargs > 0, args != nullptr);
    if (!ok) return status::invalid_arguments;

    auto *sycl_stream_impl
            = utils::downcast<dnnl::impl::xpu::sycl::stream_impl_t *>(
                    stream->impl());

    stream->before_exec_hook();

    if (deps_ != nullptr) {
        auto deps = dnnl::impl::xpu::sycl::event_t(
                *(const std::vector<::sycl::event> *)deps_);
        sycl_stream_impl->sycl_ctx().set_deps(std::move(deps));
    }

    // run primitive
    exec_args_t exec_args;
    CHECK(cvt_primitive_args(
            primitive_iface->pd()->impl().get(), nargs, args, exec_args));

    exec_ctx_t ctx(stream, std::move(exec_args));
    CHECK(primitive_execute(primitive_iface, ctx));

    // return output event
    ::sycl::event return_event = sycl_stream_impl->get_output_event();
    if (return_event_ != nullptr)
        *(::sycl::event *)return_event_ = return_event;

    stream->after_exec_hook();

    return status::success;
}
