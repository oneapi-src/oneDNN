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

#include <CL/sycl.hpp>

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/utils.hpp"
#include "sycl/capi.hpp"
#include "sycl/sycl_engine.hpp"

using namespace dnnl::impl;

status_t dnnl_engine_create_sycl(engine_t **engine, engine_kind_t kind,
        const void *dev, const void *ctx) {
    bool args_ok = !utils::any_null(engine, dev, ctx);
    if (!args_ok) return status::invalid_arguments;

    auto ef = dnnl::impl::sycl::get_engine_factory(kind);
    if (!ef) return status::invalid_arguments;

    auto &sycl_dev = *static_cast<const cl::sycl::device *>(dev);
    auto &sycl_ctx = *static_cast<const cl::sycl::context *>(ctx);
    return ef->engine_create(engine, sycl_dev, sycl_ctx);
}

status_t dnnl_engine_get_sycl_context(engine_t *engine, void **ctx) {
    bool args_ok = true && !utils::any_null(ctx, engine)
            && engine->runtime_kind() == runtime_kind::sycl;

    if (!args_ok) return status::invalid_arguments;

    auto *sycl_engine
            = utils::downcast<dnnl::impl::sycl::sycl_engine_base_t *>(engine);
    auto &sycl_ctx = const_cast<cl::sycl::context &>(sycl_engine->context());
    *ctx = static_cast<void *>(&sycl_ctx);
    return status::success;
}

status_t dnnl_engine_get_sycl_device(engine_t *engine, void **dev) {
    bool args_ok = true && !utils::any_null(dev, engine)
            && engine->runtime_kind() == runtime_kind::sycl;

    if (!args_ok) return status::invalid_arguments;

    auto *sycl_engine
            = utils::downcast<dnnl::impl::sycl::sycl_engine_base_t *>(engine);
    auto &sycl_dev = const_cast<cl::sycl::device &>(sycl_engine->device());
    *dev = static_cast<void *>(&sycl_dev);
    return status::success;
}
