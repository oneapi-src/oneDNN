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

#include "oneapi/dnnl/dnnl_sycl.h"

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/utils.hpp"

#include "xpu/sycl/engine_factory.hpp"
#include "xpu/sycl/utils.hpp"

using dnnl::impl::engine_t;
using dnnl::impl::status_t;

status_t dnnl_sycl_interop_engine_create(
        engine_t **engine, const void *dev, const void *ctx) {
    using namespace dnnl::impl;
    bool args_ok = !utils::any_null(engine, dev, ctx);
    VERROR_ENGINE(args_ok, status::invalid_arguments, VERBOSE_NULL_ARG);

    auto &sycl_dev = *static_cast<const ::sycl::device *>(dev);
    auto &sycl_ctx = *static_cast<const ::sycl::context *>(ctx);

    engine_kind_t kind;
    if (sycl_dev.is_gpu())
        kind = engine_kind::gpu;
    else if (sycl_dev.is_cpu() || dnnl::impl::xpu::sycl::is_host(sycl_dev))
        kind = engine_kind::cpu;
    else
        VERROR_ENGINE(
                false, status::invalid_arguments, VERBOSE_BAD_ENGINE_KIND);

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
    VERROR_ENGINE(kind != engine_kind::cpu, status::invalid_arguments,
            VERBOSE_BAD_ENGINE_KIND);
#endif

    auto ef = dnnl::impl::xpu::sycl::get_engine_factory(kind);
    VERROR_ENGINE(ef, status::invalid_arguments, VERBOSE_BAD_ENGINE_KIND);

    size_t index;
    CHECK(dnnl::impl::xpu::sycl::get_device_index(&index, sycl_dev));

    return ef->engine_create(engine, sycl_dev, sycl_ctx, index);
}

status_t dnnl_sycl_interop_engine_get_context(engine_t *engine, void **ctx) {
    using namespace dnnl::impl;
    bool args_ok = true && !utils::any_null(ctx, engine)
            && engine->runtime_kind() == runtime_kind::sycl;

    if (!args_ok) return status::invalid_arguments;

    const auto *sycl_engine_impl
            = utils::downcast<const dnnl::impl::xpu::sycl::engine_impl_t *>(
                    engine->impl());
    auto &sycl_ctx = const_cast<::sycl::context &>(sycl_engine_impl->context());
    *ctx = static_cast<void *>(&sycl_ctx);
    return status::success;
}

status_t dnnl_sycl_interop_engine_get_device(engine_t *engine, void **dev) {
    using namespace dnnl::impl;
    bool args_ok = true && !utils::any_null(dev, engine)
            && engine->runtime_kind() == runtime_kind::sycl;

    if (!args_ok) return status::invalid_arguments;

    const auto *sycl_engine_impl
            = utils::downcast<const dnnl::impl::xpu::sycl::engine_impl_t *>(
                    engine->impl());
    auto &sycl_dev = const_cast<::sycl::device &>(sycl_engine_impl->device());
    *dev = static_cast<void *>(&sycl_dev);
    return status::success;
}
