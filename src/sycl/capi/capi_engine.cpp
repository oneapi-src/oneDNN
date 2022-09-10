/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
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
#include "sycl/sycl_engine.hpp"
#include "sycl/sycl_utils.hpp"

using dnnl::impl::engine_t;
using dnnl::impl::status_t;

status_t dnnl_sycl_interop_engine_create(
        engine_t **engine, const void *dev, const void *ctx) {
    using namespace dnnl::impl;
    bool args_ok = !utils::any_null(engine, dev, ctx);
    if (!args_ok) return status::invalid_arguments;

    auto &sycl_dev = *static_cast<const ::sycl::device *>(dev);
    auto &sycl_ctx = *static_cast<const ::sycl::context *>(ctx);

    engine_kind_t kind;
    if (sycl_dev.is_gpu())
        kind = engine_kind::gpu;
    else if (sycl_dev.is_cpu() || dnnl::impl::sycl::is_host(sycl_dev))
        kind = engine_kind::cpu;
    else
        return status::invalid_arguments;

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
    if (kind == engine_kind::cpu) return status::invalid_arguments;
#endif

    auto ef = dnnl::impl::sycl::get_engine_factory(kind);
    if (!ef) return status::invalid_arguments;

    size_t index;
    CHECK(dnnl::impl::sycl::get_sycl_device_index(&index, sycl_dev));

    return ef->engine_create(engine, sycl_dev, sycl_ctx, index);
}

status_t dnnl_sycl_interop_engine_get_context(engine_t *engine, void **ctx) {
    using namespace dnnl::impl;
    bool args_ok = true && !utils::any_null(ctx, engine)
            && engine->runtime_kind() == runtime_kind::sycl;

    if (!args_ok) return status::invalid_arguments;

    auto *sycl_engine
            = utils::downcast<dnnl::impl::sycl::sycl_engine_base_t *>(engine);
    auto &sycl_ctx = const_cast<::sycl::context &>(sycl_engine->context());
    *ctx = static_cast<void *>(&sycl_ctx);
    return status::success;
}

status_t dnnl_sycl_interop_engine_get_device(engine_t *engine, void **dev) {
    using namespace dnnl::impl;
    bool args_ok = true && !utils::any_null(dev, engine)
            && engine->runtime_kind() == runtime_kind::sycl;

    if (!args_ok) return status::invalid_arguments;

    auto *sycl_engine
            = utils::downcast<dnnl::impl::sycl::sycl_engine_base_t *>(engine);
    auto &sycl_dev = const_cast<::sycl::device &>(sycl_engine->device());
    *dev = static_cast<void *>(&sycl_dev);
    return status::success;
}
