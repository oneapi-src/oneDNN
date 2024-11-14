/*******************************************************************************
* Copyright 2016-2024 Intel Corporation
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

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "engine.hpp"
#include "memory.hpp"
#include "nstl.hpp"
#include "primitive.hpp"
#include "utils.hpp"

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
#include "cpu/cpu_engine.hpp"
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "xpu/ocl/engine_factory.hpp"
#endif

#ifdef DNNL_WITH_SYCL
#include "xpu/sycl/engine_factory.hpp"
#endif

namespace dnnl {
namespace impl {

static inline std::unique_ptr<engine_factory_t> get_engine_factory(
        engine_kind_t kind, runtime_kind_t runtime_kind) {

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    if (kind == engine_kind::cpu && is_native_runtime(runtime_kind)) {
        return std::unique_ptr<engine_factory_t>(
                new cpu::cpu_engine_factory_t());
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    if (kind == engine_kind::gpu && runtime_kind == runtime_kind::ocl) {
        return std::unique_ptr<engine_factory_t>(
                new xpu::ocl::engine_factory_t(kind));
    }
#endif
#ifdef DNNL_WITH_SYCL
    if (runtime_kind == runtime_kind::sycl)
        return xpu::sycl::get_engine_factory(kind);
#endif
    return nullptr;
}

} // namespace impl
} // namespace dnnl

using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;

using dnnl::impl::engine_kind_t;
using dnnl::impl::engine_t;
using dnnl::impl::status_t;

size_t dnnl_engine_get_count(engine_kind_t kind) {
    using namespace dnnl::impl;
    auto ef = get_engine_factory(kind, get_default_runtime(kind));
    return ef != nullptr ? ef->count() : 0;
}

status_t dnnl_engine_create(
        engine_t **engine, engine_kind_t kind, size_t index) {
    using namespace dnnl::impl;
    VERROR_ENGINE(engine != nullptr, invalid_arguments, VERBOSE_NULL_ARG);

    // engine_factory creation can fail with an exception
    try {
        auto ef = get_engine_factory(kind, get_default_runtime(kind));
        VERROR_ENGINE(ef != nullptr, invalid_arguments,
                VERBOSE_INVALID_ENGINE_KIND, "", dnnl_engine_kind2str(kind));

        auto s_runtime_kind = dnnl_runtime2str(kind == engine_kind::cpu
                        ? dnnl_version()->cpu_runtime
                        : dnnl_version()->gpu_runtime);

        VERROR_ENGINE(ef->count() > 0, invalid_arguments,
                "%s %s devices queried but not found",
                get_default_runtime(kind) == runtime_kind::none
                        ? ""
                        : s_runtime_kind,
                dnnl_engine_kind2str(kind));

        VERROR_ENGINE(index < ef->count(), invalid_arguments,
                VERBOSE_INVALID_ENGINE_IDX, ef->count(),
                dnnl_engine_kind2str(kind), index);

        const status_t engine_status = ef->engine_create(engine, index);

        if (engine_status != success) {
            VERROR(common, runtime, VERBOSE_ENGINE_CREATION_FAIL,
                    dnnl_engine_kind2str(kind), index);
        }

        return engine_status;
    } catch (...) {
        VERROR(common, runtime, VERBOSE_INVALID_DEVICE_ENV,
                dnnl_engine_kind2str(kind), index);
        return runtime_error;
    }
}

status_t dnnl_engine_get_kind(engine_t *engine, engine_kind_t *kind) {
    using namespace dnnl::impl;
    if (engine == nullptr) return invalid_arguments;
    *kind = engine->kind();
    return success;
}

status_t dnnl_engine_destroy(engine_t *engine) {
    if (engine != nullptr) engine->release();
    return success;
}

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
