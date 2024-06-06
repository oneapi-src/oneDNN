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

#ifndef SYCL_ENGINE_FACTORY_HPP
#define SYCL_ENGINE_FACTORY_HPP

#include <algorithm>
#include <assert.h>
#include <cstdio>
#include <exception>
#include <memory>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/utils.hpp"
#include "gpu/intel/sycl/utils.hpp"
#include "gpu/sycl/sycl_gpu_engine.hpp"
#include "xpu/sycl/utils.hpp"

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
#include "cpu/sycl/engine.hpp"
#endif

#ifdef DNNL_SYCL_CUDA
#include "gpu/nvidia/engine.hpp"
#endif

namespace dnnl {
namespace impl {

#ifdef DNNL_SYCL_HIP
// XXX: forward declarations to avoid cuda dependencies on sycl level.
namespace gpu {
namespace amd {

status_t hip_engine_create(impl::engine_t **engine, engine_kind_t engine_kind,
        const ::sycl::device &dev, const ::sycl::context &ctx, size_t index);

} // namespace amd
} // namespace gpu
#endif

namespace sycl {

class sycl_engine_factory_t : public engine_factory_t {
public:
    sycl_engine_factory_t(engine_kind_t engine_kind)
        : engine_kind_(engine_kind) {
        assert(utils::one_of(engine_kind_, engine_kind::cpu, engine_kind::gpu));
    }

    size_t count() const override {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_NONE
        if (engine_kind_ == engine_kind::cpu) return 0;
#endif
        auto dev_type = (engine_kind_ == engine_kind::cpu)
                ? ::sycl::info::device_type::cpu
                : ::sycl::info::device_type::gpu;
        return xpu::sycl::get_devices(dev_type).size();
    }

    status_t engine_create(engine_t **engine, size_t index) const override;

    status_t engine_create(engine_t **engine, const ::sycl::device &dev,
            const ::sycl::context &ctx, size_t index) const;

private:
    engine_kind_t engine_kind_;
};

inline std::unique_ptr<sycl_engine_factory_t> get_engine_factory(
        engine_kind_t engine_kind) {
    return std::unique_ptr<sycl_engine_factory_t>(
            new sycl_engine_factory_t(engine_kind));
}

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif
