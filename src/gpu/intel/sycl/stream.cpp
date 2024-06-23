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

#include <map>
#include <memory>
#include <CL/cl.h>

#include "common/verbose.hpp"

#include "xpu/sycl/stream_profiler.hpp"

#include "gpu/intel/sycl/stream.hpp"

#include "gpu/intel/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace sycl {

status_t stream_t::init() {
    if ((flags() & stream_flags::in_order) == 0
            && (flags() & stream_flags::out_of_order) == 0)
        return status::invalid_arguments;

    if (is_profiling_enabled())
        profiler_ = utils::make_unique<xpu::sycl::stream_profiler_t>(this);

    const auto &sycl_engine_impl
            = *utils::downcast<const xpu::sycl::engine_impl_t *>(
                    engine()->impl());
    auto &sycl_ctx = sycl_engine_impl.context();
    auto &sycl_dev = sycl_engine_impl.device();

    // If queue_ is not set then construct it
    if (!impl()->queue()) {
        ::sycl::property_list props;
        if (is_profiling_enabled() && sycl_dev.is_gpu()) {
            props = (flags() & stream_flags::in_order)
                    ? ::sycl::property_list {::sycl::property::queue::
                                                     in_order {},
                            ::sycl::property::queue::enable_profiling {}}
                    : ::sycl::property_list {
                            ::sycl::property::queue::enable_profiling {}};
        } else {
            props = (flags() & stream_flags::in_order)
                    ? ::sycl::
                            property_list {::sycl::property::queue::in_order {}}
                    : ::sycl::property_list {};
        }
        impl()->set_queue(::sycl::queue(sycl_ctx, sycl_dev, props));
    } else {
        // TODO: Compare device and context of the engine with those of the
        // queue after SYCL adds support for device/context comparison.
        //
        // For now perform some simple checks.
        auto sycl_dev = queue().get_device();
        bool args_ok = true
                && IMPLICATION(
                        engine()->kind() == engine_kind::gpu, sycl_dev.is_gpu())
                && IMPLICATION(engine()->kind() == engine_kind::cpu,
                        (sycl_dev.is_cpu() || xpu::sycl::is_host(sycl_dev)));
        if (!args_ok) return status::invalid_arguments;
    }

    if (is_profiling_enabled() && sycl_dev.is_gpu() && !queue().is_in_order()) {
        VERROR(common, dpcpp,
                "DPC++ kernel profiling is not supported with out-of-order "
                "queues");
        return status::invalid_arguments;
    }

    return status::success;
}

void stream_t::before_exec_hook() {
    if (is_profiling_enabled()) profiler_->start_profiling();
}

void stream_t::after_exec_hook() {
    sycl_ctx().set_deps(xpu::sycl::event_t());
    if (is_profiling_enabled()) profiler_->stop_profiling();
}

} // namespace sycl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
