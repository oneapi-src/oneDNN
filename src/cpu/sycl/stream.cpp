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

#include "cpu/sycl/stream.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace sycl {

status_t stream_t::init() {
    if ((flags() & stream_flags::in_order) == 0
            && (flags() & stream_flags::out_of_order) == 0)
        return status::invalid_arguments;

    const auto &sycl_engine_impl
            = *utils::downcast<const xpu::sycl::engine_impl_t *>(
                    engine()->impl());
    auto &sycl_ctx = sycl_engine_impl.context();
    auto &sycl_dev = sycl_engine_impl.device();

    // If queue_ is not set then construct it
    if (!impl()->queue()) {
        const auto props = (flags() & stream_flags::in_order)
                ? ::sycl::property_list {::sycl::property::queue::in_order {}}
                : ::sycl::property_list {};
        impl()->set_queue(::sycl::queue(sycl_ctx, sycl_dev, props));
    } else {
        // TODO: Compare device and context of the engine with those of the
        // queue after SYCL adds support for device/context comparison.
        //
        // For now perform some simple checks.
        auto sycl_dev = queue().get_device();
        const bool args_ok = engine()->kind() == engine_kind::cpu
                && (sycl_dev.is_cpu() || xpu::sycl::is_host(sycl_dev));
        if (!args_ok) return status::invalid_arguments;
    }

    return status::success;
}

void stream_t::after_exec_hook() {
    sycl_ctx().set_deps(xpu::sycl::event_t());
}

} // namespace sycl
} // namespace cpu
} // namespace impl
} // namespace dnnl
