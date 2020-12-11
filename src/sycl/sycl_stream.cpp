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

#include "sycl/sycl_stream.hpp"

#include "gpu/ocl/ocl_utils.hpp"
#include "sycl/sycl_engine.hpp"

#include <map>
#include <memory>
#include <CL/cl.h>

namespace dnnl {
namespace impl {
namespace sycl {

status_t sycl_stream_t::init() {
    if ((flags() & stream_flags::in_order) == 0
            && (flags() & stream_flags::out_of_order) == 0)
        return status::invalid_arguments;

    // If queue_ is not set then construct it
    if (!queue_) {
        auto &sycl_engine = *utils::downcast<sycl_engine_base_t *>(engine());
        auto &sycl_ctx = sycl_engine.context();
        auto &sycl_dev = sycl_engine.device();

        cl::sycl::property_list props = (flags() & stream_flags::in_order)
                ? cl::sycl::
                        property_list {cl::sycl::property::queue::in_order {}}
                : cl::sycl::property_list {};
        queue_.reset(new cl::sycl::queue(sycl_ctx, sycl_dev, props));
    } else {
        // TODO: Compare device and context of the engine with those of the
        // queue after SYCL adds support for device/context comparison.
        //
        // For now perform some simple checks.
        auto sycl_dev = queue_->get_device();
        bool args_ok = true
                && IMPLICATION(
                        engine()->kind() == engine_kind::gpu, sycl_dev.is_gpu())
                && IMPLICATION(engine()->kind() == engine_kind::cpu,
                        (sycl_dev.is_cpu() || sycl_dev.is_host()));
        if (!args_ok) return status::invalid_arguments;
    }

    return status::success;
}

} // namespace sycl
} // namespace impl
} // namespace dnnl
