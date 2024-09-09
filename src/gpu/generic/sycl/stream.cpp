/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "common/verbose.hpp"

#include "gpu/generic/sycl/engine.hpp"
#include "gpu/generic/sycl/stream.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

status_t stream_t::init() {
    if ((flags() & stream_flags::in_order) == 0
            && (flags() & stream_flags::out_of_order) == 0)
        return status::invalid_arguments;

    VCONDCHECK(primitive, create, check, stream,
            is_profiling_enabled() == false, status::unimplemented,
            VERBOSE_PROFILING_UNSUPPORTED);

    // If queue_ is not set then construct it
    auto &sycl_engine = *utils::downcast<generic::sycl::engine_t *>(engine());

    if (!impl()->queue()) {
        auto &sycl_ctx = sycl_engine.context();
        auto &sycl_dev = sycl_engine.device();
        ::sycl::property_list prop_list;
        if (flags() & stream_flags::in_order)
            prop_list = {::sycl::property::queue::in_order {}};
        impl()->set_queue(::sycl::queue(sycl_ctx, sycl_dev, prop_list));
    } else {
        auto sycl_dev = queue().get_device();
        bool args_ok = engine()->kind() == engine_kind::gpu
                && (sycl_dev.is_gpu() || sycl_dev.is_accelerator());
        if (!args_ok) return status::invalid_arguments;
    }

    return status::success;
}

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl
