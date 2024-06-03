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

#include <memory>

#include "cpu/sycl/engine.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace sycl {

status_t engine_create(impl::engine_t **engine, const ::sycl::device &dev,
        const ::sycl::context &ctx, size_t index) {
    std::unique_ptr<cpu::sycl::engine_t, engine_deleter_t> e(
            (new cpu::sycl::engine_t(dev, ctx, index)));
    if (!e) return status::out_of_memory;

    CHECK(e->init());
    *engine = e.release();

    return status::success;
}
} // namespace sycl
} // namespace cpu
} // namespace impl
} // namespace dnnl
