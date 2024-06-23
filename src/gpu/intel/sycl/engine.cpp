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

#include "common/memory.hpp"

#include "xpu/sycl/memory_storage.hpp"

#include "gpu/intel/sycl/compat.hpp"
#include "gpu/intel/sycl/device_info.hpp"
#include "gpu/intel/sycl/engine.hpp"
#include "gpu/intel/sycl/stream.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace sycl {

status_t engine_create(impl::engine_t **engine, engine_kind_t engine_kind,
        const ::sycl::device &dev, const ::sycl::context &ctx, size_t index) {
    std::unique_ptr<intel::sycl::engine_t, engine_deleter_t> e(
            (new intel::sycl::engine_t(dev, ctx, index)));
    if (!e) return status::out_of_memory;

    CHECK(e->init());
    *engine = e.release();

    return status::success;
}

status_t engine_t::create_memory_storage(
        memory_storage_t **storage, unsigned flags, size_t size, void *handle) {
    return impl()->create_memory_storage(storage, this, flags, size, handle);
}

status_t engine_t::create_stream(
        impl::stream_t **stream, impl::stream_impl_t *stream_impl) {
    return gpu::intel::sycl::stream_t::create_stream(stream, this, stream_impl);
}

status_t engine_t::init_device_info() {
    device_info_.reset(new gpu::intel::sycl::device_info_t());
    CHECK(device_info_->init(this));
    return status::success;
}

} // namespace sycl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
