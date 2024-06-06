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

#include "common/engine.hpp"

#include "xpu/sycl/engine_impl.hpp"
#include "xpu/sycl/memory_storage.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace sycl {

status_t engine_impl_t::create_memory_storage(memory_storage_t **storage,
        engine_t *engine, unsigned flags, size_t size, void *handle) const {
    std::unique_ptr<memory_storage_t> _storage;

    // This interface is expected to be used by the engine that holds this
    // engine_impl_t. Other use cases are possible but not expected at this
    // point.
    assert(engine->impl() == this);

    if (flags & memory_flags_t::prefer_device_usm) {
        _storage.reset(new xpu::sycl::usm_memory_storage_t(
                engine, ::sycl::usm::alloc::device));
    } else
        _storage.reset(new xpu::sycl::buffer_memory_storage_t(engine));

    if (!_storage) return status::out_of_memory;

    status_t status = _storage->init(flags, size, handle);
    if (status != status::success) return status;

    *storage = _storage.release();
    return status::success;
}

} // namespace sycl
} // namespace xpu
} // namespace impl
} // namespace dnnl
