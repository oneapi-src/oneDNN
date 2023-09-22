/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#include "sycl/sycl_engine_base.hpp"

#include "common/memory.hpp"
#include "common/memory_storage.hpp"
#include "sycl/sycl_compat.hpp"
#include "sycl/sycl_device_info.hpp"
#include "sycl/sycl_memory_storage.hpp"
#include "sycl/sycl_stream.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

status_t sycl_engine_base_t::create_memory_storage(
        memory_storage_t **storage, unsigned flags, size_t size, void *handle) {
    std::unique_ptr<memory_storage_t> _storage;

    if (flags & memory_flags_t::prefer_device_usm) {
        _storage.reset(new sycl_usm_memory_storage_t(
                this, ::sycl::usm::alloc::device));
    } else
        _storage.reset(new sycl_buffer_memory_storage_t(this));

    if (!_storage) return status::out_of_memory;

    status_t status = _storage->init(flags, size, handle);
    if (status != status::success) return status;

    *storage = _storage.release();
    return status::success;
}

status_t sycl_engine_base_t::create_stream(stream_t **stream, unsigned flags) {
    return sycl_stream_t::create_stream(stream, this, flags);
}
status_t sycl_engine_base_t::create_stream(
        stream_t **stream, ::sycl::queue &queue) {
    return sycl_stream_t::create_stream(stream, this, queue);
}

status_t sycl_engine_base_t::init_device_info() {
    device_info_.reset(new sycl_device_info_t());
    CHECK(device_info_->init(this));
    return status::success;
}

} // namespace sycl
} // namespace impl
} // namespace dnnl
