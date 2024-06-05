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

#include "sycl/sycl_engine_base.hpp"

#include "common/memory.hpp"
#include "common/memory_storage.hpp"
#include "gpu/intel/sycl/compat.hpp"
#include "sycl/sycl_device_info.hpp"
#include "sycl/sycl_stream.hpp"
#include "xpu/sycl/memory_storage.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

status_t sycl_engine_base_t::create_memory_storage(
        memory_storage_t **storage, unsigned flags, size_t size, void *handle) {
    return impl()->create_memory_storage(storage, this, flags, size, handle);
}

status_t sycl_engine_base_t::create_stream(
        impl::stream_t **stream, impl::stream_impl_t *stream_impl) {
    return sycl_stream_t::create_stream(stream, this, stream_impl);
}

status_t sycl_engine_base_t::init_device_info() {
    device_info_.reset(new sycl_device_info_t());
    CHECK(device_info_->init(this));
    return status::success;
}

} // namespace sycl
} // namespace impl
} // namespace dnnl
